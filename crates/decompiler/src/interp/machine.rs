//! The execution machine: operand access, stepping, and D3D math.

use dxbc::shex::{ComponentSelect, Instruction, Opcode, Operand, OperandIndex, RegisterType};

use super::{Halt, Lanes, Machine, missing};

impl Machine<'_> {
    /// Evaluates a conditional opcode's scalar test.
    pub(super) fn test(
        &mut self,
        instruction: &Instruction,
        operands: &[Operand],
    ) -> Result<bool, Halt> {
        let operand = operands.first().ok_or_else(|| missing("test operand"))?;
        let value = self.read(operand)?;
        Ok((value[0] != 0) == instruction.test_nonzero)
    }

    /// Reads one source operand, fully applying selection and modifiers.
    pub(super) fn read(&mut self, operand: &Operand) -> Result<Lanes, Halt> {
        let base = self.locate(operand)?;
        let mut lanes: Lanes = match &operand.components {
            ComponentSelect::Swizzle(swizzle) => {
                core::array::from_fn(|lane| base[usize::from(swizzle[lane])])
            }
            ComponentSelect::Scalar(select) => [base[usize::from(*select)]; 4],
            ComponentSelect::Mask(_)
            | ComponentSelect::OneComponent
            | ComponentSelect::ZeroComponent => base,
        };
        if operand.abs {
            for lane in &mut lanes {
                *lane &= 0x7fff_ffff;
            }
        }
        if operand.negate {
            for lane in &mut lanes {
                *lane ^= 0x8000_0000;
            }
        }
        // The float modifiers imply float meaning: canonicalize NaN so a
        // negate hoisted across a NaN-producing op stays comparable.
        if operand.abs || operand.negate {
            for lane in &mut lanes {
                *lane = canonical(*lane);
            }
        }
        Ok(lanes)
    }

    /// Resolves an operand index that may be relative.
    pub(super) fn dynamic_index(
        &mut self,
        operand: &Operand,
        position: usize,
    ) -> Result<u32, Halt> {
        match operand.indices.get(position) {
            Some(OperandIndex::Imm32(value)) => Ok(*value),
            Some(OperandIndex::Relative(inner)) => Ok(self.read(inner)?[0]),
            Some(OperandIndex::RelativePlusImm(base, inner)) => {
                Ok(self.read(inner)?[0].wrapping_add(*base))
            }
            None => Ok(0),
            Some(other) => Err(Halt::Unsupported(format!("operand index {other:?}"))),
        }
    }

    /// The raw value of one operand's register, before selection.
    pub(super) fn locate(&mut self, operand: &Operand) -> Result<Lanes, Halt> {
        match operand.reg_type {
            RegisterType::Temp => {
                let index = self.dynamic_index(operand, 0)?;
                Ok(self.temps.get(&index).copied().unwrap_or([0; 4]))
            }
            RegisterType::Input | RegisterType::InputControlPoint => {
                if operand.indices.len() == 2 {
                    // Geometry-shader input: v[vertex][register].
                    let vertex = self.dynamic_index(operand, 0)?;
                    let index = self.dynamic_index(operand, 1)?;
                    let key = self
                        .keys
                        .inputs
                        .get(&index)
                        .copied()
                        .unwrap_or(u64::from(index));
                    let mixed = key ^ u64::from(vertex).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                    return Ok(self.world.value(1, mixed));
                }
                let index = self.dynamic_index(operand, 0)?;
                let key = self
                    .keys
                    .inputs
                    .get(&index)
                    .copied()
                    .unwrap_or(u64::from(index));
                Ok(self.world.value(1, key))
            }
            RegisterType::ConstantBuffer => {
                let buffer = self.dynamic_index(operand, 0)?;
                let element = self.dynamic_index(operand, 1)?;
                Ok(self
                    .world
                    .value(2u64 + (u64::from(buffer) << 32), u64::from(element)))
            }
            RegisterType::ImmConstBuffer => {
                let element = self.dynamic_index(operand, 0)? as usize;
                let row = self
                    .immediate_constants
                    .get(element)
                    .copied()
                    .unwrap_or([0.0; 4]);
                Ok(row.map(f32::to_bits))
            }
            RegisterType::IndexableTemp => {
                let array = self.dynamic_index(operand, 0)?;
                let element = self.dynamic_index(operand, 1)?;
                Ok(self
                    .indexable
                    .get(&(array, element))
                    .copied()
                    .unwrap_or([0; 4]))
            }
            RegisterType::Immediate32 => {
                let values = &operand.immediate_values;
                Ok(core::array::from_fn(|lane| {
                    if values.len() == 1 {
                        values[0]
                    } else {
                        values.get(lane).copied().unwrap_or(0)
                    }
                }))
            }
            RegisterType::OutputControlPointID => Ok([self.control_point_id; 4]),
            RegisterType::ForkInstanceID => Ok([self.fork_instance_id; 4]),
            RegisterType::PatchConstant => {
                let register = self.dynamic_index(operand, 0)?;
                Ok(core::array::from_fn(|lane| {
                    let lane = u8::try_from(lane).unwrap_or(3);
                    self.keys
                        .patch
                        .get(&(register, lane))
                        .map_or(0, |&(row, offset)| {
                            self.world.value(5, row)[usize::from(offset.min(3))]
                        })
                }))
            }
            RegisterType::DomainLocation => Ok(self.world.value(3, 6)),
            RegisterType::ThreadID
            | RegisterType::ThreadGroupID
            | RegisterType::ThreadIDInGroup
            | RegisterType::Coverage
            | RegisterType::InputPrimitiveID => {
                let tag = match operand.reg_type {
                    RegisterType::ThreadID => 0,
                    RegisterType::ThreadGroupID => 1,
                    RegisterType::ThreadIDInGroup => 2,
                    RegisterType::Coverage => 3,
                    _ => 4,
                };
                let value = self.world.value(3, tag);
                // Scalar system values hold one value in every lane, so
                // both sides read the same bits through any component.
                if matches!(
                    operand.reg_type,
                    RegisterType::Coverage | RegisterType::InputPrimitiveID
                ) {
                    return Ok([value[0]; 4]);
                }
                Ok(value)
            }
            other => Err(Halt::Unsupported(format!("register file {other:?}"))),
        }
    }

    /// Writes one destination operand through its mask, with saturation.
    /// Records one typed UAV store as an observable output.
    pub(super) fn store_uav(&mut self, operands: &[Operand]) -> Result<(), Halt> {
        let [uav, coordinate, value] = operands else {
            return Err(Halt::Unsupported("store_uav_typed operand shape".into()));
        };
        let slot = self.dynamic_index(uav, 0)?;
        let mask = match &uav.components {
            ComponentSelect::Mask(mask) => *mask,
            _ => 0b1111,
        };
        let coordinate = self.read(coordinate)?;
        let dimensions = match self
            .uav_dimensions
            .get(&slot)
            .copied()
            .unwrap_or("texture2d")
        {
            "texture1d" | "buffer" => 1,
            "texture3d" | "texture2darray" => 3,
            _ => 2,
        };
        let mut canonical = [0u32; 4];
        canonical[..dimensions].copy_from_slice(&coordinate[..dimensions]);
        let value = self.read(value)?;
        let mut masked = [0u32; 4];
        for lane in 0..4 {
            if mask & (1 << lane) != 0 {
                masked[lane] = value[lane];
            }
        }
        self.outputs.uav_stores.push((slot, canonical, masked));
        Ok(())
    }

    pub(super) fn write(
        &mut self,
        operand: &Operand,
        value: Lanes,
        saturate: bool,
    ) -> Result<(), Halt> {
        let value = if saturate { value.map(sat) } else { value };
        let mask = match &operand.components {
            ComponentSelect::Mask(mask) => *mask,
            ComponentSelect::OneComponent => 0b0001,
            _ => 0b1111,
        };
        if let Some(trace) = self.trace.as_mut() {
            let mut written = [0u32; 4];
            for lane in 0..4 {
                if mask & (1 << lane) != 0 {
                    written[lane] = value[lane];
                }
            }
            trace.push((self.trace_opcode, mask, written));
        }
        let masked = |slot: &mut Lanes| {
            for lane in 0..4 {
                if mask & (1 << lane) != 0 {
                    slot[lane] = value[lane];
                }
            }
        };
        match operand.reg_type {
            RegisterType::Null => Ok(()),
            RegisterType::Temp => {
                let index = self.dynamic_index(operand, 0)?;
                masked(self.temps.entry(index).or_insert([0; 4]));
                Ok(())
            }
            RegisterType::Output => {
                let index = self.dynamic_index(operand, 0)?;
                if self.patch_phase {
                    // Patch constants key per signature row and lane.
                    for lane in 0u8..4 {
                        if mask & (1 << lane) == 0 {
                            continue;
                        }
                        if let Some(&(row, offset)) = self.keys.patch.get(&(index, lane)) {
                            let slot = self.outputs.patch.entry(row).or_insert([0; 4]);
                            slot[usize::from(offset.min(3))] = value[usize::from(lane)];
                        }
                    }
                    return Ok(());
                }
                let key = self
                    .keys
                    .outputs
                    .get(&index)
                    .copied()
                    .unwrap_or(u64::from(index));
                masked(self.outputs.outputs.entry(key).or_insert([0; 4]));
                Ok(())
            }
            RegisterType::OutputDepth => {
                self.outputs.depth = Some(value[0]);
                Ok(())
            }
            RegisterType::OutputCoverageMask => {
                self.outputs.coverage = Some(value[0]);
                Ok(())
            }
            RegisterType::IndexableTemp => {
                let array = self.dynamic_index(operand, 0)?;
                let element = self.dynamic_index(operand, 1)?;
                masked(self.indexable.entry((array, element)).or_insert([0; 4]));
                Ok(())
            }
            other => Err(Halt::Unsupported(format!("destination file {other:?}"))),
        }
    }

    /// Executes one computational instruction.
    #[expect(clippy::too_many_lines, reason = "one arm per interpreted opcode")]
    #[expect(
        clippy::float_cmp,
        reason = "eq/ne interpret D3D's exact IEEE comparisons"
    )]
    pub(super) fn step(
        &mut self,
        instruction: &Instruction,
        operands: &[Operand],
    ) -> Result<(), Halt> {
        let opcode = instruction.opcode;
        let (dests, sources) = split_operands(opcode, operands);
        let read = |machine: &mut Self, position: usize| -> Result<Lanes, Halt> {
            let operand = *sources.get(position).ok_or_else(|| missing("source"))?;
            machine.read(operand)
        };
        // Dual-destination opcodes write both halves and finish here.
        match opcode {
            Opcode::UDiv => {
                let a = read(self, 0)?;
                let b = read(self, 1)?;
                let quotient: Lanes =
                    core::array::from_fn(|lane| a[lane].checked_div(b[lane]).unwrap_or(u32::MAX));
                let remainder: Lanes =
                    core::array::from_fn(|lane| a[lane].checked_rem(b[lane]).unwrap_or(u32::MAX));
                let mut halves = dests.iter();
                if let Some(first) = halves.next() {
                    self.write(first, quotient, false)?;
                }
                if let Some(second) = halves.next() {
                    self.write(second, remainder, false)?;
                }
                return Ok(());
            }
            Opcode::IMul | Opcode::UMul => {
                let a = read(self, 0)?;
                let b = read(self, 1)?;
                let low: Lanes = core::array::from_fn(|lane| a[lane].wrapping_mul(b[lane]));
                let mut halves = dests.iter();
                if let Some(first) = halves.next()
                    && first.reg_type != RegisterType::Null
                {
                    return Err(Halt::Unsupported("high multiply half".into()));
                }
                if let Some(second) = halves.next() {
                    self.write(second, low, instruction.saturate)?;
                }
                return Ok(());
            }
            Opcode::Sincos => {
                let input = read(self, 0)?;
                let sine = input.map(|bits| canonical(f32::from_bits(bits).sin().to_bits()));
                let cosine = input.map(|bits| canonical(f32::from_bits(bits).cos().to_bits()));
                let mut halves = dests.iter();
                if let Some(first) = halves.next() {
                    self.write(first, sine, instruction.saturate)?;
                }
                if let Some(second) = halves.next() {
                    self.write(second, cosine, instruction.saturate)?;
                }
                return Ok(());
            }
            _ => {}
        }
        let value: Lanes = match opcode {
            Opcode::Mov => read(self, 0)?,
            Opcode::Movc => {
                let condition = read(self, 0)?;
                let yes = read(self, 1)?;
                let no = read(self, 2)?;
                core::array::from_fn(|lane| {
                    if condition[lane] == 0 {
                        no[lane]
                    } else {
                        yes[lane]
                    }
                })
            }
            Opcode::Add => lanes2(read(self, 0)?, read(self, 1)?, f32_add),
            Opcode::Mul => lanes2(read(self, 0)?, read(self, 1)?, f32_mul),
            Opcode::Div => lanes2(read(self, 0)?, read(self, 1)?, f32_div),
            Opcode::Mad => {
                let a = read(self, 0)?;
                let b = read(self, 1)?;
                let c = read(self, 2)?;
                core::array::from_fn(|lane| f32_add(f32_mul(a[lane], b[lane]), c[lane]))
            }
            Opcode::Min => lanes2(read(self, 0)?, read(self, 1)?, f32_min),
            Opcode::Max => lanes2(read(self, 0)?, read(self, 1)?, f32_max),
            Opcode::Frc => lanes1(read(self, 0)?, |value| value - value.floor()),
            Opcode::Sqrt => lanes1(read(self, 0)?, f32::sqrt),
            Opcode::Rsq => lanes1(read(self, 0)?, |value| 1.0 / value.sqrt()),
            Opcode::Rcp => lanes1(read(self, 0)?, |value| 1.0 / value),
            Opcode::Exp => lanes1(read(self, 0)?, f32::exp2),
            Opcode::Log => lanes1(read(self, 0)?, f32::log2),
            Opcode::RoundNe => lanes1(read(self, 0)?, round_even),
            Opcode::RoundNi => lanes1(read(self, 0)?, f32::floor),
            Opcode::RoundPi => lanes1(read(self, 0)?, f32::ceil),
            Opcode::RoundZ => lanes1(read(self, 0)?, f32::trunc),
            Opcode::Dp2 => dot(read(self, 0)?, read(self, 1)?, 2),
            Opcode::Dp3 => dot(read(self, 0)?, read(self, 1)?, 3),
            Opcode::Dp4 => dot(read(self, 0)?, read(self, 1)?, 4),
            Opcode::Lt => mask2(read(self, 0)?, read(self, 1)?, |a, b| {
                f32::from_bits(a) < f32::from_bits(b)
            }),
            Opcode::Ge => mask2(read(self, 0)?, read(self, 1)?, |a, b| {
                f32::from_bits(a) >= f32::from_bits(b)
            }),
            Opcode::Eq => mask2(read(self, 0)?, read(self, 1)?, |a, b| {
                f32::from_bits(a) == f32::from_bits(b)
            }),
            Opcode::Ne => mask2(read(self, 0)?, read(self, 1)?, |a, b| {
                f32::from_bits(a) != f32::from_bits(b)
            }),
            Opcode::ILt => mask2(read(self, 0)?, read(self, 1)?, |a, b| signed(a) < signed(b)),
            Opcode::IGe => mask2(read(self, 0)?, read(self, 1)?, |a, b| {
                signed(a) >= signed(b)
            }),
            Opcode::IEq => mask2(read(self, 0)?, read(self, 1)?, |a, b| a == b),
            Opcode::INe => mask2(read(self, 0)?, read(self, 1)?, |a, b| a != b),
            Opcode::ULt => mask2(read(self, 0)?, read(self, 1)?, |a, b| a < b),
            Opcode::UGe => mask2(read(self, 0)?, read(self, 1)?, |a, b| a >= b),
            Opcode::Iadd => lanes2(read(self, 0)?, read(self, 1)?, u32::wrapping_add),
            Opcode::Bfi => {
                let width = read(self, 0)?;
                let offset = read(self, 1)?;
                let insert = read(self, 2)?;
                let base = read(self, 3)?;
                core::array::from_fn(|lane| {
                    let w = width[lane] & 31;
                    let o = offset[lane] & 31;
                    let bits = 1u32.wrapping_shl(w).wrapping_sub(1).wrapping_shl(o);
                    (insert[lane].wrapping_shl(o) & bits) | (base[lane] & !bits)
                })
            }
            Opcode::IMad | Opcode::UMad => {
                let a = read(self, 0)?;
                let b = read(self, 1)?;
                let c = read(self, 2)?;
                core::array::from_fn(|lane| a[lane].wrapping_mul(b[lane]).wrapping_add(c[lane]))
            }
            Opcode::IMin => lanes2(read(self, 0)?, read(self, 1)?, |a, b| {
                if signed(a) < signed(b) { a } else { b }
            }),
            Opcode::IMax => lanes2(read(self, 0)?, read(self, 1)?, |a, b| {
                if signed(a) > signed(b) { a } else { b }
            }),
            Opcode::UMin => lanes2(read(self, 0)?, read(self, 1)?, u32::min),
            Opcode::UMax => lanes2(read(self, 0)?, read(self, 1)?, u32::max),
            Opcode::INeg => read(self, 0)?.map(|bits| unsigned(signed(bits).wrapping_neg())),
            Opcode::And => lanes2(read(self, 0)?, read(self, 1)?, |a, b| a & b),
            Opcode::Or => lanes2(read(self, 0)?, read(self, 1)?, |a, b| a | b),
            Opcode::Xor => lanes2(read(self, 0)?, read(self, 1)?, |a, b| a ^ b),
            Opcode::Not => read(self, 0)?.map(|bits| !bits),
            Opcode::Ishl => lanes2(read(self, 0)?, read(self, 1)?, |a, b| {
                a.wrapping_shl(b & 31)
            }),
            Opcode::Ishr => lanes2(read(self, 0)?, read(self, 1)?, |a, b| {
                unsigned(signed(a) >> (b & 31))
            }),
            Opcode::Ushr => lanes2(read(self, 0)?, read(self, 1)?, |a, b| a >> (b & 31)),
            Opcode::Itof => read(self, 0)?.map(|bits| {
                #[expect(clippy::cast_precision_loss, reason = "SM4 itof rounds to nearest")]
                let value = signed(bits) as f32;
                value.to_bits()
            }),
            Opcode::Utof => read(self, 0)?.map(|bits| {
                #[expect(clippy::cast_precision_loss, reason = "SM4 utof rounds to nearest")]
                let value = bits as f32;
                value.to_bits()
            }),
            Opcode::Ftoi => read(self, 0)?.map(|bits| {
                let value = f32::from_bits(bits);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "saturating cast is the spec"
                )]
                let converted = value as i32;
                if value.is_nan() {
                    0
                } else {
                    unsigned(converted)
                }
            }),
            Opcode::Ftou => read(self, 0)?.map(|bits| {
                let value = f32::from_bits(bits);
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "saturating cast is the spec"
                )]
                #[expect(clippy::cast_sign_loss, reason = "saturating cast is the spec")]
                let converted = value as u32;
                if value.is_nan() { 0 } else { converted }
            }),
            Opcode::Sample
            | Opcode::SampleL
            | Opcode::SampleB
            | Opcode::Ld
            | Opcode::LdMs
            | Opcode::Gather4
            | Opcode::LdStructured => self.resource_fetch(instruction, &sources, operands)?,
            Opcode::Resinfo => resinfo_value(self, instruction, &sources, operands)?,
            Opcode::DerivRtx
            | Opcode::DerivRty
            | Opcode::DerivRtxCoarse
            | Opcode::DerivRtxFine
            | Opcode::DerivRtyCoarse
            | Opcode::DerivRtyFine => {
                // Derivatives need quad context the interpreter does not
                // model; a deterministic pure function of the operand
                // stands in — identical on both sides by construction.
                let input = read(self, 0)?;
                let axis = u64::from(instruction.opcode.to_u32());
                core::array::from_fn(|lane| {
                    let raw = self.world.mix(
                        axis.wrapping_add(541),
                        u64::from(input[lane]),
                        lane as u64,
                        11,
                    );
                    #[expect(clippy::cast_precision_loss, reason = "modulus bounds the value")]
                    let value = (raw % 64) as f32 / 256.0;
                    value.to_bits()
                })
            }
            other => return Err(Halt::Unsupported(format!("opcode {}", other.name()))),
        };
        let dest = dests.first().ok_or_else(|| missing("destination"))?;
        self.write(dest, value, instruction.saturate)
    }

    /// A deterministic resource fetch over the canonical access
    /// identity: exactly the lanes the operation consumes — matching
    /// the renderer's own truncation — so dead encoding lanes never
    /// perturb the value.
    #[expect(clippy::too_many_lines, reason = "one addressing rule per fetch kind")]
    pub(super) fn resource_fetch(
        &mut self,
        instruction: &Instruction,
        sources: &[&Operand],
        operands: &[Operand],
    ) -> Result<Lanes, Halt> {
        let resource = operands
            .iter()
            .find(|operand| matches!(operand.reg_type, RegisterType::Resource | RegisterType::Uav))
            .ok_or_else(|| Halt::Unsupported("fetch without resource".into()))?;
        let Some(OperandIndex::Imm32(index)) = resource.indices.first() else {
            return Err(Halt::Unsupported("computed resource index".into()));
        };
        let index = *index;
        let dimension = self.dimensions.get(&index).copied().unwrap_or("texture2d");
        let values: Vec<&Operand> = sources
            .iter()
            .copied()
            .filter(|operand| {
                !matches!(
                    operand.reg_type,
                    RegisterType::Resource | RegisterType::Sampler | RegisterType::Uav
                )
            })
            .collect();
        let coordinate_operand = *values.first().ok_or_else(|| missing("coordinate"))?;
        let coordinate = self.read(coordinate_operand)?;
        let extra = |machine: &mut Self| -> Result<u32, Halt> {
            let operand = *values.get(1).ok_or_else(|| missing("fetch argument"))?;
            Ok(machine.read(operand)?[0])
        };
        let mut address: Vec<u32> = Vec::new();
        match instruction.opcode {
            Opcode::Sample | Opcode::SampleB | Opcode::SampleL | Opcode::Gather4 => {
                let lanes: usize = match dimension {
                    "texture1d" => 1,
                    "texture3d" | "texturecube" | "texture2darray" => 3,
                    "texturecubearray" => 4,
                    _ => 2,
                };
                address.extend_from_slice(&coordinate[..lanes]);
                match instruction.opcode {
                    Opcode::SampleL | Opcode::SampleB => address.push(extra(self)?),
                    Opcode::Gather4 => {
                        // The gathered channel comes from the sampler
                        // operand's swizzle.
                        let channel = operands
                            .iter()
                            .find(|operand| operand.reg_type == RegisterType::Sampler)
                            .map_or(0, |operand| match &operand.components {
                                ComponentSelect::Scalar(lane) => u32::from(*lane),
                                ComponentSelect::Swizzle(swizzle) => u32::from(swizzle[0]),
                                _ => 0,
                            });
                        address.push(channel);
                    }
                    _ => {}
                }
            }
            Opcode::Ld => {
                let (lanes, mip): (usize, bool) = match dimension {
                    "texture1d" => (1, true),
                    "texture3d" | "texture2darray" => (3, true),
                    "buffer" => (1, false),
                    _ => (2, true),
                };
                address.extend_from_slice(&coordinate[..lanes]);
                if mip {
                    address.push(coordinate[3]);
                }
            }
            Opcode::LdMs => {
                address.extend_from_slice(&coordinate[..2]);
                address.push(extra(self)?);
            }
            Opcode::LdStructured => {
                // Per-dword model: result lane `l` reads dword
                // `offset/4 + swizzle[l]` of the addressed element, the
                // same contract the renderer spells.
                let element = coordinate[0];
                let offset = extra(self)?;
                let swizzle = match &resource.components {
                    ComponentSelect::Swizzle(swizzle) => *swizzle,
                    _ => [0, 1, 2, 3],
                };
                return Ok(core::array::from_fn(|lane| {
                    let dword = (offset / 4) + u32::from(swizzle[lane]);
                    self.world.mix(
                        u64::from(index).wrapping_add(1013),
                        u64::from(element),
                        u64::from(dword),
                        7,
                    )
                }));
            }
            other => {
                return Err(Halt::Unsupported(format!("fetch opcode {}", other.name())));
            }
        }
        if let Some(offsets) = instruction.tex_offsets {
            for offset in offsets {
                address.push(u32::from(offset.to_ne_bytes()[0]));
            }
        }
        address.push(instruction.opcode.to_u32());
        let fetched = self.world.fetch(u64::from(index), &address);
        // The resource swizzle selects among the fetched lanes.
        Ok(match &resource.components {
            ComponentSelect::Swizzle(swizzle) => {
                core::array::from_fn(|lane| fetched[usize::from(swizzle[lane])])
            }
            _ => fetched,
        })
    }
}

/// The `resinfo` result: deterministic fake dimensions, swizzled by the
/// resource operand.
fn resinfo_value(
    machine: &mut Machine<'_>,
    instruction: &Instruction,
    sources: &[&Operand],
    operands: &[Operand],
) -> Result<Lanes, Halt> {
    let mip_operand = *sources.first().ok_or_else(|| missing("resinfo mip"))?;
    let mip = machine.read(mip_operand)?[0] & 15;
    let width = 256u32 >> mip.min(8);
    let height = 128u32 >> mip.min(7);
    let mips = 9u32;
    let uint = instruction.resinfo_return_type == Some(2);
    let encode = |value: u32| -> u32 {
        if uint {
            value
        } else {
            #[expect(clippy::cast_precision_loss, reason = "small dimension values")]
            let as_float = value as f32;
            as_float.to_bits()
        }
    };
    let dims = [encode(width), encode(height), encode(0), encode(mips)];
    let resource = operands
        .iter()
        .find(|operand| operand.reg_type == RegisterType::Resource)
        .ok_or_else(|| Halt::Unsupported("resinfo without resource".into()))?;
    Ok(match &resource.components {
        ComponentSelect::Swizzle(swizzle) => {
            core::array::from_fn(|lane| dims[usize::from(swizzle[lane])])
        }
        _ => dims,
    })
}

/// Splits an instruction's operands into destinations and sources.
fn split_operands(opcode: Opcode, operands: &[Operand]) -> (Vec<&Operand>, Vec<&Operand>) {
    let mut dests = Vec::new();
    let mut sources = Vec::new();
    let expected = expected_dests(opcode);
    for operand in operands {
        let positional_dest = dests.len() < expected && sources.is_empty();
        match operand.reg_type {
            RegisterType::Null if positional_dest => dests.push(operand),
            RegisterType::Temp
            | RegisterType::Output
            | RegisterType::OutputDepth
            | RegisterType::OutputCoverageMask
            | RegisterType::IndexableTemp
                if positional_dest
                    && matches!(
                        operand.components,
                        ComponentSelect::Mask(_) | ComponentSelect::OneComponent
                    ) =>
            {
                dests.push(operand);
            }
            _ => sources.push(operand),
        }
    }
    (dests, sources)
}

/// Destination-operand count of one opcode.
fn expected_dests(opcode: Opcode) -> usize {
    match opcode {
        Opcode::UDiv | Opcode::Sincos | Opcode::IMul | Opcode::UMul | Opcode::Swapc => 2,
        _ => 1,
    }
}

/// One canonical bit pattern for every NaN: D3D leaves NaN sign and
/// payload unspecified, so compilations may legally disagree on them —
/// canonicalizing every arithmetic result makes both sides of a
/// differential deterministic.
fn canonical(bits: u32) -> u32 {
    if f32::from_bits(bits).is_nan() {
        0x7fc0_0000
    } else {
        bits
    }
}

fn lanes1(value: Lanes, mapper: impl Fn(f32) -> f32) -> Lanes {
    value.map(|bits| canonical(mapper(f32::from_bits(bits)).to_bits()))
}

fn lanes2(a: Lanes, b: Lanes, mapper: impl Fn(u32, u32) -> u32) -> Lanes {
    core::array::from_fn(|lane| mapper(a[lane], b[lane]))
}

fn mask2(a: Lanes, b: Lanes, predicate: impl Fn(u32, u32) -> bool) -> Lanes {
    lanes2(a, b, |a, b| if predicate(a, b) { u32::MAX } else { 0 })
}

fn dot(a: Lanes, b: Lanes, components: usize) -> Lanes {
    let mut total = 0.0f32.to_bits();
    for lane in 0..components {
        total = f32_add(total, f32_mul(a[lane], b[lane]));
    }
    [total; 4]
}

#[expect(
    clippy::float_cmp,
    reason = "the half-way test is exact by construction"
)]
fn round_even(value: f32) -> f32 {
    let rounded = value.round();
    if (value - value.trunc()).abs() == 0.5 && rounded % 2.0 != 0.0 {
        rounded - value.signum()
    } else {
        rounded
    }
}

fn signed(bits: u32) -> i32 {
    #[expect(
        clippy::cast_possible_wrap,
        reason = "the wrap is the reinterpretation"
    )]
    let value = bits as i32;
    value
}

fn unsigned(value: i32) -> u32 {
    #[expect(clippy::cast_sign_loss, reason = "the wrap is the reinterpretation")]
    let bits = value as u32;
    bits
}

fn sat(bits: u32) -> u32 {
    let value = f32::from_bits(bits);
    if value.is_nan() {
        0
    } else {
        value.clamp(0.0, 1.0).to_bits()
    }
}

fn f32_add(a: u32, b: u32) -> u32 {
    canonical((f32::from_bits(a) + f32::from_bits(b)).to_bits())
}

fn f32_mul(a: u32, b: u32) -> u32 {
    canonical((f32::from_bits(a) * f32::from_bits(b)).to_bits())
}

fn f32_div(a: u32, b: u32) -> u32 {
    canonical((f32::from_bits(a) / f32::from_bits(b)).to_bits())
}

/// D3D `min`: a NaN operand yields the other operand.
fn f32_min(a: u32, b: u32) -> u32 {
    let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
    canonical(if fa.is_nan() {
        b
    } else if fb.is_nan() || fa < fb {
        a
    } else {
        b
    })
}

/// D3D `max`: a NaN operand yields the other operand.
fn f32_max(a: u32, b: u32) -> u32 {
    let (fa, fb) = (f32::from_bits(a), f32::from_bits(b));
    canonical(if fa.is_nan() {
        b
    } else if fb.is_nan() || fa > fb {
        a
    } else {
        b
    })
}
