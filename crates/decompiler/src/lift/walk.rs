//! The structured-marker walk: operands, control flow, and computation.

use cfglib::ir::rtl::{Expr, Place, ScalarType, Statement, ValueShape};
use dxbc::shex::{ComponentSelect, Instruction, Opcode, Operand, OperandIndex, RegisterType};

use crate::dialect::{AluHead, Binding, EffectOp, MemRef, RegFile, Register, RtlOp, ShaderDialect};

use super::{
    LiftExpr, Unsupported, Walk, coordinate_lanes, map_lanes, mask_lanes, modified,
    operand_register, operand_scalar, result_scalar, select_lane, unsupported,
};

impl Walk {
    #[expect(clippy::too_many_lines, reason = "one arm per operand file")]
    pub(super) fn source(
        &mut self,
        operand: &Operand,
        lanes: &[u8],
        scalar: ScalarType,
    ) -> Result<LiftExpr, Unsupported> {
        // Immediates carry no swizzle: their lanes are the destination
        // lanes themselves.
        let mapped = if matches!(
            operand.reg_type,
            RegisterType::Immediate32 | RegisterType::Immediate64
        ) {
            lanes.to_vec()
        } else {
            map_lanes(&operand.components, lanes)?
        };
        let width = u8::try_from(mapped.len()).unwrap_or(u8::MAX);
        let base = match operand.reg_type {
            RegisterType::Immediate32 => {
                let bits: Vec<u64> = mapped
                    .iter()
                    .map(|&lane| {
                        let index = usize::from(lane);
                        let values = &operand.immediate_values;
                        u64::from(if values.len() == 1 {
                            values[0]
                        } else {
                            values.get(index).copied().unwrap_or(0)
                        })
                    })
                    .collect();
                Expr::Const {
                    bits,
                    shape: ValueShape::vector(scalar, width),
                }
            }
            RegisterType::Immediate64 => return unsupported("64-bit immediate"),
            RegisterType::ConstantBuffer
            | RegisterType::ImmConstBuffer
            | RegisterType::IndexableTemp => {
                let file = match operand.reg_type {
                    RegisterType::ConstantBuffer => RegFile::ConstantBuffer,
                    RegisterType::ImmConstBuffer => RegFile::ImmediateConstantBuffer,
                    _ => RegFile::IndexableTemp,
                };
                let (reference, address) = self.memory_reference(file, operand, mapped.clone())?;
                Expr::Apply {
                    operator: RtlOp::LoadMem(reference),
                    operands: address.into_iter().collect(),
                    shape: ValueShape::vector(scalar, width),
                }
            }
            RegisterType::InputControlPoint if operand.indices.len() == 2 => {
                // A hull fork-phase control-point read: vicp[point][reg],
                // where the register index may be computed.
                let Some(OperandIndex::Imm32(vertex)) = operand.indices.first() else {
                    return unsupported("computed control-point index");
                };
                match operand.indices.get(1) {
                    Some(OperandIndex::Imm32(register)) => Expr::Apply {
                        operator: RtlOp::LoadInput {
                            vertex: *vertex,
                            register: *register,
                            lanes: mapped,
                        },
                        operands: Vec::new(),
                        shape: ValueShape::vector(scalar, width),
                    },
                    Some(
                        OperandIndex::Relative(inner) | OperandIndex::RelativePlusImm(_, inner),
                    ) => {
                        let base = match operand.indices.get(1) {
                            Some(OperandIndex::RelativePlusImm(base, _)) => *base,
                            _ => 0,
                        };
                        if inner.negate || inner.abs {
                            return unsupported("modified control-point index");
                        }
                        let lane = select_lane(&inner.components);
                        let index = self.source(inner, &[lane], ScalarType::U32)?;
                        Expr::Apply {
                            operator: RtlOp::LoadInputDynamic {
                                vertex: *vertex,
                                base,
                                lanes: mapped,
                            },
                            operands: vec![index],
                            shape: ValueShape::vector(scalar, width),
                        }
                    }
                    _ => return unsupported("computed control-point index"),
                }
            }
            RegisterType::Input if operand.indices.len() == 2 => {
                // A geometry-shader input: v[vertex][register].
                let (Some(OperandIndex::Imm32(vertex)), Some(OperandIndex::Imm32(register))) =
                    (operand.indices.first(), operand.indices.get(1))
                else {
                    return unsupported("computed geometry input index");
                };
                Expr::Apply {
                    operator: RtlOp::LoadInput {
                        vertex: *vertex,
                        register: *register,
                        lanes: mapped,
                    },
                    operands: Vec::new(),
                    shape: ValueShape::vector(scalar, width),
                }
            }
            _ => {
                let register = operand_register(operand)?;
                Expr::Read {
                    storage: register,
                    lanes: mapped,
                    scalar,
                }
            }
        };
        Ok(modified(base, operand))
    }

    /// Builds an addressable-file reference and its optional relative
    /// address value.
    pub(super) fn memory_reference(
        &mut self,
        file: RegFile,
        operand: &Operand,
        lanes: Vec<u8>,
    ) -> Result<(MemRef, Option<LiftExpr>), Unsupported> {
        let mut indices = Vec::new();
        let mut relative_to = None;
        let mut address = None;
        for (position, index) in operand.indices.iter().enumerate() {
            match index {
                OperandIndex::Imm32(value) => indices.push(*value),
                OperandIndex::Relative(inner) | OperandIndex::RelativePlusImm(_, inner) => {
                    if relative_to.is_some() {
                        return unsupported("multiple relative operand indices");
                    }
                    let base = match index {
                        OperandIndex::RelativePlusImm(base, _) => *base,
                        _ => 0,
                    };
                    indices.push(base);
                    if inner.negate || inner.abs {
                        return unsupported("modified relative address");
                    }
                    let lane = select_lane(&inner.components);
                    address = Some(self.source(inner, &[lane], ScalarType::I32)?);
                    relative_to = Some(position);
                }
                OperandIndex::Imm64(_) | OperandIndex::RelativePlusImm64(..) => {
                    return unsupported("64-bit operand index");
                }
            }
        }
        Ok((
            MemRef {
                file,
                indices,
                relative_to,
                lanes,
            },
            address,
        ))
    }

    /// Lowers one generic (non-declaration) instruction.
    pub(super) fn generic(
        &mut self,
        instruction: &Instruction,
        operands: &[Operand],
        span: (u32, u32),
    ) -> Result<(), Unsupported> {
        match instruction.opcode {
            Opcode::If => self.open_if(instruction, operands, span),
            Opcode::Else => self.open_else(),
            Opcode::EndIf => self.close_if(),
            Opcode::Loop => {
                self.open_loop();
                Ok(())
            }
            Opcode::EndLoop => self.close_loop(),
            Opcode::Break => self.transfer_break(false),
            Opcode::Breakc => {
                self.condition(instruction, operands, span)?;
                self.transfer_break(true)
            }
            Opcode::Continue => self.transfer_continue(false),
            Opcode::Continuec => {
                self.condition(instruction, operands, span)?;
                self.transfer_continue(true)
            }
            Opcode::Ret => {
                self.emit(Statement::Return { values: Vec::new() }, span);
                self.structure.terminate();
                Ok(())
            }
            Opcode::Retc => {
                self.condition(instruction, operands, span)?;
                self.conditional_return(span)
            }
            Opcode::Discard => {
                let condition = self.condition_expr(instruction, operands)?;
                self.emit(
                    Statement::Effect {
                        operation: EffectOp::Discard,
                        operands: vec![condition],
                        effects: Vec::new(),
                        may_throw: false,
                    },
                    span,
                );
                Ok(())
            }
            Opcode::StoreUavTyped => self.store_uav(instruction, operands, span),
            Opcode::DclStream => {
                // Stream declarations carry no code; only stream 0 is
                // modeled.
                match operands.first().and_then(|operand| operand.indices.first()) {
                    Some(OperandIndex::Imm32(index)) if *index != 0 => {
                        unsupported("multiple geometry streams")
                    }
                    _ => Ok(()),
                }
            }
            Opcode::Emit | Opcode::EmitStream => self.stream_event(operands, EffectOp::Emit, span),
            Opcode::Cut | Opcode::CutStream => self.stream_event(operands, EffectOp::Cut, span),
            Opcode::EmitThenCut | Opcode::EmitThenCutStream => {
                self.stream_event(operands, EffectOp::Emit, span)?;
                self.stream_event(operands, EffectOp::Cut, span)
            }
            Opcode::Switch | Opcode::Case | Opcode::Default | Opcode::EndSwitch => {
                unsupported("switch flow")
            }
            Opcode::Call | Opcode::Callc | Opcode::Label => unsupported("subroutine calls"),
            Opcode::Nop => Ok(()),
            Opcode::Sync => {
                self.emit(
                    Statement::Effect {
                        operation: EffectOp::Bare(instruction.opcode),
                        operands: Vec::new(),
                        effects: Vec::new(),
                        may_throw: false,
                    },
                    span,
                );
                Ok(())
            }
            _ => self.alu(instruction, operands, span),
        }
    }

    /// Emits one stream event (`emit`/`cut`), stream 0 only.
    pub(super) fn stream_event(
        &mut self,
        operands: &[Operand],
        operation: EffectOp,
        span: (u32, u32),
    ) -> Result<(), Unsupported> {
        let stream = operands
            .first()
            .and_then(|operand| match operand.indices.first() {
                Some(OperandIndex::Imm32(index)) => Some(*index),
                None => Some(0),
                _ => None,
            })
            .unwrap_or(0);
        if stream != 0 {
            return unsupported("multiple geometry streams");
        }
        self.emit(
            Statement::Effect {
                operation,
                operands: Vec::new(),
                effects: Vec::new(),
                may_throw: false,
            },
            span,
        );
        Ok(())
    }

    /// Lowers one typed UAV store.
    pub(super) fn store_uav(
        &mut self,
        _instruction: &Instruction,
        operands: &[Operand],
        span: (u32, u32),
    ) -> Result<(), Unsupported> {
        let [uav, coordinate, value] = operands else {
            return unsupported("store_uav_typed operand shape");
        };
        let ComponentSelect::Mask(mask) = uav.components else {
            return unsupported("store_uav_typed without a mask");
        };
        let index = match uav.indices.first() {
            Some(OperandIndex::Imm32(index)) => *index,
            _ => return unsupported("uav with a computed index"),
        };
        let lanes = mask_lanes(mask);
        let (dimension, return_type) = self
            .declarations
            .uavs
            .iter()
            .find(|(declared, ..)| *declared == index)
            .map_or(("texture2d", ScalarType::F32), |(_, dimension, kinds)| {
                let scalar = match kinds[0] {
                    dxbc::shex::ReturnType::Uint => ScalarType::U32,
                    dxbc::shex::ReturnType::Sint => ScalarType::I32,
                    _ => ScalarType::F32,
                };
                (dimension, scalar)
            });
        let coordinate_lanes: &[u8] = match dimension {
            "texture1d" | "buffer" => &[0],
            "texture3d" | "texture2darray" => &[0, 1, 2],
            _ => &[0, 1],
        };
        let coordinate = self.source(coordinate, coordinate_lanes, ScalarType::U32)?;
        let value = self.source(value, &lanes, return_type)?;
        self.emit(
            Statement::Effect {
                operation: EffectOp::StoreUav {
                    register: Register {
                        file: RegFile::Uav,
                        index,
                    },
                    lanes,
                },
                operands: vec![coordinate, value],
                effects: Vec::new(),
                may_throw: false,
            },
            span,
        );
        Ok(())
    }

    /// Builds the tested-condition expression of one conditional opcode.
    pub(super) fn condition_expr(
        &mut self,
        instruction: &Instruction,
        operands: &[Operand],
    ) -> Result<LiftExpr, Unsupported> {
        let Some(operand) = operands.first() else {
            return unsupported("conditional without a tested operand");
        };
        if operand.negate || operand.abs {
            return unsupported("modified branch condition");
        }
        let lane = select_lane(&operand.components);
        let tested = self.source(operand, &[lane], ScalarType::U32)?;
        Ok(Expr::Apply {
            operator: RtlOp::Test {
                nonzero: instruction.test_nonzero,
            },
            operands: vec![tested],
            shape: ValueShape::scalar(ScalarType::Bool),
        })
    }

    /// Emits the branch statement of one conditional opcode.
    pub(super) fn condition(
        &mut self,
        instruction: &Instruction,
        operands: &[Operand],
        span: (u32, u32),
    ) -> Result<(), Unsupported> {
        let condition = self.condition_expr(instruction, operands)?;
        self.emit(Statement::Branch { condition }, span);
        Ok(())
    }

    pub(super) fn open_if(
        &mut self,
        instruction: &Instruction,
        operands: &[Operand],
        span: (u32, u32),
    ) -> Result<(), Unsupported> {
        self.condition(instruction, operands, span)?;
        Ok(self.structure.open_if()?)
    }

    pub(super) fn open_else(&mut self) -> Result<(), Unsupported> {
        Ok(self.structure.open_else()?)
    }

    pub(super) fn close_if(&mut self) -> Result<(), Unsupported> {
        Ok(self.structure.close_if()?)
    }

    pub(super) fn open_loop(&mut self) {
        // The header must exist eagerly: it is the back-edge target. The
        // preceding block falls through into it.
        let (walk, mut sink) = self.parts();
        walk.open_loop(&mut sink);
    }

    pub(super) fn close_loop(&mut self) -> Result<(), Unsupported> {
        let (walk, mut sink) = self.parts();
        Ok(walk.close_loop(&mut sink)?)
    }

    /// A `break`/`breakc`: `conditional` when the branch just landed.
    pub(super) fn transfer_break(&mut self, conditional: bool) -> Result<(), Unsupported> {
        Ok(self.structure.break_out(conditional)?)
    }

    pub(super) fn transfer_continue(&mut self, conditional: bool) -> Result<(), Unsupported> {
        let (walk, mut sink) = self.parts();
        Ok(walk.continue_loop(&mut sink, conditional)?)
    }

    pub(super) fn conditional_return(&mut self, span: (u32, u32)) -> Result<(), Unsupported> {
        let ret_block = if let Some(existing) = self.ret_block {
            existing
        } else {
            let created = self.builder.new_block("ret");
            self.builder
                .append(
                    created,
                    Statement::Return { values: Vec::new() },
                    Some(span),
                )
                .expect("return statements are always valid");
            self.ret_block = Some(created);
            created
        };
        let (walk, mut sink) = self.parts();
        Ok(walk.conditional_transfer(&mut sink, ret_block)?)
    }

    /// Lowers a retained decoded instruction with dest/source operands.
    #[expect(clippy::too_many_lines, reason = "one arm per destination kind")]
    pub(super) fn alu(
        &mut self,
        instruction: &Instruction,
        operands: &[Operand],
        span: (u32, u32),
    ) -> Result<(), Unsupported> {
        let mut dests: Vec<(&Operand, Vec<u8>, usize)> = Vec::new();
        let mut sources: Vec<&Operand> = Vec::new();
        let mut bindings: Vec<&Operand> = Vec::new();
        let dual = matches!(instruction.opcode, Opcode::Sincos | Opcode::UDiv);
        for (position, operand) in operands.iter().enumerate() {
            match operand.reg_type {
                RegisterType::Null => {}
                RegisterType::Resource | RegisterType::Sampler | RegisterType::Uav => {
                    bindings.push(operand);
                }
                RegisterType::Immediate32 | RegisterType::Immediate64 => sources.push(operand),
                _ if matches!(operand.components, ComponentSelect::Mask(_))
                    && (dests.is_empty() || (dual && dests.len() == 1)) =>
                {
                    let ComponentSelect::Mask(mask) = operand.components else {
                        unreachable!()
                    };
                    dests.push((operand, mask_lanes(mask), position));
                }
                _ if matches!(operand.components, ComponentSelect::Mask(_)) => {
                    return unsupported(format!(
                        "second destination operand for {}",
                        instruction.opcode.name()
                    ));
                }
                _ if dests.is_empty()
                    && sources.is_empty()
                    && matches!(operand.components, ComponentSelect::OneComponent)
                    && matches!(
                        operand.reg_type,
                        RegisterType::OutputDepth | RegisterType::OutputCoverageMask
                    ) =>
                {
                    dests.push((operand, vec![0], position));
                }
                _ => sources.push(operand),
            }
        }
        if dests.is_empty() {
            // A destination-less retained operation.
            self.emit(
                Statement::Effect {
                    operation: EffectOp::Bare(instruction.opcode),
                    operands: Vec::new(),
                    effects: Vec::new(),
                    may_throw: false,
                },
                span,
            );
            return Ok(());
        }
        // Build one value per destination half, then split the halves
        // into effect stores (outputs, indexable temps) and register
        // assignments. Effects run first: their values, like every value
        // here, are built from pre-statement reads.
        let mut assignments: Vec<(Place<ShaderDialect>, LiftExpr)> = Vec::new();
        let mut effects: Vec<(EffectOp, Vec<LiftExpr>)> = Vec::new();
        for (operand, lanes, position) in &dests {
            let value = self.alu_value(instruction, &bindings, &sources, lanes, *position)?;
            match operand.reg_type {
                RegisterType::IndexableTemp => {
                    let (reference, address) =
                        self.memory_reference(RegFile::IndexableTemp, operand, Vec::new())?;
                    let mut store_operands = vec![value];
                    store_operands.extend(address);
                    let mut reference = reference;
                    reference.lanes.clone_from(lanes);
                    effects.push((EffectOp::StoreMem(reference), store_operands));
                }
                RegisterType::Output
                | RegisterType::OutputDepth
                | RegisterType::OutputCoverageMask => {
                    // A dynamically indexed output (`o[expr]`, instanced
                    // hull phases) carries its index as a second operand.
                    if operand.reg_type == RegisterType::Output
                        && matches!(
                            operand.indices.first(),
                            Some(OperandIndex::Relative(_) | OperandIndex::RelativePlusImm(..))
                        )
                    {
                        let (base, inner) = match operand.indices.first() {
                            Some(OperandIndex::Relative(inner)) => (0u32, inner),
                            Some(OperandIndex::RelativePlusImm(base, inner)) => (*base, inner),
                            _ => unreachable!(),
                        };
                        if inner.negate || inner.abs {
                            return unsupported("modified output index");
                        }
                        let lane = select_lane(&inner.components);
                        let index = self.source(inner, &[lane], ScalarType::U32)?;
                        effects.push((
                            EffectOp::StoreOutput {
                                register: Register {
                                    file: RegFile::Output,
                                    index: base,
                                },
                                lanes: lanes.clone(),
                                dynamic: true,
                            },
                            vec![value, index],
                        ));
                        continue;
                    }
                    let register = operand_register(operand)?;
                    effects.push((
                        EffectOp::StoreOutput {
                            register,
                            lanes: lanes.clone(),
                            dynamic: false,
                        },
                        vec![value],
                    ));
                }
                _ => {
                    let register = operand_register(operand)?;
                    assignments.push((
                        Place {
                            storage: register,
                            lanes: lanes.clone(),
                        },
                        value,
                    ));
                }
            }
        }
        for (operation, operands) in effects {
            self.emit(
                Statement::Effect {
                    operation,
                    operands,
                    effects: Vec::new(),
                    may_throw: false,
                },
                span,
            );
        }
        if !assignments.is_empty() {
            self.emit(
                Statement::Transfer {
                    assignments,
                    effects: Vec::new(),
                    may_throw: false,
                },
                span,
            );
        }
        Ok(())
    }

    /// Builds the value expression of one destination half.
    #[expect(clippy::too_many_lines, reason = "one arm per operand pattern")]
    pub(super) fn alu_value(
        &mut self,
        instruction: &Instruction,
        bindings: &[&Operand],
        sources: &[&Operand],
        dest_lanes: &[u8],
        dest_position: usize,
    ) -> Result<LiftExpr, Unsupported> {
        let opcode = instruction.opcode;
        let width = u8::try_from(dest_lanes.len()).unwrap_or(u8::MAX);
        let source = |walk: &mut Self, position: usize, scalar: ScalarType| {
            let operand = sources
                .get(position)
                .copied()
                .ok_or_else(|| Unsupported(format!("missing source {position}")))?;
            walk.source(operand, dest_lanes, scalar)
        };
        let vector = |walk: &mut Self, position: usize, count: u8, scalar: ScalarType| {
            let operand = sources
                .get(position)
                .copied()
                .ok_or_else(|| Unsupported(format!("missing source {position}")))?;
            let full: Vec<u8> = (0..count).collect();
            walk.source(operand, &full, scalar)
        };
        let scalar_arg = |walk: &mut Self, position: usize, scalar: ScalarType| {
            let operand = sources
                .get(position)
                .copied()
                .ok_or_else(|| Unsupported(format!("missing source {position}")))?;
            walk.source(operand, &[select_lane(&operand.components)], scalar)
        };
        // Componentwise operand interpretation per opcode family.
        let operand_scalar = operand_scalar(opcode);
        let result = result_scalar(opcode, bindings, &self.declarations);
        let mut operands: Vec<LiftExpr> = Vec::new();
        match opcode {
            Opcode::Dp2 => {
                operands.push(vector(self, 0, 2, ScalarType::F32)?);
                operands.push(vector(self, 1, 2, ScalarType::F32)?);
            }
            Opcode::Dp3 => {
                operands.push(vector(self, 0, 3, ScalarType::F32)?);
                operands.push(vector(self, 1, 3, ScalarType::F32)?);
            }
            Opcode::Dp4 => {
                operands.push(vector(self, 0, 4, ScalarType::F32)?);
                operands.push(vector(self, 1, 4, ScalarType::F32)?);
            }
            Opcode::Movc => {
                operands.push(source(self, 0, ScalarType::U32)?);
                operands.push(source(self, 1, ScalarType::Bits)?);
                operands.push(source(self, 2, ScalarType::Bits)?);
            }
            Opcode::Sample | Opcode::SampleL | Opcode::SampleB => {
                let coordinates = coordinate_lanes(bindings, &self.declarations, false);
                let operand = sources
                    .first()
                    .copied()
                    .ok_or_else(|| Unsupported("sample without a coordinate".into()))?;
                operands.push(self.source(operand, coordinates, ScalarType::F32)?);
                if sources.len() > 1 {
                    operands.push(scalar_arg(self, 1, ScalarType::F32)?);
                }
            }
            Opcode::Ld => {
                let coordinates = coordinate_lanes(bindings, &self.declarations, true);
                let operand = sources
                    .first()
                    .copied()
                    .ok_or_else(|| Unsupported("ld without a coordinate".into()))?;
                operands.push(self.source(operand, coordinates, ScalarType::I32)?);
            }
            Opcode::LdMs => {
                let operand = sources
                    .first()
                    .copied()
                    .ok_or_else(|| Unsupported("ld_ms without a coordinate".into()))?;
                operands.push(self.source(operand, &[0, 1], ScalarType::I32)?);
                operands.push(scalar_arg(self, 1, ScalarType::I32)?);
            }
            Opcode::Gather4 => {
                let operand = sources
                    .first()
                    .copied()
                    .ok_or_else(|| Unsupported("gather4 without a coordinate".into()))?;
                operands.push(self.source(operand, &[0, 1], ScalarType::F32)?);
            }
            Opcode::LdStructured => {
                operands.push(scalar_arg(self, 0, ScalarType::U32)?);
                operands.push(scalar_arg(self, 1, ScalarType::U32)?);
            }
            Opcode::Resinfo => {
                operands.push(scalar_arg(self, 0, ScalarType::U32)?);
            }
            _ => {
                for position in 0..sources.len() {
                    operands.push(source(self, position, operand_scalar)?);
                }
            }
        }
        let head = AluHead {
            opcode,
            saturate: instruction.saturate,
            dest_position,
            bindings: Self::alu_bindings(bindings, dest_lanes)?,
            tex_offsets: instruction.tex_offsets,
            resinfo_return_type: instruction.resinfo_return_type,
        };
        Ok(Expr::Apply {
            operator: RtlOp::Alu(head),
            operands,
            shape: ValueShape::vector(result, width),
        })
    }

    /// Resolves binding operands, mapping their swizzles through the
    /// destination lanes.
    pub(super) fn alu_bindings(
        bindings: &[&Operand],
        dest_lanes: &[u8],
    ) -> Result<Vec<Binding>, Unsupported> {
        bindings
            .iter()
            .map(|operand| {
                let file = match operand.reg_type {
                    RegisterType::Resource => RegFile::Resource,
                    RegisterType::Sampler => RegFile::Sampler,
                    _ => RegFile::Uav,
                };
                let index = match operand.indices.first() {
                    Some(OperandIndex::Imm32(index)) => *index,
                    _ => return unsupported("binding with a computed index"),
                };
                let lanes = match &operand.components {
                    ComponentSelect::Swizzle(swizzle) => dest_lanes
                        .iter()
                        .map(|&lane| swizzle[usize::from(lane)])
                        .collect(),
                    ComponentSelect::Scalar(lane) => vec![*lane; dest_lanes.len()],
                    _ => dest_lanes.to_vec(),
                };
                Ok(Binding {
                    register: Register { file, index },
                    lanes,
                })
            })
            .collect()
    }
}
