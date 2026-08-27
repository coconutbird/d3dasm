//! Value rendering: expression trees, operations, and memory places.

use cfglib::ir::hlil::{ExpressionId, ExpressionKind};
use cfglib::ir::rtl::{ScalarType, ValueShape, VarExpr};
use dxbc::shex::Opcode;

use crate::dialect::{AluHead, MemRef, RegFile, Register, RtlOp, SystemInput};
use crate::lift::Unsupported;

use super::{
    CB_NAMING_MISS, Expr, HullMode, Op, Rendered, Renderer, binary, call1, call2, cast_name,
    literal, scalar_keyword, select_call_lanes, splat, suffix, texel_offsets, width_text,
};

impl Renderer<'_> {
    pub(super) fn value(
        &self,
        expr: &Expr,
        operands: &[ExpressionId],
        cursor: &mut usize,
        context: ScalarType,
    ) -> Result<Rendered, Unsupported> {
        match expr {
            VarExpr::Read {
                positions, scalar, ..
            } => {
                let wanted = if *scalar == ScalarType::Bits {
                    context
                } else {
                    *scalar
                };
                self.read(positions, wanted, operands, cursor)
            }
            VarExpr::Const { bits, shape } => {
                let wanted = if shape.scalar == ScalarType::Bits {
                    context
                } else {
                    shape.scalar
                };
                Ok(Rendered::atom(literal(bits, wanted), wanted))
            }
            VarExpr::Apply {
                operator,
                operands: inner,
                shape,
            } => self.apply(operator, inner, *shape, operands, cursor, context),
            VarExpr::Reinterpret { operand, shape } => {
                let value = self.value(operand, operands, cursor, context)?;
                Ok(Rendered::atom(
                    format!("{}({})", cast_name(shape.scalar), value.text),
                    shape.scalar,
                ))
            }
            VarExpr::Compose { parts, shape } => {
                let wanted = if shape.scalar == ScalarType::Bits {
                    context
                } else {
                    shape.scalar
                };
                let rendered: Vec<String> = parts
                    .iter()
                    .map(|part| {
                        self.value(part, operands, cursor, wanted)
                            .map(|part| part.cast_bare(wanted))
                    })
                    .collect::<Result<_, _>>()?;
                Ok(Rendered::atom(
                    format!(
                        "{}{}({})",
                        scalar_keyword(wanted),
                        shape.lanes,
                        rendered.join(", ")
                    ),
                    wanted,
                ))
            }
        }
    }

    /// Renders one web read (or an inlined producer at that position).
    pub(super) fn read(
        &self,
        positions: &[u8],
        wanted: ScalarType,
        operands: &[ExpressionId],
        cursor: &mut usize,
    ) -> Result<Rendered, Unsupported> {
        let id = operands
            .get(*cursor)
            .copied()
            .ok_or_else(|| Unsupported("read without an operand".into()))?;
        *cursor += 1;
        match self.expression(id)?.kind() {
            ExpressionKind::Variable(variable) => {
                let (index, web) = self.hlil_web(*variable)?;
                if let Some(Register {
                    file: RegFile::PatchConstant,
                    index: register,
                }) = web.storage
                {
                    return self.patch_read(register, web, positions, wanted);
                }
                if let Some(Register {
                    file: RegFile::System(SystemInput::ForkInstanceId),
                    ..
                }) = web.storage
                {
                    // Instances unroll: the id is a literal per copy.
                    let value = Rendered::atom(format!("{}u", self.fork_instance), ScalarType::U32);
                    return Ok(Rendered::atom(value.cast(wanted), wanted));
                }
                let name = &self.web_names[index];
                let holder = self.web_holder(web);
                let selected = if Self::is_input_web(web) {
                    // Interface registers are full float4 storage: map
                    // web positions back to the storage lanes they name.
                    let lanes: Vec<u8> = positions
                        .iter()
                        .map(|&position| web.lanes.get(usize::from(position)).copied().unwrap_or(0))
                        .collect();
                    format!("{name}.{}", suffix(&lanes))
                } else {
                    let full: Vec<u8> = (0..web.shape.lanes.max(1)).collect();
                    if positions == full.as_slice() {
                        name.clone()
                    } else {
                        format!("{name}.{}", suffix(positions))
                    }
                };
                let value = Rendered::atom(selected, holder);
                Ok(Rendered::atom(value.cast(wanted), wanted))
            }
            ExpressionKind::Operation {
                operation:
                    Op::Assign {
                        positions: written,
                        value,
                        ..
                    },
                operands: inner,
            } => {
                let mut inner_cursor = 0usize;
                let rendered = self.value(value, inner, &mut inner_cursor, wanted)?;
                // Remap the read's positions through the producer's
                // written order.
                let identity = positions == written.as_slice();
                if identity {
                    Ok(Rendered {
                        text: rendered.cast(wanted),
                        scalar: wanted,
                        atomic: true,
                    })
                } else {
                    let mapped: Vec<u8> = positions
                        .iter()
                        .map(|position| {
                            written
                                .iter()
                                .position(|written| written == position)
                                .and_then(|index| u8::try_from(index).ok())
                                .ok_or_else(|| {
                                    Unsupported("read outside the inlined producer".into())
                                })
                        })
                        .collect::<Result<_, _>>()?;
                    let swizzled = format!("{}.{}", rendered.wrapped(), suffix(&mapped));
                    let value = Rendered::atom(swizzled, rendered.scalar);
                    Ok(Rendered::atom(value.cast(wanted), wanted))
                }
            }
            _ => Err(Unsupported("read of an unexpected expression".into())),
        }
    }

    /// Renders one operator application.
    pub(super) fn apply(
        &self,
        operator: &RtlOp,
        inner: &[Expr],
        shape: ValueShape,
        operands: &[ExpressionId],
        cursor: &mut usize,
        context: ScalarType,
    ) -> Result<Rendered, Unsupported> {
        let produced = if shape.scalar == ScalarType::Bits {
            context
        } else {
            shape.scalar
        };
        match operator {
            RtlOp::Neg => {
                let value = self.value(&inner[0], operands, cursor, context)?;
                let scalar = value.scalar;
                Ok(Rendered::composite(format!("-{}", value.wrapped()), scalar))
            }
            RtlOp::Abs => {
                let value = self.value(&inner[0], operands, cursor, context)?;
                let scalar = value.scalar;
                Ok(Rendered::atom(format!("abs({})", value.text), scalar))
            }
            RtlOp::Test { nonzero } => {
                let value = self.value(&inner[0], operands, cursor, ScalarType::U32)?;
                let text = value.cast(ScalarType::U32);
                let symbol = if *nonzero { "!=" } else { "==" };
                Ok(Rendered::composite(
                    format!("{text} {symbol} 0u"),
                    ScalarType::Bool,
                ))
            }
            RtlOp::LoadInput {
                vertex,
                register,
                lanes,
            } => {
                let holder = if self
                    .semantics
                    .inputs
                    .get(register)
                    .is_some_and(|row| row.integral)
                {
                    ScalarType::U32
                } else {
                    ScalarType::F32
                };
                let patch = if self.hull_mode == HullMode::None {
                    "vertices"
                } else {
                    "patch"
                };
                let selected = format!("{patch}[{vertex}].v{register}.{}", suffix(lanes));
                let value = Rendered::atom(selected, holder);
                Ok(Rendered::atom(value.cast(produced), produced))
            }
            RtlOp::LoadInputDynamic {
                vertex,
                base,
                lanes,
            } => {
                let index = self
                    .value(&inner[0], operands, cursor, ScalarType::U32)?
                    .cast(ScalarType::U32);
                let element = if *base == 0 {
                    format!("_icp{vertex}[{index}]")
                } else {
                    format!("_icp{vertex}[{index} + {base}]")
                };
                let selected = format!("{element}.{}", suffix(lanes));
                let value = Rendered::atom(selected, ScalarType::F32);
                Ok(Rendered::atom(value.cast(produced), produced))
            }
            RtlOp::LoadMem(reference) => {
                let address = inner.first();
                let place = self.memory(reference, address, operands, cursor)?;
                let selected = if reference.lanes.as_slice() == [0, 1, 2, 3] {
                    place
                } else {
                    format!("{place}.{}", suffix(&reference.lanes))
                };
                let holder = match reference.file {
                    RegFile::ConstantBuffer => ScalarType::F32,
                    _ => ScalarType::U32,
                };
                let value = Rendered::atom(selected, holder);
                Ok(Rendered::atom(value.cast(produced), produced))
            }
            RtlOp::Alu(head) => self.alu(head, inner, shape, produced, operands, cursor),
        }
    }

    /// Renders one retained decoded operation.
    #[expect(clippy::too_many_lines, reason = "one arm per opcode")]
    pub(super) fn alu(
        &self,
        head: &AluHead,
        inner: &[Expr],
        shape: ValueShape,
        produced: ScalarType,
        operands: &[ExpressionId],
        cursor: &mut usize,
    ) -> Result<Rendered, Unsupported> {
        let opcode = head.opcode;
        let width = shape.lanes;
        let operand_context = match opcode {
            // Transports render their data operands in the surrounding
            // interpretation, never forcing a float round-trip.
            Opcode::Mov | Opcode::Movc => produced,
            _ => ScalarType::F32,
        };
        macro_rules! arg {
            ($index:expr) => {{
                let expr = inner
                    .get($index)
                    .ok_or_else(|| Unsupported(format!("missing operand {}", $index)))?;
                self.value(expr, operands, cursor, operand_context)
            }};
        }
        let rendered = match opcode {
            Opcode::Mov => {
                let value = arg!(0)?;
                Rendered {
                    text: value.cast_bare(produced),
                    scalar: produced,
                    atomic: value.atomic,
                }
            }
            Opcode::Movc => {
                let condition = {
                    let expr = &inner[0];
                    if let Some(text) = self.inlined_comparison(expr, operands, cursor)? {
                        text
                    } else {
                        let value = self.value(expr, operands, cursor, ScalarType::U32)?;
                        format!("{} != 0u", value.cast(ScalarType::U32))
                    }
                };
                let left = arg!(1)?.cast(produced);
                let right = arg!(2)?.cast(produced);
                Rendered::composite(format!("({condition}) ? {left} : {right}"), produced)
            }
            Opcode::Add | Opcode::Iadd => binary(&arg!(0)?, "+", &arg!(1)?, produced),
            Opcode::Mul => binary(&arg!(0)?, "*", &arg!(1)?, produced),
            Opcode::Div => binary(&arg!(0)?, "/", &arg!(1)?, produced),
            Opcode::Mad | Opcode::IMad | Opcode::UMad => {
                let a = arg!(0)?.wrapped();
                let b = arg!(1)?.wrapped();
                let c = arg!(2)?.wrapped();
                Rendered::composite(format!("{a} * {b} + {c}"), produced)
            }
            Opcode::Min | Opcode::IMin | Opcode::UMin => {
                call2("min", &arg!(0)?, &arg!(1)?, produced)
            }
            Opcode::Max | Opcode::IMax | Opcode::UMax => {
                call2("max", &arg!(0)?, &arg!(1)?, produced)
            }
            Opcode::Frc => call1("frac", &arg!(0)?, produced),
            Opcode::Sqrt => call1("sqrt", &arg!(0)?, produced),
            Opcode::Rsq => call1("rsqrt", &arg!(0)?, produced),
            Opcode::Exp => call1("exp2", &arg!(0)?, produced),
            Opcode::Log => call1("log2", &arg!(0)?, produced),
            Opcode::RoundNe => call1("round", &arg!(0)?, produced),
            Opcode::RoundNi => call1("floor", &arg!(0)?, produced),
            Opcode::RoundPi => call1("ceil", &arg!(0)?, produced),
            Opcode::RoundZ => call1("trunc", &arg!(0)?, produced),
            Opcode::Rcp => call1("rcp", &arg!(0)?, produced),
            Opcode::DerivRtx => call1("ddx", &arg!(0)?, produced),
            Opcode::DerivRty => call1("ddy", &arg!(0)?, produced),
            Opcode::DerivRtxCoarse => call1("ddx_coarse", &arg!(0)?, produced),
            Opcode::DerivRtxFine => call1("ddx_fine", &arg!(0)?, produced),
            Opcode::DerivRtyCoarse => call1("ddy_coarse", &arg!(0)?, produced),
            Opcode::DerivRtyFine => call1("ddy_fine", &arg!(0)?, produced),
            Opcode::Dp2 | Opcode::Dp3 | Opcode::Dp4 => call2("dot", &arg!(0)?, &arg!(1)?, produced),
            Opcode::INeg => {
                let value = arg!(0)?;
                Rendered::composite(format!("-{}", value.wrapped()), produced)
            }
            Opcode::And => binary(&arg!(0)?, "&", &arg!(1)?, produced),
            Opcode::Or => binary(&arg!(0)?, "|", &arg!(1)?, produced),
            Opcode::Xor => binary(&arg!(0)?, "^", &arg!(1)?, produced),
            Opcode::Not => {
                let value = arg!(0)?;
                Rendered::composite(format!("~{}", value.wrapped()), produced)
            }
            Opcode::Ishl => binary(&arg!(0)?, "<<", &arg!(1)?, produced),
            Opcode::Ishr | Opcode::Ushr => binary(&arg!(0)?, ">>", &arg!(1)?, produced),
            Opcode::Itof | Opcode::Utof => {
                let value = arg!(0)?;
                Rendered::atom(
                    format!("float{}({})", width_text(width), value.text),
                    produced,
                )
            }
            Opcode::Ftoi => {
                let value = arg!(0)?;
                Rendered::atom(
                    format!("int{}({})", width_text(width), value.text),
                    produced,
                )
            }
            Opcode::Ftou => {
                let value = arg!(0)?;
                Rendered::atom(
                    format!("uint{}({})", width_text(width), value.text),
                    produced,
                )
            }
            Opcode::Lt
            | Opcode::Ge
            | Opcode::Eq
            | Opcode::Ne
            | Opcode::ILt
            | Opcode::IGe
            | Opcode::IEq
            | Opcode::INe
            | Opcode::ULt
            | Opcode::UGe => {
                let comparison = self.comparison(head, inner, operands, cursor)?;
                let ones = splat("0xffffffffu", width);
                let zeros = splat("0u", width);
                Rendered::composite(format!("({comparison}) ? {ones} : {zeros}"), produced)
            }
            Opcode::Bfi => {
                // Bitfield insert: bits = ((1 << width) - 1) << offset;
                // dest = ((insert << offset) & bits) | (base & ~bits).
                let ones = splat("1u", width);
                let masks = splat("31u", width);
                let w = arg!(0)?.cast(produced);
                let o = arg!(1)?.cast(produced);
                let insert = arg!(2)?.cast(produced);
                let base = arg!(3)?.cast(produced);
                let bits = format!("((({ones} << ({w} & {masks})) - {ones}) << ({o} & {masks}))");
                Rendered::composite(
                    format!("(({insert} << ({o} & {masks})) & {bits}) | ({base} & ~{bits})"),
                    produced,
                )
            }
            Opcode::Sincos => {
                // Operand slot 0 is sin, slot 1 cos; the surviving
                // destination selects the meaning.
                let intrinsic = if head.dest_position == 1 {
                    "cos"
                } else {
                    "sin"
                };
                call1(intrinsic, &arg!(0)?, produced)
            }
            Opcode::UDiv => {
                // Operand slot 0 is the quotient, slot 1 the remainder;
                // the surviving destination selects the meaning.
                let symbol = if head.dest_position == 1 { "%" } else { "/" };
                binary(&arg!(0)?, symbol, &arg!(1)?, produced)
            }
            Opcode::Sample | Opcode::SampleL | Opcode::SampleB => {
                self.sample(head, inner, operands, cursor, produced)?
            }
            Opcode::Ld => self.texel_load(head, inner, operands, cursor, produced)?,
            Opcode::LdMs => self.texel_load_ms(head, inner, operands, cursor, produced)?,
            Opcode::Gather4 => self.gather(head, inner, operands, cursor, produced)?,
            Opcode::LdStructured => self.structured_load(head, inner, shape, operands, cursor)?,
            other => {
                return Err(Unsupported(format!("opcode {}", other.name())));
            }
        };
        if head.saturate {
            Ok(Rendered::atom(
                format!("saturate({})", rendered.text),
                rendered.scalar,
            ))
        } else {
            Ok(rendered)
        }
    }

    pub(super) fn sample(
        &self,
        head: &AluHead,
        inner: &[Expr],
        operands: &[ExpressionId],
        cursor: &mut usize,
        produced: ScalarType,
    ) -> Result<Rendered, Unsupported> {
        let [resource, sampler] = head.bindings.as_slice() else {
            return Err(Unsupported("sample without resource and sampler".into()));
        };
        let coordinate = self
            .value(&inner[0], operands, cursor, ScalarType::F32)?
            .cast_bare(ScalarType::F32);
        let texture = self.texture_name(resource.register.index);
        let sampler = self.sampler_name(sampler.register.index);
        let dimensions = inner[0].shape().lanes.min(3);
        let offsets = texel_offsets(head, usize::from(dimensions));
        let call = match head.opcode {
            Opcode::SampleL => {
                let level = self
                    .value(&inner[1], operands, cursor, ScalarType::F32)?
                    .cast_bare(ScalarType::F32);
                format!("{texture}.SampleLevel({sampler}, {coordinate}, {level}{offsets})")
            }
            Opcode::SampleB => {
                let bias = self
                    .value(&inner[1], operands, cursor, ScalarType::F32)?
                    .cast_bare(ScalarType::F32);
                format!("{texture}.SampleBias({sampler}, {coordinate}, {bias}{offsets})")
            }
            _ => format!("{texture}.Sample({sampler}, {coordinate}{offsets})"),
        };
        Ok(Rendered::atom(
            select_call_lanes(call, &resource.lanes),
            produced,
        ))
    }

    pub(super) fn texel_load(
        &self,
        head: &AluHead,
        inner: &[Expr],
        operands: &[ExpressionId],
        cursor: &mut usize,
        produced: ScalarType,
    ) -> Result<Rendered, Unsupported> {
        let [resource] = head.bindings.as_slice() else {
            return Err(Unsupported("ld without a resource".into()));
        };
        let coordinate = self
            .value(&inner[0], operands, cursor, ScalarType::I32)?
            .cast_bare(ScalarType::I32);
        let dimensions = usize::from(inner[0].shape().lanes.max(2)) - 1;
        let offsets = texel_offsets(head, dimensions.max(1));
        let call = format!(
            "{}.Load({coordinate}{offsets})",
            self.texture_name(resource.register.index)
        );
        Ok(Rendered::atom(
            select_call_lanes(call, &resource.lanes),
            produced,
        ))
    }

    pub(super) fn texel_load_ms(
        &self,
        head: &AluHead,
        inner: &[Expr],
        operands: &[ExpressionId],
        cursor: &mut usize,
        produced: ScalarType,
    ) -> Result<Rendered, Unsupported> {
        let [resource] = head.bindings.as_slice() else {
            return Err(Unsupported("ld_ms without a resource".into()));
        };
        let coordinate = self
            .value(&inner[0], operands, cursor, ScalarType::I32)?
            .cast_bare(ScalarType::I32);
        let sample = self
            .value(&inner[1], operands, cursor, ScalarType::I32)?
            .cast_bare(ScalarType::I32);
        let call = format!(
            "{}.Load({coordinate}, {sample})",
            self.texture_name(resource.register.index)
        );
        Ok(Rendered::atom(
            select_call_lanes(call, &resource.lanes),
            produced,
        ))
    }

    pub(super) fn gather(
        &self,
        head: &AluHead,
        inner: &[Expr],
        operands: &[ExpressionId],
        cursor: &mut usize,
        produced: ScalarType,
    ) -> Result<Rendered, Unsupported> {
        let [resource, sampler] = head.bindings.as_slice() else {
            return Err(Unsupported("gather4 without resource and sampler".into()));
        };
        let coordinate = self
            .value(&inner[0], operands, cursor, ScalarType::F32)?
            .cast_bare(ScalarType::F32);
        let channel = sampler.lanes.first().copied().unwrap_or(0);
        let intrinsic = match channel {
            1 => "GatherGreen",
            2 => "GatherBlue",
            3 => "GatherAlpha",
            _ => "GatherRed",
        };
        let offsets = texel_offsets(head, 2);
        let call = format!(
            "{}.{intrinsic}({}, {coordinate}{offsets})",
            self.texture_name(resource.register.index),
            self.sampler_name(sampler.register.index)
        );
        Ok(Rendered::atom(
            select_call_lanes(call, &resource.lanes),
            produced,
        ))
    }

    /// Renders one `ld_structured` element read: per result lane, one
    /// dword of the element at the byte offset, selected by the resource
    /// swizzle.
    pub(super) fn structured_load(
        &self,
        head: &AluHead,
        inner: &[Expr],
        shape: ValueShape,
        operands: &[ExpressionId],
        cursor: &mut usize,
    ) -> Result<Rendered, Unsupported> {
        let [resource] = head.bindings.as_slice() else {
            return Err(Unsupported("ld_structured without a resource".into()));
        };
        let element = self
            .value(&inner[0], operands, cursor, ScalarType::U32)?
            .cast(ScalarType::U32);
        // The byte offset must be an immediate to name a struct field.
        let VarExpr::Const { bits, .. } = &inner[1] else {
            return Err(Unsupported("computed structured offset".into()));
        };
        let offset_bits = u32::try_from(bits.first().copied().unwrap_or(0)).unwrap_or(0);
        if offset_bits % 4 != 0 {
            return Err(Unsupported("unaligned structured offset".into()));
        }
        let base = offset_bits / 4;
        let texture = self.texture_name(resource.register.index);
        let rendered: Vec<String> = resource
            .lanes
            .iter()
            .map(|&lane| {
                let dword = base + u32::from(lane);
                format!("{texture}[{element}].value[{dword}]")
            })
            .collect();
        Ok(Rendered::atom(
            if rendered.len() == 1 {
                rendered.into_iter().next().expect("one lane renders")
            } else {
                format!("uint{}({})", shape.lanes, rendered.join(", "))
            },
            ScalarType::U32,
        ))
    }

    /// Renders one `resinfo` as a `GetDimensions` block assigning the
    /// selected dimensions into the destination place.
    pub(super) fn resinfo(
        &mut self,
        place: &str,
        target_scalar: ScalarType,
        head: &AluHead,
        inner: &[Expr],
        operands: &[ExpressionId],
    ) -> Result<(), Unsupported> {
        let [resource] = head.bindings.as_slice() else {
            return Err(Unsupported("resinfo without a resource".into()));
        };
        let dimension = self
            .declarations
            .resources
            .iter()
            .find(|(index, ..)| *index == resource.register.index)
            .map_or("texture2d", |(_, dimension, _)| dimension);
        let mut cursor = 0usize;
        let mip = self
            .value(&inner[0], operands, &mut cursor, ScalarType::U32)?
            .cast_bare(ScalarType::U32);
        let texture = self.texture_name(resource.register.index);
        // Result layout: [width, height, depth-or-elements, mip count].
        let (arguments, filled): (&str, [bool; 4]) = match dimension {
            "texture1d" => ("rw, rm", [true, false, false, true]),
            "texture3d" | "texture2darray" | "texturecubearray" => {
                ("rw, rh, rd, rm", [true, true, true, true])
            }
            _ => ("rw, rh, rm", [true, true, false, true]),
        };
        let uint = head.resinfo_return_type == Some(2);
        let produced = if uint {
            ScalarType::U32
        } else {
            ScalarType::F32
        };
        let scalars = ["rw", "rh", "rd", "rm"];
        let selected: Vec<String> = resource
            .lanes
            .iter()
            .map(|&lane| {
                let index = usize::from(lane);
                if filled[index] {
                    let scalar = scalars[index];
                    if uint {
                        scalar.to_owned()
                    } else {
                        format!("float({scalar})")
                    }
                } else if uint {
                    "0u".to_owned()
                } else {
                    "0.0".to_owned()
                }
            })
            .collect();
        let constructor = if uint { "uint" } else { "float" };
        let value = if selected.len() == 1 {
            selected.into_iter().next().expect("one lane renders")
        } else {
            format!("{constructor}{}({})", selected.len(), selected.join(", "))
        };
        let value = Rendered::atom(value, produced).cast_bare(target_scalar);
        self.line("{");
        self.indent += 1;
        self.line("uint rw, rh, rd, rm;");
        self.line("rd = 0; rw = 0; rh = 0; rm = 0;");
        self.line(&format!("{texture}.GetDimensions({mip}, {arguments});"));
        self.line(&format!("{place} = {value};"));
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    /// Renders one addressable-file place expression.
    pub(super) fn memory(
        &self,
        reference: &MemRef,
        address: Option<&Expr>,
        operands: &[ExpressionId],
        cursor: &mut usize,
    ) -> Result<String, Unsupported> {
        if reference.file == RegFile::ConstantBuffer {
            return self.constant_buffer_place(reference, address, operands, cursor);
        }
        // The element index position depends on the file: `icb` has no
        // register-selecting index.
        let (base, element_position) = match reference.file {
            RegFile::IndexableTemp => (
                format!("x{}", reference.indices.first().copied().unwrap_or(0)),
                1,
            ),
            RegFile::ImmediateConstantBuffer => ("icb".to_owned(), 0),
            _ => return Err(Unsupported("addressable file".into())),
        };
        let Some(&offset) = reference.indices.get(element_position) else {
            return Err(Unsupported("unindexed addressable read".into()));
        };
        let index = self.memory_index(reference, address, operands, cursor, offset)?;
        // A dynamically indexed icb read is bounds-defined in DXBC (out
        // of range reads zero) but undefined for an HLSL const array,
        // where fxc lowers assuming the index is in range. An explicit
        // guard restores the exact semantics — applied by the caller via
        // the read path for icb loads.
        if reference.file == RegFile::ImmediateConstantBuffer && reference.relative_to.is_some() {
            let count = self.declarations.immediate_constants.len();
            return Ok(format!(
                "(({index}) < {count}u ? icb[{index}] : uint4(0u, 0u, 0u, 0u))"
            ));
        }
        Ok(format!("{base}[{index}]"))
    }

    pub(super) fn memory_index(
        &self,
        reference: &MemRef,
        address: Option<&Expr>,
        operands: &[ExpressionId],
        cursor: &mut usize,
        offset: u32,
    ) -> Result<String, Unsupported> {
        if reference.relative_to.is_some() {
            let address =
                address.ok_or_else(|| Unsupported("relative reference without address".into()))?;
            let wanted = if reference.file == RegFile::ImmediateConstantBuffer {
                ScalarType::U32
            } else {
                ScalarType::I32
            };
            let rendered = self.value(address, operands, cursor, wanted)?.cast(wanted);
            if offset == 0 {
                Ok(rendered)
            } else {
                Ok(format!("{rendered} + {offset}"))
            }
        } else {
            Ok(offset.to_string())
        }
    }

    /// Renders one constant-buffer element place, resolving RDEF-named
    /// fields when they cover the accessed register.
    pub(super) fn constant_buffer_place(
        &self,
        reference: &MemRef,
        address: Option<&Expr>,
        operands: &[ExpressionId],
        cursor: &mut usize,
    ) -> Result<String, Unsupported> {
        let slot = reference.indices.first().copied().unwrap_or(0);
        let Some(&offset) = reference.indices.get(1) else {
            return Err(Unsupported("unindexed constant-buffer read".into()));
        };
        let index_text = if reference.relative_to.is_some() {
            let address =
                address.ok_or_else(|| Unsupported("relative reference without address".into()))?;
            Some(
                self.value(address, operands, cursor, ScalarType::I32)?
                    .cast(ScalarType::I32),
            )
        } else {
            None
        };
        if let Some(fields) = self
            .resources
            .constant_buffers
            .get(&slot)
            .and_then(|naming| naming.fields.as_ref())
        {
            let field = fields
                .iter()
                .find(|field| offset >= field.register && offset < field.register + field.span)
                .ok_or_else(|| Unsupported(CB_NAMING_MISS.into()))?;
            if field.span == 1 {
                if index_text.is_some() {
                    // Dynamic indexing through a single-register field
                    // has no spelling; the caller retries unnamed.
                    return Err(Unsupported(CB_NAMING_MISS.into()));
                }
                return Ok(field.name.clone());
            }
            let base = offset - field.register;
            let index = match index_text {
                Some(text) if base == 0 => text,
                Some(text) => format!("{text} + {base}"),
                None => base.to_string(),
            };
            return Ok(format!("{}[{index}]", field.name));
        }
        let index = match index_text {
            Some(text) if offset == 0 => text,
            Some(text) => format!("{text} + {offset}"),
            None => offset.to_string(),
        };
        Ok(format!("cb{slot}[{index}]"))
    }
}
