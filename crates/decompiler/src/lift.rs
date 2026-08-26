//! SM4/SM5 structured instruction stream → shader-dialect MLIL.
//!
//! The walk mirrors the bytecode's structured markers (`if`/`else`/
//! `endif`, `loop`/`endloop`, conditional transfers) into explicit blocks
//! and edges, materializing non-register operand reads (immediates,
//! constant buffers, indexable temporaries) as pure value producers so
//! HLIL expression inlining folds them back into their consumers. Blocks
//! are created lazily when an instruction lands, so no empty semantic
//! block ever reaches verification.

use std::collections::BTreeMap;

use cfglib::BlockId;
use cfglib::ir::mlil::{FunctionBuilder, TypedVariable};
use dxbc::shex::{
    ComponentSelect, Instruction, InstructionKind, Opcode, Operand, OperandIndex, Program,
    RegisterType,
};

use crate::dialect::{
    AluOp, Binding, DestAccess, Edge, Function, Immediate, MemRef, Op, RegFile, Register, Role,
    ShaderDialect, SourceAccess, SystemInput, ValueType, VariableId,
};

/// One declared input register row.
#[derive(Debug, Clone)]
pub struct InputRow {
    /// The input register.
    pub register: Register,
    /// Declared component selection.
    pub components: ComponentSelect,
    /// System-value name, when declared.
    pub system_value: Option<&'static str>,
    /// Pixel-shader interpolation mode string, when declared.
    pub interpolation: Option<&'static str>,
}

/// A declaration retained for header rendering.
#[derive(Debug, Clone)]
pub struct Declarations {
    /// `dcl_sampler` register indices with their mode strings.
    pub samplers: Vec<(u32, &'static str)>,
    /// `dcl_resource` entries: register, dimension, return types.
    pub resources: Vec<(u32, &'static str, [dxbc::shex::ReturnType; 4])>,
    /// `dcl_constantbuffer` entries: register and vector count.
    pub constant_buffers: Vec<(u32, u32)>,
    /// Input register rows: register, components, system value,
    /// interpolation mode.
    pub inputs: Vec<InputRow>,
    /// Output register indices with any system-value name.
    pub outputs: Vec<(Register, ComponentSelect, Option<&'static str>)>,
    /// `dcl_indexableTemp` entries: register, length, component count.
    pub indexable_temps: Vec<(u32, u32, u32)>,
    /// Immediate constant buffer rows, when declared.
    pub immediate_constants: Vec<[f32; 4]>,
    /// `dcl_resource_structured` entries: register and byte stride.
    pub structured: Vec<(u32, u32)>,
    /// `dcl_thread_group` sizes for compute shaders.
    pub thread_group: Option<(u32, u32, u32)>,
}

/// One lifted shader body plus its rendering metadata.
pub struct LiftedShader {
    /// The verified MLIL function.
    pub function: Function,
    /// Declarations for the HLSL header.
    pub declarations: Declarations,
}

/// A shape the lift does not model yet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unsupported(pub String);

impl core::fmt::Display for Unsupported {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "unsupported shader shape: {}", self.0)
    }
}

impl std::error::Error for Unsupported {}

fn unsupported<T>(message: impl Into<String>) -> Result<T, Unsupported> {
    Err(Unsupported(message.into()))
}

/// One structural nesting frame during the walk.
enum Frame {
    If {
        /// Where the false edge resumes when no `else` intervenes, or
        /// the pending else entry.
        pending_false: Option<(BlockId, Edge)>,
        /// Transfers waiting for the merge block.
        pending_merge: Vec<(BlockId, Edge)>,
        /// Whether the `else` arm was entered.
        saw_else: bool,
    },
    Loop {
        /// The loop header (back-edge target).
        header: BlockId,
        /// Transfers waiting for the loop exit.
        pending_exit: Vec<(BlockId, Edge)>,
    },
}

struct Walk {
    builder: FunctionBuilder<ShaderDialect>,
    variables: BTreeMap<Register, VariableId>,
    outputs: Vec<VariableId>,
    /// The open block instructions are appended to, if any.
    current: Option<BlockId>,
    /// Edges waiting to enter the next materialized block.
    pending: Vec<(BlockId, Edge)>,
    frames: Vec<Frame>,
    blocks: u32,
    /// Whether any block was ever materialized (guards a degenerate
    /// empty program).
    ret_block: Option<BlockId>,
}

impl Walk {
    /// The block instructions land in, materializing it if needed.
    fn landing(&mut self) -> BlockId {
        if let Some(block) = self.current {
            return block;
        }
        let block = self.builder.new_block(format!("b{}", self.blocks));
        self.blocks += 1;
        for (source, edge) in self.pending.drain(..) {
            self.builder
                .add_edge(source, block, edge, None)
                .expect("pending edges name live blocks");
        }
        self.current = Some(block);
        block
    }

    fn variable(&mut self, register: Register) -> VariableId {
        if let Some(&variable) = self.variables.get(&register) {
            return variable;
        }
        let role = match register.file {
            RegFile::Input | RegFile::System(_) => Role::Input,
            RegFile::Output | RegFile::OutputDepth | RegFile::OutputCoverage => Role::Output,
            _ => Role::Temp,
        };
        let variable = self
            .builder
            .declare_variable(role, Some(register))
            .expect("variable identity space is far from exhausted");
        self.variables.insert(register, variable);
        if matches!(role, Role::Output) {
            self.outputs.push(variable);
        }
        variable
    }

    fn fresh_temp(&mut self) -> VariableId {
        self.builder
            .declare_variable(Role::Temp, None)
            .expect("variable identity space is far from exhausted")
    }

    fn append(
        &mut self,
        operation: Op,
        uses: Vec<TypedVariable<ShaderDialect>>,
        defs: Vec<TypedVariable<ShaderDialect>>,
        span: (u32, u32),
    ) {
        let block = self.landing();
        self.builder
            .append_instruction(block, operation, uses, defs, false, Some(span))
            .expect("instruction identity space is far from exhausted");
    }
}

fn typed(variable: VariableId) -> TypedVariable<ShaderDialect> {
    TypedVariable::new(variable, ValueType::VEC4)
}

/// Maps a variable-backed register file, or `None` for files that
/// materialize differently.
fn variable_file(register: RegisterType) -> Option<RegFile> {
    match register {
        RegisterType::Temp => Some(RegFile::Temp),
        RegisterType::Input => Some(RegFile::Input),
        RegisterType::Output => Some(RegFile::Output),
        RegisterType::OutputDepth => Some(RegFile::OutputDepth),
        RegisterType::OutputCoverageMask => Some(RegFile::OutputCoverage),
        RegisterType::ThreadID => Some(RegFile::System(SystemInput::ThreadId)),
        RegisterType::ThreadGroupID => Some(RegFile::System(SystemInput::ThreadGroupId)),
        RegisterType::ThreadIDInGroup => Some(RegFile::System(SystemInput::ThreadIdInGroup)),
        RegisterType::Coverage => Some(RegFile::System(SystemInput::Coverage)),
        RegisterType::InputPrimitiveID => Some(RegFile::System(SystemInput::PrimitiveId)),
        _ => None,
    }
}

/// The register a variable-backed operand names.
fn operand_register(operand: &Operand) -> Result<Register, Unsupported> {
    let Some(file) = variable_file(operand.reg_type) else {
        return unsupported(format!("register file {:?}", operand.reg_type));
    };
    let index = match operand.indices.first() {
        None => 0,
        Some(OperandIndex::Imm32(index)) => *index,
        Some(other) => return unsupported(format!("register index {other:?}")),
    };
    Ok(Register { file, index })
}

fn identity_access() -> SourceAccess {
    SourceAccess {
        components: ComponentSelect::Swizzle([0, 1, 2, 3]),
        negate: false,
        abs: false,
    }
}

impl Walk {
    /// Reads one source operand: returns the variable use aligned with
    /// the returned access, materializing constants and addressable
    /// loads first.
    fn read_source(
        &mut self,
        operand: &Operand,
        span: (u32, u32),
    ) -> Result<(VariableId, SourceAccess), Unsupported> {
        let access = SourceAccess {
            components: operand.components.clone(),
            negate: operand.negate,
            abs: operand.abs,
        };
        match operand.reg_type {
            RegisterType::Immediate32 => {
                let value = self.fresh_temp();
                self.append(
                    Op::Const(Immediate {
                        bits: operand.immediate_values.to_vec(),
                    }),
                    Vec::new(),
                    vec![typed(value)],
                    span,
                );
                Ok((
                    value,
                    SourceAccess {
                        components: ComponentSelect::Swizzle([0, 1, 2, 3]),
                        negate: operand.negate,
                        abs: operand.abs,
                    },
                ))
            }
            RegisterType::ConstantBuffer
            | RegisterType::ImmConstBuffer
            | RegisterType::IndexableTemp => {
                let file = match operand.reg_type {
                    RegisterType::ConstantBuffer => RegFile::ConstantBuffer,
                    RegisterType::ImmConstBuffer => RegFile::ImmediateConstantBuffer,
                    _ => RegFile::IndexableTemp,
                };
                let (reference, address) = self.memory_reference(file, operand, span)?;
                let value = self.fresh_temp();
                let uses = address.map(typed).into_iter().collect();
                self.append(Op::Load(reference), uses, vec![typed(value)], span);
                Ok((
                    value,
                    SourceAccess {
                        components: ComponentSelect::Swizzle([0, 1, 2, 3]),
                        negate: operand.negate,
                        abs: operand.abs,
                    },
                ))
            }
            _ => {
                let register = operand_register(operand)?;
                Ok((self.variable(register), access))
            }
        }
    }

    /// Builds an addressable-file reference, materializing at most one
    /// relative address as a variable.
    fn memory_reference(
        &mut self,
        file: RegFile,
        operand: &Operand,
        span: (u32, u32),
    ) -> Result<(MemRef, Option<VariableId>), Unsupported> {
        let mut indices = Vec::new();
        let mut relative = None;
        let mut address = None;
        for (position, index) in operand.indices.iter().enumerate() {
            match index {
                OperandIndex::Imm32(value) => indices.push(*value),
                OperandIndex::Relative(inner) | OperandIndex::RelativePlusImm(_, inner) => {
                    if relative.is_some() {
                        return unsupported("multiple relative operand indices");
                    }
                    let base = match index {
                        OperandIndex::RelativePlusImm(base, _) => *base,
                        _ => 0,
                    };
                    indices.push(base);
                    let (variable, access) = self.read_source(inner, span)?;
                    if access.negate || access.abs {
                        return unsupported("modified relative address");
                    }
                    relative = Some((position, access.components));
                    address = Some(variable);
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
                relative_to: relative.as_ref().map(|(position, _)| *position),
                relative_select: relative.map(|(_, components)| components),
                components: operand.components.clone(),
            },
            address,
        ))
    }

    /// Lowers one generic (non-declaration) instruction.
    fn generic(
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
                let uses = self.outputs.iter().copied().map(typed).collect();
                self.append(Op::Ret, uses, Vec::new(), span);
                self.current = None;
                Ok(())
            }
            Opcode::Retc => {
                self.condition(instruction, operands, span)?;
                self.conditional_return(span);
                Ok(())
            }
            Opcode::Discard => {
                let (variable, access) = self.read_source(&operands[0], span)?;
                self.append(
                    Op::Discard {
                        nonzero: instruction.test_nonzero,
                        components: access.components,
                    },
                    vec![typed(variable)],
                    Vec::new(),
                    span,
                );
                Ok(())
            }
            Opcode::Switch | Opcode::Case | Opcode::Default | Opcode::EndSwitch => {
                unsupported("switch flow")
            }
            Opcode::Call | Opcode::Callc | Opcode::Label => unsupported("subroutine calls"),
            Opcode::Nop => Ok(()),
            _ => self.alu(instruction, operands, span),
        }
    }

    /// Reads a conditional opcode's single tested source and emits the
    /// branch instruction.
    fn condition(
        &mut self,
        instruction: &Instruction,
        operands: &[Operand],
        span: (u32, u32),
    ) -> Result<(), Unsupported> {
        let Some(operand) = operands.first() else {
            return unsupported("conditional without a tested operand");
        };
        let (variable, access) = self.read_source(operand, span)?;
        if access.negate || access.abs {
            return unsupported("modified branch condition");
        }
        self.append(
            Op::Branch {
                nonzero: instruction.test_nonzero,
                components: access.components,
            },
            vec![typed(variable)],
            Vec::new(),
            span,
        );
        Ok(())
    }

    fn open_if(
        &mut self,
        instruction: &Instruction,
        operands: &[Operand],
        span: (u32, u32),
    ) -> Result<(), Unsupported> {
        self.condition(instruction, operands, span)?;
        let block = self.current.expect("the branch just landed");
        self.frames.push(Frame::If {
            pending_false: Some((block, Edge::False)),
            pending_merge: Vec::new(),
            saw_else: false,
        });
        self.current = None;
        self.pending = vec![(block, Edge::True)];
        Ok(())
    }

    fn open_else(&mut self) -> Result<(), Unsupported> {
        let arm_end = self.current.take();
        let stale = core::mem::take(&mut self.pending);
        let Some(Frame::If {
            pending_false,
            pending_merge,
            saw_else,
        }) = self.frames.last_mut()
        else {
            return unsupported("else outside if");
        };
        if *saw_else {
            return unsupported("second else in one if");
        }
        *saw_else = true;
        if let Some(block) = arm_end {
            pending_merge.push((block, Edge::Fall));
        } else {
            pending_merge.extend(stale);
        }
        let Some(false_entry) = pending_false.take() else {
            return unsupported("else with no false entry");
        };
        self.pending = vec![false_entry];
        Ok(())
    }

    fn close_if(&mut self) -> Result<(), Unsupported> {
        let arm_end = self.current.take();
        let stale = core::mem::take(&mut self.pending);
        let Some(Frame::If {
            pending_false,
            mut pending_merge,
            ..
        }) = self.frames.pop()
        else {
            return unsupported("endif outside if");
        };
        if let Some(block) = arm_end {
            pending_merge.push((block, Edge::Fall));
        } else {
            pending_merge.extend(stale);
        }
        if let Some(false_entry) = pending_false {
            pending_merge.push(false_entry);
        }
        self.pending = pending_merge;
        Ok(())
    }

    fn open_loop(&mut self) {
        // The header must exist eagerly: it is the back-edge target. The
        // preceding block falls through into it.
        if let Some(block) = self.current.take() {
            self.pending.push((block, Edge::Fall));
        }
        let header = self.landing();
        self.frames.push(Frame::Loop {
            header,
            pending_exit: Vec::new(),
        });
    }

    fn close_loop(&mut self) -> Result<(), Unsupported> {
        let latch = self.current.take();
        let stale = core::mem::take(&mut self.pending);
        let Some(Frame::Loop {
            header,
            pending_exit,
        }) = self.frames.pop()
        else {
            return unsupported("endloop outside loop");
        };
        if !stale.is_empty() {
            return unsupported("loop body ends unmaterialized");
        }
        if let Some(block) = latch {
            self.builder
                .add_edge(block, header, Edge::Fall, None)
                .expect("loop header is live");
        }
        self.pending = pending_exit;
        Ok(())
    }

    /// A `break`/`breakc`: `conditional` when the branch just landed.
    fn transfer_break(&mut self, conditional: bool) -> Result<(), Unsupported> {
        let Some(block) = self.current else {
            return unsupported("transfer outside a materialized block");
        };
        let Some(Frame::Loop { pending_exit, .. }) = self
            .frames
            .iter_mut()
            .rev()
            .find(|frame| matches!(frame, Frame::Loop { .. }))
        else {
            return unsupported("break outside loop");
        };
        if conditional {
            pending_exit.push((block, Edge::True));
            self.current = None;
            self.pending = vec![(block, Edge::False)];
        } else {
            pending_exit.push((block, Edge::Fall));
            self.current = None;
            self.pending = Vec::new();
        }
        Ok(())
    }

    fn transfer_continue(&mut self, conditional: bool) -> Result<(), Unsupported> {
        let Some(block) = self.current else {
            return unsupported("transfer outside a materialized block");
        };
        let Some(Frame::Loop { header, .. }) = self
            .frames
            .iter()
            .rev()
            .find(|frame| matches!(frame, Frame::Loop { .. }))
        else {
            return unsupported("continue outside loop");
        };
        let header = *header;
        let edge = if conditional { Edge::True } else { Edge::Fall };
        self.builder
            .add_edge(block, header, edge, None)
            .expect("loop header is live");
        self.current = None;
        self.pending = if conditional {
            vec![(block, Edge::False)]
        } else {
            Vec::new()
        };
        Ok(())
    }

    fn conditional_return(&mut self, span: (u32, u32)) {
        let block = self.current.expect("the branch just landed");
        let ret_block = if let Some(existing) = self.ret_block {
            existing
        } else {
            let created = self.builder.new_block("ret");
            let uses = self.outputs.iter().copied().map(typed).collect();
            self.builder
                .append_instruction(created, Op::Ret, uses, Vec::new(), false, Some(span))
                .expect("instruction identity space is far from exhausted");
            self.ret_block = Some(created);
            created
        };
        self.builder
            .add_edge(block, ret_block, Edge::True, None)
            .expect("return block is live");
        self.current = None;
        self.pending = vec![(block, Edge::False)];
    }

    /// Lowers a retained decoded instruction with dest/source operands.
    fn alu(
        &mut self,
        instruction: &Instruction,
        operands: &[Operand],
        span: (u32, u32),
    ) -> Result<(), Unsupported> {
        let mut dests: Vec<(&Operand, DestAccess, usize)> = Vec::new();
        let mut sources = Vec::new();
        let mut bindings = Vec::new();
        let mut uses = Vec::new();
        let mut source_operands: Vec<&Operand> = Vec::new();
        let dual = matches!(instruction.opcode, Opcode::Sincos | Opcode::UDiv);
        for (position, operand) in operands.iter().enumerate() {
            match operand.reg_type {
                RegisterType::Null => {}
                RegisterType::Resource | RegisterType::Sampler | RegisterType::Uav => {
                    let file = match operand.reg_type {
                        RegisterType::Resource => RegFile::Resource,
                        RegisterType::Sampler => RegFile::Sampler,
                        _ => RegFile::Uav,
                    };
                    let index = match operand.indices.first() {
                        Some(OperandIndex::Imm32(index)) => *index,
                        _ => return unsupported("binding with a computed index"),
                    };
                    bindings.push(Binding {
                        register: Register { file, index },
                        components: operand.components.clone(),
                    });
                }
                RegisterType::Immediate32 | RegisterType::Immediate64 => {
                    let (variable, access) = self.read_source(operand, span)?;
                    uses.push(typed(variable));
                    sources.push(access);
                    source_operands.push(operand);
                }
                _ if matches!(operand.components, ComponentSelect::Mask(_))
                    && (dests.is_empty() || (dual && dests.len() == 1)) =>
                {
                    let ComponentSelect::Mask(mask) = operand.components else {
                        unreachable!()
                    };
                    dests.push((operand, DestAccess { mask }, position));
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
                    dests.push((operand, DestAccess { mask: 0b0001 }, position));
                }
                _ => {
                    let (variable, access) = self.read_source(operand, span)?;
                    uses.push(typed(variable));
                    sources.push(access);
                    source_operands.push(operand);
                }
            }
        }
        // Dual-destination opcodes split into one pure instruction per
        // surviving destination; the operand position selects the half.
        if dests.len() > 1 {
            return self.split_dual(instruction, &dests, &source_operands, &bindings, span);
        }
        self.finish_alu(instruction, dests.pop(), uses, sources, bindings, span)
    }

    /// Splits one dual-destination instruction into a pure instruction
    /// per surviving destination.
    ///
    /// The halves execute atomically in the machine but become a
    /// statement sequence here: a half whose destination register feeds
    /// the shared sources must run last, or the other half reads a
    /// clobbered value. If both destinations feed the sources no order
    /// is sound.
    fn split_dual(
        &mut self,
        instruction: &Instruction,
        dests: &[(&Operand, DestAccess, usize)],
        source_operands: &[&Operand],
        bindings: &[Binding],
        span: (u32, u32),
    ) -> Result<(), Unsupported> {
        let source_lanes = |operand: &Operand| -> u8 {
            match &operand.components {
                ComponentSelect::Swizzle(swizzle) => {
                    swizzle.iter().fold(0u8, |bits, &lane| bits | 1 << lane)
                }
                ComponentSelect::Scalar(lane) => 1 << lane,
                _ => 0b1111,
            }
        };
        let feeds_sources = |dest: &Operand, mask: u8| {
            source_operands.iter().any(|source| {
                source.reg_type == dest.reg_type
                    && source.indices.first() == dest.indices.first()
                    && source_lanes(source) & mask != 0
            })
        };
        let clobbering: Vec<bool> = dests
            .iter()
            .map(|(operand, access, _)| feeds_sources(operand, access.mask))
            .collect();
        if clobbering.iter().all(|&clobbers| clobbers) {
            return unsupported(format!(
                "both destinations of {} overwrite its sources",
                instruction.opcode.name()
            ));
        }
        let mut ordered: Vec<usize> = (0..dests.len()).collect();
        ordered.sort_by_key(|&half| clobbering[half]);
        for &half in &ordered {
            let (operand, access, position) = (dests[half].0, dests[half].1.clone(), dests[half].2);
            let register = operand_register(operand)?;
            let variable = self.variable(register);
            // Re-materialize the operands per half: every constant
            // stays single-use, so it inlines as a native literal
            // instead of round-tripping through a float register.
            let mut split_uses = Vec::new();
            let mut split_sources = Vec::new();
            for source_operand in source_operands {
                let (operand_variable, operand_access) = self.read_source(source_operand, span)?;
                split_uses.push(typed(operand_variable));
                split_sources.push(operand_access);
            }
            let merges = !access.full() && register.file == RegFile::Temp;
            if merges {
                split_uses.push(typed(variable));
                split_sources.push(identity_access());
            }
            self.append(
                Op::Alu(AluOp {
                    opcode: instruction.opcode,
                    saturate: instruction.saturate,
                    test_nonzero: instruction.test_nonzero,
                    dest: Some(access),
                    dest_position: position,
                    sources: split_sources,
                    bindings: bindings.to_vec(),
                    tex_offsets: instruction.tex_offsets,
                    resinfo_return_type: instruction.resinfo_return_type,
                    merges,
                }),
                split_uses,
                vec![typed(variable)],
                span,
            );
        }
        Ok(())
    }

    /// Commits one decoded instruction after operand classification.
    #[allow(clippy::too_many_arguments, reason = "one call site")]
    fn finish_alu(
        &mut self,
        instruction: &Instruction,
        dest: Option<(&Operand, DestAccess, usize)>,
        mut uses: Vec<TypedVariable<ShaderDialect>>,
        mut sources: Vec<SourceAccess>,
        bindings: Vec<Binding>,
        span: (u32, u32),
    ) -> Result<(), Unsupported> {
        let mut defs = Vec::new();
        let mut dest_access = None;
        let mut dest_position = 0usize;
        let mut merges = false;
        if let Some((operand, access, position)) = dest {
            dest_position = position;
            if operand.reg_type == RegisterType::IndexableTemp {
                // Writes into x#[...] are stores, not variable defs.
                let (reference, address) =
                    self.memory_reference(RegFile::IndexableTemp, operand, span)?;
                let alu = AluOp {
                    opcode: instruction.opcode,
                    saturate: instruction.saturate,
                    test_nonzero: instruction.test_nonzero,
                    dest: Some(access),
                    dest_position,
                    sources,
                    bindings,
                    tex_offsets: instruction.tex_offsets,
                    resinfo_return_type: instruction.resinfo_return_type,
                    merges: false,
                };
                // Compute into a temp, then store it. The store's uses
                // are [address..., value]: the HLIL store shape takes
                // the LAST use as the stored value.
                let value = self.fresh_temp();
                let mut store_uses = Vec::new();
                if let Some(address) = address {
                    store_uses.push(typed(address));
                }
                store_uses.push(typed(value));
                self.append(Op::Alu(alu), uses, vec![typed(value)], span);
                self.append(Op::Store(reference), store_uses, Vec::new(), span);
                return Ok(());
            }
            let register = operand_register(operand)?;
            let variable = self.variable(register);
            if !access.full() && register.file == RegFile::Temp {
                // Partial writes read-modify-write the register.
                uses.push(typed(variable));
                sources.push(identity_access());
                merges = true;
            }
            defs.push(typed(variable));
            dest_access = Some(access);
        }
        // A plain full-register move is the dialect's copy.
        let operation = if instruction.opcode == Opcode::Mov
            && !instruction.saturate
            && dest_access.as_ref().is_some_and(DestAccess::full)
            && sources.len() == 1
            && !sources[0].negate
            && !sources[0].abs
            && sources[0].components == ComponentSelect::Swizzle([0, 1, 2, 3])
            && bindings.is_empty()
        {
            Op::Copy
        } else {
            Op::Alu(AluOp {
                opcode: instruction.opcode,
                saturate: instruction.saturate,
                test_nonzero: instruction.test_nonzero,
                dest: dest_access,
                dest_position,
                sources,
                bindings,
                tex_offsets: instruction.tex_offsets,
                resinfo_return_type: instruction.resinfo_return_type,
                merges,
            })
        };
        self.append(operation, uses, defs, span);
        Ok(())
    }
}

/// Lifts one decoded program into a verified MLIL function.
///
/// # Errors
///
/// Returns [`Unsupported`] for shader shapes outside the modeled subset
/// (switch flow, subroutines, exotic operand addressing) and for
/// verification failures during construction.
pub fn lift_program(program: &Program) -> Result<LiftedShader, Unsupported> {
    if matches!(program.shader_type, "gs" | "hs" | "ds") {
        return unsupported(format!("{} stage", program.shader_type));
    }
    let source = format!(
        "{}_{}_{}",
        program.shader_type, program.major_version, program.minor_version
    );
    let mut walk = Walk {
        builder: FunctionBuilder::new(source),
        variables: BTreeMap::new(),
        outputs: Vec::new(),
        current: None,
        pending: Vec::new(),
        frames: Vec::new(),
        blocks: 0,
        ret_block: None,
    };
    let entry = walk.builder.entry();
    walk.pending.push((entry, Edge::Entry));

    let mut declarations = Declarations {
        samplers: Vec::new(),
        resources: Vec::new(),
        constant_buffers: Vec::new(),
        inputs: Vec::new(),
        outputs: Vec::new(),
        indexable_temps: Vec::new(),
        immediate_constants: Vec::new(),
        structured: Vec::new(),
        thread_group: None,
    };

    for (index, instruction) in program.instructions.iter().enumerate() {
        let span = (
            u32::try_from(index).unwrap_or(u32::MAX),
            u32::try_from(index + 1).unwrap_or(u32::MAX),
        );
        collect_declaration(instruction, &mut declarations, &mut walk);
        if let InstructionKind::Generic { operands, .. } = &instruction.kind {
            walk.generic(instruction, operands, span)?;
        }
    }
    if !walk.frames.is_empty() {
        return unsupported("unclosed structured region");
    }

    let parameters = walk
        .variables
        .iter()
        .filter(|(register, _)| matches!(register.file, RegFile::Input | RegFile::System(_)))
        .map(|(_, &variable)| variable)
        .collect();
    let returns = walk.outputs.iter().map(|_| ValueType::VEC4).collect();
    walk.builder
        .set_signature(cfglib::ir::mlil::Signature::<ShaderDialect>::new(
            parameters, returns,
        ))
        .map_err(|error| Unsupported(format!("signature: {error}")))?;

    let function = walk
        .builder
        .finish()
        .map_err(|error| Unsupported(format!("verification: {error}")))?;
    // A walk bug that strands blocks must fail loudly, never render a
    // silently truncated body.
    let mut reachable = std::collections::BTreeSet::new();
    let mut frontier = vec![function.cfg().entry()];
    while let Some(block) = frontier.pop() {
        if reachable.insert(block) {
            frontier.extend(function.cfg().successors(block));
        }
    }
    if reachable.len() != function.cfg().block_count() {
        return unsupported("stranded unreachable blocks");
    }
    Ok(LiftedShader {
        function,
        declarations,
    })
}

/// Records declaration instructions for the rendering header.
fn collect_declaration(
    instruction: &Instruction,
    declarations: &mut Declarations,
    walk: &mut Walk,
) {
    match &instruction.kind {
        InstructionKind::DclSampler { mode, operands, .. } => {
            if let Some(operand) = operands.first()
                && let Some(OperandIndex::Imm32(index)) = operand.indices.first()
            {
                declarations.samplers.push((*index, mode));
            }
        }
        InstructionKind::DclResource {
            dimension,
            return_type,
            operands,
            ..
        } => {
            if let Some(operand) = operands.first()
                && let Some(OperandIndex::Imm32(index)) = operand.indices.first()
            {
                declarations
                    .resources
                    .push((*index, dimension, *return_type));
            }
        }
        InstructionKind::DclConstantBuffer { operands, size, .. } => {
            if let Some(operand) = operands.first()
                && let Some(OperandIndex::Imm32(register)) = operand.indices.first()
            {
                declarations.constant_buffers.push((*register, *size));
            }
        }
        InstructionKind::DclInput {
            interpolation,
            system_value,
            operands,
        } => {
            if let Some(operand) = operands.first()
                && let Ok(register) = operand_register(operand)
            {
                walk.variable(register);
                if !declarations
                    .inputs
                    .iter()
                    .any(|row| row.register == register)
                {
                    declarations.inputs.push(InputRow {
                        register,
                        components: operand.components.clone(),
                        system_value: system_value.and_then(dxbc::shex::SystemValue::name),
                        interpolation: *interpolation,
                    });
                }
            }
        }
        InstructionKind::DclOutput {
            system_value,
            operands,
        } => {
            collect_io(&mut declarations.outputs, operands, *system_value, walk);
        }
        InstructionKind::DclResourceStructured {
            stride, operands, ..
        } => {
            if let Some(operand) = operands.first()
                && let Some(OperandIndex::Imm32(index)) = operand.indices.first()
            {
                declarations.structured.push((*index, *stride));
            }
        }
        InstructionKind::DclThreadGroup { x, y, z } => {
            declarations.thread_group = Some((*x, *y, *z));
        }
        InstructionKind::CustomData {
            subtype: dxbc::shex::CustomDataType::ImmediateConstantBuffer,
            values,
            ..
        } => {
            declarations.immediate_constants.clone_from(values);
        }
        InstructionKind::DclIndexableTemp {
            reg,
            size,
            components,
        } => {
            declarations
                .indexable_temps
                .push((*reg, *size, *components));
        }
        _ => {}
    }
}

/// Records one input or output declaration row, deduplicating split
/// component declarations of the same register.
fn collect_io(
    rows: &mut Vec<(Register, ComponentSelect, Option<&'static str>)>,
    operands: &[Operand],
    system_value: Option<dxbc::shex::SystemValue>,
    walk: &mut Walk,
) {
    if let Some(operand) = operands.first()
        && let Ok(register) = operand_register(operand)
    {
        walk.variable(register);
        if !rows.iter().any(|(declared, ..)| *declared == register) {
            rows.push((
                register,
                operand.components.clone(),
                system_value.and_then(dxbc::shex::SystemValue::name),
            ));
        }
    }
}
