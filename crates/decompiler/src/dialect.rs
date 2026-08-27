//! SM4/SM5 shader dialect over cfglib's generic IR contracts.
//!
//! One dialect type implements the RTL, MLIL, and HLIL levels. The
//! decoded program (the language's own low-level form) lifts into RTL —
//! typed parallel transfers over raw registers — and cfglib's web
//! lifting recovers typed MLIL variables from register lanes. The MLIL
//! and HLIL operation vocabulary is the lifted statement itself:
//! expression trees over web variables, with resource bindings and
//! addressable-memory references carried in operator payloads.

use std::collections::BTreeMap;

use cfglib::ir::hlil::{self, Lifted};
use cfglib::ir::mlil::{
    AnalysisDialect, Dialect, InstructionMetadata, VerificationIssue, VerifyDialect,
};
use cfglib::ir::rtl::{self, LiftedStatement, ScalarType, ValueShape, VarExpr};
use cfglib::{EdgeKind, FlowEffect, Vocabulary};
use dxbc::shex::Opcode;

/// One MLIL/HLIL function over the shader dialect.
pub type Function = cfglib::ir::mlil::Function<ShaderDialect>;
/// One MLIL instruction over the shader dialect.
pub type Instruction = cfglib::ir::mlil::Instruction<ShaderDialect>;
/// One MLIL variable identity over the shader dialect.
pub type VariableId = cfglib::ir::mlil::VariableId;
/// The operation vocabulary shared by MLIL and HLIL: lifted statements.
pub type Op = LiftedStatement<ShaderDialect>;
/// One typed expression over web variables.
pub type Expr = VarExpr<ShaderDialect>;

/// The Direct3D SM4/SM5 semantic dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShaderDialect;

/// Observable effects of shader operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effect {
    /// Reads addressable state (resources, indexable temporaries).
    Read,
    /// Writes addressable state (UAVs, indexable temporaries).
    Write,
    /// Writes the shader interface (outputs, depth, coverage).
    Export,
    /// Control-adjacent side effect (`discard`, `sync`).
    Control,
}

/// Semantic role of one variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    /// A web recovered from temporaries, or a synthetic value.
    Temp,
    /// Shader input (`v#` and system-value inputs).
    Input,
}

/// A register file the dialect models as storage or memory references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RegFile {
    /// `r#` temporaries.
    Temp,
    /// `v#` inputs.
    Input,
    /// `o#` outputs.
    Output,
    /// `oDepth`.
    OutputDepth,
    /// `oMask`.
    OutputCoverage,
    /// `cb#[...]` constant buffers.
    ConstantBuffer,
    /// Immediate constant buffer.
    ImmediateConstantBuffer,
    /// `x#[...]` indexable temporaries.
    IndexableTemp,
    /// `t#` shader resource views.
    Resource,
    /// `s#` samplers.
    Sampler,
    /// `u#` unordered access views.
    Uav,
    /// `vpc#` tessellation patch-constant inputs.
    PatchConstant,
    /// System-value input registers (thread ids, coverage, primitive id).
    System(SystemInput),
}

/// System-value input registers modeled as read-only inputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum SystemInput {
    /// `vThreadID`.
    ThreadId,
    /// `vThreadGroupID`.
    ThreadGroupId,
    /// `vThreadIDInGroup`.
    ThreadIdInGroup,
    /// `vCoverage`.
    Coverage,
    /// `vPrim` (primitive id).
    PrimitiveId,
    /// `vDomain` (tessellator domain location).
    DomainPoint,
    /// `vOutputControlPointID` (hull control-point phase).
    ControlPointId,
    /// `vForkInstanceID` (instanced hull fork phase).
    ForkInstanceId,
}

/// The native identity of one register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Register {
    /// Register file.
    pub file: RegFile,
    /// Register index within the file.
    pub index: u32,
}

/// A reference into an addressable register file. The selected lanes are
/// already mapped through the destination mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemRef {
    /// Register file being addressed.
    pub file: RegFile,
    /// Immediate index chain (`cb0[3]` carries `[0, 3]`).
    pub indices: Vec<u32>,
    /// Which immediate index a relative address operand offsets, when
    /// the reference is dynamically indexed.
    pub relative_to: Option<usize>,
    /// Mapped element lanes selected by the read (empty for stores).
    pub lanes: Vec<u8>,
}

/// One binding operand (resource, sampler, UAV) carried in an operator
/// payload rather than as dataflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    /// Bound register.
    pub register: Register,
    /// Mapped lanes the binding's swizzle selects for the destination.
    pub lanes: Vec<u8>,
}

/// The retained head of one decoded computational instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AluHead {
    /// Decoded opcode.
    pub opcode: Opcode,
    /// Saturate modifier.
    pub saturate: bool,
    /// The operand slot the destination came from — dual-destination
    /// opcodes (`udiv`, `sincos`) select their meaning by position.
    pub dest_position: usize,
    /// Resource and sampler bindings in operand order.
    pub bindings: Vec<Binding>,
    /// Texel offsets from the extended opcode token.
    pub tex_offsets: Option<[i8; 3]>,
    /// `resinfo` return-type selector (0 float, 1 rcp-float, 2 uint).
    pub resinfo_return_type: Option<u32>,
}

/// Pure RTL operators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RtlOp {
    /// A retained decoded instruction applied to its source values.
    Alu(AluHead),
    /// Float negation source modifier.
    Neg,
    /// Float absolute-value source modifier.
    Abs,
    /// A read of an addressable file (`cb#[...]`, `x#[...]`, icb); the
    /// optional operand is the relative address.
    LoadMem(MemRef),
    /// A geometry-shader input read: one vertex of the input primitive.
    LoadInput {
        /// The primitive vertex index.
        vertex: u32,
        /// The input register within the vertex.
        register: u32,
        /// Mapped lanes selected by the read.
        lanes: Vec<u8>,
    },
    /// A control-point read whose register index is computed; the
    /// operand is the index.
    LoadInputDynamic {
        /// The control-point index.
        vertex: u32,
        /// The immediate base added to the computed register index.
        base: u32,
        /// Mapped lanes selected by the read.
        lanes: Vec<u8>,
    },
    /// A scalar zero test producing a boolean.
    Test {
        /// Polarity: `true` is non-zero.
        nonzero: bool,
    },
}

/// Effect-bearing RTL operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectOp {
    /// A write into an addressable file (`x#[...]`); operands are the
    /// value and the optional relative address.
    StoreMem(MemRef),
    /// A write of interface output lanes; the operands are the value
    /// and, when dynamic, the register index.
    StoreOutput {
        /// The output register (the immediate base when dynamic).
        register: Register,
        /// Written lanes in value order.
        lanes: Vec<u8>,
        /// Whether a trailing operand dynamically indexes the register.
        dynamic: bool,
    },
    /// A typed UAV store; operands are the coordinate and the value.
    StoreUav {
        /// The UAV register.
        register: Register,
        /// Written lanes in value order.
        lanes: Vec<u8>,
    },
    /// A geometry-shader vertex emit to stream 0.
    Emit,
    /// A geometry-shader strip cut on stream 0.
    Cut,
    /// `discard_z` / `discard_nz`; the operand is the tested condition.
    Discard,
    /// A destination-less retained instruction (`sync`, `nop`-likes).
    Bare(Opcode),
}

/// Edge payload carrying the structural meaning of each transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Edge {
    /// Synthetic entry edge.
    Entry,
    /// Sequential flow.
    Fall,
    /// Taken branch of a conditional.
    True,
    /// Not-taken branch of a conditional.
    False,
}

impl Vocabulary for ShaderDialect {
    type ValueType = ValueShape;
    type Effect = Effect;
    type Source = String;
    type SourceSpan = (u32, u32);
    type SourcePoint = u32;
    type VariableRole = Role;
    type NativeVariable = Register;

    fn span_is_empty(span: &Self::SourceSpan) -> bool {
        span.0 >= span.1
    }

    fn span_contains(span: &Self::SourceSpan, point: &Self::SourcePoint) -> bool {
        span.0 <= *point && *point < span.1
    }
}

impl rtl::Dialect for ShaderDialect {
    type Operator = RtlOp;
    type EffectOp = EffectOp;
    type Edge = Edge;

    fn mnemonic(operator: &Self::Operator) -> &str {
        match operator {
            RtlOp::Alu(head) => head.opcode.name(),
            RtlOp::Neg => "neg",
            RtlOp::Abs => "abs",
            RtlOp::LoadMem(_) => "load",
            RtlOp::LoadInput { .. } => "load_input",
            RtlOp::LoadInputDynamic { .. } => "load_input_dynamic",
            RtlOp::Test { .. } => "test",
        }
    }

    fn effect_mnemonic(operation: &Self::EffectOp) -> &str {
        match operation {
            EffectOp::StoreMem(_) => "store",
            EffectOp::StoreOutput { .. } => "store_output",
            EffectOp::StoreUav { .. } => "store_uav",
            EffectOp::Emit => "emit",
            EffectOp::Cut => "cut",
            EffectOp::Discard => "discard",
            EffectOp::Bare(opcode) => opcode.name(),
        }
    }

    fn edge_kind(edge: &Self::Edge) -> EdgeKind {
        match edge {
            Edge::Entry | Edge::Fall => EdgeKind::Fallthrough,
            Edge::True => EdgeKind::ConditionalTrue,
            Edge::False => EdgeKind::ConditionalFalse,
        }
    }

    fn is_entry_edge(edge: &Self::Edge) -> bool {
        *edge == Edge::Entry
    }
}

impl rtl::Lift for ShaderDialect {
    fn value_type(shape: ValueShape) -> ValueShape {
        shape
    }

    fn web_role(storage: Option<&Register>) -> Role {
        match storage.map(|register| register.file) {
            Some(RegFile::Input | RegFile::System(_) | RegFile::PatchConstant) => Role::Input,
            _ => Role::Temp,
        }
    }

    fn operation(statement: LiftedStatement<Self>) -> Op {
        statement
    }
}

impl Dialect for ShaderDialect {
    type Operation = Op;
    type Edge = Edge;

    fn instruction_metadata(
        operation: &Self::Operation,
        _may_throw: bool,
    ) -> InstructionMetadata<Self::Effect> {
        let (effects, flow) = match operation {
            LiftedStatement::Assign { effects, value, .. } => {
                let mut effects = effects.clone();
                add_read_effects(value, &mut effects);
                (effects, FlowEffect::Fallthrough)
            }
            LiftedStatement::Effect {
                operation,
                effects,
                operands,
            } => {
                let mut effects = effects.clone();
                for operand in operands {
                    add_read_effects(operand, &mut effects);
                }
                let base = match operation {
                    EffectOp::StoreMem(_) | EffectOp::StoreUav { .. } => Some(Effect::Write),
                    EffectOp::StoreOutput { .. } | EffectOp::Emit | EffectOp::Cut => {
                        Some(Effect::Export)
                    }
                    EffectOp::Discard | EffectOp::Bare(_) => Some(Effect::Control),
                };
                if let Some(effect) = base
                    && !effects.contains(&effect)
                {
                    effects.push(effect);
                }
                (effects, FlowEffect::Fallthrough)
            }
            LiftedStatement::Branch { .. } => (Vec::new(), FlowEffect::ConditionalJump),
            LiftedStatement::Return { .. } => (Vec::new(), FlowEffect::Return),
        };
        InstructionMetadata::new(effects, flow, false)
    }

    fn mnemonic(operation: &Self::Operation) -> &str {
        match operation {
            LiftedStatement::Assign { value, .. } => value_mnemonic(value),
            LiftedStatement::Effect { operation, .. } => {
                <Self as rtl::Dialect>::effect_mnemonic(operation)
            }
            LiftedStatement::Branch { .. } => "branch",
            LiftedStatement::Return { .. } => "ret",
        }
    }

    fn edge_kind(edge: &Self::Edge) -> EdgeKind {
        <Self as rtl::Dialect>::edge_kind(edge)
    }

    fn is_entry_edge(edge: &Self::Edge) -> bool {
        <Self as rtl::Dialect>::is_entry_edge(edge)
    }
}

fn value_mnemonic(value: &Expr) -> &str {
    match value {
        VarExpr::Read { .. } => "mov",
        VarExpr::Const { .. } => "const",
        VarExpr::Apply { operator, .. } => <ShaderDialect as rtl::Dialect>::mnemonic(operator),
        VarExpr::Reinterpret { .. } => "bitcast",
        VarExpr::Compose { .. } => "compose",
    }
}

/// Adds the observable-read effects of resource accesses nested in one
/// expression: samples and loads must be neither eliminated nor
/// duplicated.
fn add_read_effects(value: &Expr, effects: &mut Vec<Effect>) {
    match value {
        VarExpr::Apply {
            operator, operands, ..
        } => {
            let reads = match operator {
                RtlOp::Alu(head) => {
                    let name = head.opcode.name();
                    name.starts_with("sample")
                        || name.starts_with("ld")
                        || name.starts_with("gather")
                        || name.starts_with("bufinfo")
                        || name.starts_with("resinfo")
                        || name.starts_with("atomic")
                        || name.starts_with("imm_atomic")
                }
                _ => false,
            };
            if reads && !effects.contains(&Effect::Read) {
                effects.push(Effect::Read);
            }
            for operand in operands {
                add_read_effects(operand, effects);
            }
        }
        VarExpr::Reinterpret { operand, .. } => add_read_effects(operand, effects),
        VarExpr::Compose { parts, .. } => {
            for part in parts {
                add_read_effects(part, effects);
            }
        }
        VarExpr::Read { .. } | VarExpr::Const { .. } => {}
    }
}

impl AnalysisDialect for ShaderDialect {
    type Constant = Vec<u64>;
    type ExpressionOperator = Opcode;
    type Callee = u32;

    fn is_copy(_operation: &Self::Operation) -> bool {
        false
    }

    fn expression_operator(operation: &Self::Operation) -> Option<Self::ExpressionOperator> {
        match operation {
            LiftedStatement::Assign {
                value:
                    VarExpr::Apply {
                        operator: RtlOp::Alu(head),
                        ..
                    },
                ..
            } => Some(head.opcode),
            _ => None,
        }
    }

    fn constant(operation: &Self::Operation) -> Option<Self::Constant> {
        match operation {
            LiftedStatement::Assign {
                value: VarExpr::Const { bits, .. },
                ..
            } => Some(bits.clone()),
            _ => None,
        }
    }

    fn fold_constant(
        _instruction: &Instruction,
        _known: &BTreeMap<VariableId, Self::Constant>,
    ) -> Option<(VariableId, Self::Constant)> {
        None
    }

    fn callee(_operation: &Self::Operation) -> Option<Self::Callee> {
        None
    }
}

impl VerifyDialect for ShaderDialect {
    fn verify(_function: &Function, _issues: &mut Vec<VerificationIssue>) {}
}

impl hlil::Dialect for ShaderDialect {
    type Operation = Op;
    type Constant = Vec<u64>;

    fn mnemonic(operation: &Self::Operation) -> &str {
        <Self as Dialect>::mnemonic(operation)
    }
}

impl hlil::VerifyDialect for ShaderDialect {
    fn verify(
        _function: &hlil::Function<Self>,
        _issues: &mut Vec<cfglib::ir::hlil::VerificationIssue>,
    ) {
    }
}

impl hlil::LiftDialect for ShaderDialect {
    fn lift_operation(operation: &Op) -> Lifted<Op> {
        match operation {
            LiftedStatement::Branch { .. } => Lifted::BranchOperation(operation.clone()),
            LiftedStatement::Return { .. } => Lifted::Return,
            other => Lifted::Operation(other.clone()),
        }
    }

    fn case_values(_edge: &Edge) -> Vec<Vec<u64>> {
        Vec::new()
    }

    fn void_type() -> ValueShape {
        ValueShape::vector(ScalarType::Bits, 0)
    }

    fn temporary_role() -> Option<Role> {
        Some(Role::Temp)
    }

    fn previous_value_operand(operation: &Op) -> Option<usize> {
        match operation {
            LiftedStatement::Assign {
                merges: true,
                value,
                ..
            } => {
                let mut count = 0usize;
                value.for_each_read(&mut |_| count += 1);
                Some(count)
            }
            // `resinfo` renders as a statement block (`GetDimensions`),
            // never as an expression: declaring a previous-value slot one
            // past the reads pins the definition out of inlining.
            LiftedStatement::Assign { value, .. }
                if matches!(
                    value,
                    VarExpr::Apply {
                        operator: RtlOp::Alu(AluHead {
                            opcode: Opcode::Resinfo,
                            ..
                        }),
                        ..
                    }
                ) =>
            {
                let mut count = 0usize;
                value.for_each_read(&mut |_| count += 1);
                Some(count)
            }
            _ => None,
        }
    }

    fn negate_operation(operation: &Op) -> Option<Op> {
        let LiftedStatement::Branch { condition } = operation else {
            return None;
        };
        let VarExpr::Apply {
            operator: RtlOp::Test { nonzero },
            operands,
            shape,
        } = condition
        else {
            return None;
        };
        Some(LiftedStatement::Branch {
            condition: VarExpr::Apply {
                operator: RtlOp::Test { nonzero: !nonzero },
                operands: operands.clone(),
                shape: *shape,
            },
        })
    }

    fn evaluation_commutes(
        moved_effects: &[Effect],
        _moved_may_throw: bool,
        crossed_effects: &[Effect],
        _crossed_may_throw: bool,
    ) -> bool {
        // Shader reads never alias writes we reorder across: only pure
        // reads may commute with each other.
        let read_only = |effects: &[Effect]| effects.iter().all(|effect| *effect == Effect::Read);
        read_only(moved_effects) && read_only(crossed_effects)
    }
}
