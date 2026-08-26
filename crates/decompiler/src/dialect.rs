//! SM4/SM5 shader dialect over cfglib's generic IR contracts.
//!
//! One dialect type implements both the MLIL and HLIL levels: registers
//! are dataflow variables, non-register operand reads (immediates,
//! constant buffers, indexable temporaries) materialize as pure value
//! producers so expression inlining can fold them into consumers, and
//! everything else stays a typed wrapper around the decoded instruction.

use std::collections::BTreeMap;

use cfglib::ir::hlil::{self, Lifted};
use cfglib::ir::mlil::{
    AnalysisDialect, Dialect, InstructionMetadata, VerificationIssue, VerifyDialect,
};
use cfglib::{EdgeKind, FlowEffect, Vocabulary};
use dxbc::shex::{ComponentSelect, Opcode};

/// One MLIL/HLIL function over the shader dialect.
pub type Function = cfglib::ir::mlil::Function<ShaderDialect>;
/// One MLIL instruction over the shader dialect.
pub type Instruction = cfglib::ir::mlil::Instruction<ShaderDialect>;
/// One MLIL variable identity over the shader dialect.
pub type VariableId = cfglib::ir::mlil::VariableId;

/// The Direct3D SM4/SM5 semantic dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ShaderDialect;

/// The value type of one occurrence: a register-shaped vector of 32-bit
/// lanes. SM4/SM5 registers are typeless; lane interpretation belongs to
/// each consuming operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ValueType {
    /// Number of 32-bit components (1 or 4).
    pub components: u8,
}

impl ValueType {
    /// The full register shape.
    pub const VEC4: Self = Self { components: 4 };
}

/// Observable effects of shader operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Effect {
    /// Reads addressable state (resources, indexable temporaries).
    Read,
    /// Writes addressable state (UAVs, indexable temporaries).
    Write,
    /// Control-adjacent side effect (`discard`, `sync`).
    Control,
}

/// Semantic role of one register variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    /// General-purpose temporary (`r#`) or a materialized operand value.
    Temp,
    /// Shader input (`v#` and system-value inputs).
    Input,
    /// Shader output (`o#`, `oDepth`, `oMask`).
    Output,
}

/// A register file the dialect models as variables or memory references.
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
}

/// The native identity of one register variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Register {
    /// Register file.
    pub file: RegFile,
    /// Register index within the file.
    pub index: u32,
}

/// One source operand's access metadata, positionally aligned with the
/// instruction's variable uses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceAccess {
    /// Component selection applied to the read.
    pub components: ComponentSelect,
    /// Negate modifier.
    pub negate: bool,
    /// Absolute-value modifier.
    pub abs: bool,
}

/// The destination operand's access metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DestAccess {
    /// Component write mask (bit per lane).
    pub mask: u8,
}

impl DestAccess {
    /// Whether every lane is written.
    #[must_use]
    pub fn full(&self) -> bool {
        self.mask == 0b1111
    }
}

/// A reference into an addressable register file, with at most one
/// materialized relative address (the instruction's trailing use).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemRef {
    /// Register file being addressed.
    pub file: RegFile,
    /// Immediate index chain (`cb0[3]` carries `[0, 3]`).
    pub indices: Vec<u32>,
    /// Which immediate index the materialized relative address offsets,
    /// when the reference is dynamically indexed.
    pub relative_to: Option<usize>,
    /// Component selection of the relative address operand.
    pub relative_select: Option<ComponentSelect>,
    /// Component selection applied to the read.
    pub components: ComponentSelect,
}

/// An immediate literal operand's raw 32-bit lanes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Immediate {
    /// Raw lane bits.
    pub bits: Vec<u32>,
}

/// One binding operand (resource, sampler, UAV) carried in an
/// instruction's payload rather than as a dataflow variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    /// Bound register.
    pub register: Register,
    /// Component selection on the binding operand.
    pub components: ComponentSelect,
}

/// A general decoded instruction retained with its access metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AluOp {
    /// Decoded opcode.
    pub opcode: Opcode,
    /// Saturate modifier.
    pub saturate: bool,
    /// Conditional test polarity for testing opcodes.
    pub test_nonzero: bool,
    /// Destination access, when the opcode writes a register.
    pub dest: Option<DestAccess>,
    /// The operand slot the destination came from — dual-destination
    /// opcodes (`udiv`, `sincos`) select their meaning by position.
    pub dest_position: usize,
    /// Source accesses, positionally aligned with the variable uses.
    pub sources: Vec<SourceAccess>,
    /// Resource and sampler bindings in operand order.
    pub bindings: Vec<Binding>,
    /// Texel offsets from the extended opcode token.
    pub tex_offsets: Option<[i8; 3]>,
    /// `resinfo` return-type selector (0 float, 1 rcp-float, 2 uint).
    pub resinfo_return_type: Option<u32>,
    /// Whether the lift appended a trailing read of the destination's
    /// previous value (partial-mask writes read-modify-write).
    pub merges: bool,
}

/// The dialect's operation vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// Materialized immediate literal.
    Const(Immediate),
    /// Pure read of an addressable file (`cb#[...]`, `x#[...]`, icb).
    Load(MemRef),
    /// Write into an addressable file (`x#[...]`).
    Store(MemRef),
    /// Full-register unmodified move between plain register variables.
    Copy,
    /// A structured conditional (`if_z` / `if_nz` and the conditional
    /// transfers lowered to branches).
    Branch {
        /// Test polarity: `true` branches when the component is non-zero.
        nonzero: bool,
        /// Component selection of the tested source.
        components: ComponentSelect,
    },
    /// `ret` — uses every live output so liveness keeps output writes.
    Ret,
    /// `discard_z` / `discard_nz` — flags the invocation discarded and
    /// continues executing.
    Discard {
        /// Test polarity.
        nonzero: bool,
        /// Component selection of the tested source.
        components: ComponentSelect,
    },
    /// Every other decoded instruction.
    Alu(AluOp),
}

impl Op {
    /// Compact stable mnemonic.
    #[must_use]
    pub fn mnemonic(&self) -> &'static str {
        match self {
            Self::Const(_) => "const",
            Self::Load(_) => "load",
            Self::Store(_) => "store",
            Self::Copy => "copy",
            Self::Branch { .. } => "branch",
            Self::Ret => "ret",
            Self::Discard { .. } => "discard",
            Self::Alu(alu) => alu.opcode.name(),
        }
    }
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
    type ValueType = ValueType;
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

impl Dialect for ShaderDialect {
    type Operation = Op;
    type Edge = Edge;

    fn instruction_metadata(
        operation: &Self::Operation,
        _may_throw: bool,
    ) -> InstructionMetadata<Self::Effect> {
        let (effects, flow) = match operation {
            Op::Const(_) | Op::Copy | Op::Load(_) => (Vec::new(), FlowEffect::Fallthrough),
            Op::Store(_) => (vec![Effect::Write], FlowEffect::Fallthrough),
            Op::Branch { .. } => (Vec::new(), FlowEffect::ConditionalJump),
            Op::Ret => (Vec::new(), FlowEffect::Return),
            Op::Discard { .. } => (vec![Effect::Control], FlowEffect::Fallthrough),
            Op::Alu(alu) => (alu_effects(alu), FlowEffect::Fallthrough),
        };
        InstructionMetadata::new(effects, flow, false)
    }

    fn mnemonic(operation: &Self::Operation) -> &str {
        operation.mnemonic()
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

/// Effects of one retained decoded instruction.
fn alu_effects(alu: &AluOp) -> Vec<Effect> {
    // Resource reads stay observable so they are neither eliminated nor
    // duplicated; UAV and shared-memory traffic is a real write.
    let name = alu.opcode.name();
    if name.starts_with("atomic") || name.starts_with("imm_atomic") || name.starts_with("store") {
        vec![Effect::Read, Effect::Write]
    } else if name.starts_with("sample")
        || name.starts_with("ld")
        || name.starts_with("gather")
        || name.starts_with("bufinfo")
        || name.starts_with("resinfo")
    {
        vec![Effect::Read]
    } else if matches!(alu.opcode, Opcode::Sync) {
        vec![Effect::Control]
    } else {
        Vec::new()
    }
}

impl AnalysisDialect for ShaderDialect {
    type Constant = Immediate;
    type ExpressionOperator = Opcode;
    type Callee = u32;

    fn is_copy(operation: &Self::Operation) -> bool {
        matches!(operation, Op::Copy)
    }

    fn expression_operator(operation: &Self::Operation) -> Option<Self::ExpressionOperator> {
        match operation {
            Op::Alu(alu) => Some(alu.opcode),
            _ => None,
        }
    }

    fn constant(operation: &Self::Operation) -> Option<Self::Constant> {
        match operation {
            Op::Const(immediate) => Some(immediate.clone()),
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
    type Constant = Immediate;

    fn mnemonic(operation: &Self::Operation) -> &str {
        operation.mnemonic()
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
            Op::Branch { .. } => Lifted::BranchOperation(operation.clone()),
            Op::Ret => Lifted::Return,
            Op::Store(_) => Lifted::Store {
                location: match operation {
                    Op::Store(reference) => Op::Load(reference.clone()),
                    _ => unreachable!(),
                },
            },
            other => Lifted::Operation(other.clone()),
        }
    }

    fn case_values(_edge: &Edge) -> Vec<Immediate> {
        Vec::new()
    }

    fn void_type() -> ValueType {
        ValueType { components: 0 }
    }

    fn temporary_role() -> Option<Role> {
        Some(Role::Temp)
    }

    fn previous_value_operand(operation: &Op) -> Option<usize> {
        match operation {
            Op::Alu(alu) if alu.merges => Some(alu.sources.len() - 1),
            _ => None,
        }
    }

    fn negate_operation(operation: &Op) -> Option<Op> {
        let Op::Branch {
            nonzero,
            components,
        } = operation
        else {
            return None;
        };
        Some(Op::Branch {
            nonzero: !nonzero,
            components: components.clone(),
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
