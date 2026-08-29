//! SM4/SM5 instruction and variable adapters.
use cfglib::{
    FlowControl, FlowEffect, InstrInfo, MemoryAccessKind, MemoryAtomicity, MemoryEventInfo,
};
use dxbc::shex::{ComponentSelect, Instruction, Opcode, Operand, OperandIndex, RegisterType};

mod memory;

use memory::analyze_memory_events;
pub use memory::{Sm4Fence, Sm4MemoryAlias, Sm4MemoryEvent, Sm4MemoryLocation, Sm4MemoryRegion};

/// SM4/SM5 side-effect vocabulary reported through [`cfglib::EffectInfo`].
///
/// Shader-native categories: resource loads/stores, exports observable
/// outside the invocation (render targets, streams, `discard`), subroutine
/// calls, and synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sm4Effect {
    /// Reads a resource or thread-group shared memory.
    ResourceRead,
    /// Writes a resource or thread-group shared memory.
    ResourceWrite,
    /// Observable output: render-target/stream export, emit/cut, discard.
    Export,
    /// Calls a subroutine or interface method.
    Call,
    /// Memory-ordering or execution-synchronization boundary.
    Sync,
    /// A store through a dynamically-indexed indexable temp (`x#[reg]`).
    ///
    /// The written slot is statically unknowable, so the store reports NO
    /// def (a may-def must not kill other slots' reaching definitions) and
    /// carries this effect instead, keeping it visible to liveness, purity,
    /// and dead-code elimination. Loads through dynamic indices keep their
    /// structural identities; cross-slot def/use edges between static and
    /// dynamic accesses are not represented by ordinary variable def/use
    /// edges. They remain explicit in the instruction's generic memory-event
    /// stream.
    IndexableWrite,
    /// Unclassified effect (abort, debug break, unknown opcode).
    Unknown,
}

/// Stable, ordered representation of an SM4/SM5 register type.
///
/// The decoded `dxbc` type deliberately does not impose an ordering. SSA and
/// data-flow maps need one, so the adapter retains the token format's raw
/// register-type value without changing its meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sm4RegisterType(u32);

impl Sm4RegisterType {
    /// Construct a register type from its token-format value.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    /// Return the token-format value.
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    /// Convert back to the decoded `dxbc` register type.
    #[must_use]
    pub fn decoded(self) -> RegisterType {
        RegisterType::from_u32(self.0)
    }

    fn is_data(self) -> bool {
        !matches!(
            self.decoded(),
            RegisterType::Immediate32
                | RegisterType::Immediate64
                | RegisterType::Label
                | RegisterType::Null
                | RegisterType::Stream
                | RegisterType::FunctionBody
                | RegisterType::FunctionTable
        )
    }

    fn is_observable_output(self) -> bool {
        matches!(
            self.decoded(),
            RegisterType::Output
                | RegisterType::OutputDepth
                | RegisterType::OutputCoverageMask
                | RegisterType::FunctionOutput
                | RegisterType::OutputControlPoint
                | RegisterType::PatchConstant
                | RegisterType::OutputDepthGE
                | RegisterType::OutputDepthLE
                | RegisterType::OutputStencilRef
        )
    }
}

impl From<RegisterType> for Sm4RegisterType {
    fn from(register_type: RegisterType) -> Self {
        Self(register_type.to_u32())
    }
}

/// One atomic component of an SM4/SM5 register.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sm4Component {
    /// The X component.
    X,
    /// The Y component.
    Y,
    /// The Z component.
    Z,
    /// The W component.
    W,
    /// A scalar or object-like operand without vector components.
    Whole,
    /// A component selector retained from malformed or future bytecode.
    Unknown(u8),
}

impl Sm4Component {
    fn from_raw(raw: u8) -> Self {
        match raw {
            0 => Self::X,
            1 => Self::Y,
            2 => Self::Z,
            3 => Self::W,
            value => Self::Unknown(value),
        }
    }
}

/// One index dimension of an SM4/SM5 register reference.
///
/// Relative address expressions retain the register component that supplies
/// the dynamic index. This keeps `x0[r1.x]` distinct from `x0[r1.y]` while the
/// address register is also reported as a normal instruction use.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sm4Index {
    /// Immediate 32-bit index.
    Immediate32(u32),
    /// Immediate 64-bit index.
    Immediate64(u64),
    /// Fully relative index.
    Relative(Box<Sm4Variable>),
    /// Immediate base plus a relative index.
    RelativePlusImmediate {
        /// Immediate base offset.
        offset: u32,
        /// Register component supplying the relative offset.
        relative: Box<Sm4Variable>,
    },
    /// 64-bit immediate base plus a relative index.
    RelativePlusImmediate64 {
        /// Immediate base offset.
        offset: u64,
        /// Register component supplying the relative offset.
        relative: Box<Sm4Variable>,
    },
}

/// Register identity without a component selection.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sm4Register {
    register_type: Sm4RegisterType,
    indices: Vec<Sm4Index>,
}

impl Sm4Register {
    /// Construct a register identity.
    #[must_use]
    pub fn new(register_type: RegisterType, indices: Vec<Sm4Index>) -> Self {
        Self {
            register_type: register_type.into(),
            indices,
        }
    }

    /// Return the register type.
    #[must_use]
    pub const fn register_type(&self) -> Sm4RegisterType {
        self.register_type
    }

    /// Return the register's index dimensions.
    #[must_use]
    pub fn indices(&self) -> &[Sm4Index] {
        &self.indices
    }

    fn from_operand(operand: &Operand) -> Self {
        Self {
            register_type: operand.reg_type.into(),
            indices: operand.indices.iter().map(convert_index).collect(),
        }
    }
}

/// Component-granular SM4/SM5 variable identity used by SSA and data flow.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sm4Variable {
    register: Sm4Register,
    component: Sm4Component,
}

impl Sm4Variable {
    /// Construct a component-granular variable identity.
    #[must_use]
    pub const fn new(register: Sm4Register, component: Sm4Component) -> Self {
        Self {
            register,
            component,
        }
    }

    /// Return the underlying register identity.
    #[must_use]
    pub const fn register(&self) -> &Sm4Register {
        &self.register
    }

    /// Return the selected atomic component.
    #[must_use]
    pub const fn component(&self) -> Sm4Component {
        self.component
    }

    fn relative(operand: &Operand) -> Self {
        let component = selected_components(&operand.components)
            .into_iter()
            .next()
            .unwrap_or(Sm4Component::Whole);
        Self::new(Sm4Register::from_operand(operand), component)
    }
}

/// A decoded SM4/SM5 instruction with cached data-flow information.
///
/// Construction derives component-granular uses and definitions plus ordered
/// memory events once. The cached data then satisfies [`InstrInfo`] and
/// [`MemoryEventInfo`] without allocating during analysis.
#[derive(Debug, Clone, PartialEq)]
pub struct Sm4Instruction {
    instruction: Instruction,
    uses: Vec<Sm4Variable>,
    defs: Vec<Sm4Variable>,
    effects: Vec<Sm4Effect>,
    memory_events: Vec<Sm4MemoryEvent>,
}

impl Sm4Instruction {
    /// Adapt a decoded instruction for CFG and data-flow analysis.
    #[must_use]
    pub fn new(instruction: Instruction) -> Self {
        let analysis = analyze_operands(&instruction);
        let memory_events = analyze_memory_events(&instruction);
        let mut effects = analyze_effects(&instruction, &analysis.defs);
        if analysis.dynamic_indexable_write {
            push_unique(&mut effects, Sm4Effect::IndexableWrite);
        }
        Self {
            instruction,
            uses: analysis.uses,
            defs: analysis.defs,
            effects,
            memory_events,
        }
    }

    /// Borrow the decoded instruction.
    #[must_use]
    pub const fn instruction(&self) -> &Instruction {
        &self.instruction
    }

    /// Consume the adapter and recover the decoded instruction.
    #[must_use]
    pub fn into_instruction(self) -> Instruction {
        self.instruction
    }
}

impl From<Instruction> for Sm4Instruction {
    fn from(instruction: Instruction) -> Self {
        Self::new(instruction)
    }
}

impl AsRef<Instruction> for Sm4Instruction {
    fn as_ref(&self) -> &Instruction {
        self.instruction()
    }
}

impl FlowControl for Sm4Instruction {
    fn flow_effect(&self) -> FlowEffect {
        match self.instruction.opcode {
            Opcode::If => FlowEffect::ConditionalOpen,
            Opcode::Else => FlowEffect::ConditionalAlternate,
            Opcode::EndIf => FlowEffect::ConditionalClose,
            Opcode::Switch => FlowEffect::SwitchOpen,
            Opcode::Case | Opcode::Default => FlowEffect::SwitchCase,
            Opcode::EndSwitch => FlowEffect::SwitchClose,
            Opcode::Loop => FlowEffect::LoopOpen,
            Opcode::EndLoop => FlowEffect::LoopClose,
            Opcode::Break => FlowEffect::Break,
            Opcode::Breakc => FlowEffect::ConditionalBreak,
            Opcode::Continue => FlowEffect::Continue,
            Opcode::Continuec => FlowEffect::ConditionalContinue,
            Opcode::Ret => FlowEffect::Return,
            Opcode::Retc => FlowEffect::ConditionalReturn,
            Opcode::Call | Opcode::InterfaceCall => FlowEffect::Call,
            Opcode::Callc => FlowEffect::ConditionalCall,
            Opcode::Discard => FlowEffect::Fallthrough,
            Opcode::Abort => FlowEffect::Terminate,
            Opcode::Label => FlowEffect::Label,
            opcode if is_declaration(opcode) => FlowEffect::Declaration,
            _ => FlowEffect::Fallthrough,
        }
    }
}

impl cfglib::DisplayInstr for Sm4Instruction {
    fn mnemonic(&self) -> std::borrow::Cow<'_, str> {
        std::borrow::Cow::Borrowed(self.instruction.opcode.name())
    }
}

impl InstrInfo for Sm4Instruction {
    type Variable = Sm4Variable;

    fn uses(&self) -> &[Self::Variable] {
        &self.uses
    }

    fn defs(&self) -> &[Self::Variable] {
        &self.defs
    }
}

impl cfglib::EffectInfo for Sm4Instruction {
    type Effect = Sm4Effect;

    fn effects(&self) -> &[Sm4Effect] {
        &self.effects
    }
}

impl MemoryEventInfo for Sm4Instruction {
    type Location = Sm4MemoryLocation;
    type Fence = Sm4Fence;

    fn memory_events(
        &self,
    ) -> impl Iterator<Item = cfglib::MemoryEvent<Self::Location, Self::Variable, Self::Fence>>
    {
        self.memory_events.iter().cloned()
    }
}

impl cfglib::CallInfo for Sm4Instruction {
    type Callee = u32;

    fn callee(&self) -> Option<u32> {
        if !matches!(
            self.instruction.opcode,
            Opcode::Call | Opcode::Callc | Opcode::InterfaceCall
        ) {
            return None;
        }
        self.instruction
            .operands()
            .iter()
            .find(|operand| operand.reg_type == RegisterType::Label)
            .and_then(|operand| match operand.indices.first() {
                Some(&OperandIndex::Imm32(value)) => Some(value),
                _ => None,
            })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OperandLayout {
    None,
    Sources,
    Destinations(usize),
    Memory {
        destinations: usize,
        target: usize,
        address_start: usize,
        address_count: usize,
        kind: MemoryAccessKind,
        atomicity: MemoryAtomicity,
    },
}

impl OperandLayout {
    const fn is_source(self, index: usize) -> bool {
        match self {
            Self::None => false,
            Self::Sources => true,
            Self::Destinations(count) => index >= count,
            Self::Memory {
                destinations,
                target,
                ..
            } => index >= destinations && index != target,
        }
    }

    const fn is_destination(self, index: usize) -> bool {
        match self {
            Self::Destinations(count) => index < count,
            Self::Memory { destinations, .. } => index < destinations,
            Self::None | Self::Sources => false,
        }
    }

    const fn memory_access(self, index: usize) -> Option<(MemoryAccessKind, MemoryAtomicity)> {
        match self {
            Self::Memory {
                target,
                kind,
                atomicity,
                ..
            } if index == target => Some((kind, atomicity)),
            _ => None,
        }
    }

    const fn is_memory_address(self, index: usize) -> bool {
        match self {
            Self::Memory {
                address_start,
                address_count,
                ..
            } => index >= address_start && index < address_start + address_count,
            Self::None | Self::Sources | Self::Destinations(_) => false,
        }
    }
}

fn operand_layout(opcode: Opcode) -> OperandLayout {
    if is_declaration(opcode) {
        return OperandLayout::None;
    }
    if let Some(layout) = memory_operand_layout(opcode) {
        return layout;
    }

    match opcode {
        Opcode::IMul
        | Opcode::Sincos
        | Opcode::UDiv
        | Opcode::UMul
        | Opcode::Uaddc
        | Opcode::Usubb
        | Opcode::Swapc
        | Opcode::Gather4Feedback
        | Opcode::Gather4CFeedback
        | Opcode::Gather4PoFeedback
        | Opcode::Gather4PoCFeedback
        | Opcode::LdFeedback
        | Opcode::LdMsFeedback
        | Opcode::LdUavTypedFeedback
        | Opcode::LdRawFeedback
        | Opcode::LdStructuredFeedback
        | Opcode::SampleLFeedback
        | Opcode::SampleCLzFeedback
        | Opcode::SampleClampFeedback
        | Opcode::SampleBClampFeedback
        | Opcode::SampleDClampFeedback
        | Opcode::SampleCClampFeedback => OperandLayout::Destinations(2),

        Opcode::Break
        | Opcode::Breakc
        | Opcode::Call
        | Opcode::Callc
        | Opcode::Case
        | Opcode::Continue
        | Opcode::Continuec
        | Opcode::Cut
        | Opcode::Default
        | Opcode::Discard
        | Opcode::Else
        | Opcode::Emit
        | Opcode::EmitThenCut
        | Opcode::EndIf
        | Opcode::EndLoop
        | Opcode::EndSwitch
        | Opcode::If
        | Opcode::Label
        | Opcode::Loop
        | Opcode::Nop
        | Opcode::Ret
        | Opcode::Retc
        | Opcode::Switch
        | Opcode::EmitStream
        | Opcode::CutStream
        | Opcode::EmitThenCutStream
        | Opcode::InterfaceCall
        | Opcode::Sync
        | Opcode::Abort
        | Opcode::DebugBreak
        | Opcode::Unknown(_) => OperandLayout::Sources,

        _ => OperandLayout::Destinations(1),
    }
}

fn memory_operand_layout(opcode: Opcode) -> Option<OperandLayout> {
    let layout = match opcode {
        Opcode::StoreUavTyped | Opcode::StoreRaw => OperandLayout::Memory {
            destinations: 0,
            target: 0,
            address_start: 1,
            address_count: 1,
            kind: MemoryAccessKind::Write,
            atomicity: MemoryAtomicity::NonAtomic,
        },
        Opcode::StoreStructured => OperandLayout::Memory {
            destinations: 0,
            target: 0,
            address_start: 1,
            address_count: 2,
            kind: MemoryAccessKind::Write,
            atomicity: MemoryAtomicity::NonAtomic,
        },
        Opcode::AtomicAnd
        | Opcode::AtomicOr
        | Opcode::AtomicXor
        | Opcode::AtomicCmpStore
        | Opcode::AtomicIAdd
        | Opcode::AtomicIMax
        | Opcode::AtomicIMin
        | Opcode::AtomicUMax
        | Opcode::AtomicUMin => OperandLayout::Memory {
            destinations: 0,
            target: 0,
            address_start: 1,
            address_count: 1,
            kind: MemoryAccessKind::ReadModifyWrite,
            atomicity: MemoryAtomicity::Atomic,
        },
        Opcode::ImmAtomicAlloc | Opcode::ImmAtomicConsume => OperandLayout::Memory {
            destinations: 1,
            target: 1,
            address_start: 2,
            address_count: 0,
            kind: MemoryAccessKind::ReadModifyWrite,
            atomicity: MemoryAtomicity::Atomic,
        },
        Opcode::ImmAtomicIAdd
        | Opcode::ImmAtomicAnd
        | Opcode::ImmAtomicOr
        | Opcode::ImmAtomicXor
        | Opcode::ImmAtomicExch
        | Opcode::ImmAtomicCmpExch
        | Opcode::ImmAtomicIMax
        | Opcode::ImmAtomicIMin
        | Opcode::ImmAtomicUMax
        | Opcode::ImmAtomicUMin => OperandLayout::Memory {
            destinations: 1,
            target: 1,
            address_start: 2,
            address_count: 1,
            kind: MemoryAccessKind::ReadModifyWrite,
            atomicity: MemoryAtomicity::Atomic,
        },
        _ => return None,
    };
    Some(layout)
}

/// The def/use analysis result: the variable sets plus whether the
/// instruction stores through a dynamically-indexed indexable temp.
struct OperandAnalysis {
    uses: Vec<Sm4Variable>,
    defs: Vec<Sm4Variable>,
    /// A write through `x#[reg]` hit a statically unknowable slot. Such a
    /// store is reported as NO def (a may-def must not kill other slots'
    /// reaching definitions) and instead declares
    /// [`Sm4Effect::IndexableWrite`] so liveness/DCE can never delete it.
    dynamic_indexable_write: bool,
}

/// Whether `operand` addresses an indexable temp (`x#`) through a
/// register-relative index — a slot no static identity can name.
fn is_dynamic_indexable(operand: &Operand) -> bool {
    operand.reg_type == RegisterType::IndexableTemp
        && operand.indices.iter().any(|index| {
            matches!(
                index,
                OperandIndex::Relative(_)
                    | OperandIndex::RelativePlusImm(..)
                    | OperandIndex::RelativePlusImm64(..)
            )
        })
}

fn analyze_operands(instruction: &Instruction) -> OperandAnalysis {
    let operands = instruction.operands();
    let layout = operand_layout(instruction.opcode);
    let mut uses = Vec::new();
    let mut defs = Vec::new();
    let mut dynamic_indexable_write = false;

    for (index, operand) in operands.iter().enumerate() {
        if layout.is_destination(index) {
            dynamic_indexable_write |= collect_definition(operand, &mut uses, &mut defs);
        } else if layout.memory_access(index).is_some() {
            // The memory object is a location, not an SSA definition. Only
            // registers used to compute a relative binding index are value
            // dependencies.
            collect_index_uses(operand, &mut uses);
        } else if layout.is_source(index) {
            collect_use(operand, &mut uses);
        }
    }

    OperandAnalysis {
        uses,
        defs,
        dynamic_indexable_write,
    }
}

fn collect_use(operand: &Operand, uses: &mut Vec<Sm4Variable>) {
    collect_index_uses(operand, uses);
    collect_selected_variables(operand, uses);
}

/// Collect a destination operand. Returns whether it was a dynamic
/// indexable-temp store (reported as effect, not def — see
/// [`OperandAnalysis::dynamic_indexable_write`]).
fn collect_definition(
    operand: &Operand,
    uses: &mut Vec<Sm4Variable>,
    defs: &mut Vec<Sm4Variable>,
) -> bool {
    collect_index_uses(operand, uses);
    if is_dynamic_indexable(operand) {
        return true;
    }
    collect_selected_variables(operand, defs);
    false
}

fn collect_index_uses(operand: &Operand, uses: &mut Vec<Sm4Variable>) {
    for index in &operand.indices {
        match index {
            OperandIndex::Imm32(_) | OperandIndex::Imm64(_) => {}
            OperandIndex::Relative(relative)
            | OperandIndex::RelativePlusImm(_, relative)
            | OperandIndex::RelativePlusImm64(_, relative) => collect_use(relative, uses),
        }
    }
}

fn collect_selected_variables(operand: &Operand, variables: &mut Vec<Sm4Variable>) {
    for variable in operand_variables(operand) {
        push_unique(variables, variable);
    }
}

fn operand_variables(operand: &Operand) -> Vec<Sm4Variable> {
    let register = Sm4Register::from_operand(operand);
    if !register.register_type.is_data() {
        return Vec::new();
    }

    selected_components(&operand.components)
        .into_iter()
        .map(|component| Sm4Variable::new(register.clone(), component))
        .collect()
}

fn selected_components(selection: &ComponentSelect) -> Vec<Sm4Component> {
    match selection {
        ComponentSelect::ZeroComponent => vec![Sm4Component::Whole],
        ComponentSelect::OneComponent => vec![Sm4Component::X],
        ComponentSelect::Mask(mask) => (0_u8..4)
            .filter(|component| mask & (1 << component) != 0)
            .map(Sm4Component::from_raw)
            .collect(),
        ComponentSelect::Swizzle(swizzle) => swizzle
            .iter()
            .copied()
            .map(Sm4Component::from_raw)
            .collect(),
        ComponentSelect::Scalar(component) => {
            vec![Sm4Component::from_raw(*component)]
        }
    }
}

fn convert_index(index: &OperandIndex) -> Sm4Index {
    match index {
        OperandIndex::Imm32(value) => Sm4Index::Immediate32(*value),
        OperandIndex::Imm64(value) => Sm4Index::Immediate64(*value),
        OperandIndex::Relative(relative) => {
            Sm4Index::Relative(Box::new(Sm4Variable::relative(relative)))
        }
        OperandIndex::RelativePlusImm(offset, relative) => Sm4Index::RelativePlusImmediate {
            offset: *offset,
            relative: Box::new(Sm4Variable::relative(relative)),
        },
        OperandIndex::RelativePlusImm64(offset, relative) => Sm4Index::RelativePlusImmediate64 {
            offset: *offset,
            relative: Box::new(Sm4Variable::relative(relative)),
        },
    }
}

fn push_unique<T: PartialEq>(values: &mut Vec<T>, value: T) {
    if !values.contains(&value) {
        values.push(value);
    }
}

fn analyze_effects(instruction: &Instruction, defs: &[Sm4Variable]) -> Vec<Sm4Effect> {
    let mut effects = Vec::new();

    if memory::resource_read_region(instruction.opcode).is_some() {
        push_unique(&mut effects, Sm4Effect::ResourceRead);
    }
    if let OperandLayout::Memory { kind, .. } = operand_layout(instruction.opcode) {
        if kind.reads() {
            push_unique(&mut effects, Sm4Effect::ResourceRead);
        }
        if kind.writes() {
            push_unique(&mut effects, Sm4Effect::ResourceWrite);
        }
    }

    match instruction.opcode {
        Opcode::Call | Opcode::Callc | Opcode::InterfaceCall => {
            push_unique(&mut effects, Sm4Effect::Call);
        }
        Opcode::Discard => {
            push_unique(&mut effects, Sm4Effect::Export);
            push_unique(&mut effects, Sm4Effect::Sync);
        }
        Opcode::Cut
        | Opcode::Emit
        | Opcode::EmitThenCut
        | Opcode::EmitStream
        | Opcode::CutStream
        | Opcode::EmitThenCutStream => push_unique(&mut effects, Sm4Effect::Export),
        Opcode::Sync => push_unique(&mut effects, Sm4Effect::Sync),
        Opcode::Abort | Opcode::DebugBreak | Opcode::Unknown(_) => {
            push_unique(&mut effects, Sm4Effect::Unknown);
        }
        _ => {}
    }

    if defs
        .iter()
        .any(|variable| variable.register.register_type.is_observable_output())
    {
        push_unique(&mut effects, Sm4Effect::Export);
    }

    effects
}

/// Return whether an opcode is a declaration rather than executable code.
pub(crate) fn is_declaration(opcode: Opcode) -> bool {
    matches!(
        opcode,
        Opcode::DclGlobalFlags
            | Opcode::DclInput
            | Opcode::DclInputSgv
            | Opcode::DclInputSiv
            | Opcode::DclInputPs
            | Opcode::DclInputPsSgv
            | Opcode::DclInputPsSiv
            | Opcode::DclOutput
            | Opcode::DclOutputSgv
            | Opcode::DclOutputSiv
            | Opcode::DclResource
            | Opcode::DclSampler
            | Opcode::DclConstantBuffer
            | Opcode::DclTemps
            | Opcode::DclIndexableTemp
            | Opcode::DclIndexRange
            | Opcode::DclGsInputPrimitive
            | Opcode::DclGsOutputPrimitiveTopology
            | Opcode::DclMaxOutputVertexCount
            | Opcode::DclGsInstanceCount
            | Opcode::DclOutputControlPointCount
            | Opcode::DclInputControlPointCount
            | Opcode::DclTessDomain
            | Opcode::DclTessPartitioning
            | Opcode::DclTessOutputPrimitive
            | Opcode::DclHsMaxTessFactor
            | Opcode::DclHsForkPhaseInstanceCount
            | Opcode::DclHsJoinPhaseInstanceCount
            | Opcode::DclThreadGroup
            | Opcode::DclUnorderedAccessViewTyped
            | Opcode::DclUnorderedAccessViewRaw
            | Opcode::DclUnorderedAccessViewStructured
            | Opcode::DclThreadGroupSharedMemoryRaw
            | Opcode::DclThreadGroupSharedMemoryStructured
            | Opcode::DclResourceRaw
            | Opcode::DclResourceStructured
            | Opcode::DclStream
            | Opcode::DclFunctionBody
            | Opcode::DclFunctionTable
            | Opcode::DclInterface
            | Opcode::HsDecls
            | Opcode::HsControlPointPhase
            | Opcode::HsForkPhase
            | Opcode::HsJoinPhase
            | Opcode::CustomData
    )
}
