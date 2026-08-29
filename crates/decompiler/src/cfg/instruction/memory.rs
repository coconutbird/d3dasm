//! SM4/SM5 memory-event vocabulary and instruction classification.

use cfglib::{MemoryAccessKind, MemoryAlias, MemoryAtomicity, MemoryEvent, MemoryEventAccess};
use dxbc::shex::{Instruction, Opcode, Operand, RegisterType};

use super::{Sm4Index, Sm4Register, Sm4Variable, collect_index_uses, collect_use, operand_layout};

/// The part of one SM4/SM5 storage object selected by an access.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Sm4MemoryRegion {
    /// The resource, buffer, or addressable array contents.
    Contents,
    /// Resource shape or sampling metadata.
    Metadata,
    /// The hidden counter associated with a count or append UAV.
    HiddenCounter,
}

/// A conservative SM4/SM5 memory location.
///
/// The register identifies the syntactically selected storage object or
/// addressable array slot. Dynamic byte/element addresses remain ordinary
/// instruction operands and are linked from
/// [`MemoryEventAccess::address_uses`]; this location deliberately denotes the
/// containing object or region rather than claiming a concrete runtime byte.
/// The event's [`cfglib::MemoryEventSite`] maps back to the decoded instruction
/// when a consumer needs its complete constant and vector address expression.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sm4MemoryLocation {
    register: Sm4Register,
    region: Sm4MemoryRegion,
}

impl Sm4MemoryLocation {
    /// Creates a conservative location within one storage object.
    #[must_use]
    pub const fn new(register: Sm4Register, region: Sm4MemoryRegion) -> Self {
        Self { register, region }
    }

    /// The bound storage object or addressable register-array slot.
    #[must_use]
    pub const fn register(&self) -> &Sm4Register {
        &self.register
    }

    /// The selected contents, metadata, or hidden-counter region.
    #[must_use]
    pub const fn region(&self) -> Sm4MemoryRegion {
        self.region
    }
}

/// Conservative default alias relation for SM4/SM5 memory locations.
///
/// Shader-local indexable temporaries, immediate constant buffers, and
/// thread-group shared objects are separated when unequal fixed indices prove
/// they cannot overlap. Runtime-bound resources, UAVs, and constant buffers
/// may refer to overlapping storage even when their register kinds or slots
/// differ, so locations in the same region are merged by default.
///
/// A caller with descriptor-table and resource-view information can pass a
/// narrower [`MemoryAlias`] implementation to [`cfglib::MemorySSA`]. Returning
/// `false` is a proof of disjointness, so unknown bindings must remain aliased.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct Sm4MemoryAlias;

impl MemoryAlias<Sm4MemoryLocation> for Sm4MemoryAlias {
    fn may_alias(&self, left: &Sm4MemoryLocation, right: &Sm4MemoryLocation) -> bool {
        if left.region != right.region {
            return false;
        }
        let left_type = left.register.register_type().decoded();
        let right_type = right.register.register_type().decoded();
        if is_runtime_bound_storage(left_type) && is_runtime_bound_storage(right_type) {
            return true;
        }
        if left_type != right_type {
            return false;
        }
        match left_type {
            RegisterType::IndexableTemp
            | RegisterType::ImmConstBuffer
            | RegisterType::ThreadGroupSharedMemory => {
                indices_may_overlap(left.register.indices(), right.register.indices())
            }
            _ => left.register == right.register,
        }
    }
}

const fn is_runtime_bound_storage(register_type: RegisterType) -> bool {
    matches!(
        register_type,
        RegisterType::Resource | RegisterType::Uav | RegisterType::ConstantBuffer
    )
}

fn indices_may_overlap(left: &[Sm4Index], right: &[Sm4Index]) -> bool {
    if left.len() != right.len() {
        return true;
    }
    left.iter().zip(right).all(
        |(left, right)| match (fixed_index(left), fixed_index(right)) {
            (Some(left), Some(right)) => left == right,
            _ => true,
        },
    )
}

const fn fixed_index(index: &Sm4Index) -> Option<u64> {
    match index {
        Sm4Index::Immediate32(value) => Some(*value as u64),
        Sm4Index::Immediate64(value) => Some(*value),
        Sm4Index::Relative(_)
        | Sm4Index::RelativePlusImmediate { .. }
        | Sm4Index::RelativePlusImmediate64 { .. } => None,
    }
}

/// SM5 synchronization scopes carried by a memory-fence event.
///
/// These are the four published token flags after shifting token bits 11–14
/// down to bits 0–3. An explicit `sync` retains its encoded flags; `discard`
/// emits its specified implicit global-UAV fence. The requested global UAV
/// scope can be narrowed by the runtime for a UAV not declared globally
/// coherent, so retaining the request leaves that binding-dependent decision
/// to later analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Sm4Fence(u8);

impl Sm4Fence {
    const THREADS_IN_GROUP: u8 = 1 << 0;
    const THREAD_GROUP_SHARED_MEMORY: u8 = 1 << 1;
    const UAV_MEMORY_GROUP: u8 = 1 << 2;
    const UAV_MEMORY_GLOBAL: u8 = 1 << 3;
    const ALL: u8 = Self::THREADS_IN_GROUP
        | Self::THREAD_GROUP_SHARED_MEMORY
        | Self::UAV_MEMORY_GROUP
        | Self::UAV_MEMORY_GLOBAL;

    /// Creates fence flags from the decoded four-bit representation.
    #[must_use]
    pub const fn from_raw(raw: u8) -> Self {
        Self(raw & Self::ALL)
    }

    /// Returns the decoded four-bit representation.
    #[must_use]
    pub const fn raw(self) -> u8 {
        self.0
    }

    /// Whether all threads in the group rendezvous at the instruction.
    #[must_use]
    pub const fn synchronizes_threads_in_group(self) -> bool {
        self.0 & Self::THREADS_IN_GROUP != 0
    }

    /// Whether thread-group shared memory is fenced for the group.
    #[must_use]
    pub const fn fences_thread_group_shared_memory(self) -> bool {
        self.0 & Self::THREAD_GROUP_SHARED_MEMORY != 0
    }

    /// Whether UAV memory is fenced within the current thread group.
    #[must_use]
    pub const fn fences_uav_memory_in_group(self) -> bool {
        self.0 & Self::UAV_MEMORY_GROUP != 0
    }

    /// Whether the instruction requests a global UAV-memory fence.
    #[must_use]
    pub const fn fences_uav_memory_globally(self) -> bool {
        self.0 & Self::UAV_MEMORY_GLOBAL != 0
    }
}

/// One generic memory event emitted by an adapted SM4/SM5 instruction.
///
/// Access events pair the logical storage location with the
/// [`Sm4Variable`] values used to select its binding or element. Texture and
/// sampling events conservatively retain every non-resource source operand,
/// including sampler or comparison inputs. Runtime descriptor aliasing is
/// handled conservatively by [`Sm4MemoryAlias`] unless the caller supplies a
/// binding-aware relation.
pub type Sm4MemoryEvent = MemoryEvent<Sm4MemoryLocation, Sm4Variable, Sm4Fence>;

pub(super) fn analyze_memory_events(instruction: &Instruction) -> Vec<Sm4MemoryEvent> {
    let layout = operand_layout(instruction.opcode);
    let operands = instruction.operands();
    let mut events = Vec::new();

    // Source expressions are evaluated before the instruction's principal
    // memory operation and before an addressable destination is committed.
    for (index, operand) in operands.iter().enumerate() {
        if layout.is_source(index) && is_addressable_value_storage(operand.reg_type) {
            let mut address_uses = Vec::new();
            collect_index_uses(operand, &mut address_uses);
            push_access(
                &mut events,
                operand,
                Sm4MemoryRegion::Contents,
                address_uses,
                MemoryAccessKind::Read,
                MemoryAtomicity::NonAtomic,
            );
        }
    }

    if let Some(region) = resource_read_region(instruction.opcode) {
        if let Some((resource_index, resource)) = operands
            .iter()
            .enumerate()
            .find(|(_, operand)| is_resource_storage(operand.reg_type))
        {
            let mut address_uses = Vec::new();
            collect_index_uses(resource, &mut address_uses);
            for (index, operand) in operands.iter().enumerate() {
                if index != resource_index && layout.is_source(index) {
                    collect_use(operand, &mut address_uses);
                }
            }
            push_access(
                &mut events,
                resource,
                region,
                address_uses,
                MemoryAccessKind::Read,
                MemoryAtomicity::NonAtomic,
            );
        }
    }

    for (index, operand) in operands.iter().enumerate() {
        if let Some((kind, atomicity)) = layout.memory_access(index) {
            let region = if matches!(
                instruction.opcode,
                Opcode::ImmAtomicAlloc | Opcode::ImmAtomicConsume
            ) {
                Sm4MemoryRegion::HiddenCounter
            } else {
                Sm4MemoryRegion::Contents
            };
            let mut address_uses = Vec::new();
            collect_index_uses(operand, &mut address_uses);
            for (address_index, address) in operands.iter().enumerate() {
                if layout.is_memory_address(address_index) {
                    collect_use(address, &mut address_uses);
                }
            }
            push_access(&mut events, operand, region, address_uses, kind, atomicity);
        }
    }

    for (index, operand) in operands.iter().enumerate() {
        if layout.is_destination(index) && operand.reg_type == RegisterType::IndexableTemp {
            let mut address_uses = Vec::new();
            collect_index_uses(operand, &mut address_uses);
            push_access(
                &mut events,
                operand,
                Sm4MemoryRegion::Contents,
                address_uses,
                MemoryAccessKind::Write,
                MemoryAtomicity::NonAtomic,
            );
        }
    }

    match instruction.opcode {
        Opcode::Sync => events.push(MemoryEvent::Fence(Sm4Fence::from_raw(
            instruction.sync_flags,
        ))),
        // SM5 specifies discard as an implicit sync_uglobal ordering
        // boundary, even though it has no explicit sync token flags.
        Opcode::Discard => events.push(MemoryEvent::Fence(Sm4Fence::from_raw(
            Sm4Fence::UAV_MEMORY_GLOBAL,
        ))),
        _ => {}
    }

    events
}

fn push_access(
    events: &mut Vec<Sm4MemoryEvent>,
    operand: &Operand,
    region: Sm4MemoryRegion,
    address_uses: Vec<Sm4Variable>,
    kind: MemoryAccessKind,
    atomicity: MemoryAtomicity,
) {
    let location = Sm4MemoryLocation::new(Sm4Register::from_operand(operand), region);
    events.push(MemoryEvent::Access(MemoryEventAccess::new(
        location,
        address_uses,
        kind,
        atomicity,
    )));
}

const fn is_addressable_value_storage(register_type: RegisterType) -> bool {
    matches!(
        register_type,
        RegisterType::ConstantBuffer | RegisterType::ImmConstBuffer | RegisterType::IndexableTemp
    )
}

const fn is_resource_storage(register_type: RegisterType) -> bool {
    matches!(
        register_type,
        RegisterType::Resource | RegisterType::Uav | RegisterType::ThreadGroupSharedMemory
    )
}

pub(super) const fn resource_read_region(opcode: Opcode) -> Option<Sm4MemoryRegion> {
    match opcode {
        Opcode::Resinfo
        | Opcode::Lod
        | Opcode::SamplePos
        | Opcode::SampleInfo
        | Opcode::BufInfo => Some(Sm4MemoryRegion::Metadata),
        Opcode::Ld
        | Opcode::LdMs
        | Opcode::Sample
        | Opcode::SampleC
        | Opcode::SampleCLz
        | Opcode::SampleL
        | Opcode::SampleD
        | Opcode::SampleB
        | Opcode::Gather4
        | Opcode::Gather4C
        | Opcode::Gather4Po
        | Opcode::Gather4PoC
        | Opcode::LdUavTyped
        | Opcode::LdRaw
        | Opcode::LdStructured
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
        | Opcode::SampleCClampFeedback => Some(Sm4MemoryRegion::Contents),
        _ => None,
    }
}
