use cfglib::{
    Cfg, DominatorTree, EffectInfo, InstrInfo, MemoryAccessKind, MemoryAlias, MemoryAtomicity,
    MemoryDefinition, MemoryEvent, MemoryEventInfo, MemoryEventSite, MemoryOperations, MemorySSA,
    ProgramPoint, SsaForm,
};
use decompiler::cfg::{
    Sm4Component, Sm4Effect, Sm4Index, Sm4Instruction, Sm4MemoryAlias, Sm4MemoryLocation,
    Sm4MemoryRegion, Sm4Register, Sm4Variable, build_cfg,
};
use dxbc::shex::{
    ComponentSelect, Immediates, Indices, Instruction, InstructionKind, MinPrecision, Opcode,
    Operand, OperandIndex, RegisterType, SmallU32Vec, decode,
};

fn instruction_token(opcode: Opcode, dword_length: u32) -> u32 {
    (dword_length << 24) | opcode.to_u32()
}

fn destination_token(register_type: RegisterType, mask: u8) -> u32 {
    0x0010_0002 | (register_type.to_u32() << 12) | (u32::from(mask) << 4)
}

fn scalar_source_token(register_type: RegisterType, component: u8) -> u32 {
    0x0010_000A | (register_type.to_u32() << 12) | (u32::from(component) << 4)
}

fn swizzled_source_token(register_type: RegisterType, swizzle: [u8; 4]) -> u32 {
    let encoded = u32::from(swizzle[0])
        | (u32::from(swizzle[1]) << 2)
        | (u32::from(swizzle[2]) << 4)
        | (u32::from(swizzle[3]) << 6);
    0x0010_0006 | (register_type.to_u32() << 12) | (encoded << 4)
}

fn append_mov(
    dwords: &mut Vec<u32>,
    destination_type: RegisterType,
    destination: u32,
    destination_mask: u8,
    source_type: RegisterType,
    source: u32,
    source_component: u8,
) {
    dwords.extend([
        instruction_token(Opcode::Mov, 5),
        destination_token(destination_type, destination_mask),
        destination,
        scalar_source_token(source_type, source_component),
        source,
    ]);
}

fn encode_program(mut instructions: Vec<u32>) -> Vec<u8> {
    let mut dwords = vec![(1 << 16) | (5 << 4), 0];
    dwords.append(&mut instructions);
    dwords[1] = u32::try_from(dwords.len()).unwrap();
    dwords.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn variable(
    register_type: RegisterType,
    register_index: u32,
    component: Sm4Component,
) -> Sm4Variable {
    Sm4Variable::new(
        Sm4Register::new(register_type, vec![Sm4Index::Immediate32(register_index)]),
        component,
    )
}

fn operand(
    register_type: RegisterType,
    components: ComponentSelect,
    indices: impl IntoIterator<Item = OperandIndex>,
) -> Operand {
    Operand {
        reg_type: register_type,
        components,
        negate: false,
        abs: false,
        min_precision: MinPrecision::Default,
        non_uniform: false,
        indices: indices.into_iter().collect::<Indices>(),
        immediate_values: Immediates::new(),
    }
}

fn generic_instruction(opcode: Opcode, operands: impl IntoIterator<Item = Operand>) -> Instruction {
    Instruction {
        opcode,
        saturate: false,
        test_nonzero: false,
        precise_mask: 0,
        resinfo_return_type: None,
        sync_flags: 0,
        tex_offsets: None,
        resource_dim: None,
        resource_return_type: None,
        kind: InstructionKind::Generic {
            operands: operands.into_iter().collect(),
            trailing_tokens: SmallU32Vec::new(),
        },
    }
}

#[test]
fn decoded_shex_uses_component_granular_variables() {
    let instructions = vec![
        instruction_token(Opcode::Mov, 5),
        destination_token(RegisterType::Temp, 0b0011),
        0,
        swizzled_source_token(RegisterType::Input, [2, 3, 2, 3]),
        1,
        instruction_token(Opcode::Ret, 1),
    ];
    let bytes = encode_program(instructions);
    let program = decode(&bytes).unwrap();
    let instruction = Sm4Instruction::new(program.instructions[0].clone());

    assert_eq!(
        instruction.defs(),
        [
            variable(RegisterType::Temp, 0, Sm4Component::X),
            variable(RegisterType::Temp, 0, Sm4Component::Y),
        ]
    );
    assert_eq!(
        instruction.uses(),
        [
            variable(RegisterType::Input, 1, Sm4Component::Z),
            variable(RegisterType::Input, 1, Sm4Component::W),
        ]
    );
}

#[test]
fn decoded_shex_builds_renamed_ssa_across_a_branch() {
    let mut instructions = Vec::new();
    append_mov(
        &mut instructions,
        RegisterType::Temp,
        0,
        0b0001,
        RegisterType::Input,
        0,
        0,
    );
    instructions.extend([
        instruction_token(Opcode::If, 3),
        scalar_source_token(RegisterType::Input, 0),
        3,
    ]);
    append_mov(
        &mut instructions,
        RegisterType::Temp,
        0,
        0b0001,
        RegisterType::Input,
        1,
        0,
    );
    instructions.push(instruction_token(Opcode::Else, 1));
    append_mov(
        &mut instructions,
        RegisterType::Temp,
        0,
        0b0001,
        RegisterType::Input,
        2,
        0,
    );
    instructions.push(instruction_token(Opcode::EndIf, 1));
    append_mov(
        &mut instructions,
        RegisterType::Output,
        0,
        0b0001,
        RegisterType::Temp,
        0,
        0,
    );
    instructions.push(instruction_token(Opcode::Ret, 1));

    let bytes = encode_program(instructions);
    let program = decode(&bytes).unwrap();
    let cfg = build_cfg(&program).unwrap();
    let dominators = DominatorTree::compute(&cfg);
    let ssa = SsaForm::compute(&cfg, &dominators);
    let temporary = variable(RegisterType::Temp, 0, Sm4Component::X);
    let output = variable(RegisterType::Output, 0, Sm4Component::X);

    let (merge, phi) = ssa
        .phis()
        .find(|(_, phi)| phi.result.variable == temporary)
        .unwrap();
    assert_eq!(phi.operands.len(), 2);
    assert_ne!(phi.operands[0].1, phi.operands[1].1);

    let output_index = cfg
        .block(merge)
        .instructions()
        .iter()
        .position(|instruction| instruction.defs().contains(&output))
        .unwrap();
    let native_output = &cfg.block(merge).instructions()[output_index];
    assert!(native_output.effects().contains(&Sm4Effect::Export));
    let output_uses = &ssa
        .instruction(ProgramPoint {
            block: merge,
            inst_idx: output_index,
        })
        .unwrap()
        .uses;
    assert_eq!(output_uses, core::slice::from_ref(&phi.result));
}

#[test]
fn latest_shex_relative_index_is_retained_and_read() {
    let address = operand(
        RegisterType::Temp,
        ComponentSelect::Scalar(0),
        [OperandIndex::Imm32(2)],
    );
    let source = operand(
        RegisterType::IndexableTemp,
        ComponentSelect::Scalar(1),
        [
            OperandIndex::Imm32(0),
            OperandIndex::RelativePlusImm64(1_u64 << 40, Box::new(address)),
        ],
    );
    let destination = operand(
        RegisterType::Temp,
        ComponentSelect::Mask(0b0001),
        [OperandIndex::Imm32(0)],
    );
    let instruction = Sm4Instruction::new(generic_instruction(Opcode::Mov, [destination, source]));
    let address_variable = variable(RegisterType::Temp, 2, Sm4Component::X);
    let indexed_variable = Sm4Variable::new(
        Sm4Register::new(
            RegisterType::IndexableTemp,
            vec![
                Sm4Index::Immediate32(0),
                Sm4Index::RelativePlusImmediate64 {
                    offset: 1_u64 << 40,
                    relative: Box::new(address_variable.clone()),
                },
            ],
        ),
        Sm4Component::Y,
    );

    assert_eq!(
        instruction.uses(),
        [address_variable.clone(), indexed_variable]
    );
    let events: Vec<_> = instruction.memory_events().collect();
    assert!(matches!(
        events.as_slice(),
        [MemoryEvent::Access(access)]
            if access.kind == MemoryAccessKind::Read
                && access.address_uses() == [address_variable]
                && access.location.region() == Sm4MemoryRegion::Contents
                && access.location.register().register_type().decoded()
                    == RegisterType::IndexableTemp
    ));
}

#[test]
fn multi_result_and_memory_roles_are_explicit() {
    let udiv = Sm4Instruction::new(generic_instruction(
        Opcode::UDiv,
        [
            operand(
                RegisterType::Temp,
                ComponentSelect::Mask(1),
                [OperandIndex::Imm32(0)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Mask(1),
                [OperandIndex::Imm32(1)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Scalar(0),
                [OperandIndex::Imm32(2)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Scalar(0),
                [OperandIndex::Imm32(3)],
            ),
        ],
    ));
    assert_eq!(
        udiv.defs(),
        [
            variable(RegisterType::Temp, 0, Sm4Component::X),
            variable(RegisterType::Temp, 1, Sm4Component::X),
        ]
    );
    assert_eq!(
        udiv.uses(),
        [
            variable(RegisterType::Temp, 2, Sm4Component::X),
            variable(RegisterType::Temp, 3, Sm4Component::X),
        ]
    );

    let target = Sm4Variable::new(
        Sm4Register::new(RegisterType::Uav, vec![Sm4Index::Immediate32(0)]),
        Sm4Component::Whole,
    );
    let store = Sm4Instruction::new(generic_instruction(
        Opcode::StoreUavTyped,
        [
            operand(
                RegisterType::Uav,
                ComponentSelect::ZeroComponent,
                [OperandIndex::Imm32(0)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Scalar(0),
                [OperandIndex::Imm32(4)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Swizzle([0, 1, 2, 3]),
                [OperandIndex::Imm32(5)],
            ),
        ],
    ));
    assert!(!store.uses().contains(&target));
    assert!(store.defs().is_empty());
    assert!(store.effects().contains(&Sm4Effect::ResourceWrite));
    assert_eq!(store.memory_operations(), MemoryOperations::WRITE);
    let events: Vec<_> = store.memory_events().collect();
    assert!(matches!(
        events.as_slice(),
        [MemoryEvent::Access(access)]
            if access.kind == MemoryAccessKind::Write
                && access.atomicity == MemoryAtomicity::NonAtomic
                && access.address_uses() == &store.uses()[..1]
                && access.location.region() == Sm4MemoryRegion::Contents
                && access.location.register() == target.register()
    ));
}

#[test]
fn load_exposes_its_location_and_address_data_flow() {
    let load = Sm4Instruction::new(generic_instruction(
        Opcode::LdRaw,
        [
            operand(
                RegisterType::Temp,
                ComponentSelect::Mask(1),
                [OperandIndex::Imm32(0)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Scalar(0),
                [OperandIndex::Imm32(1)],
            ),
            operand(
                RegisterType::Resource,
                ComponentSelect::Swizzle([0, 1, 2, 3]),
                [OperandIndex::Imm32(3)],
            ),
        ],
    ));
    let load_events: Vec<_> = load.memory_events().collect();
    assert!(matches!(
        load_events.as_slice(),
        [MemoryEvent::Access(access)]
            if access.kind == MemoryAccessKind::Read
                && access.atomicity == MemoryAtomicity::NonAtomic
                && access.address_uses() == &load.uses()[..1]
                && access.location.region() == Sm4MemoryRegion::Contents
                && access.location.register().register_type().decoded() == RegisterType::Resource
    ));
    assert_eq!(load.memory_operations(), MemoryOperations::READ);
}

#[test]
fn lod_reads_resource_metadata_without_claiming_texel_contents() {
    let lod = Sm4Instruction::new(generic_instruction(
        Opcode::Lod,
        [
            operand(
                RegisterType::Temp,
                ComponentSelect::Mask(0b0011),
                [OperandIndex::Imm32(0)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Swizzle([0, 1, 2, 3]),
                [OperandIndex::Imm32(1)],
            ),
            operand(
                RegisterType::Resource,
                ComponentSelect::Swizzle([0, 1, 2, 3]),
                [OperandIndex::Imm32(2)],
            ),
            operand(
                RegisterType::Sampler,
                ComponentSelect::ZeroComponent,
                [OperandIndex::Imm32(3)],
            ),
        ],
    ));
    let events: Vec<_> = lod.memory_events().collect();

    assert!(matches!(
        events.as_slice(),
        [MemoryEvent::Access(access)]
            if access.kind == MemoryAccessKind::Read
                && access.location.region() == Sm4MemoryRegion::Metadata
    ));
}

#[test]
fn atomics_and_counters_expose_compound_memory_data_flow() {
    let atomic = Sm4Instruction::new(generic_instruction(
        Opcode::AtomicIAdd,
        [
            operand(
                RegisterType::ThreadGroupSharedMemory,
                ComponentSelect::ZeroComponent,
                [OperandIndex::Imm32(0)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Scalar(0),
                [OperandIndex::Imm32(1)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Scalar(0),
                [OperandIndex::Imm32(2)],
            ),
        ],
    ));
    let atomic_events: Vec<_> = atomic.memory_events().collect();
    assert!(matches!(
        atomic_events.as_slice(),
        [MemoryEvent::Access(access)]
            if access.kind == MemoryAccessKind::ReadModifyWrite
                && access.atomicity == MemoryAtomicity::Atomic
                && access.address_uses() == &atomic.uses()[..1]
                && access.location.register().register_type().decoded()
                    == RegisterType::ThreadGroupSharedMemory
    ));
    assert_eq!(
        atomic.memory_operations(),
        MemoryOperations::READ_MODIFY_WRITE
    );

    let counter = Sm4Instruction::new(generic_instruction(
        Opcode::ImmAtomicAlloc,
        [
            operand(
                RegisterType::Temp,
                ComponentSelect::Mask(1),
                [OperandIndex::Imm32(4)],
            ),
            operand(
                RegisterType::Uav,
                ComponentSelect::ZeroComponent,
                [OperandIndex::Imm32(2)],
            ),
        ],
    ));
    let counter_events: Vec<_> = counter.memory_events().collect();
    assert!(matches!(
        counter_events.as_slice(),
        [MemoryEvent::Access(access)]
            if access.kind == MemoryAccessKind::ReadModifyWrite
                && access.atomicity == MemoryAtomicity::Atomic
                && access.address_uses().is_empty()
                && access.location.region() == Sm4MemoryRegion::HiddenCounter
    ));
    assert_eq!(
        counter.memory_operations(),
        MemoryOperations::READ_MODIFY_WRITE
    );
}

#[test]
fn sync_exposes_fence_without_claiming_a_memory_access() {
    let mut sync = generic_instruction(Opcode::Sync, []);
    sync.sync_flags = 0b1111;
    let sync = Sm4Instruction::new(sync);
    let sync_events: Vec<_> = sync.memory_events().collect();
    assert!(matches!(
        sync_events.as_slice(),
        [MemoryEvent::Fence(fence)]
            if fence.synchronizes_threads_in_group()
                && fence.fences_thread_group_shared_memory()
                && fence.fences_uav_memory_in_group()
                && fence.fences_uav_memory_globally()
    ));
    assert_eq!(sync.memory_operations(), MemoryOperations::NONE);
}

#[test]
fn discard_exposes_its_implicit_global_uav_fence() {
    let discard = Sm4Instruction::new(generic_instruction(
        Opcode::Discard,
        [operand(
            RegisterType::Temp,
            ComponentSelect::Scalar(0),
            [OperandIndex::Imm32(0)],
        )],
    ));
    let events: Vec<_> = discard.memory_events().collect();

    assert!(matches!(
        events.as_slice(),
        [MemoryEvent::Fence(fence)]
            if fence.fences_uav_memory_globally()
                && !fence.fences_uav_memory_in_group()
                && !fence.fences_thread_group_shared_memory()
                && !fence.synchronizes_threads_in_group()
    ));
    assert!(discard.effects().contains(&Sm4Effect::Sync));
    assert_eq!(discard.memory_operations(), MemoryOperations::NONE);
}

fn memory_location(
    register_type: RegisterType,
    indices: Vec<Sm4Index>,
    region: Sm4MemoryRegion,
) -> Sm4MemoryLocation {
    Sm4MemoryLocation::new(Sm4Register::new(register_type, indices), region)
}

#[test]
fn sm4_aliasing_is_conservative_for_bindings_and_precise_for_local_arrays() {
    let resource = memory_location(
        RegisterType::Resource,
        vec![Sm4Index::Immediate32(0)],
        Sm4MemoryRegion::Contents,
    );
    let uav = memory_location(
        RegisterType::Uav,
        vec![Sm4Index::Immediate32(7)],
        Sm4MemoryRegion::Contents,
    );
    let metadata = memory_location(
        RegisterType::Resource,
        vec![Sm4Index::Immediate32(0)],
        Sm4MemoryRegion::Metadata,
    );
    assert!(Sm4MemoryAlias.may_alias(&resource, &uav));
    assert!(!Sm4MemoryAlias.may_alias(&resource, &metadata));

    let dynamic_index =
        Sm4Index::Relative(Box::new(variable(RegisterType::Temp, 3, Sm4Component::X)));
    let static_element = memory_location(
        RegisterType::IndexableTemp,
        vec![Sm4Index::Immediate32(0), Sm4Index::Immediate32(2)],
        Sm4MemoryRegion::Contents,
    );
    let dynamic_element = memory_location(
        RegisterType::IndexableTemp,
        vec![Sm4Index::Immediate32(0), dynamic_index.clone()],
        Sm4MemoryRegion::Contents,
    );
    let other_array = memory_location(
        RegisterType::IndexableTemp,
        vec![Sm4Index::Immediate32(1), dynamic_index],
        Sm4MemoryRegion::Contents,
    );
    assert!(Sm4MemoryAlias.may_alias(&static_element, &dynamic_element));
    assert!(!Sm4MemoryAlias.may_alias(&static_element, &other_array));

    let group_zero = memory_location(
        RegisterType::ThreadGroupSharedMemory,
        vec![Sm4Index::Immediate32(0)],
        Sm4MemoryRegion::Contents,
    );
    let group_one = memory_location(
        RegisterType::ThreadGroupSharedMemory,
        vec![Sm4Index::Immediate32(1)],
        Sm4MemoryRegion::Contents,
    );
    assert!(!Sm4MemoryAlias.may_alias(&group_zero, &group_one));
}

#[test]
fn sm4_memory_ssa_traces_runtime_aliased_storage_and_ssa_addresses() {
    let store = Sm4Instruction::new(generic_instruction(
        Opcode::StoreUavTyped,
        [
            operand(
                RegisterType::Uav,
                ComponentSelect::ZeroComponent,
                [OperandIndex::Imm32(0)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Scalar(0),
                [OperandIndex::Imm32(4)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Swizzle([0, 1, 2, 3]),
                [OperandIndex::Imm32(5)],
            ),
        ],
    ));
    let load = Sm4Instruction::new(generic_instruction(
        Opcode::LdRaw,
        [
            operand(
                RegisterType::Temp,
                ComponentSelect::Mask(1),
                [OperandIndex::Imm32(0)],
            ),
            operand(
                RegisterType::Temp,
                ComponentSelect::Scalar(0),
                [OperandIndex::Imm32(6)],
            ),
            operand(
                RegisterType::Resource,
                ComponentSelect::Swizzle([0, 1, 2, 3]),
                [OperandIndex::Imm32(1)],
            ),
        ],
    ));
    let mut cfg = Cfg::new();
    cfg.block_mut(cfg.entry()).push(store);
    cfg.block_mut(cfg.entry()).push(load);
    let memory = MemorySSA::compute(&cfg, &Sm4MemoryAlias);
    let store_site = MemoryEventSite::new(
        ProgramPoint {
            block: cfg.entry(),
            inst_idx: 0,
        },
        0,
    );
    let load_site = MemoryEventSite::new(
        ProgramPoint {
            block: cfg.entry(),
            inst_idx: 1,
        },
        0,
    );

    assert_eq!(
        memory.reaching_definition(load_site),
        Some(&MemoryDefinition::Event { site: store_site })
    );
    let load_event = memory.event(load_site).unwrap();
    assert_eq!(
        load_event
            .location()
            .unwrap()
            .register()
            .register_type()
            .decoded(),
        RegisterType::Resource
    );

    let dominators = DominatorTree::compute(&cfg);
    let values = SsaForm::compute(&cfg, &dominators);
    let address_values = load_event.ssa_address_uses(&values).unwrap();
    assert_eq!(address_values.len(), 1);
    assert_eq!(
        address_values[0].variable,
        variable(RegisterType::Temp, 6, Sm4Component::X)
    );
    assert!(address_values[0].is_live_in());
}
