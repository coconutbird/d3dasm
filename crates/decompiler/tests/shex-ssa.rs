use cfglib::{DominatorTree, EffectInfo, InstrInfo, ProgramPoint, SsaForm};
use decompiler::cfg::Sm4Effect;
use decompiler::cfg::{
    Sm4Component, Sm4Index, Sm4Instruction, Sm4Register, Sm4Variable, build_cfg,
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

    assert_eq!(instruction.uses(), [address_variable, indexed_variable]);
}

#[test]
fn multi_result_and_read_modify_write_roles_are_explicit() {
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
    assert!(store.uses().contains(&target));
    assert_eq!(store.defs(), core::slice::from_ref(&target));
    assert!(store.effects().contains(&Sm4Effect::ResourceWrite));
}
