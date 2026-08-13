//! Shared helpers for the `.d3dasm` integration tests.
//!
//! These tests exercise only the public API (`serialize` / `assemble` /
//! `decode` / `encode` plus the public IR types), so they live in `tests/`.

use d3dasm::{assemble, serialize};
use dxbc::shex::{
    ComponentSelect, Indices, Instruction, InstructionKind, MinPrecision, Opcode, Operand,
    OperandIndex, Program, RegisterType, SmallU32Vec, decode, encode,
};
use smallvec::{SmallVec, smallvec};

pub const XYZW: ComponentSelect = ComponentSelect::Swizzle([0, 1, 2, 3]);
pub const MASK_ALL: ComponentSelect = ComponentSelect::Mask(0xF);

pub fn insn(opcode: Opcode, kind: InstructionKind) -> Instruction {
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
        kind,
    }
}

pub fn program(instructions: Vec<Instruction>) -> Program {
    Program {
        shader_type: "ps",
        major_version: 5,
        minor_version: 0,
        instructions,
        warnings: Vec::new(),
        fourcc: *b"SHEX",
    }
}

pub fn generic(opcode: Opcode, operands: Vec<Operand>) -> Instruction {
    insn(
        opcode,
        InstructionKind::Generic {
            operands: operands.into_iter().collect(),
            trailing_tokens: SmallU32Vec::default(),
        },
    )
}

pub fn temp(reg: u32, comp: ComponentSelect) -> Operand {
    reg_op(
        RegisterType::Temp,
        comp,
        smallvec![OperandIndex::Imm32(reg)],
    )
}

pub fn reg_op(reg_type: RegisterType, components: ComponentSelect, indices: Indices) -> Operand {
    Operand {
        reg_type,
        components,
        negate: false,
        abs: false,
        min_precision: MinPrecision::default(),
        non_uniform: false,
        indices,
        immediate_values: SmallVec::new(),
    }
}

/// Assert the full round-trip through the text format: both IR-equality
/// (`assemble(serialize(canon)) == canon`) and byte-identity. `canon` is the
/// decoded IR, the clean reference.
pub fn rt(p: &Program) {
    let bytes = encode(p);
    let canon = decode(&bytes).expect("decode failed");
    let text = serialize(&canon);
    let parsed =
        assemble(&text).unwrap_or_else(|e| panic!("parse error: {e}\n--- text ---\n{text}"));
    assert_eq!(
        canon, parsed,
        "IR round-trip mismatch\n--- text ---\n{text}"
    );
    assert_eq!(
        encode(&canon),
        encode(&parsed),
        "byte round-trip mismatch\n--- text ---\n{text}"
    );
}
