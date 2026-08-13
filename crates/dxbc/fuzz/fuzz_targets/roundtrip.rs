//! Coverage-guided round-trip fuzzer for the lossless `.d3dasm` text format.
//!
//! `arbitrary::Unstructured` turns the libFuzzer input bytes into a structured
//! `Program`, which is then driven through `encode -> decode -> serialize ->
//! assemble -> encode`. Both IR-equality and byte-identity must hold; any
//! mismatch (or a failed `assemble`) panics, which libFuzzer reports.
//!
//! Run with:  cargo +nightly fuzz run roundtrip

#![cfg_attr(not(test), no_main)]

use arbitrary::Unstructured;
use libfuzzer_sys::fuzz_target;
use smallvec::SmallVec;

use d3dasm::{assemble, serialize};
use dxbc::shex::{
    ComponentSelect, Immediates, Indices, Instruction, InstructionKind, MinPrecision, Opcode,
    Operand, OperandIndex, Operands, Program, RegisterType, decode, encode,
};

const FUZZ_OPCODES: &[Opcode] = &[
    Opcode::Add,
    Opcode::Mul,
    Opcode::Mad,
    Opcode::Mov,
    Opcode::Movc,
    Opcode::Dp2,
    Opcode::Dp3,
    Opcode::Dp4,
    Opcode::Min,
    Opcode::Max,
    Opcode::And,
    Opcode::Or,
    Opcode::Ishl,
    Opcode::Iadd,
    Opcode::Sample,
    Opcode::SampleL,
    Opcode::Ld,
    Opcode::Ret,
    Opcode::If,
    Opcode::EndIf,
];

const SHADER_TYPES: &[&str] = &["ps", "vs", "gs", "hs", "ds", "cs"];

type R<T> = arbitrary::Result<T>;

fn gen_components(u: &mut Unstructured<'_>) -> R<ComponentSelect> {
    Ok(match u.int_in_range(0u8..=4)? {
        0 => ComponentSelect::ZeroComponent,
        1 => ComponentSelect::OneComponent,
        2 => ComponentSelect::Mask(u.int_in_range(0u8..=15)?),
        3 => ComponentSelect::Swizzle([
            u.int_in_range(0u8..=3)?,
            u.int_in_range(0u8..=3)?,
            u.int_in_range(0u8..=3)?,
            u.int_in_range(0u8..=3)?,
        ]),
        _ => ComponentSelect::Scalar(u.int_in_range(0u8..=3)?),
    })
}

/// A simple, always-encodable operand used inside relative index brackets.
fn gen_simple_operand(u: &mut Unstructured<'_>) -> R<Operand> {
    let mut indices: Indices = SmallVec::new();
    indices.push(OperandIndex::Imm32(u.int_in_range(0u32..=7)?));
    Ok(Operand {
        reg_type: RegisterType::Temp,
        components: ComponentSelect::Scalar(u.int_in_range(0u8..=3)?),
        negate: u.arbitrary()?,
        abs: false,
        min_precision: MinPrecision::Default,
        non_uniform: false,
        indices,
        immediate_values: SmallVec::new(),
    })
}

fn gen_index(u: &mut Unstructured<'_>, allow_relative: bool) -> R<OperandIndex> {
    let hi = if allow_relative { 3 } else { 1 };
    Ok(match u.int_in_range(0u8..=hi)? {
        0 => OperandIndex::Imm32(u.int_in_range(0u32..=63)?),
        1 => OperandIndex::Imm64(u.int_in_range(0u64..=63)?),
        2 => OperandIndex::Relative(Box::new(gen_simple_operand(u)?)),
        _ => OperandIndex::RelativePlusImm(
            u.int_in_range(0u32..=63)?,
            Box::new(gen_simple_operand(u)?),
        ),
    })
}

fn gen_operand(u: &mut Unstructured<'_>) -> R<Operand> {
    // ~1 in 8 operands is an inline immediate.
    if u.int_in_range(0u8..=7)? == 0 {
        let (n, comp) = match u.int_in_range(0u8..=2)? {
            0 => (0u32, ComponentSelect::ZeroComponent),
            1 => (1, ComponentSelect::OneComponent),
            _ => (4, gen_components(u)?),
        };
        let mut values: Immediates = SmallVec::new();
        for _ in 0..n {
            values.push(u.arbitrary()?);
        }
        return Ok(Operand {
            reg_type: RegisterType::Immediate32,
            components: comp,
            negate: u.arbitrary()?,
            abs: u.arbitrary()?,
            min_precision: MinPrecision::Default,
            non_uniform: false,
            indices: SmallVec::new(),
            immediate_values: values,
        });
    }

    // Any register file (0..=45 covers all known plus Unknown(41..=45)).
    let mut reg_type = RegisterType::from_u32(u.int_in_range(0u32..=45)?);
    if matches!(
        reg_type,
        RegisterType::Immediate32 | RegisterType::Immediate64
    ) {
        reg_type = RegisterType::Temp;
    }

    let nidx = u.int_in_range(0u8..=3)?;
    let mut indices: Indices = SmallVec::new();
    for i in 0..nidx {
        indices.push(gen_index(u, i + 1 == nidx)?);
    }

    Ok(Operand {
        reg_type,
        components: gen_components(u)?,
        negate: u.arbitrary()?,
        abs: u.arbitrary()?,
        min_precision: MinPrecision::Default,
        non_uniform: false,
        indices,
        immediate_values: SmallVec::new(),
    })
}

fn gen_instruction(u: &mut Unstructured<'_>) -> R<Instruction> {
    let opcode = if u.int_in_range(0u8..=19)? == 0 {
        Opcode::Unknown(u.int_in_range(218u32..=0x7FF)?)
    } else {
        *u.choose(FUZZ_OPCODES)?
    };

    let nops = u.int_in_range(0u8..=4)?;
    let mut operands: Operands = SmallVec::new();
    for _ in 0..nops {
        operands.push(gen_operand(u)?);
    }

    let mut i = Instruction {
        opcode,
        saturate: u.arbitrary()?,
        test_nonzero: u.arbitrary()?,
        precise_mask: u.int_in_range(0u8..=15)?,
        resinfo_return_type: None,
        sync_flags: 0,
        tex_offsets: None,
        resource_dim: None,
        resource_return_type: None,
        kind: InstructionKind::Generic {
            operands,
            trailing_tokens: SmallVec::new(),
        },
    };
    if u.int_in_range(0u8..=3)? == 0 {
        i.resinfo_return_type = Some(u.int_in_range(0u32..=2)?);
    }
    if u.int_in_range(0u8..=3)? == 0 {
        i.sync_flags = u.int_in_range(0u8..=15)?;
    }
    if u.int_in_range(0u8..=3)? == 0 {
        i.tex_offsets = Some([
            u.int_in_range(-8i8..=7)?,
            u.int_in_range(-8i8..=7)?,
            u.int_in_range(-8i8..=7)?,
        ]);
    }
    if u.int_in_range(0u8..=3)? == 0 {
        i.resource_dim =
            Some(2 | (u.int_in_range(0u32..=12)? << 6) | (u.int_in_range(0u32..=3)? << 11));
    }
    if u.int_in_range(0u8..=3)? == 0 {
        i.resource_return_type = Some(
            3 | (u.int_in_range(0u32..=15)? << 6)
                | (u.int_in_range(0u32..=15)? << 10)
                | (u.int_in_range(0u32..=15)? << 14)
                | (u.int_in_range(0u32..=15)? << 18),
        );
    }
    Ok(i)
}

fn gen_program(u: &mut Unstructured<'_>) -> R<Program> {
    let shader_type = *u.choose(SHADER_TYPES)?;
    let n = u.int_in_range(1u8..=8)?;
    let mut instructions = Vec::new();
    for _ in 0..n {
        instructions.push(gen_instruction(u)?);
    }
    Ok(Program {
        shader_type,
        major_version: 5,
        minor_version: u.int_in_range(0u32..=1)?,
        instructions,
        warnings: Vec::new(),
        fourcc: if u.arbitrary()? { *b"SHEX" } else { *b"SHDR" },
    })
}

fuzz_target!(|data: &[u8]| {
    let mut u = Unstructured::new(data);
    let Ok(prog) = gen_program(&mut u) else {
        return;
    };

    let bytes = encode(&prog);
    let Ok(canon) = decode(&bytes) else {
        return;
    };

    let text = serialize(&canon);
    let parsed = assemble(&text).unwrap_or_else(|e| panic!("assemble failed: {e}\n{text}"));

    assert_eq!(canon, parsed, "IR round-trip mismatch\n{text}");
    assert_eq!(
        encode(&canon),
        encode(&parsed),
        "byte round-trip mismatch\n{text}"
    );
});
