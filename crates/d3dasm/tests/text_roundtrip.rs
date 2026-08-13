//! End-to-end test: synthesize a DXBC container holding a SHEX chunk, then
//! drive the full library path `parse -> serialize -> assemble -> encode` and
//! assert the shader chunk re-encodes byte-identically.

use d3dasm::dxbc;
use dxbc::chunks::WritableChunk;
use dxbc::shex::{
    ComponentSelect, Instruction, InstructionKind, MinPrecision, OperandIndex, Program,
    RegisterType,
};
use dxbc::shex::{Opcode, Operand};

fn temp(reg: u32, components: ComponentSelect) -> Operand {
    Operand {
        reg_type: RegisterType::Temp,
        components,
        negate: false,
        abs: false,
        min_precision: Default::default(),
        non_uniform: false,
        indices: core::iter::once(OperandIndex::Imm32(reg)).collect(),
        immediate_values: dxbc::shex::Immediates::new(),
    }
}

fn insn(opcode: Opcode, kind: InstructionKind) -> Instruction {
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

fn generic(opcode: Opcode, operands: Vec<Operand>) -> Instruction {
    insn(
        opcode,
        InstructionKind::Generic {
            operands: operands.into_iter().collect(),
            trailing_tokens: Default::default(),
        },
    )
}

#[test]
fn container_text_roundtrip() {
    let mut tagged_source = temp(1, ComponentSelect::Swizzle([0, 1, 2, 3]));
    tagged_source.min_precision = MinPrecision::Float16;
    tagged_source.non_uniform = true;
    let program = Program {
        shader_type: "ps",
        major_version: 5,
        minor_version: 0,
        instructions: vec![
            insn(
                Opcode::DclGlobalFlags,
                InstructionKind::DclGlobalFlags {
                    flags: core::iter::once("refactoringAllowed").collect(),
                },
            ),
            insn(Opcode::DclTemps, InstructionKind::DclTemps { count: 1 }),
            generic(
                Opcode::Mov,
                vec![
                    temp(0, ComponentSelect::Mask(0xF)),
                    tagged_source,
                ],
            ),
            generic(Opcode::Ret, vec![]),
        ],
        warnings: Vec::new(),
        fourcc: *b"SHEX",
    };

    // Encode to a SHEX chunk and wrap it in a real DXBC container.
    let shex_bytes = dxbc::shex::encode(&program);
    let container = dxbc::container::build_dxbc(&[WritableChunk {
        fourcc: *b"SHEX",
        data: shex_bytes.clone(),
    }]);

    // Drive the full library path through the container.
    let shaders = d3dasm::parse(&container);
    assert_eq!(shaders.len(), 1, "expected one shader in container");
    let prog = shaders[0]
        .program()
        .expect("container has a shader program");

    let text = d3dasm::serialize(prog);
    let parsed = d3dasm::assemble(&text).unwrap_or_else(|e| panic!("assemble failed: {e}\n{text}"));
    let reencoded = dxbc::shex::encode(&parsed);

    assert_eq!(
        reencoded, shex_bytes,
        "SHEX chunk did not re-encode byte-identically\n--- .d3dasm ---\n{text}"
    );

    // The text should look like assembly (sanity check on a couple of lines).
    assert!(
        text.starts_with("profile=ps_5_0\n"),
        "unexpected profile line:\n{text}"
    );
    assert!(
        text.contains("\nmov "),
        "expected a mov instruction:\n{text}"
    );
    assert!(
        text.contains("@min16f@nonuniform"),
        "expected lossless extended-operand tags:\n{text}"
    );
}
