//! End-to-end test for the full-container forensic `.d3dasm` document:
//! synthesize a container, serialize it, and assert it reassembles
//! byte-identically (preserving the header hash and all non-program chunks).

use d3dasm::dxbc;
use dxbc::chunks::WritableChunk;
use dxbc::shex::{Instruction, InstructionKind, Opcode, Program};

fn ret_program() -> Vec<u8> {
    let ret = Instruction {
        opcode: Opcode::Ret,
        saturate: false,
        test_nonzero: false,
        precise_mask: 0,
        resinfo_return_type: None,
        sync_flags: 0,
        tex_offsets: None,
        resource_dim: None,
        resource_return_type: None,
        kind: InstructionKind::Generic {
            operands: dxbc::shex::Operands::new(),
            trailing_tokens: dxbc::shex::SmallU32Vec::new(),
        },
    };
    let program = Program {
        shader_type: "ps",
        major_version: 5,
        minor_version: 0,
        instructions: vec![ret],
        warnings: Vec::new(),
        fourcc: *b"SHEX",
    };
    dxbc::shex::encode(&program)
}

/// Build a container with a valid (recomputed) header checksum.
fn valid_container(chunks: &[WritableChunk], version: u32) -> Vec<u8> {
    let mut bytes = dxbc::container::build_dxbc_with_header(chunks, version, &[0u8; 16]);
    let digest = dxbc::checksum::dxbc_checksum(&bytes[20..]);
    bytes[4..20].copy_from_slice(&digest);
    bytes
}

#[test]
fn container_document_roundtrip() {
    let shex = ret_program();
    let priv_data = vec![0xde, 0xad, 0xbe, 0xef, 0x01, 0x02, 0x03];

    // Build a container: editable SHEX program + an opaque PRIV chunk.
    let original = valid_container(
        &[
            WritableChunk {
                fourcc: *b"SHEX",
                data: shex.clone(),
            },
            WritableChunk {
                fourcc: *b"PRIV",
                data: priv_data.clone(),
            },
        ],
        1,
    );

    let shaders = d3dasm::parse(&original);
    assert_eq!(shaders.len(), 1);

    let text = d3dasm::container_doc::serialize(&shaders[0]);

    // Forensic header + directives are present.
    assert!(text.contains("// Container:"), "missing header:\n{text}");
    assert!(text.contains("PRIV"), "missing chunk inventory:\n{text}");
    assert!(
        text.contains(".code SHEX"),
        "missing program chunk:\n{text}"
    );
    assert!(text.contains("\nret"), "missing disassembly body:\n{text}");
    assert!(text.contains(".chunk PRIV"), "missing raw chunk:\n{text}");

    // Reassembles byte-identically, header hash and all.
    let rebuilt = d3dasm::container_doc::assemble(&text)
        .unwrap_or_else(|e| panic!("assemble failed: {e}\n{text}"));
    assert_eq!(rebuilt, original, "container not byte-identical\n{text}");
    // The header hash is a freshly computed (valid) checksum of the content.
    assert_eq!(
        &rebuilt[4..20],
        &dxbc::checksum::dxbc_checksum(&rebuilt[20..])[..],
        "header hash not recomputed"
    );
}

#[test]
fn assemble_all_handles_multiple_containers() {
    let shex = ret_program();
    let one = valid_container(
        &[WritableChunk {
            fourcc: *b"SHEX",
            data: shex.clone(),
        }],
        1,
    );
    // Two back-to-back containers (an archive).
    let mut archive = one.clone();
    archive.extend_from_slice(&one);

    let mut doc = String::new();
    for shader in &d3dasm::parse(&archive) {
        doc.push_str(&d3dasm::container_doc::serialize(shader));
    }
    let rebuilt = d3dasm::container_doc::assemble_all(&doc).expect("assemble_all");
    assert_eq!(rebuilt, archive, "archive not byte-identical");
}
