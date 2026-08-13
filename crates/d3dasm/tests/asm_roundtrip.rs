//! Targeted round-trip tests for the lossless `.d3dasm` text format.
//!
//! Each case constructs IR, encodes to canonical bytes, then asserts that
//! `assemble(serialize(decode(bytes)))` reproduces both the IR and the bytes.

mod common;

use common::*;
use d3dasm::{assemble, serialize};
use dxbc::shex::{
    ComponentSelect, CustomDataType, InstructionKind, Opcode, Operand, OperandIndex, Program,
    RegisterType, ReturnType, decode, encode,
};
use smallvec::{SmallVec, smallvec};

#[test]
fn ret_only() {
    rt(&program(vec![generic(Opcode::Ret, vec![])]));
}

#[test]
fn mov_mask_swizzle() {
    rt(&program(vec![
        generic(Opcode::Mov, vec![temp(0, MASK_ALL), temp(1, XYZW)]),
        generic(Opcode::Ret, vec![]),
    ]));
}

#[test]
fn add_three_operands() {
    rt(&program(vec![
        generic(
            Opcode::Add,
            vec![temp(0, MASK_ALL), temp(1, XYZW), temp(2, XYZW)],
        ),
        generic(Opcode::Ret, vec![]),
    ]));
}

#[test]
fn partial_mask_and_swizzle() {
    rt(&program(vec![
        generic(
            Opcode::Mul,
            vec![
                temp(0, ComponentSelect::Mask(0b0011)),
                temp(1, ComponentSelect::Swizzle([1, 1, 2, 3])),
                temp(2, ComponentSelect::Scalar(2)),
            ],
        ),
        generic(Opcode::Ret, vec![]),
    ]));
}

#[test]
fn negate_and_abs() {
    let mut a = temp(1, XYZW);
    a.negate = true;
    let mut b = temp(2, XYZW);
    b.abs = true;
    let mut c = temp(3, XYZW);
    c.negate = true;
    c.abs = true;
    rt(&program(vec![
        generic(Opcode::Mad, vec![temp(0, MASK_ALL), a, b, c]),
        generic(Opcode::Ret, vec![]),
    ]));
}

#[test]
fn immediate_operand() {
    let imm = Operand {
        reg_type: RegisterType::Immediate32,
        components: XYZW,
        negate: false,
        abs: false,
        min_precision: Default::default(),
        non_uniform: false,
        indices: SmallVec::new(),
        immediate_values: smallvec![0x3f80_0000, 0x4000_0000, 0x0000_0000, 0xc080_0000],
    };
    rt(&program(vec![
        generic(Opcode::Mov, vec![temp(0, MASK_ALL), imm]),
        generic(Opcode::Ret, vec![]),
    ]));
}

#[test]
fn constant_buffer_operand() {
    // cb0[16].xyzw — two-dimensional immediate index.
    let cb = reg_op(
        RegisterType::ConstantBuffer,
        XYZW,
        smallvec![OperandIndex::Imm32(0), OperandIndex::Imm32(16)],
    );
    rt(&program(vec![
        generic(Opcode::Mov, vec![temp(0, MASK_ALL), cb]),
        generic(Opcode::Ret, vec![]),
    ]));
}

#[test]
fn relative_index_operand() {
    // cb0[r1.x + 2]
    let inner = temp(1, ComponentSelect::Scalar(0));
    let cb = reg_op(
        RegisterType::ConstantBuffer,
        XYZW,
        smallvec![
            OperandIndex::Imm32(0),
            OperandIndex::RelativePlusImm(2, Box::new(inner))
        ],
    );
    rt(&program(vec![
        generic(Opcode::Mov, vec![temp(0, MASK_ALL), cb]),
        generic(Opcode::Ret, vec![]),
    ]));
}

#[test]
fn saturate_modifier() {
    let mut i = generic(
        Opcode::Add,
        vec![temp(0, MASK_ALL), temp(1, XYZW), temp(2, XYZW)],
    );
    i.saturate = true;
    rt(&program(vec![i, generic(Opcode::Ret, vec![])]));
}

#[test]
fn dcl_temps() {
    rt(&program(vec![insn(
        Opcode::DclTemps,
        InstructionKind::DclTemps { count: 3 },
    )]));
}

#[test]
fn dcl_global_flags() {
    rt(&program(vec![insn(
        Opcode::DclGlobalFlags,
        InstructionKind::DclGlobalFlags {
            flags: smallvec!["refactoringAllowed"],
        },
    )]));
}

#[test]
fn dcl_thread_group() {
    rt(&program(vec![insn(
        Opcode::DclThreadGroup,
        InstructionKind::DclThreadGroup { x: 8, y: 8, z: 1 },
    )]));
}

#[test]
fn dcl_constant_buffer() {
    let cb = reg_op(
        RegisterType::ConstantBuffer,
        ComponentSelect::Swizzle([0, 1, 2, 3]),
        smallvec![OperandIndex::Imm32(0), OperandIndex::Imm32(8)],
    );
    rt(&program(vec![insn(
        Opcode::DclConstantBuffer,
        InstructionKind::DclConstantBuffer {
            access: "immediateIndexed",
            operands: smallvec![cb],
            size: 8,
            space: 0,
        },
    )]));
}

#[test]
fn dcl_sampler() {
    let s = reg_op(
        RegisterType::Sampler,
        ComponentSelect::ZeroComponent,
        smallvec![OperandIndex::Imm32(0)],
    );
    rt(&program(vec![insn(
        Opcode::DclSampler,
        InstructionKind::DclSampler {
            mode: "default",
            operands: smallvec![s],
            space: 0,
        },
    )]));
}

#[test]
fn dcl_resource_texture2d() {
    let t = reg_op(
        RegisterType::Resource,
        ComponentSelect::Swizzle([0, 1, 2, 3]),
        smallvec![OperandIndex::Imm32(0)],
    );
    rt(&program(vec![insn(
        Opcode::DclResource,
        InstructionKind::DclResource {
            dimension: "texture2d",
            sample_count: 0,
            return_type: [ReturnType::Float; 4],
            operands: smallvec![t],
            space: 0,
        },
    )]));
}

#[test]
fn dcl_input_ps_interp() {
    let v = reg_op(
        RegisterType::Input,
        ComponentSelect::Mask(0b0011),
        smallvec![OperandIndex::Imm32(0)],
    );
    rt(&program(vec![insn(
        Opcode::DclInputPs,
        InstructionKind::DclInput {
            interpolation: Some("linear"),
            system_value: None,
            operands: smallvec![v],
        },
    )]));
}

#[test]
fn dcl_output() {
    let o = reg_op(
        RegisterType::Output,
        ComponentSelect::Mask(0xF),
        smallvec![OperandIndex::Imm32(0)],
    );
    rt(&program(vec![insn(
        Opcode::DclOutput,
        InstructionKind::DclOutput {
            system_value: None,
            operands: smallvec![o],
        },
    )]));
}

#[test]
fn customdata_icb() {
    rt(&program(vec![insn(
        Opcode::CustomData,
        InstructionKind::CustomData {
            subtype: CustomDataType::ImmediateConstantBuffer,
            values: vec![[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            raw: Vec::new(),
            raw_dword_count: 10,
        },
    )]));
}

#[test]
fn customdata_opaque_payload() {
    // Opaque/comment/debuginfo payloads must round-trip byte-identically.
    rt(&program(vec![insn(
        Opcode::CustomData,
        InstructionKind::CustomData {
            subtype: CustomDataType::Opaque,
            values: Vec::new(),
            raw: vec![
                0xdead_beef,
                0x0000_0001,
                0xffff_ffff,
                0x1234_5678,
                0x9abc_def0,
            ],
            raw_dword_count: 7,
        },
    )]));
}

#[test]
fn customdata_comment_empty() {
    rt(&program(vec![insn(
        Opcode::CustomData,
        InstructionKind::CustomData {
            subtype: CustomDataType::Comment,
            values: Vec::new(),
            raw: Vec::new(),
            raw_dword_count: 2,
        },
    )]));
}

#[test]
fn shdr_fourcc_preserved() {
    // The SHDR FourCC must survive the text round-trip (encode ignores it, so
    // this is a serialize/parse-level property).
    let p = Program {
        shader_type: "vs",
        major_version: 4,
        minor_version: 0,
        instructions: vec![generic(Opcode::Ret, vec![])],
        warnings: Vec::new(),
        fourcc: *b"SHDR",
    };
    let text = serialize(&p);
    assert!(
        text.starts_with("profile=vs_4_0 fourcc=SHDR\n"),
        "text:\n{text}"
    );
    let p2 = assemble(&text).expect("parse failed");
    assert_eq!(p2, p, "SHDR round-trip mismatch\n{text}");
}

#[test]
fn declarations_have_no_garbage_modifiers() {
    // A texture3d resource sets token0 bit 13 (part of the dimension field);
    // decode must NOT surface that as `saturate`.
    let t = reg_op(
        RegisterType::Resource,
        ComponentSelect::Swizzle([0, 1, 2, 3]),
        smallvec![OperandIndex::Imm32(0)],
    );
    let prog = program(vec![insn(
        Opcode::DclResource,
        InstructionKind::DclResource {
            dimension: "texture3d",
            sample_count: 0,
            return_type: [ReturnType::Float; 4],
            operands: smallvec![t],
            space: 0,
        },
    )]);
    let canon = decode(&encode(&prog)).unwrap();
    assert!(
        !canon.instructions[0].saturate,
        "decode leaked garbage saturate"
    );
    assert_eq!(
        canon.instructions[0].precise_mask, 0,
        "garbage precise_mask"
    );
    rt(&prog);
}

#[test]
fn float_immediate_edges() {
    // 1.0, -0.0, +inf, NaN — NaN must fall back to hex, the rest stay pretty.
    let imm = Operand {
        reg_type: RegisterType::Immediate32,
        components: XYZW,
        negate: false,
        abs: false,
        min_precision: Default::default(),
        non_uniform: false,
        indices: SmallVec::new(),
        immediate_values: smallvec![0x3f80_0000, 0x8000_0000, 0x7f80_0000, 0x7fc0_0000],
    };
    rt(&program(vec![
        generic(Opcode::Mov, vec![temp(0, MASK_ALL), imm]),
        generic(Opcode::Ret, vec![]),
    ]));
}

#[test]
fn comments_are_ignored() {
    let p = program(vec![
        generic(Opcode::Mov, vec![temp(0, MASK_ALL), temp(1, XYZW)]),
        generic(Opcode::Ret, vec![]),
    ]);
    let canon = decode(&encode(&p)).unwrap();
    let text = serialize(&canon);
    let commented = format!(
        "// a header comment\n{}\n// a dangling comment",
        text.replace("\nret", "  // trailing comment\nret")
    );
    let parsed = assemble(&commented).expect("parse with comments");
    assert_eq!(
        encode(&canon),
        encode(&parsed),
        "comments changed the result"
    );
}

#[test]
fn resource_extended_tokens() {
    // A sample with resource_dim + resource_return_type extended tokens must
    // render readably (_res/_rt) and round-trip.
    let mut i = generic(
        Opcode::Sample,
        vec![
            temp(0, MASK_ALL),
            temp(1, XYZW),
            reg_op(
                RegisterType::Resource,
                XYZW,
                smallvec![OperandIndex::Imm32(0)],
            ),
            reg_op(
                RegisterType::Sampler,
                ComponentSelect::ZeroComponent,
                smallvec![OperandIndex::Imm32(0)],
            ),
        ],
    );
    // type 2, dim=texture2d(3), stride=0 ; type 3, four floats(5)
    i.resource_dim = Some(2 | (3 << 6));
    i.resource_return_type = Some(3 | (5 << 6) | (5 << 10) | (5 << 14) | (5 << 18));
    let canon = decode(&encode(&program(vec![
        i.clone(),
        generic(Opcode::Ret, vec![]),
    ])))
    .unwrap();
    let text = serialize(&canon);
    assert!(
        text.contains("_res(texture2d)"),
        "expected readable res:\n{text}"
    );
    assert!(
        text.contains("_rt(float,float,float,float)"),
        "expected readable rt:\n{text}"
    );
    rt(&program(vec![i, generic(Opcode::Ret, vec![])]));
}

#[test]
fn unknown_opcode_roundtrip() {
    rt(&program(vec![
        generic(Opcode::Unknown(253), vec![temp(0, MASK_ALL), temp(1, XYZW)]),
        generic(Opcode::Ret, vec![]),
    ]));
}

#[test]
fn unknown_register_roundtrip() {
    let r = reg_op(
        RegisterType::Unknown(45),
        XYZW,
        smallvec![OperandIndex::Imm32(0)],
    );
    rt(&program(vec![
        generic(Opcode::Mov, vec![temp(0, MASK_ALL), r]),
        generic(Opcode::Ret, vec![]),
    ]));
}

#[test]
fn full_pixel_shader() {
    rt(&program(vec![
        insn(
            Opcode::DclGlobalFlags,
            InstructionKind::DclGlobalFlags {
                flags: smallvec!["refactoringAllowed"],
            },
        ),
        insn(Opcode::DclTemps, InstructionKind::DclTemps { count: 1 }),
        generic(Opcode::Mov, vec![temp(0, MASK_ALL), temp(1, XYZW)]),
        generic(Opcode::Ret, vec![]),
    ]));
}

#[test]
fn vector_immediate_has_no_trailing_colon() {
    // A 4-component inline immediate decodes to mask-mode with an empty mask.
    // Its component selection is implied by the value count, so it must serialize
    // as a bare `l(...)` with no dangling `:` — and still round-trip.
    let imm = Operand {
        reg_type: RegisterType::Immediate32,
        components: ComponentSelect::Mask(0),
        negate: false,
        abs: false,
        min_precision: Default::default(),
        non_uniform: false,
        indices: SmallVec::new(),
        immediate_values: smallvec![0x3f80_0000, 0x3f80_0000, 0, 0],
    };
    let prog = program(vec![
        generic(Opcode::Max, vec![temp(0, MASK_ALL), temp(1, XYZW), imm]),
        generic(Opcode::Ret, vec![]),
    ]);
    let canon = decode(&encode(&prog)).unwrap();
    let text = serialize(&canon);
    assert!(!text.contains("):"), "stray colon after immediate:\n{text}");
    rt(&prog);
}

#[test]
fn dcl_resource_uses_named_tags() {
    // A resource declaration is self-describing: operand first, then `dim=`,
    // `rt=`, and an always-present `samples=` tag. Both round-trip.
    let t = |dim| {
        let r = reg_op(
            RegisterType::Resource,
            ComponentSelect::Swizzle([0, 1, 2, 3]),
            smallvec![OperandIndex::Imm32(0)],
        );
        program(vec![insn(
            Opcode::DclResource,
            InstructionKind::DclResource {
                dimension: dim,
                sample_count: 0,
                return_type: [ReturnType::Float; 4],
                operands: smallvec![r],
                space: 0,
            },
        )])
    };

    let tex2d = t("texture2d");
    let text = serialize(&decode(&encode(&tex2d)).unwrap());
    assert!(
        text.contains("dim=texture2d rt=float,float,float,float samples=0"),
        "unexpected resource text:\n{text}"
    );
    rt(&tex2d);

    let texms = t("texture2dms");
    let text = serialize(&decode(&encode(&texms)).unwrap());
    assert!(text.contains("dim=texture2dms"), "MS dim missing:\n{text}");
    assert!(text.contains("samples=0"), "samples missing:\n{text}");
    rt(&texms);
}
