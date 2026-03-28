//! SM4/SM5 shader bytecode handling: decode, encode, and format.
//!
//! The pipeline is **bytes → IR → text** (disassembly) or
//! **IR → bytes** (encoding / round-trip).
//!
//! * `decode` parses raw SHEX/SHDR chunk bytes into a `Program`.
//! * `format_program` renders a `Program` as human-readable text.
//! * `encode` serialises a `Program` back to the binary dword stream.
//! * `disassemble` is a convenience that chains decode + format.

use alloc::string::String;

mod decode;
mod encode;
mod fmt;
mod ir;
mod opcodes;

pub use self::decode::*;
pub use self::encode::*;
pub use self::fmt::*;
pub use self::ir::*;
pub use self::opcodes::*;

/// Disassemble a SHEX or SHDR chunk into human-readable text.
///
/// This is the main entry point. It decodes the raw bytes into a structured
/// IR ([`Program`]) and then formats it into text.
pub fn disassemble(data: &[u8]) -> String {
    match self::decode::decode(data) {
        Ok(program) => self::fmt::format_program(&program),
        Err(_) => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    /// Helper: build a minimal SHEX chunk (ps_5_0) from a list of instruction token slices.
    /// Returns raw bytes suitable for `disassemble()`.
    fn build_shex(shader_type: u32, instructions: &[&[u32]]) -> Vec<u8> {
        let mut dwords: Vec<u32> = Vec::new();
        // Version token: shader_type << 16 | major << 4 | minor
        dwords.push((shader_type << 16) | (5 << 4));
        // Placeholder for length (will fill in)
        dwords.push(0);
        for instr in instructions {
            dwords.extend_from_slice(instr);
        }
        // Fill in length (total dwords including header)
        dwords[1] = dwords.len() as u32;
        // Convert to bytes
        dwords.iter().flat_map(|d| d.to_le_bytes()).collect()
    }

    fn build_ps(instructions: &[&[u32]]) -> Vec<u8> {
        build_shex(0, instructions) // 0 = pixel shader
    }

    fn build_vs(instructions: &[&[u32]]) -> Vec<u8> {
        build_shex(1, instructions) // 1 = vertex shader
    }

    /// Helper: encode an instruction token with opcode and length.
    fn instr_token(opcode: u32, length: u32) -> u32 {
        (length << 24) | opcode
    }

    /// Helper: encode a temp register operand (r0-rN) with 4-component mask.
    /// For destinations: mask mode, e.g. r0.xyzw
    fn temp_dest(reg: u32, mask: u8) -> [u32; 2] {
        // num_components=2 (4-comp), sel_mode=0 (mask), op_type=0 (temp), index_dim=1 (1D)
        let token = 0x00100002 | ((mask as u32) << 4);
        [token, reg]
    }

    /// Helper: encode a temp register operand (r0-rN) with 4-component swizzle.
    /// For sources: swizzle mode, e.g. r1.xyzw
    fn temp_src(reg: u32, swizzle: u32) -> [u32; 2] {
        // num_components=2 (4-comp), sel_mode=1 (swizzle), op_type=0 (temp), index_dim=1 (1D)
        let token = 0x00100006 | (swizzle << 4);
        [token, reg]
    }

    /// Helper: encode a scalar temp source, e.g. r0.x
    fn temp_src_scalar(reg: u32, component: u32) -> [u32; 2] {
        // num_components=2 (4-comp), sel_mode=2 (scalar), op_type=0 (temp), index_dim=1 (1D)
        let token = 0x0010000A | (component << 4);
        [token, reg]
    }

    /// Swizzle constants
    const XYZW: u32 = 0b11_10_01_00; // x=0,y=1,z=2,w=3

    // Version header tests.

    #[test]
    fn version_ps_5_0() {
        let data = build_ps(&[&[instr_token(62, 1)]]); // ret
        let output = disassemble(&data);
        assert!(output.starts_with("ps_5_0\n"), "got: {output}");
    }

    #[test]
    fn version_vs_5_0() {
        let data = build_vs(&[&[instr_token(62, 1)]]); // ret
        let output = disassemble(&data);
        assert!(output.starts_with("vs_5_0\n"), "got: {output}");
    }

    #[test]
    fn version_gs_5_0() {
        let data = build_shex(2, &[&[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(output.starts_with("gs_5_0\n"), "got: {output}");
    }

    #[test]
    fn version_cs_5_0() {
        let data = build_shex(5, &[&[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(output.starts_with("cs_5_0\n"), "got: {output}");
    }

    // Simple instruction tests.

    #[test]
    fn instr_ret() {
        let data = build_ps(&[&[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(output.contains("ret\n"), "got: {output}");
    }

    #[test]
    fn instr_nop() {
        let data = build_ps(&[&[instr_token(58, 1)], &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(output.contains("nop\n"), "got: {output}");
    }

    #[test]
    fn instr_mov() {
        let d = temp_dest(0, 0xF); // r0.xyzw
        let s = temp_src(1, XYZW); // r1.xyzw
        let tokens = [instr_token(54, 5), d[0], d[1], s[0], s[1]];
        let data = build_ps(&[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(output.contains("mov r0.xyzw, r1.xyzw"), "got: {output}");
    }

    #[test]
    fn instr_add() {
        let d = temp_dest(0, 0xF);
        let s0 = temp_src(1, XYZW);
        let s1 = temp_src(2, XYZW);
        let tokens = [instr_token(0, 7), d[0], d[1], s0[0], s0[1], s1[0], s1[1]];
        let data = build_ps(&[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(
            output.contains("add r0.xyzw, r1.xyzw, r2.xyzw"),
            "got: {output}"
        );
    }

    #[test]
    fn instr_mul() {
        let d = temp_dest(0, 0xF);
        let s0 = temp_src(1, XYZW);
        let s1 = temp_src(2, XYZW);
        let tokens = [instr_token(56, 7), d[0], d[1], s0[0], s0[1], s1[0], s1[1]];
        let data = build_ps(&[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(
            output.contains("mul r0.xyzw, r1.xyzw, r2.xyzw"),
            "got: {output}"
        );
    }

    #[test]
    fn instr_mad() {
        let d = temp_dest(0, 0xF);
        let s0 = temp_src(1, XYZW);
        let s1 = temp_src(2, XYZW);
        let s2 = temp_src(3, XYZW);
        let tokens = [
            instr_token(50, 9),
            d[0],
            d[1],
            s0[0],
            s0[1],
            s1[0],
            s1[1],
            s2[0],
            s2[1],
        ];
        let data = build_ps(&[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(
            output.contains("mad r0.xyzw, r1.xyzw, r2.xyzw, r3.xyzw"),
            "got: {output}"
        );
    }

    // Saturate modifier.

    #[test]
    fn instr_mov_sat() {
        let d = temp_dest(0, 0xF);
        let s = temp_src(1, XYZW);
        // Saturate bit is bit 13
        let tokens = [instr_token(54, 5) | (1 << 13), d[0], d[1], s[0], s[1]];
        let data = build_ps(&[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(output.contains("mov_sat r0.xyzw, r1.xyzw"), "got: {output}");
    }

    // Declaration tests.

    #[test]
    fn dcl_global_flags_refactoring() {
        // opcode 106, length 1, flags bit 11 = refactoringAllowed
        let tokens = [instr_token(106, 1) | (1 << 11)];
        let data = build_ps(&[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(
            output.contains("dcl_globalFlags refactoringAllowed"),
            "got: {output}"
        );
    }

    #[test]
    fn dcl_temps() {
        // opcode 104, length 2, followed by count=4
        let tokens = [instr_token(104, 2), 4];
        let data = build_ps(&[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(output.contains("dcl_temps 4"), "got: {output}");
    }

    #[test]
    fn dcl_max_output_vertex_count() {
        // opcode 94, length 2
        let tokens = [instr_token(94, 2), 6];
        let data = build_shex(2, &[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(
            output.contains("dcl_maxOutputVertexCount 6"),
            "got: {output}"
        );
    }

    #[test]
    fn dcl_thread_group() {
        // opcode 155, length 4, followed by x, y, z
        let tokens = [instr_token(155, 4), 8, 8, 1];
        let data = build_shex(5, &[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        // dcl_thread_group goes through format_generic_instruction
        // which decodes operands — the 3 values will be treated as operands
        assert!(output.contains("dcl_thread_group"), "got: {output}");
    }

    // Control flow.

    #[test]
    fn control_flow_if_else_endif() {
        let cond = temp_src_scalar(0, 0); // r0.x
        let data = build_ps(&[
            &[instr_token(31, 3), cond[0], cond[1]], // if r0.x
            &[instr_token(58, 1)],                   // nop
            &[instr_token(18, 1)],                   // else
            &[instr_token(58, 1)],                   // nop
            &[instr_token(21, 1)],                   // endif
            &[instr_token(62, 1)],                   // ret
        ]);
        let output = disassemble(&data);
        assert!(output.contains("if"), "missing 'if' in: {output}");
        assert!(output.contains("else"), "missing 'else' in: {output}");
        assert!(output.contains("endif"), "missing 'endif' in: {output}");
    }

    #[test]
    fn control_flow_loop_endloop() {
        let data = build_ps(&[
            &[instr_token(48, 1)], // loop
            &[instr_token(2, 1)],  // break
            &[instr_token(22, 1)], // endloop
            &[instr_token(62, 1)], // ret
        ]);
        let output = disassemble(&data);
        assert!(output.contains("loop"), "missing 'loop' in: {output}");
        assert!(output.contains("break"), "missing 'break' in: {output}");
        assert!(output.contains("endloop"), "missing 'endloop' in: {output}");
    }

    // SM5 instructions.

    #[test]
    fn instr_emit_stream() {
        // opcode 117 = emit_stream, needs a stream operand
        // stream0 operand: op_type=16 (stream), index_dim=1 (1D), num_components=0
        let stream_op = (1u32 << 20) | (16 << 12); // index_dim=1, op_type=stream, 0-comp
        let tokens = [instr_token(117, 3), stream_op, 0]; // stream0
        let data = build_shex(2, &[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(output.contains("emit_stream"), "got: {output}");
    }

    #[test]
    fn instr_cut_stream() {
        let stream_op = (1u32 << 20) | (16 << 12);
        let tokens = [instr_token(118, 3), stream_op, 0];
        let data = build_shex(2, &[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(output.contains("cut_stream"), "got: {output}");
    }

    #[test]
    fn instr_hs_phases() {
        let data = build_shex(
            3,
            &[
                &[instr_token(113, 1)], // hs_decls
                &[instr_token(114, 1)], // hs_control_point_phase
                &[instr_token(115, 1)], // hs_fork_phase
                &[instr_token(62, 1)],  // ret
            ],
        );
        let output = disassemble(&data);
        assert!(output.contains("hs_decls"), "got: {output}");
        assert!(output.contains("hs_control_point_phase"), "got: {output}");
        assert!(output.contains("hs_fork_phase"), "got: {output}");
    }

    // Partial mask tests.

    #[test]
    fn instr_mov_partial_mask() {
        let d = temp_dest(0, 0x3); // r0.xy
        let s = temp_src(1, XYZW);
        let tokens = [instr_token(54, 5), d[0], d[1], s[0], s[1]];
        let data = build_ps(&[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(output.contains("mov r0.xy,"), "got: {output}");
    }

    // GS declarations.

    #[test]
    fn dcl_gs_input_primitive() {
        // opcode 93, triangle = 3 in bits [16:11]
        let tokens = [instr_token(93, 1) | (3 << 11)];
        let data = build_shex(2, &[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(
            output.contains("dcl_inputPrimitive triangle"),
            "got: {output}"
        );
    }

    #[test]
    fn dcl_gs_output_topology_trianglestrip() {
        // opcode 92, trianglestrip = 5 per D3D10_SB_PRIMITIVE_TOPOLOGY enum
        let tokens = [instr_token(92, 1) | (5 << 11)];
        let data = build_shex(2, &[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(
            output.contains("dcl_outputTopology trianglestrip"),
            "got: {output}"
        );
    }

    // Tessellation declarations.

    #[test]
    fn dcl_tess_domain_tri() {
        let tokens = [instr_token(149, 1) | (2 << 11)]; // tri = 2
        let data = build_shex(3, &[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(output.contains("dcl_tessDomain tri"), "got: {output}");
    }

    #[test]
    fn dcl_tess_partitioning_fractional_odd() {
        let tokens = [instr_token(150, 1) | (3 << 11)]; // fractional_odd = 3
        let data = build_shex(3, &[&tokens, &[instr_token(62, 1)]]);
        let output = disassemble(&data);
        assert!(
            output.contains("dcl_tessPartitioning fractional_odd"),
            "got: {output}"
        );
    }

    // Empty / minimal input.

    #[test]
    fn empty_data() {
        let output = disassemble(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn too_short() {
        let output = disassemble(&[0, 0, 0, 0]);
        assert!(output.is_empty() || output.contains("unknown"));
    }

    // Round-trip tests: decode → encode should produce identical bytes.

    /// Helper: assert that decoding then re-encoding produces identical bytes.
    fn assert_roundtrip(raw: &[u8]) {
        let program = decode::decode(raw).expect("decode failed");
        let encoded = encode::encode(&program);
        assert_eq!(
            raw,
            &encoded[..],
            "round-trip mismatch:\n  original: {:?}\n  encoded:  {:?}",
            raw.iter().collect::<Vec<_>>(),
            encoded.iter().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn roundtrip_ret() {
        assert_roundtrip(&build_ps(&[&[instr_token(62, 1)]]));
    }

    #[test]
    fn roundtrip_nop_ret() {
        assert_roundtrip(&build_ps(&[&[instr_token(58, 1)], &[instr_token(62, 1)]]));
    }

    #[test]
    fn roundtrip_mov() {
        let d = temp_dest(0, 0xF);
        let s = temp_src(1, XYZW);
        assert_roundtrip(&build_ps(&[
            &[instr_token(54, 5), d[0], d[1], s[0], s[1]],
            &[instr_token(62, 1)],
        ]));
    }

    #[test]
    fn roundtrip_add() {
        let d = temp_dest(0, 0xF);
        let s0 = temp_src(1, XYZW);
        let s1 = temp_src(2, XYZW);
        assert_roundtrip(&build_ps(&[
            &[instr_token(0, 7), d[0], d[1], s0[0], s0[1], s1[0], s1[1]],
            &[instr_token(62, 1)],
        ]));
    }

    #[test]
    fn roundtrip_dcl_temps() {
        assert_roundtrip(&build_ps(&[
            &[instr_token(104, 2), 4],
            &[instr_token(62, 1)],
        ]));
    }

    #[test]
    fn roundtrip_dcl_global_flags() {
        assert_roundtrip(&build_ps(&[
            &[instr_token(106, 1) | (1 << 11)], // refactoringAllowed
            &[instr_token(62, 1)],
        ]));
    }

    #[test]
    fn roundtrip_dcl_gs_input_primitive() {
        assert_roundtrip(&build_shex(
            2,
            &[
                &[instr_token(93, 1) | (3 << 11)], // triangle
                &[instr_token(62, 1)],
            ],
        ));
    }

    #[test]
    fn roundtrip_dcl_thread_group() {
        assert_roundtrip(&build_shex(
            5,
            &[&[instr_token(155, 4), 8, 8, 1], &[instr_token(62, 1)]],
        ));
    }

    #[test]
    fn roundtrip_mov_sat() {
        let d = temp_dest(0, 0xF);
        let s = temp_src(1, XYZW);
        assert_roundtrip(&build_ps(&[
            &[instr_token(54, 5) | (1 << 13), d[0], d[1], s[0], s[1]],
            &[instr_token(62, 1)],
        ]));
    }

    #[test]
    fn roundtrip_hs_phases() {
        assert_roundtrip(&build_shex(
            3,
            &[
                &[instr_token(113, 1)], // hs_decls
                &[instr_token(114, 1)], // hs_control_point_phase
                &[instr_token(115, 1)], // hs_fork_phase
                &[instr_token(62, 1)],
            ],
        ));
    }
}
