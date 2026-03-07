//! Decoder: raw SHEX/SHDR bytes → structured IR.

use super::ir::*;
use super::opcodes::Opcode;

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

/// Decode a SHEX/SHDR chunk into a structured [`Program`].
pub fn decode(data: &[u8]) -> Option<Program> {
    if data.len() < 8 {
        return None;
    }

    let version_token = read_u32(data, 0);
    let length_dwords = read_u32(data, 4) as usize;
    let shader_type_val = (version_token >> 16) & 0xFFFF;
    let major = (version_token >> 4) & 0xF;
    let minor = version_token & 0xF;

    let shader_type = match shader_type_val {
        0 => "ps",
        1 => "vs",
        2 => "gs",
        3 => "hs",
        4 => "ds",
        5 => "cs",
        _ => "unknown",
    };

    let mut instructions = Vec::new();
    let mut offset = 8;
    let end = (length_dwords * 4).min(data.len());

    while offset + 4 <= end {
        let token = read_u32(data, offset);
        let opcode_val = token & 0x7FF;
        let instr_len = instruction_length(token, opcode_val, data, offset, end);

        if instr_len == 0 || offset + instr_len * 4 > end {
            break;
        }

        let tokens: Vec<u32> = (0..instr_len)
            .map(|i| read_u32(data, offset + i * 4))
            .collect();

        instructions.push(decode_instruction(&tokens, opcode_val));
        offset += instr_len * 4;
    }

    Some(Program {
        shader_type,
        major_version: major,
        minor_version: minor,
        instructions,
    })
}

fn instruction_length(token: u32, opcode: u32, data: &[u8], offset: usize, end: usize) -> usize {
    let op = Opcode::from_u32(opcode);
    if matches!(op, Opcode::CustomData) {
        if offset + 8 <= end {
            return read_u32(data, offset + 4) as usize;
        }
        return 0;
    }
    let len = ((token >> 24) & 0x7F) as usize;
    if len == 0 { 1 } else { len }
}

fn decode_instruction(tokens: &[u32], opcode_val: u32) -> Instruction {
    let op = Opcode::from_u32(opcode_val);
    let token0 = tokens[0];
    let saturate = (token0 >> 13) & 1 != 0;

    let kind = match op {
        Opcode::CustomData => decode_custom_data(tokens),
        Opcode::DclGlobalFlags => decode_dcl_global_flags(token0),
        Opcode::DclInput
        | Opcode::DclInputSgv
        | Opcode::DclInputSiv
        | Opcode::DclInputPs
        | Opcode::DclInputPsSgv
        | Opcode::DclInputPsSiv => decode_dcl_input(tokens, &op),
        Opcode::DclOutput | Opcode::DclOutputSgv | Opcode::DclOutputSiv => {
            decode_dcl_output(tokens, &op)
        }
        Opcode::DclResource => decode_dcl_resource(tokens),
        Opcode::DclUnorderedAccessViewTyped => decode_dcl_uav_typed(tokens),
        Opcode::DclSampler => decode_dcl_sampler(tokens),
        Opcode::DclConstantBuffer => decode_dcl_cb(tokens),
        Opcode::DclTemps => InstructionKind::DclTemps {
            count: if tokens.len() > 1 { tokens[1] } else { 0 },
        },
        Opcode::DclIndexableTemp => decode_dcl_indexable_temp(tokens),
        Opcode::DclGsInputPrimitive => decode_dcl_gs_input(token0),
        Opcode::DclGsOutputPrimitiveTopology => decode_dcl_gs_output_topo(token0),
        Opcode::DclMaxOutputVertexCount => InstructionKind::DclMaxOutputVertexCount {
            count: if tokens.len() > 1 { tokens[1] } else { 0 },
        },
        Opcode::DclGsInstanceCount => InstructionKind::DclGsInstanceCount {
            count: if tokens.len() > 1 { tokens[1] } else { 0 },
        },
        Opcode::DclOutputControlPointCount => InstructionKind::DclOutputControlPointCount {
            count: (token0 >> 11) & 0x3F,
        },
        Opcode::DclInputControlPointCount => InstructionKind::DclInputControlPointCount {
            count: (token0 >> 11) & 0x3F,
        },
        Opcode::DclTessDomain => decode_dcl_tess_domain(token0),
        Opcode::DclTessPartitioning => decode_dcl_tess_partitioning(token0),
        Opcode::DclTessOutputPrimitive => decode_dcl_tess_output_prim(token0),
        Opcode::DclHsMaxTessFactor => InstructionKind::DclHsMaxTessFactor {
            value: if tokens.len() > 1 {
                f32::from_bits(tokens[1])
            } else {
                0.0
            },
        },
        Opcode::DclHsForkPhaseInstanceCount => InstructionKind::DclHsForkPhaseInstanceCount {
            count: if tokens.len() > 1 { tokens[1] } else { 0 },
        },
        Opcode::DclThreadGroup => InstructionKind::DclThreadGroup {
            x: if tokens.len() > 1 { tokens[1] } else { 0 },
            y: if tokens.len() > 2 { tokens[2] } else { 0 },
            z: if tokens.len() > 3 { tokens[3] } else { 0 },
        },
        Opcode::HsDecls
        | Opcode::HsControlPointPhase
        | Opcode::HsForkPhase
        | Opcode::HsJoinPhase => InstructionKind::HsPhase,
        _ => decode_generic(tokens),
    };

    Instruction {
        opcode: op,
        saturate,
        kind,
    }
}

// ── Declaration decoders ──────────────────────────────────────────────

fn decode_custom_data(tokens: &[u32]) -> InstructionKind {
    let subtype_val = (tokens[0] >> 11) & 0x1F;
    let subtype = CustomDataType::from_u32(subtype_val);
    let mut values = Vec::new();
    if subtype == CustomDataType::ImmediateConstantBuffer && tokens.len() > 2 {
        let count = (tokens.len() - 2) / 4;
        for i in 0..count {
            let base = 2 + i * 4;
            if base + 4 > tokens.len() {
                break;
            }
            values.push([
                f32::from_bits(tokens[base]),
                f32::from_bits(tokens[base + 1]),
                f32::from_bits(tokens[base + 2]),
                f32::from_bits(tokens[base + 3]),
            ]);
        }
    }
    InstructionKind::CustomData {
        subtype,
        values,
        raw_dword_count: tokens.len(),
    }
}

fn decode_dcl_global_flags(token: u32) -> InstructionKind {
    let flags_bits = (token >> 11) & 0x1FFF;
    let mut flags = Vec::new();
    let names: &[&str] = &[
        "refactoringAllowed",
        "enableDoublePrecisionFloatOps",
        "forceEarlyDepthStencil",
        "enableRawAndStructuredBuffers",
        "skipOptimization",
        "enableMinPrecision",
        "enable11_1DoubleExtensions",
        "enable11_1ShaderExtensions",
    ];
    for (i, name) in names.iter().enumerate() {
        if flags_bits & (1 << i) != 0 {
            flags.push(*name);
        }
    }
    InstructionKind::DclGlobalFlags { flags }
}

fn decode_dcl_input(tokens: &[u32], op: &Opcode) -> InstructionKind {
    let interpolation = if matches!(
        op,
        Opcode::DclInputPs | Opcode::DclInputPsSiv | Opcode::DclInputPsSgv
    ) {
        let mode = (tokens[0] >> 11) & 0xF;
        Some(match mode {
            0 => "undefined",
            1 => "constant",
            2 => "linear",
            3 => "linearCentroid",
            4 => "linearNoperspective",
            5 => "linearNoperspectiveCentroid",
            6 => "linearSample",
            7 => "linearNoperspectiveSample",
            _ => "?interp",
        })
    } else {
        None
    };
    // System value is in the last token for sgv/siv variants
    let has_sv = matches!(
        op,
        Opcode::DclInputSgv | Opcode::DclInputSiv | Opcode::DclInputPsSgv | Opcode::DclInputPsSiv
    );
    let (system_value, operand_end) = if has_sv && tokens.len() >= 3 {
        let sv = tokens[tokens.len() - 1];
        (Some(system_value_name(sv)), tokens.len() - 1)
    } else {
        (None, tokens.len())
    };
    InstructionKind::DclInput {
        interpolation,
        system_value,
        operands: decode_operands(&tokens[..operand_end], 1),
    }
}

fn decode_dcl_output(tokens: &[u32], op: &Opcode) -> InstructionKind {
    let has_sv = matches!(op, Opcode::DclOutputSgv | Opcode::DclOutputSiv);
    let (system_value, operand_end) = if has_sv && tokens.len() >= 3 {
        let sv = tokens[tokens.len() - 1];
        (Some(system_value_name(sv)), tokens.len() - 1)
    } else {
        (None, tokens.len())
    };
    InstructionKind::DclOutput {
        system_value,
        operands: decode_operands(&tokens[..operand_end], 1),
    }
}

fn decode_dcl_resource(tokens: &[u32]) -> InstructionKind {
    let dim = (tokens[0] >> 11) & 0x1F;
    let dimension = match dim {
        1 => "buffer",
        2 => "texture1d",
        3 => "texture2d",
        4 => "texture2dms",
        5 => "texture3d",
        6 => "texturecube",
        7 => "texture1darray",
        8 => "texture2darray",
        9 => "texture2dmsarray",
        10 => "texturecubearray",
        _ => "unknown",
    };
    let return_type = if tokens.len() > 2 {
        let rt = tokens[tokens.len() - 1];
        [
            ReturnType::from_u32(rt & 0xF),
            ReturnType::from_u32((rt >> 4) & 0xF),
            ReturnType::from_u32((rt >> 8) & 0xF),
            ReturnType::from_u32((rt >> 12) & 0xF),
        ]
    } else {
        [ReturnType::Unknown(0); 4]
    };
    // Operands are between token 1 and the return type token (last)
    let operand_end = if tokens.len() > 2 {
        tokens.len() - 1
    } else {
        tokens.len()
    };
    InstructionKind::DclResource {
        dimension,
        return_type,
        operands: decode_operands(&tokens[..operand_end], 1),
    }
}

fn decode_dcl_uav_typed(tokens: &[u32]) -> InstructionKind {
    let dim = (tokens[0] >> 11) & 0x1F;
    let dimension = match dim {
        1 => "buffer",
        2 => "texture1d",
        3 => "texture2d",
        4 => "texture2dms",
        5 => "texture3d",
        6 => "texturecube",
        7 => "texture1darray",
        8 => "texture2darray",
        9 => "texture2dmsarray",
        10 => "texturecubearray",
        _ => "unknown",
    };
    let return_type = if tokens.len() > 2 {
        let rt = tokens[tokens.len() - 1];
        [
            ReturnType::from_u32(rt & 0xF),
            ReturnType::from_u32((rt >> 4) & 0xF),
            ReturnType::from_u32((rt >> 8) & 0xF),
            ReturnType::from_u32((rt >> 12) & 0xF),
        ]
    } else {
        [ReturnType::Unknown(0); 4]
    };
    let operand_end = if tokens.len() > 2 {
        tokens.len() - 1
    } else {
        tokens.len()
    };
    InstructionKind::DclUavTyped {
        dimension,
        return_type,
        operands: decode_operands(&tokens[..operand_end], 1),
    }
}

fn decode_dcl_sampler(tokens: &[u32]) -> InstructionKind {
    let mode = match (tokens[0] >> 11) & 0xF {
        0 => "default",
        1 => "comparison",
        2 => "mono",
        _ => "?",
    };
    InstructionKind::DclSampler {
        mode,
        operands: decode_operands(tokens, 1),
    }
}

fn decode_dcl_cb(tokens: &[u32]) -> InstructionKind {
    let access = if (tokens[0] >> 11) & 1 == 0 {
        "immediateIndexed"
    } else {
        "dynamicIndexed"
    };
    InstructionKind::DclConstantBuffer {
        access,
        operands: decode_operands(tokens, 1),
    }
}

fn decode_dcl_indexable_temp(tokens: &[u32]) -> InstructionKind {
    if tokens.len() >= 4 {
        InstructionKind::DclIndexableTemp {
            reg: tokens[1],
            size: tokens[2],
            components: tokens[3],
        }
    } else {
        InstructionKind::DclIndexableTemp {
            reg: 0,
            size: 0,
            components: 0,
        }
    }
}

fn decode_dcl_gs_input(token: u32) -> InstructionKind {
    let prim = (token >> 11) & 0x3F;
    let primitive = if (6..=37).contains(&prim) {
        // patchlist — we can't return a dynamic string from a &'static str,
        // so we handle this in the formatter
        "patchlist"
    } else {
        match prim {
            1 => "point",
            2 => "line",
            3 => "triangle",
            4 => "lineAdj",
            5 => "triangleAdj",
            _ => "?",
        }
    };
    InstructionKind::DclGsInputPrimitive { primitive }
}

fn decode_dcl_gs_output_topo(token: u32) -> InstructionKind {
    let topo = (token >> 11) & 0x7;
    let topology = match topo {
        1 => "pointlist",
        2 => "linestrip",
        3 => "trianglestrip",
        _ => "?",
    };
    InstructionKind::DclGsOutputTopology { topology }
}

fn decode_dcl_tess_domain(token: u32) -> InstructionKind {
    let domain = match (token >> 11) & 0x3 {
        0 => "undefined",
        1 => "isoline",
        2 => "tri",
        3 => "quad",
        _ => "?",
    };
    InstructionKind::DclTessDomain { domain }
}

fn decode_dcl_tess_partitioning(token: u32) -> InstructionKind {
    let partitioning = match (token >> 11) & 0x7 {
        0 => "undefined",
        1 => "integer",
        2 => "pow2",
        3 => "fractional_odd",
        4 => "fractional_even",
        _ => "?",
    };
    InstructionKind::DclTessPartitioning { partitioning }
}

fn decode_dcl_tess_output_prim(token: u32) -> InstructionKind {
    let primitive = match (token >> 11) & 0x7 {
        0 => "undefined",
        1 => "point",
        2 => "line",
        3 => "triangle_cw",
        4 => "triangle_ccw",
        _ => "?",
    };
    InstructionKind::DclTessOutputPrimitive { primitive }
}

// ── Generic instruction decoder ───────────────────────────────────────

fn decode_generic(tokens: &[u32]) -> InstructionKind {
    let mut start = 1;
    // Skip extended opcode tokens
    if (tokens[0] >> 31) & 1 != 0 {
        while start < tokens.len() && (tokens[start] >> 31) & 1 != 0 {
            start += 1;
        }
        start += 1;
    }
    InstructionKind::Generic {
        operands: decode_operands(tokens, start),
    }
}

// ── Operand decoder ───────────────────────────────────────────────────

fn decode_operands(tokens: &[u32], start: usize) -> Vec<Operand> {
    let mut result = Vec::new();
    let mut pos = start;
    while pos < tokens.len() {
        let (op, consumed) = decode_one_operand(tokens, pos);
        if consumed == 0 {
            break;
        }
        result.push(op);
        pos += consumed;
    }
    result
}

fn decode_one_operand(tokens: &[u32], pos: usize) -> (Operand, usize) {
    if pos >= tokens.len() {
        return (empty_operand(), 0);
    }
    let token = tokens[pos];
    let num_components = token & 0x3;
    let op_type_val = (token >> 12) & 0xFF;
    let index_dim = (token >> 20) & 0x3;
    let is_extended = (token >> 31) & 1 != 0;

    let reg_type = RegisterType::from_u32(op_type_val);
    let mut consumed = 1usize;

    // Extended modifier
    let (negate, abs) = if is_extended && pos + consumed < tokens.len() {
        let ext = tokens[pos + consumed];
        consumed += 1;
        ((ext >> 6) & 1 != 0, (ext >> 7) & 1 != 0)
    } else {
        (false, false)
    };

    let components = decode_components(token, num_components);

    // Decode indices and immediates
    let mut indices = Vec::new();
    let mut immediate_values = Vec::new();

    match index_dim {
        0 => {
            if op_type_val == 4 {
                // Immediate32: consume 4 dwords
                let count = if pos + consumed + 4 <= tokens.len() {
                    4
                } else if pos + consumed < tokens.len() {
                    1
                } else {
                    0
                };
                for i in 0..count {
                    immediate_values.push(tokens[pos + consumed + i]);
                }
                consumed += count;
            }
        }
        1 => {
            let parent_token = tokens[pos];
            let idx0_repr = (parent_token >> 22) & 0x7;
            let (idx, ic) = decode_index(tokens, pos + consumed, idx0_repr);
            indices.push(idx);
            consumed += ic;
        }
        2 => {
            let parent_token = tokens[pos];
            let idx0_repr = (parent_token >> 22) & 0x7;
            let idx1_repr = (parent_token >> 25) & 0x7;
            let (idx0, ic0) = decode_index(tokens, pos + consumed, idx0_repr);
            indices.push(idx0);
            consumed += ic0;
            let (idx1, ic1) = decode_index(tokens, pos + consumed, idx1_repr);
            indices.push(idx1);
            consumed += ic1;
        }
        3 => {
            let parent_token = tokens[pos];
            let idx0_repr = (parent_token >> 22) & 0x7;
            let idx1_repr = (parent_token >> 25) & 0x7;
            let idx2_repr = (parent_token >> 28) & 0x7;
            let reprs = [idx0_repr, idx1_repr, idx2_repr];
            for repr in reprs {
                let (idx, ic) = decode_index(tokens, pos + consumed, repr);
                indices.push(idx);
                consumed += ic;
            }
        }
        _ => {}
    }

    (
        Operand {
            reg_type,
            components,
            negate,
            abs,
            indices,
            immediate_values,
        },
        consumed,
    )
}

fn decode_components(token: u32, num_components: u32) -> ComponentSelect {
    let sel_mode = (token >> 2) & 0x3;
    match num_components {
        0 => ComponentSelect::None,
        1 => {
            let mask = ((token >> 4) & 0xF) as u8;
            ComponentSelect::Mask(mask)
        }
        2 => match sel_mode {
            0 => ComponentSelect::Mask(((token >> 4) & 0xF) as u8),
            1 => {
                let mut s = [0u8; 4];
                for (i, val) in s.iter_mut().enumerate() {
                    *val = ((token >> (4 + i * 2)) & 0x3) as u8;
                }
                ComponentSelect::Swizzle(s)
            }
            2 => ComponentSelect::Scalar(((token >> 4) & 0x3) as u8),
            _ => ComponentSelect::None,
        },
        _ => ComponentSelect::None,
    }
}

fn decode_index(tokens: &[u32], pos: usize, repr: u32) -> (OperandIndex, usize) {
    match repr {
        0 => {
            if pos < tokens.len() {
                (OperandIndex::Imm32(tokens[pos]), 1)
            } else {
                (OperandIndex::Imm32(0), 0)
            }
        }
        1 => {
            if pos + 1 < tokens.len() {
                let val = tokens[pos] as u64 | ((tokens[pos + 1] as u64) << 32);
                (OperandIndex::Imm64(val), 2)
            } else {
                (OperandIndex::Imm32(0), 0)
            }
        }
        2 => {
            let (sub, consumed) = decode_one_operand(tokens, pos);
            (OperandIndex::Relative(Box::new(sub)), consumed)
        }
        3 => {
            if pos < tokens.len() {
                let imm = tokens[pos];
                let (sub, consumed) = decode_one_operand(tokens, pos + 1);
                (
                    OperandIndex::RelativePlusImm(imm, Box::new(sub)),
                    1 + consumed,
                )
            } else {
                (OperandIndex::Imm32(0), 0)
            }
        }
        _ => (OperandIndex::Imm32(0), 0),
    }
}

fn empty_operand() -> Operand {
    Operand {
        reg_type: RegisterType::Unknown(0),
        components: ComponentSelect::None,
        negate: false,
        abs: false,
        indices: Vec::new(),
        immediate_values: Vec::new(),
    }
}
