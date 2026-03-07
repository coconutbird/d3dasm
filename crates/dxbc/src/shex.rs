mod opcodes;
mod operand;

use std::fmt::Write;

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap())
}

/// Disassemble a SHEX or SHDR chunk into human-readable text.
pub fn disassemble(data: &[u8]) -> String {
    let mut out = String::new();
    if data.len() < 8 {
        return out;
    }

    let version_token = read_u32(data, 0);
    let length_dwords = read_u32(data, 4) as usize;
    let shader_type = (version_token >> 16) & 0xFFFF;
    let major = (version_token >> 4) & 0xF;
    let minor = version_token & 0xF;

    let type_name = match shader_type {
        0 => "ps",
        1 => "vs",
        2 => "gs",
        3 => "hs",
        4 => "ds",
        5 => "cs",
        _ => "unknown",
    };

    let _ = writeln!(out, "{type_name}_{major}_{minor}");

    let mut offset = 8; // skip version + length tokens
    let end = (length_dwords * 4).min(data.len());
    let mut indent = 0u32;

    while offset + 4 <= end {
        let token = read_u32(data, offset);
        let opcode_val = token & 0x7FF;
        let is_extended = (token >> 31) & 1 != 0;
        let instr_len = instruction_length(token, opcode_val, data, offset, end);

        if instr_len == 0 || offset + instr_len * 4 > end {
            let _ = writeln!(out, "  // invalid instruction at offset 0x{offset:X}");
            break;
        }

        // Collect all dwords for this instruction
        let tokens: Vec<u32> = (0..instr_len)
            .map(|i| read_u32(data, offset + i * 4))
            .collect();

        let line = disassemble_instruction(&tokens, opcode_val, is_extended, &mut indent);
        let pad = "  ".repeat(indent as usize);
        let _ = writeln!(out, "{pad}{line}");

        offset += instr_len * 4;
    }

    out
}

/// Determine instruction length in dwords.
fn instruction_length(token: u32, opcode: u32, data: &[u8], offset: usize, end: usize) -> usize {
    use opcodes::Opcode;
    let op = Opcode::from_u32(opcode);

    // Customdata has its own length encoding
    if matches!(op, Opcode::CustomData) {
        if offset + 8 <= end {
            return read_u32(data, offset + 4) as usize;
        }
        return 0;
    }

    // For declarations and most instructions, length is in bits [30:24]
    let len = ((token >> 24) & 0x7F) as usize;
    if len == 0 {
        // Some declaration opcodes use the whole next section
        return 1;
    }
    len
}

fn disassemble_instruction(
    tokens: &[u32],
    opcode_val: u32,
    _is_extended: bool,
    indent: &mut u32,
) -> String {
    use opcodes::Opcode;
    let op = Opcode::from_u32(opcode_val);
    let token0 = tokens[0];

    // Handle indent changes for control flow
    match op {
        Opcode::Else | Opcode::EndIf | Opcode::EndLoop | Opcode::EndSwitch => {
            *indent = indent.saturating_sub(1);
        }
        _ => {}
    }

    let result = match op {
        Opcode::CustomData => format_custom_data(tokens),
        Opcode::DclGlobalFlags => format_dcl_global_flags(token0),
        Opcode::DclInput
        | Opcode::DclInputSgv
        | Opcode::DclInputSiv
        | Opcode::DclInputPs
        | Opcode::DclInputPsSgv
        | Opcode::DclInputPsSiv => format_dcl_input(tokens, &op),
        Opcode::DclOutput | Opcode::DclOutputSgv | Opcode::DclOutputSiv => {
            format_dcl_output(tokens, &op)
        }
        Opcode::DclResource => format_dcl_resource(tokens),
        Opcode::DclSampler => format_dcl_sampler(tokens),
        Opcode::DclConstantBuffer => format_dcl_cb(tokens),
        Opcode::DclTemps => format_dcl_temps(tokens),
        Opcode::DclIndexableTemp => format_dcl_indexable_temp(tokens),
        Opcode::DclGsInputPrimitive => format_dcl_gs_input(token0),
        Opcode::DclGsOutputPrimitiveTopology => format_dcl_gs_output_topo(token0),
        Opcode::DclMaxOutputVertexCount => format_dcl_max_output(tokens),
        Opcode::DclGsInstanceCount => format_dcl_gs_instance_count(tokens),
        Opcode::DclOutputControlPointCount => format_dcl_output_cp_count(token0),
        Opcode::DclInputControlPointCount => format_dcl_input_cp_count(token0),
        Opcode::DclTessDomain => format_dcl_tess_domain(token0),
        Opcode::DclTessPartitioning => format_dcl_tess_partitioning(token0),
        Opcode::DclTessOutputPrimitive => format_dcl_tess_output_prim(token0),
        Opcode::DclHsMaxTessFactor => format_dcl_hs_max_tess(tokens),
        Opcode::DclHsForkPhaseInstanceCount => format_dcl_hs_fork_count(tokens),
        Opcode::HsDecls
        | Opcode::HsControlPointPhase
        | Opcode::HsForkPhase
        | Opcode::HsJoinPhase => op.name().to_string(),
        _ => format_generic_instruction(tokens, &op),
    };

    // Handle indent increases for control flow
    match op {
        Opcode::If | Opcode::Else | Opcode::Loop | Opcode::Switch => {
            let r = result;
            *indent += 1;
            r
        }
        _ => result,
    }
}

fn format_custom_data(tokens: &[u32]) -> String {
    let subtype = (tokens[0] >> 11) & 0x1F;
    let ty = match subtype {
        0 => "comment",
        1 => "debuginfo",
        2 => "opaque",
        3 => "dcl_immediateConstantBuffer",
        _ => "customdata",
    };
    if subtype == 3 && tokens.len() > 2 {
        let count = (tokens.len() - 2) / 4;
        let mut s = format!("{ty} {{");
        for i in 0..count {
            let base = 2 + i * 4;
            if base + 4 > tokens.len() {
                break;
            }
            let x = f32::from_bits(tokens[base]);
            let y = f32::from_bits(tokens[base + 1]);
            let z = f32::from_bits(tokens[base + 2]);
            let w = f32::from_bits(tokens[base + 3]);
            s.push_str(&format!("\n    {{ {x:e}, {y:e}, {z:e}, {w:e} }}"));
        }
        s.push_str("\n}");
        s
    } else {
        format!("{ty} // {subtype}, {} dwords", tokens.len())
    }
}

fn format_dcl_global_flags(token: u32) -> String {
    let flags = (token >> 11) & 0x1FFF;
    let mut parts = Vec::new();
    if flags & 1 != 0 {
        parts.push("refactoringAllowed");
    }
    if flags & 2 != 0 {
        parts.push("enableDoublePrecisionFloatOps");
    }
    if flags & 4 != 0 {
        parts.push("forceEarlyDepthStencil");
    }
    if flags & 8 != 0 {
        parts.push("enableRawAndStructuredBuffers");
    }
    if flags & 16 != 0 {
        parts.push("skipOptimization");
    }
    if flags & 32 != 0 {
        parts.push("enableMinPrecision");
    }
    if flags & 64 != 0 {
        parts.push("enable11_1DoubleExtensions");
    }
    if flags & 128 != 0 {
        parts.push("enable11_1ShaderExtensions");
    }
    format!("dcl_globalFlags {}", parts.join("|"))
}

fn format_dcl_input(tokens: &[u32], op: &opcodes::Opcode) -> String {
    let operands = operand::decode_operands(tokens, 1);
    let interp = if matches!(
        op,
        opcodes::Opcode::DclInputPs
            | opcodes::Opcode::DclInputPsSiv
            | opcodes::Opcode::DclInputPsSgv
    ) {
        let interp_mode = (tokens[0] >> 11) & 0xF;
        match interp_mode {
            0 => "",
            1 => " constant",
            2 => " linear",
            3 => " linearCentroid",
            4 => " linearNoperspective",
            5 => " linearNoperspectiveCentroid",
            6 => " linearSample",
            7 => " linearNoperspectiveSample",
            _ => " ?interp",
        }
    } else {
        ""
    };
    let name = op.name();
    if operands.is_empty() {
        format!("{name}{interp}")
    } else {
        format!("{name}{interp}, {}", operands.join(", "))
    }
}

fn format_dcl_output(tokens: &[u32], op: &opcodes::Opcode) -> String {
    let operands = operand::decode_operands(tokens, 1);
    let name = op.name();
    if operands.is_empty() {
        name.to_string()
    } else {
        format!("{name} {}", operands.join(", "))
    }
}

fn format_dcl_resource(tokens: &[u32]) -> String {
    let dim = (tokens[0] >> 11) & 0x1F;
    let dim_str = match dim {
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
    let operands = operand::decode_operands(tokens, 1);
    let ret_type = if tokens.len() > 2 {
        let rt = tokens[tokens.len() - 1];
        format_return_type(rt)
    } else {
        String::new()
    };
    format!(
        "dcl_resource_{dim_str} ({ret_type}) {}",
        operands.join(", ")
    )
}

fn format_return_type(token: u32) -> String {
    let names = [
        "",
        "unorm",
        "snorm",
        "sint",
        "uint",
        "float",
        "mixed",
        "double",
        "continued",
        "unused",
    ];
    let mut parts = Vec::new();
    for i in 0..4 {
        let t = ((token >> (i * 4)) & 0xF) as usize;
        parts.push(if t < names.len() { names[t] } else { "?" });
    }
    parts.join(",")
}

fn format_dcl_sampler(tokens: &[u32]) -> String {
    let mode = (tokens[0] >> 11) & 0xF;
    let mode_str = match mode {
        0 => "default",
        1 => "comparison",
        2 => "mono",
        _ => "?",
    };
    let operands = operand::decode_operands(tokens, 1);
    format!("dcl_sampler {}, mode_{mode_str}", operands.join(", "))
}

fn format_dcl_cb(tokens: &[u32]) -> String {
    let access = (tokens[0] >> 11) & 1;
    let access_str = if access == 0 {
        "immediateIndexed"
    } else {
        "dynamicIndexed"
    };
    let operands = operand::decode_operands(tokens, 1);
    format!("dcl_constantbuffer {}, {access_str}", operands.join(", "))
}

fn format_dcl_temps(tokens: &[u32]) -> String {
    let count = if tokens.len() > 1 { tokens[1] } else { 0 };
    format!("dcl_temps {count}")
}

fn format_dcl_indexable_temp(tokens: &[u32]) -> String {
    if tokens.len() >= 4 {
        format!(
            "dcl_indexableTemp x{}[{}], {}",
            tokens[1], tokens[2], tokens[3]
        )
    } else {
        "dcl_indexableTemp".to_string()
    }
}

fn format_dcl_gs_input(token: u32) -> String {
    let prim = (token >> 11) & 0x3F;
    let prim_str = match prim {
        1 => "point",
        2 => "line",
        3 => "triangle",
        4 => "lineAdj",
        5 => "triangleAdj",
        _ => "?",
    };
    if (6..=37).contains(&prim) {
        format!("dcl_inputPrimitive patchlist_{}", prim - 5)
    } else {
        format!("dcl_inputPrimitive {prim_str}")
    }
}

fn format_dcl_gs_output_topo(token: u32) -> String {
    let topo = (token >> 11) & 0x7;
    let topo_str = match topo {
        1 => "pointlist",
        2 => "linestrip",
        3 => "trianglestrip",
        _ => "?",
    };
    format!("dcl_outputTopology {topo_str}")
}

fn format_dcl_max_output(tokens: &[u32]) -> String {
    let count = if tokens.len() > 1 { tokens[1] } else { 0 };
    format!("dcl_maxOutputVertexCount {count}")
}

fn format_dcl_gs_instance_count(tokens: &[u32]) -> String {
    let count = if tokens.len() > 1 { tokens[1] } else { 0 };
    format!("dcl_gsInstanceCount {count}")
}

fn format_dcl_output_cp_count(token: u32) -> String {
    let count = (token >> 11) & 0x3F;
    format!("dcl_outputControlPointCount {count}")
}

fn format_dcl_input_cp_count(token: u32) -> String {
    let count = (token >> 11) & 0x3F;
    format!("dcl_inputControlPointCount {count}")
}

fn format_dcl_tess_domain(token: u32) -> String {
    let domain = (token >> 11) & 0x3;
    let s = match domain {
        0 => "undefined",
        1 => "isoline",
        2 => "tri",
        3 => "quad",
        _ => "?",
    };
    format!("dcl_tessDomain {s}")
}

fn format_dcl_tess_partitioning(token: u32) -> String {
    let p = (token >> 11) & 0x7;
    let s = match p {
        0 => "undefined",
        1 => "integer",
        2 => "pow2",
        3 => "fractional_odd",
        4 => "fractional_even",
        _ => "?",
    };
    format!("dcl_tessPartitioning {s}")
}

fn format_dcl_tess_output_prim(token: u32) -> String {
    let p = (token >> 11) & 0x7;
    let s = match p {
        0 => "undefined",
        1 => "point",
        2 => "line",
        3 => "triangle_cw",
        4 => "triangle_ccw",
        _ => "?",
    };
    format!("dcl_tessOutputPrimitive {s}")
}

fn format_dcl_hs_max_tess(tokens: &[u32]) -> String {
    let val = if tokens.len() > 1 {
        f32::from_bits(tokens[1])
    } else {
        0.0
    };
    format!("dcl_hsMaxTessFactor {val}")
}

fn format_dcl_hs_fork_count(tokens: &[u32]) -> String {
    let count = if tokens.len() > 1 { tokens[1] } else { 0 };
    format!("dcl_hsForkPhaseInstanceCount {count}")
}

fn format_generic_instruction(tokens: &[u32], op: &opcodes::Opcode) -> String {
    let name = op.name();
    let sat = if (tokens[0] >> 13) & 1 != 0 {
        "_sat"
    } else {
        ""
    };

    // Decode operands starting after the opcode token (and any extended tokens)
    let mut start = 1;
    // Skip extended opcode tokens
    if (tokens[0] >> 31) & 1 != 0 {
        while start < tokens.len() && (tokens[start] >> 31) & 1 != 0 {
            start += 1;
        }
        start += 1; // skip the last extended token
    }

    let operands = operand::decode_operands(tokens, start);
    if operands.is_empty() {
        format!("{name}{sat}")
    } else {
        format!("{name}{sat} {}", operands.join(", "))
    }
}
