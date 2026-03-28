//! Decoder: raw SHEX/SHDR bytes → structured IR.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use nostdio::{ReadLe, Seek, SeekFrom, SliceCursor};

use super::ir::*;
use super::opcodes::Opcode;

/// Error returned when decoding a SHEX/SHDR chunk fails.
#[derive(Debug, Clone)]
pub struct DecodeError {
    /// Human-readable description of the decode failure.
    pub message: String,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SHEX decode error: {}", self.message)
    }
}

/// Decode a SHEX/SHDR chunk into a structured [`Program`].
pub fn decode(data: &[u8]) -> Result<Program, DecodeError> {
    decode_with_fourcc(data, *b"SHEX")
}

/// Decode a SHEX/SHDR chunk into a structured [`Program`], preserving the original FourCC.
pub fn decode_with_fourcc(data: &[u8], fourcc: [u8; 4]) -> Result<Program, DecodeError> {
    if data.len() < 8 {
        return Err(DecodeError {
            message: format!(
                "SHEX/SHDR chunk too short: {} bytes (minimum 8)",
                data.len()
            ),
        });
    }

    let mut c = SliceCursor::new(data);
    let version_token = c.read_u32_le().map_err(|_| DecodeError {
        message: String::from("failed to read version token"),
    })?;
    let length_dwords = c.read_u32_le().map_err(|_| DecodeError {
        message: String::from("failed to read length"),
    })? as usize;
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
    let mut warnings = Vec::new();
    let mut offset = 8;
    let end = (length_dwords * 4).min(data.len());
    // Reuse a single token buffer across the decode loop to avoid
    // allocating a fresh Vec<u32> for every instruction.
    let mut tokens: Vec<u32> = Vec::new();

    if length_dwords * 4 > data.len() {
        warnings.push(format!(
            "SHEX header claims {} dwords ({} bytes) but chunk is only {} bytes",
            length_dwords,
            length_dwords * 4,
            data.len()
        ));
    }

    while offset + 4 <= end {
        c.seek(SeekFrom::Start(offset as u64)).unwrap();
        let token = c.read_u32_le().unwrap();
        let opcode_val = token & 0x7FF;
        let instr_len = instruction_length(token, opcode_val, &mut c, offset, end);

        if instr_len == 0 {
            warnings.push(format!(
                "instruction at byte offset {offset} has zero length, stopping"
            ));
            break;
        }
        if offset + instr_len * 4 > end {
            warnings.push(format!(
                "instruction at byte offset {offset} extends past end of chunk \
                 (needs {} bytes, only {} available)",
                instr_len * 4,
                end - offset
            ));
            break;
        }

        c.seek(SeekFrom::Start(offset as u64)).unwrap();
        tokens.clear();
        tokens.reserve(instr_len);
        for _ in 0..instr_len {
            tokens.push(c.read_u32_le().unwrap());
        }

        instructions.push(decode_instruction(&tokens, opcode_val));
        offset += instr_len * 4;
    }

    Ok(Program {
        shader_type,
        major_version: major,
        minor_version: minor,
        instructions,
        warnings,
        fourcc,
    })
}

fn instruction_length(
    token: u32,
    opcode: u32,
    c: &mut SliceCursor<'_>,
    offset: usize,
    end: usize,
) -> usize {
    let op = Opcode::from_u32(opcode);
    if matches!(op, Opcode::CustomData) {
        if offset + 8 <= end {
            c.seek(SeekFrom::Start((offset + 4) as u64)).unwrap();
            return c.read_u32_le().unwrap() as usize;
        }
        return 0;
    }
    let len = ((token >> 24) & 0x7F) as usize;
    if len == 0 { 1 } else { len }
}

fn decode_instruction(tokens: &[u32], opcode_val: u32) -> Instruction {
    let op = Opcode::from_u32(opcode_val);
    let token0 = tokens[0];

    // Saturate (bit 13) — not applicable to Sync (which reuses bits 11-14).
    let saturate = op != Opcode::Sync && (token0 >> 13) & 1 != 0;

    // Conditional test condition: bit 18 (0 = zero, 1 = non-zero).
    let test_nonzero = matches!(
        op,
        Opcode::If
            | Opcode::Breakc
            | Opcode::Callc
            | Opcode::Continuec
            | Opcode::Discard
            | Opcode::Retc
    ) && ((token0 >> 18) & 1 != 0);

    // SM5 precise modifier: bits 19-22.
    let precise_mask = ((token0 >> 19) & 0xF) as u8;

    // Extract resinfo return type (bits [11:12]) for resinfo opcode.
    let resinfo_return_type = if op == Opcode::Resinfo {
        Some((token0 >> 11) & 0x3)
    } else {
        None
    };

    // Sync flags: bits 11-14 (only meaningful for Sync opcode).
    let sync_flags = if op == Opcode::Sync {
        ((token0 >> 11) & 0xF) as u8
    } else {
        0
    };

    // Parse all extended opcode tokens from the instruction header chain.
    let (tex_offsets, resource_dim, resource_return_type) = decode_extended_tokens(tokens);

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
        Opcode::DclUnorderedAccessViewRaw => decode_dcl_uav_raw(tokens),
        Opcode::DclUnorderedAccessViewStructured => decode_dcl_uav_structured(tokens),
        Opcode::DclResourceRaw => decode_dcl_resource_raw(tokens),
        Opcode::DclResourceStructured => decode_dcl_resource_structured(tokens),
        Opcode::DclStream => {
            // dcl_stream has a single stream operand — decode it generically
            decode_generic(tokens)
        }
        Opcode::DclIndexRange => decode_dcl_index_range(tokens),
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
        Opcode::DclHsForkPhaseInstanceCount | Opcode::DclHsJoinPhaseInstanceCount => {
            InstructionKind::DclHsForkPhaseInstanceCount {
                count: if tokens.len() > 1 { tokens[1] } else { 0 },
            }
        }
        Opcode::DclThreadGroup => InstructionKind::DclThreadGroup {
            x: if tokens.len() > 1 { tokens[1] } else { 0 },
            y: if tokens.len() > 2 { tokens[2] } else { 0 },
            z: if tokens.len() > 3 { tokens[3] } else { 0 },
        },
        Opcode::DclFunctionBody => decode_dcl_function_body(tokens),
        Opcode::DclFunctionTable => decode_dcl_function_table(tokens),
        Opcode::DclInterface => decode_dcl_interface(tokens),
        Opcode::HsDecls
        | Opcode::HsControlPointPhase
        | Opcode::HsForkPhase
        | Opcode::HsJoinPhase => InstructionKind::HsPhase,
        _ => decode_generic(tokens),
    };

    Instruction {
        opcode: op,
        saturate,
        test_nonzero,
        precise_mask,
        resinfo_return_type,
        sync_flags,
        tex_offsets,
        resource_dim,
        resource_return_type,
        kind,
    }
}

// Declaration decoders.

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
    // When the instruction carries a system value, the last token is the
    // SV enum — decode operands only from the tokens *before* it so the
    // greedy operand decoder doesn't consume the SV token as a degenerate
    // operand.
    let has_sv = matches!(
        op,
        Opcode::DclInputSgv | Opcode::DclInputSiv | Opcode::DclInputPsSgv | Opcode::DclInputPsSiv
    );
    let system_value = if has_sv && tokens.len() >= 3 {
        Some(system_value_name(tokens[tokens.len() - 1]))
    } else {
        None
    };
    let operand_end = if has_sv && tokens.len() >= 3 {
        tokens.len() - 1
    } else {
        tokens.len()
    };
    let operands = decode_operands(&tokens[..operand_end], 1);
    InstructionKind::DclInput {
        interpolation,
        system_value,
        operands,
    }
}

fn decode_dcl_output(tokens: &[u32], op: &Opcode) -> InstructionKind {
    let has_sv = matches!(op, Opcode::DclOutputSgv | Opcode::DclOutputSiv);
    let system_value = if has_sv && tokens.len() >= 3 {
        Some(system_value_name(tokens[tokens.len() - 1]))
    } else {
        None
    };
    let operand_end = if has_sv && tokens.len() >= 3 {
        tokens.len() - 1
    } else {
        tokens.len()
    };
    let operands = decode_operands(&tokens[..operand_end], 1);
    InstructionKind::DclOutput {
        system_value,
        operands,
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
    // MSAA sample count in bits 16-22 (relevant for texture2dms/texture2dmsarray).
    let sample_count = (tokens[0] >> 16) & 0x7F;
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
        sample_count,
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
    // UAV flags: bit 16 = globallyCoherent, bit 17 = rasterizer ordered (ROV).
    let flags = (tokens[0] >> 16) & 0xFF;
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
        flags,
        return_type,
        operands: decode_operands(&tokens[..operand_end], 1),
    }
}

fn decode_dcl_uav_raw(tokens: &[u32]) -> InstructionKind {
    let flags = (tokens[0] >> 16) & 0xFF;
    InstructionKind::DclUavRaw {
        flags,
        operands: decode_operands(tokens, 1),
    }
}

fn decode_dcl_uav_structured(tokens: &[u32]) -> InstructionKind {
    let flags = (tokens[0] >> 16) & 0xFF;
    // Last token is the byte stride
    let stride = if tokens.len() >= 3 {
        tokens[tokens.len() - 1]
    } else {
        0
    };
    let operand_end = if tokens.len() >= 3 {
        tokens.len() - 1
    } else {
        tokens.len()
    };
    InstructionKind::DclUavStructured {
        flags,
        stride,
        operands: decode_operands(&tokens[..operand_end], 1),
    }
}

fn decode_dcl_resource_raw(tokens: &[u32]) -> InstructionKind {
    InstructionKind::DclResourceRaw {
        operands: decode_operands(tokens, 1),
    }
}

fn decode_dcl_resource_structured(tokens: &[u32]) -> InstructionKind {
    // Last token is the byte stride
    let stride = if tokens.len() >= 3 {
        tokens[tokens.len() - 1]
    } else {
        0
    };
    let operand_end = if tokens.len() >= 3 {
        tokens.len() - 1
    } else {
        tokens.len()
    };
    InstructionKind::DclResourceStructured {
        stride,
        operands: decode_operands(&tokens[..operand_end], 1),
    }
}

fn decode_dcl_index_range(tokens: &[u32]) -> InstructionKind {
    // Last token is the register count
    let count = if tokens.len() >= 3 {
        tokens[tokens.len() - 1]
    } else {
        0
    };
    let operand_end = if tokens.len() >= 3 {
        tokens.len() - 1
    } else {
        tokens.len()
    };
    InstructionKind::DclIndexRange {
        operands: decode_operands(&tokens[..operand_end], 1),
        count,
    }
}

/// Walk the extended opcode token chain and extract all extended data.
///
/// Returns `(tex_offsets, resource_dim, resource_return_type)`.
fn decode_extended_tokens(tokens: &[u32]) -> (Option<[i8; 3]>, Option<u32>, Option<u32>) {
    let token0 = tokens[0];
    if (token0 >> 31) & 1 == 0 || tokens.len() < 2 {
        return (None, None, None);
    }

    let mut tex_offsets = None;
    let mut resource_dim = None;
    let mut resource_return_type = None;
    let mut idx = 1;

    loop {
        if idx >= tokens.len() {
            break;
        }
        let ext = tokens[idx];
        let ext_type = ext & 0x3F;

        match ext_type {
            // Type 1 = D3D10_SB_EXTENDED_OPCODE_SAMPLE_CONTROLS
            1 => {
                let u = sign_extend_4bit((ext >> 9) & 0xF);
                let v = sign_extend_4bit((ext >> 13) & 0xF);
                let w = sign_extend_4bit((ext >> 17) & 0xF);
                tex_offsets = Some([u, v, w]);
            }
            // Type 2 = D3D10_SB_EXTENDED_OPCODE_RESOURCE_DIM
            2 => {
                resource_dim = Some(ext);
            }
            // Type 3 = D3D10_SB_EXTENDED_OPCODE_RESOURCE_RETURN_TYPE
            3 => {
                resource_return_type = Some(ext);
            }
            _ => {}
        }

        // Bit 31 of the extended token indicates whether another extended token follows.
        if (ext >> 31) & 1 == 0 {
            break;
        }
        idx += 1;
    }

    (tex_offsets, resource_dim, resource_return_type)
}

/// Sign-extend a 4-bit value to i8.
fn sign_extend_4bit(val: u32) -> i8 {
    if val & 0x8 != 0 {
        (val | 0xFFFFFFF0) as i32 as i8
    } else {
        val as i8
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
    InstructionKind::DclGsInputPrimitive {
        primitive: GsPrimitive::from_raw(prim),
    }
}

fn decode_dcl_gs_output_topo(token: u32) -> InstructionKind {
    let topo = (token >> 11) & 0xF;
    InstructionKind::DclGsOutputTopology {
        topology: GsOutputTopology::from_raw(topo),
    }
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

fn decode_dcl_function_body(tokens: &[u32]) -> InstructionKind {
    // dcl_function_body fb<N>
    // Token layout: [opcode_token, function_body_index]
    InstructionKind::DclFunctionBody {
        index: if tokens.len() > 1 { tokens[1] } else { 0 },
    }
}

fn decode_dcl_function_table(tokens: &[u32]) -> InstructionKind {
    // dcl_function_table ft<N> = { fb0, fb1, ... }
    // Token layout: [opcode_token, table_index, count, body0, body1, ...]
    let table_index = if tokens.len() > 1 { tokens[1] } else { 0 };
    let count = if tokens.len() > 2 {
        tokens[2] as usize
    } else {
        0
    };
    let body_indices: Vec<u32> = tokens.iter().skip(3).take(count).copied().collect();
    InstructionKind::DclFunctionTable {
        table_index,
        body_indices,
    }
}

fn decode_dcl_interface(tokens: &[u32]) -> InstructionKind {
    // dcl_interface fp<N>[<num_call_sites>][<num_types>] = { ft0, ft1, ... }
    // Token layout: [opcode_token, interface_index, num_call_sites, num_types, ft0, ft1, ...]
    let interface_index = if tokens.len() > 1 { tokens[1] } else { 0 };
    let num_call_sites = if tokens.len() > 2 { tokens[2] } else { 0 };
    let num_types = if tokens.len() > 3 {
        tokens[3] as usize
    } else {
        0
    };
    let table_indices: Vec<u32> = tokens.iter().skip(4).take(num_types).copied().collect();
    InstructionKind::DclInterface {
        interface_index,
        num_call_sites,
        table_indices,
    }
}

// Generic instruction decoder.

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

// Operand decoder.

/// Decode operands starting at `start`, returning the operands and the
/// token index just past the last consumed operand token.
fn decode_operands_with_pos(tokens: &[u32], start: usize) -> (Vec<Operand>, usize) {
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
    (result, pos)
}

fn decode_operands(tokens: &[u32], start: usize) -> Vec<Operand> {
    decode_operands_with_pos(tokens, start).0
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
    let num_components = components.num_components();

    // Decode indices and immediates
    let mut indices = Vec::new();
    let mut immediate_values = Vec::new();

    match index_dim {
        0 => {
            if op_type_val == 4 {
                // Immediate32: dword count depends on num_components
                //   0 → 0 components (no data)
                //   1 → 1 component  (1 dword)
                //   2 → N-component  (4 dwords for Immediate32)
                let count = match num_components {
                    0 => 0,
                    1 => 1,
                    _ => 4,
                };
                let available = (tokens.len() - (pos + consumed)).min(count);
                for i in 0..available {
                    immediate_values.push(tokens[pos + consumed + i]);
                }
                consumed += available;
            } else if op_type_val == 5 {
                // Immediate64: each component is 2 dwords (64-bit)
                let components = match num_components {
                    0 => 0,
                    1 => 1,
                    _ => 4,
                };
                let dword_count = components * 2;
                let available = (tokens.len() - (pos + consumed)).min(dword_count);
                for i in 0..available {
                    immediate_values.push(tokens[pos + consumed + i]);
                }
                consumed += available;
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
        0 => ComponentSelect::ZeroComponent,
        1 => ComponentSelect::OneComponent,
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
            _ => ComponentSelect::ZeroComponent,
        },
        _ => ComponentSelect::ZeroComponent,
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
        components: ComponentSelect::ZeroComponent,
        negate: false,
        abs: false,
        indices: Vec::new(),
        immediate_values: Vec::new(),
    }
}
