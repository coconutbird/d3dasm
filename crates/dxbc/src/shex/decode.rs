//! Decoder: raw SHEX/SHDR bytes → structured IR.

use alloc::boxed::Box;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use nostdio::{Cursor, ReadLe, Seek, SeekFrom};
use smallvec::{SmallVec, smallvec};

use super::ir::{
    ComponentSelect, CustomDataType, GsOutputTopology, GsPrimitive, Instruction, InstructionKind,
    MinPrecision, Operand, OperandIndex, Operands, Program, RegisterType, ReturnType, SmallU32Vec,
    SystemValue,
};
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
///
/// # Errors
///
/// Returns [`DecodeError`] when the chunk header or instruction stream cannot
/// be read.
pub fn decode(data: &[u8]) -> Result<Program, DecodeError> {
    decode_with_fourcc(data, *b"SHEX")
}

/// Decode a SHEX/SHDR chunk into a structured [`Program`], preserving the original `FourCC`.
///
/// # Errors
///
/// Returns [`DecodeError`] when the chunk header or instruction stream cannot
/// be read.
pub fn decode_with_fourcc(data: &[u8], fourcc: [u8; 4]) -> Result<Program, DecodeError> {
    if !matches!(&fourcc, b"SHEX" | b"SHDR") {
        return Err(DecodeError {
            message: format!("invalid shader-code FourCC: {fourcc:?}"),
        });
    }
    if data.len() < 8 {
        return Err(DecodeError {
            message: format!(
                "SHEX/SHDR chunk too short: {} bytes (minimum 8)",
                data.len()
            ),
        });
    }
    if data.len() % 4 != 0 {
        return Err(DecodeError {
            message: format!("SHEX/SHDR size {} is not DWORD-aligned", data.len()),
        });
    }

    let mut c = Cursor::new(data);
    let version_token = c.read_u32_le().map_err(|_| DecodeError {
        message: String::from("failed to read version token"),
    })?;
    let length_dwords = c.read_u32_le().map_err(|_| DecodeError {
        message: String::from("failed to read length"),
    })? as usize;
    let shader_type_val = (version_token >> 16) & 0xFFFF;
    let major = (version_token >> 4) & 0xF;
    let minor = version_token & 0xF;

    if version_token & 0x0000_FF00 != 0 {
        return Err(DecodeError {
            message: format!("version token has reserved bits set: 0x{version_token:08X}"),
        });
    }
    if !matches!((major, minor), (4, 0 | 1) | (5, 0 | 1)) {
        return Err(DecodeError {
            message: format!("unsupported shader model {major}.{minor}"),
        });
    }
    if shader_type_val > 5 {
        return Err(DecodeError {
            message: format!("invalid tokenized-program type {shader_type_val}"),
        });
    }

    let declared_bytes = length_dwords.checked_mul(4).ok_or_else(|| DecodeError {
        message: String::from("declared shader length overflows usize"),
    })?;
    if length_dwords < 2 || declared_bytes != data.len() {
        return Err(DecodeError {
            message: format!(
                "shader length is {length_dwords} dwords ({declared_bytes} bytes), actual size is {} bytes",
                data.len()
            ),
        });
    }

    let shader_type = match shader_type_val {
        0 => "ps",
        1 => "vs",
        2 => "gs",
        3 => "hs",
        4 => "ds",
        5 => "cs",
        _ => unreachable!("program type validated above"),
    };

    let mut instructions = Vec::new();
    let mut offset = 8;
    let end = declared_bytes;
    // Reuse a single token buffer across the decode loop to avoid
    // allocating a fresh Vec<u32> for every instruction.
    let mut tokens: Vec<u32> = Vec::new();

    while offset + 4 <= end {
        c.seek(SeekFrom::Start(offset as u64))
            .map_err(|_| DecodeError {
                message: format!("failed to seek to instruction at byte offset {offset}"),
            })?;
        let token = c.read_u32_le().map_err(|_| DecodeError {
            message: format!("failed to read instruction at byte offset {offset}"),
        })?;
        let opcode_val = token & 0x7FF;
        let instr_len = instruction_length(token, opcode_val, &mut c, offset, end).unwrap_or(0);

        if instr_len == 0 {
            return Err(DecodeError {
                message: format!("instruction at byte offset {offset} has zero length"),
            });
        }
        let instr_bytes = instr_len.checked_mul(4).ok_or_else(|| DecodeError {
            message: format!("instruction length overflows at byte offset {offset}"),
        })?;
        if offset + instr_bytes > end {
            return Err(DecodeError {
                message: format!(
                    "instruction at byte offset {offset} extends past end of chunk (needs {instr_bytes} bytes, only {} available)",
                    end - offset
                ),
            });
        }

        c.seek(SeekFrom::Start(offset as u64))
            .map_err(|_| DecodeError {
                message: format!("failed to seek to instruction at byte offset {offset}"),
            })?;
        tokens.clear();
        tokens.reserve(instr_len);
        for _ in 0..instr_len {
            tokens.push(c.read_u32_le().map_err(|_| DecodeError {
                message: format!("failed to read instruction at byte offset {offset}"),
            })?);
        }

        let shader_model_5_1_or_later = major > 5 || (major == 5 && minor > 0);
        let instruction = decode_instruction(
            &tokens,
            opcode_val,
            shader_model_5_1_or_later,
        );
        if let InstructionKind::Generic { operands, .. } = &instruction.kind {
            if let Some(expected) = instruction.opcode.expected_operand_count() {
                if operands.len() != expected {
                    return Err(DecodeError {
                        message: format!(
                            "{} at byte offset {offset} has {} operands; specification requires {expected}",
                            instruction.opcode.name(),
                            operands.len()
                        ),
                    });
                }
            }
        }
        instructions.push(instruction);
        offset += instr_bytes;
    }

    if offset != end {
        return Err(DecodeError {
            message: format!("{} trailing shader-code bytes", end - offset),
        });
    }

    let program = Program {
        shader_type,
        major_version: major,
        minor_version: minor,
        instructions,
        warnings: Vec::new(),
        fourcc,
    };
    let encoded = super::encode::encode(&program);
    if encoded != data {
        let difference = encoded
            .iter()
            .zip(data)
            .position(|(encoded, original)| encoded != original)
            .unwrap_or(encoded.len().min(data.len()));
        return Err(DecodeError {
            message: format!(
                "unsupported or non-canonical token encoding at byte offset {difference}"
            ),
        });
    }
    Ok(program)
}

fn instruction_length(
    token: u32,
    opcode: u32,
    c: &mut Cursor<&[u8]>,
    offset: usize,
    end: usize,
) -> Option<usize> {
    let op = Opcode::from_u32(opcode);
    if matches!(op, Opcode::CustomData)
        || (token >> 31 != 0
            && matches!(op, Opcode::DclInterface | Opcode::DclFunctionTable))
    {
        if offset + 8 <= end {
            c.seek(SeekFrom::Start((offset + 4) as u64)).ok()?;
            return usize::try_from(c.read_u32_le().ok()?).ok();
        }
        return None;
    }
    let len = ((token >> 24) & 0x7F) as usize;
    Some(len)
}

fn decode_instruction(
    tokens: &[u32],
    opcode_val: u32,
    shader_model_5_1_or_later: bool,
) -> Instruction {
    let op = Opcode::from_u32(opcode_val);
    let primary_token = tokens[0];

    // Saturate (bit 13) — not applicable to Sync (which reuses bits 11-14).
    let saturate = op != Opcode::Sync && (primary_token >> 13) & 1 != 0;

    // Conditional test condition: bit 18 (0 = zero, 1 = non-zero).
    let test_nonzero = matches!(
        op,
        Opcode::If
            | Opcode::Breakc
            | Opcode::Callc
            | Opcode::Continuec
            | Opcode::Discard
            | Opcode::Retc
    ) && ((primary_token >> 18) & 1 != 0);

    // SM5 precise modifier: bits 19-22.
    let precise_mask = ((primary_token >> 19) & 0xF) as u8;

    // Extract resinfo return type (bits [11:12]) for resinfo opcode.
    let resinfo_return_type = if matches!(op, Opcode::Resinfo | Opcode::SampleInfo) {
        Some((primary_token >> 11) & 0x3)
    } else {
        None
    };

    // Sync flags: bits 11-14 (only meaningful for Sync opcode).
    let sync_flags = if op == Opcode::Sync {
        ((primary_token >> 11) & 0xF) as u8
    } else {
        0
    };

    // Parse all extended opcode tokens from the instruction header chain.
    let (tex_offsets, resource_dim, resource_return_type) = if matches!(
        op,
        Opcode::DclInterface | Opcode::DclFunctionTable
    ) {
        (None, None, None)
    } else {
        decode_extended_tokens(tokens)
    };

    let kind = decode_instruction_kind(tokens, op, shader_model_5_1_or_later);

    // Token0 bits 13 and 19-22 encode saturate / precise for ALU and phase
    // instructions, but for declaration opcodes those same bit positions carry
    // payload (dimension, sample count, etc.). The encoder reconstructs
    // declaration token0 bits from the typed kind and ignores these modifier
    // fields, so suppress them here to keep the IR free of payload garbage.
    let is_modifier_kind = matches!(
        kind,
        InstructionKind::Generic { .. } | InstructionKind::HsPhase
    );

    Instruction {
        opcode: op,
        saturate: saturate && is_modifier_kind,
        test_nonzero: test_nonzero && is_modifier_kind,
        precise_mask: if is_modifier_kind { precise_mask } else { 0 },
        resinfo_return_type: if is_modifier_kind {
            resinfo_return_type
        } else {
            None
        },
        sync_flags: if is_modifier_kind { sync_flags } else { 0 },
        tex_offsets,
        resource_dim,
        resource_return_type,
        kind,
    }
}

fn decode_instruction_kind(
    tokens: &[u32],
    op: Opcode,
    shader_model_5_1_or_later: bool,
) -> InstructionKind {
    let primary_token = tokens[0];
    match op {
        Opcode::CustomData => decode_custom_data(tokens),
        Opcode::DclGlobalFlags => decode_dcl_global_flags(primary_token),
        Opcode::DclInput
        | Opcode::DclInputSgv
        | Opcode::DclInputSiv
        | Opcode::DclInputPs
        | Opcode::DclInputPsSgv
        | Opcode::DclInputPsSiv => decode_dcl_input(tokens, op),
        Opcode::DclOutput | Opcode::DclOutputSgv | Opcode::DclOutputSiv => {
            decode_dcl_output(tokens, op)
        }
        Opcode::DclResource => decode_dcl_resource(tokens, shader_model_5_1_or_later),
        Opcode::DclUnorderedAccessViewTyped => {
            decode_dcl_uav_typed(tokens, shader_model_5_1_or_later)
        }
        Opcode::DclUnorderedAccessViewRaw => {
            decode_dcl_uav_raw(tokens, shader_model_5_1_or_later)
        }
        Opcode::DclUnorderedAccessViewStructured => {
            decode_dcl_uav_structured(tokens, shader_model_5_1_or_later)
        }
        Opcode::DclThreadGroupSharedMemoryRaw => decode_dcl_tgsm_raw(tokens),
        Opcode::DclThreadGroupSharedMemoryStructured => decode_dcl_tgsm_structured(tokens),
        Opcode::DclResourceRaw => decode_dcl_resource_raw(tokens, shader_model_5_1_or_later),
        Opcode::DclResourceStructured => {
            decode_dcl_resource_structured(tokens, shader_model_5_1_or_later)
        }
        Opcode::DclStream => {
            // dcl_stream has a single stream operand — decode it generically
            decode_generic(tokens, op)
        }
        Opcode::DclIndexRange => decode_dcl_index_range(tokens),
        Opcode::DclSampler => decode_dcl_sampler(tokens, shader_model_5_1_or_later),
        Opcode::DclConstantBuffer => decode_dcl_cb(tokens, shader_model_5_1_or_later),
        Opcode::DclTemps => InstructionKind::DclTemps {
            count: if tokens.len() > 1 { tokens[1] } else { 0 },
        },
        Opcode::DclIndexableTemp => decode_dcl_indexable_temp(tokens),
        Opcode::DclGsInputPrimitive => decode_dcl_gs_input(primary_token),
        Opcode::DclGsOutputPrimitiveTopology => decode_dcl_gs_output_topo(primary_token),
        Opcode::DclMaxOutputVertexCount => InstructionKind::DclMaxOutputVertexCount {
            count: if tokens.len() > 1 { tokens[1] } else { 0 },
        },
        Opcode::DclGsInstanceCount => InstructionKind::DclGsInstanceCount {
            count: if tokens.len() > 1 { tokens[1] } else { 0 },
        },
        Opcode::DclOutputControlPointCount => InstructionKind::DclOutputControlPointCount {
            count: (primary_token >> 11) & 0x3F,
        },
        Opcode::DclInputControlPointCount => InstructionKind::DclInputControlPointCount {
            count: (primary_token >> 11) & 0x3F,
        },
        Opcode::DclTessDomain => decode_dcl_tess_domain(primary_token),
        Opcode::DclTessPartitioning => decode_dcl_tess_partitioning(primary_token),
        Opcode::DclTessOutputPrimitive => decode_dcl_tess_output_prim(primary_token),
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
        Opcode::DclHsJoinPhaseInstanceCount => {
            InstructionKind::DclHsJoinPhaseInstanceCount {
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
        Opcode::InterfaceCall => decode_interface_call(tokens),
        Opcode::HsDecls
        | Opcode::HsControlPointPhase
        | Opcode::HsForkPhase
        | Opcode::HsJoinPhase => InstructionKind::HsPhase,
        _ => decode_generic(tokens, op),
    }
}

// Declaration decoders.

fn decode_custom_data(tokens: &[u32]) -> InstructionKind {
    let subtype_val = (tokens[0] >> 11) & 0x1F;
    let subtype = CustomDataType::from_u32(subtype_val);
    let mut values = Vec::new();
    let mut raw = Vec::new();
    if subtype == CustomDataType::ImmediateConstantBuffer {
        if tokens.len() > 2 {
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
    } else if tokens.len() > 2 {
        // Preserve the raw payload of non-ICB blocks so they re-encode exactly.
        raw.extend_from_slice(&tokens[2..]);
    }
    InstructionKind::CustomData {
        subtype,
        values,
        raw,
        raw_dword_count: tokens.len(),
    }
}

fn decode_dcl_global_flags(token: u32) -> InstructionKind {
    let flags_bits = (token >> 11) & 0x1FFF;
    let mut flags = SmallVec::new();
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

fn decode_dcl_input(tokens: &[u32], op: Opcode) -> InstructionKind {
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
        Some(SystemValue::from_u32(tokens[tokens.len() - 1]))
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

fn decode_dcl_output(tokens: &[u32], op: Opcode) -> InstructionKind {
    let has_sv = matches!(op, Opcode::DclOutputSgv | Opcode::DclOutputSiv);
    let system_value = if has_sv && tokens.len() >= 3 {
        Some(SystemValue::from_u32(tokens[tokens.len() - 1]))
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

fn decode_dcl_resource(tokens: &[u32], shader_model_5_1_or_later: bool) -> InstructionKind {
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
    let (operands, pos) = decode_single_operand(tokens, 1);
    let return_type = if pos < tokens.len() {
        let rt = tokens[pos];
        [
            ReturnType::from_u32(rt & 0xF),
            ReturnType::from_u32((rt >> 4) & 0xF),
            ReturnType::from_u32((rt >> 8) & 0xF),
            ReturnType::from_u32((rt >> 12) & 0xF),
        ]
    } else {
        [ReturnType::Unknown(0); 4]
    };
    let space = if shader_model_5_1_or_later {
        tokens.get(pos + 1).copied().unwrap_or(0)
    } else {
        0
    };
    InstructionKind::DclResource {
        dimension,
        sample_count,
        return_type,
        operands,
        space,
    }
}

fn decode_dcl_uav_typed(tokens: &[u32], shader_model_5_1_or_later: bool) -> InstructionKind {
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
    let (operands, pos) = decode_single_operand(tokens, 1);
    let return_type = if pos < tokens.len() {
        let rt = tokens[pos];
        [
            ReturnType::from_u32(rt & 0xF),
            ReturnType::from_u32((rt >> 4) & 0xF),
            ReturnType::from_u32((rt >> 8) & 0xF),
            ReturnType::from_u32((rt >> 12) & 0xF),
        ]
    } else {
        [ReturnType::Unknown(0); 4]
    };
    let space = if shader_model_5_1_or_later {
        tokens.get(pos + 1).copied().unwrap_or(0)
    } else {
        0
    };
    InstructionKind::DclUavTyped {
        dimension,
        flags,
        return_type,
        operands,
        space,
    }
}

fn decode_dcl_uav_raw(tokens: &[u32], shader_model_5_1_or_later: bool) -> InstructionKind {
    let flags = (tokens[0] >> 16) & 0xFF;
    let (operands, pos) = decode_single_operand(tokens, 1);
    InstructionKind::DclUavRaw {
        flags,
        operands,
        space: if shader_model_5_1_or_later {
            tokens.get(pos).copied().unwrap_or(0)
        } else {
            0
        },
    }
}

fn decode_dcl_uav_structured(
    tokens: &[u32],
    shader_model_5_1_or_later: bool,
) -> InstructionKind {
    let flags = (tokens[0] >> 16) & 0xFF;
    let (operands, pos) = decode_single_operand(tokens, 1);
    InstructionKind::DclUavStructured {
        flags,
        stride: tokens.get(pos).copied().unwrap_or(0),
        operands,
        space: if shader_model_5_1_or_later {
            tokens.get(pos + 1).copied().unwrap_or(0)
        } else {
            0
        },
    }
}

fn decode_dcl_tgsm_raw(tokens: &[u32]) -> InstructionKind {
    let (operands, pos) = decode_single_operand(tokens, 1);
    InstructionKind::DclThreadGroupSharedMemoryRaw {
        operands,
        byte_count: tokens.get(pos).copied().unwrap_or(0),
    }
}

fn decode_dcl_tgsm_structured(tokens: &[u32]) -> InstructionKind {
    let (operands, pos) = decode_single_operand(tokens, 1);
    InstructionKind::DclThreadGroupSharedMemoryStructured {
        operands,
        stride: tokens.get(pos).copied().unwrap_or(0),
        count: tokens.get(pos + 1).copied().unwrap_or(0),
    }
}

fn decode_dcl_resource_raw(tokens: &[u32], shader_model_5_1_or_later: bool) -> InstructionKind {
    let (operands, pos) = decode_single_operand(tokens, 1);
    InstructionKind::DclResourceRaw {
        operands,
        space: if shader_model_5_1_or_later {
            tokens.get(pos).copied().unwrap_or(0)
        } else {
            0
        },
    }
}

fn decode_dcl_resource_structured(
    tokens: &[u32],
    shader_model_5_1_or_later: bool,
) -> InstructionKind {
    let (operands, pos) = decode_single_operand(tokens, 1);
    InstructionKind::DclResourceStructured {
        stride: tokens.get(pos).copied().unwrap_or(0),
        operands,
        space: if shader_model_5_1_or_later {
            tokens.get(pos + 1).copied().unwrap_or(0)
        } else {
            0
        },
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
    let primary_token = tokens[0];
    if (primary_token >> 31) & 1 == 0 || tokens.len() < 2 {
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
            // Type 2 = D3D10_SB_EXTENDED_OPCODE_RESOURCE_DIM.
            // Mask off the continuation bit (31): it is structural (re-derived
            // by the encoder), not part of the semantic dimension/stride value.
            2 => {
                resource_dim = Some(ext & 0x7FFF_FFFF);
            }
            // Type 3 = D3D10_SB_EXTENDED_OPCODE_RESOURCE_RETURN_TYPE
            3 => {
                resource_return_type = Some(ext & 0x7FFF_FFFF);
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
    let nibble = i8::try_from(val & 0xF).expect("masked nibble fits in i8");
    if nibble & 0x8 != 0 {
        nibble - 16
    } else {
        nibble
    }
}

fn decode_dcl_sampler(tokens: &[u32], shader_model_5_1_or_later: bool) -> InstructionKind {
    let mode = match (tokens[0] >> 11) & 0xF {
        0 => "default",
        1 => "comparison",
        2 => "mono",
        _ => "?",
    };
    let (operands, pos) = decode_single_operand(tokens, 1);
    InstructionKind::DclSampler {
        mode,
        operands,
        space: if shader_model_5_1_or_later {
            tokens.get(pos).copied().unwrap_or(0)
        } else {
            0
        },
    }
}

fn decode_dcl_cb(tokens: &[u32], shader_model_5_1_or_later: bool) -> InstructionKind {
    let access = if (tokens[0] >> 11) & 1 == 0 {
        "immediateIndexed"
    } else {
        "dynamicIndexed"
    };
    let (operands, pos) = decode_single_operand(tokens, 1);
    let size = if shader_model_5_1_or_later {
        tokens.get(pos).copied().unwrap_or(0)
    } else {
        operands
            .first()
            .and_then(|operand| operand.indices.get(1))
            .and_then(|index| match index {
                OperandIndex::Imm32(value) => Some(*value),
                _ => None,
            })
            .unwrap_or(0)
    };
    InstructionKind::DclConstantBuffer {
        access,
        operands,
        size,
        space: if shader_model_5_1_or_later {
            tokens.get(pos + 1).copied().unwrap_or(0)
        } else {
            0
        },
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
    let start = usize::from(tokens.first().is_some_and(|token| token >> 31 != 0)) + 1;
    let table_index = tokens.get(start).copied().unwrap_or(0);
    let count = if tokens.len() > start + 1 {
        tokens[start + 1] as usize
    } else {
        0
    };
    let body_indices: SmallU32Vec = tokens
        .iter()
        .skip(start + 2)
        .take(count)
        .copied()
        .collect();
    InstructionKind::DclFunctionTable {
        table_index,
        body_indices,
    }
}

fn decode_dcl_interface(tokens: &[u32]) -> InstructionKind {
    let dynamically_indexed = tokens.first().is_some_and(|token| token & (1 << 11) != 0);
    let start = usize::from(tokens.first().is_some_and(|token| token >> 31 != 0)) + 1;
    let interface_index = tokens.get(start).copied().unwrap_or(0);
    let expected_table_size = tokens.get(start + 1).copied().unwrap_or(0);
    let lengths = tokens.get(start + 2).copied().unwrap_or(0);
    let table_length = usize::from((lengths & 0xFFFF) as u16);
    let array_length = (lengths >> 16) as u16;
    let table_indices: SmallU32Vec = tokens
        .iter()
        .skip(start + 3)
        .take(table_length)
        .copied()
        .collect();
    InstructionKind::DclInterface {
        dynamically_indexed,
        interface_index,
        expected_table_size,
        array_length,
        table_indices,
    }
}

fn decode_interface_call(tokens: &[u32]) -> InstructionKind {
    let (operands, _) = decode_single_operand(tokens, 2);
    InstructionKind::InterfaceCall {
        function_index: tokens.get(1).copied().unwrap_or(0),
        operands,
    }
}

// Generic instruction decoder.

fn decode_generic(tokens: &[u32], op: Opcode) -> InstructionKind {
    let mut start = 1;
    // Skip extended opcode tokens
    if (tokens[0] >> 31) & 1 != 0 {
        while start < tokens.len() && (tokens[start] >> 31) & 1 != 0 {
            start += 1;
        }
        start += 1;
    }
    let (operands, end) = if let Some(expected) = op.expected_operand_count() {
        decode_operand_count(tokens, start, expected)
    } else {
        // Reserved and unknown opcodes have no published operand grammar.
        // Preserve their payload opaquely instead of guessing where operand
        // boundaries lie.
        (SmallVec::new(), start)
    };
    InstructionKind::Generic {
        operands,
        trailing_tokens: tokens
            .get(end..)
            .unwrap_or_default()
            .iter()
            .copied()
            .collect(),
    }
}

// Operand decoder.

/// Decode operands starting at `start`, returning the operands and the
/// token index just past the last consumed operand token.
fn decode_operands_with_pos(tokens: &[u32], start: usize) -> (Operands, usize) {
    let mut result = SmallVec::new();
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

/// Decode at most `count` operands, leaving every remaining DWORD available
/// for opaque instruction-tail preservation.
fn decode_operand_count(tokens: &[u32], start: usize, count: usize) -> (Operands, usize) {
    let mut result = SmallVec::new();
    let mut pos = start;
    for _ in 0..count {
        if pos >= tokens.len() {
            break;
        }
        let (op, consumed) = decode_one_operand(tokens, pos);
        if consumed == 0 {
            break;
        }
        result.push(op);
        pos += consumed;
    }
    (result, pos)
}

fn decode_operands(tokens: &[u32], start: usize) -> Operands {
    decode_operands_with_pos(tokens, start).0
}

fn decode_single_operand(tokens: &[u32], start: usize) -> (Operands, usize) {
    if start >= tokens.len() {
        return (SmallVec::new(), start);
    }
    let (operand, consumed) = decode_one_operand(tokens, start);
    if consumed == 0 {
        return (SmallVec::new(), start);
    }
    (smallvec![operand], start + consumed)
}

fn decode_one_operand(tokens: &[u32], pos: usize) -> (Operand, usize) {
    decode_one_operand_at_depth(tokens, pos, 0)
}

fn decode_one_operand_at_depth(tokens: &[u32], pos: usize, depth: usize) -> (Operand, usize) {
    const MAX_OPERAND_DEPTH: usize = 64;
    if depth >= MAX_OPERAND_DEPTH {
        return (empty_operand(), 0);
    }
    if pos >= tokens.len() {
        return (empty_operand(), 0);
    }
    let token = tokens[pos];
    let num_components = token & 0x3;
    let op_type_val = (token >> 12) & 0xFF;
    let index_dim = (token >> 20) & 0x3;
    let is_extended = (token >> 31) & 1 != 0;

    let reg_type = RegisterType::from_u32(op_type_val);
    if matches!(reg_type, RegisterType::Immediate32 | RegisterType::Immediate64)
        && index_dim != 0
    {
        return (empty_operand(), 0);
    }
    let mut consumed = 1usize;

    // Extended operand tokens precede index/immediate data. The public token
    // format currently defines only type 1 (modifier), but permits a chained
    // sequence for forward compatibility.
    let mut negate = false;
    let mut abs = false;
    let mut min_precision = MinPrecision::Default;
    let mut non_uniform = false;
    let mut has_more_extensions = is_extended;
    if has_more_extensions && pos + consumed >= tokens.len() {
        return (empty_operand(), 0);
    }
    while has_more_extensions && pos + consumed < tokens.len() {
        let ext = tokens[pos + consumed];
        consumed += 1;
        has_more_extensions = ext >> 31 != 0;
        if ext & 0x3F == 1 {
            let modifier = (ext >> 6) & 0xFF;
            negate = matches!(modifier, 1 | 3);
            abs = matches!(modifier, 2 | 3);
            min_precision = MinPrecision::from_u32((ext >> 14) & 0x7);
            non_uniform = ext & (1 << 17) != 0;
        }
    }
    if has_more_extensions {
        return (empty_operand(), 0);
    }

    let components = decode_components(token, num_components);
    let num_components = components.num_components();

    // Decode indices and immediates
    let mut indices = SmallVec::new();
    let mut immediate_values = SmallVec::new();

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
                if tokens.len().saturating_sub(pos + consumed) < count {
                    return (empty_operand(), 0);
                }
                for i in 0..count {
                    immediate_values.push(tokens[pos + consumed + i]);
                }
                consumed += count;
            } else if op_type_val == 5 {
                // The tokenized-program format stores the raw immediate64
                // payload in the same 1-or-4 DWORD component slots used by
                // Microsoft's parser. A four-component token therefore holds
                // two packed 64-bit values.
                let dword_count = match num_components {
                    0 => 0,
                    1 => 1,
                    _ => 4,
                };
                if tokens.len().saturating_sub(pos + consumed) < dword_count {
                    return (empty_operand(), 0);
                }
                for i in 0..dword_count {
                    immediate_values.push(tokens[pos + consumed + i]);
                }
                consumed += dword_count;
            }
        }
        1 => {
            let parent_token = tokens[pos];
            let idx0_repr = (parent_token >> 22) & 0x7;
            let (idx, ic) = decode_index(tokens, pos + consumed, idx0_repr, depth);
            if ic == 0 {
                return (empty_operand(), 0);
            }
            indices.push(idx);
            consumed += ic;
        }
        2 => {
            let parent_token = tokens[pos];
            let idx0_repr = (parent_token >> 22) & 0x7;
            let idx1_repr = (parent_token >> 25) & 0x7;
            let (idx0, ic0) = decode_index(tokens, pos + consumed, idx0_repr, depth);
            if ic0 == 0 {
                return (empty_operand(), 0);
            }
            indices.push(idx0);
            consumed += ic0;
            let (idx1, ic1) = decode_index(tokens, pos + consumed, idx1_repr, depth);
            if ic1 == 0 {
                return (empty_operand(), 0);
            }
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
                let (idx, ic) = decode_index(tokens, pos + consumed, repr, depth);
                if ic == 0 {
                    return (empty_operand(), 0);
                }
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
            min_precision,
            non_uniform,
            indices,
            immediate_values,
        },
        consumed,
    )
}

fn decode_components(token: u32, num_components: u32) -> ComponentSelect {
    let sel_mode = (token >> 2) & 0x3;
    match num_components {
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

fn decode_index(
    tokens: &[u32],
    pos: usize,
    repr: u32,
    parent_depth: usize,
) -> (OperandIndex, usize) {
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
                let val = u64::from(tokens[pos]) | (u64::from(tokens[pos + 1]) << 32);
                (OperandIndex::Imm64(val), 2)
            } else {
                (OperandIndex::Imm32(0), 0)
            }
        }
        2 => {
            let (sub, consumed) = decode_one_operand_at_depth(tokens, pos, parent_depth + 1);
            if consumed == 0 {
                (OperandIndex::Imm32(0), 0)
            } else {
                (OperandIndex::Relative(Box::new(sub)), consumed)
            }
        }
        3 => {
            if pos < tokens.len() {
                let imm = tokens[pos];
                let (sub, consumed) =
                    decode_one_operand_at_depth(tokens, pos + 1, parent_depth + 1);
                if consumed == 0 {
                    (OperandIndex::Imm32(0), 0)
                } else {
                    (
                        OperandIndex::RelativePlusImm(imm, Box::new(sub)),
                        1 + consumed,
                    )
                }
            } else {
                (OperandIndex::Imm32(0), 0)
            }
        }
        4 => {
            if pos + 1 < tokens.len() {
                let low = u64::from(tokens[pos]);
                let high = u64::from(tokens[pos + 1]);
                let immediate = low | (high << 32);
                let (sub, consumed) =
                    decode_one_operand_at_depth(tokens, pos + 2, parent_depth + 1);
                if consumed == 0 {
                    (OperandIndex::Imm32(0), 0)
                } else {
                    (
                        OperandIndex::RelativePlusImm64(immediate, Box::new(sub)),
                        2 + consumed,
                    )
                }
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
        min_precision: MinPrecision::Default,
        non_uniform: false,
        indices: smallvec![],
        immediate_values: smallvec![],
    }
}
