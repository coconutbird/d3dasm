//! Encoder: structured IR → raw SHEX/SHDR bytes.

use alloc::vec::Vec;

use super::ir::{
    ComponentSelect, CustomDataType, Instruction, InstructionKind, MinPrecision, Operand,
    OperandIndex, Program, RegisterType, ReturnType, SystemValue,
};
use super::opcodes::Opcode;

fn wire_u32(value: usize) -> u32 {
    u32::try_from(value).expect("SHEX dword count exceeds u32")
}

/// Encode a [`Program`] into raw SHEX/SHDR chunk bytes.
///
/// This is the inverse of [`super::decode::decode`]. The returned byte
/// slice is suitable for embedding directly as a DXBC chunk payload.
///
/// # Panics
///
/// Panics if the encoded program exceeds a size representable by SHEX fields.
#[must_use]
pub fn encode(program: &Program) -> Vec<u8> {
    let mut dwords: Vec<u32> = Vec::new();

    // Version token: shader_type << 16 | major << 4 | minor
    let shader_type_val = match program.shader_type {
        "vs" => 1,
        "gs" => 2,
        "hs" => 3,
        "ds" => 4,
        "cs" => 5,
        _ => 0u32, // Pixel shaders and unknown stage names.
    };
    dwords.push((shader_type_val << 16) | (program.major_version << 4) | program.minor_version);

    // Placeholder for length (dword count including header).
    dwords.push(0);

    let shader_model_5_1_or_later =
        program.major_version > 5 || (program.major_version == 5 && program.minor_version > 0);
    for instr in &program.instructions {
        encode_instruction(instr, shader_model_5_1_or_later, &mut dwords);
    }

    // Fix up total length.
    dwords[1] = wire_u32(dwords.len());

    // Convert to little-endian bytes.
    dwords.iter().flat_map(|d| d.to_le_bytes()).collect()
}

/// Encode a single instruction into the dword stream.
fn encode_instruction(instr: &Instruction, shader_model_5_1_or_later: bool, out: &mut Vec<u32>) {
    let opcode_val = instr.opcode.to_u32();

    if let InstructionKind::CustomData {
        subtype,
        values,
        raw,
        raw_dword_count,
    } = &instr.kind
    {
        encode_custom_data(opcode_val, *subtype, values, raw, *raw_dword_count, out);
        return;
    }

    // Reserve a slot for the primary instruction token; we'll patch it later.
    let token0_idx = out.len();
    out.push(0);

    // Build extended opcode token chain. Tokens are emitted in order:
    // sample_controls (type 1), resource_dim (type 2), resource_return_type (type 3).
    // Bit 31 of each token is set if another extended token follows.
    let ext_tokens = build_extended_tokens(instr);
    let has_extended_opcode = !ext_tokens.is_empty();
    for &ext in &ext_tokens {
        out.push(ext);
    }

    // Encode instruction-kind-specific payload.
    encode_kind(instr, shader_model_5_1_or_later, out);

    // Compute instruction length and build token0.
    // Reconstruct bits 11-23 from typed IR fields (no raw bit bucket).
    let mut instr_len = out.len() - token0_idx;
    let has_extended_length = instr_len > 127
        && matches!(
            &instr.kind,
            InstructionKind::DclFunctionTable { .. } | InstructionKind::DclInterface { .. }
        );
    if has_extended_length {
        assert!(
            !has_extended_opcode,
            "long interface declarations cannot also carry extended opcode tokens"
        );
        instr_len = instr_len
            .checked_add(1)
            .expect("SHEX instruction length overflows usize");
        out.insert(token0_idx + 1, wire_u32(instr_len));
    } else {
        assert!(instr_len <= 127, "SHEX instruction exceeds 127 dwords");
    }
    let mut token0 = (opcode_val & 0x7FF) | build_token0_opdata(instr);
    if has_extended_opcode || has_extended_length {
        token0 |= 1 << 31;
    }
    token0 |= (wire_u32(instr_len) & 0x7F) << 24;

    out[token0_idx] = token0;
}

/// Reconstruct the opcode-specific bits (11–23) of token0 from typed IR fields.
///
/// Declaration opcodes embed their payload entirely in these bits (dimension,
/// interpolation mode, flags, etc.), while non-declaration opcodes use them
/// for saturate, resinfo return type, test condition, precise mask, and sync.
fn build_token0_opdata(instr: &Instruction) -> u32 {
    if let Some(bits) = declaration_token0(&instr.kind) {
        return bits;
    }

    // Non-declaration opcodes: reconstruct from instruction-level typed fields.
    let mut bits = 0u32;
    if instr.opcode == Opcode::Sync {
        bits |= (u32::from(instr.sync_flags) & 0xF) << 11;
    } else {
        if instr.saturate {
            bits |= 1 << 13;
        }
        if let Some(rt) = instr.resinfo_return_type {
            bits |= (rt & 0x3) << 11;
        }
    }
    if instr.test_nonzero {
        bits |= 1 << 18;
    }
    bits | ((u32::from(instr.precise_mask) & 0xF) << 19)
}

/// Build token0 payload bits for declaration instructions.
fn declaration_token0(kind: &InstructionKind) -> Option<u32> {
    Some(match kind {
        InstructionKind::DclGlobalFlags { flags } => {
            let names: &[&str] = &[
                "refactoringAllowed",
                "enableDoublePrecisionFloatOps",
                "forceEarlyDepthStencil",
                "enableRawAndStructuredBuffers",
                "skipOptimization",
                "enableMinPrecision",
                "enable11_1DoubleExtensions",
                "enable11_1ShaderExtensions",
                "allResourcesBound",
            ];
            let mut bits = 0u32;
            for flag in flags {
                if let Some(i) = names.iter().position(|n| n == flag) {
                    bits |= 1 << i;
                }
            }
            bits << 11
        }
        InstructionKind::DclInput { interpolation, .. } => interpolation.map_or(0, |interp| {
            let value = match interp {
                "constant" => 1u32,
                "linear" => 2,
                "linearCentroid" => 3,
                "linearNoperspective" => 4,
                "linearNoperspectiveCentroid" => 5,
                "linearSample" => 6,
                "linearNoperspectiveSample" => 7,
                _ => 0,
            };
            value << 11
        }),
        InstructionKind::DclResource {
            dimension,
            sample_count,
            ..
        } => (dimension_to_u32(dimension) << 11) | ((sample_count & 0x7F) << 16),
        InstructionKind::DclSampler { mode, .. } => {
            let value = match *mode {
                "comparison" => 1,
                "mono" => 2,
                _ => 0u32, // Default and unknown sampler modes.
            };
            value << 11
        }
        InstructionKind::DclConstantBuffer { access, .. } => {
            u32::from(*access == "dynamicIndexed") << 11
        }
        InstructionKind::DclGsInputPrimitive { primitive } => (primitive.to_raw() & 0x3F) << 11,
        InstructionKind::DclGsOutputTopology { topology } => (topology.to_raw() & 0x3F) << 11,
        InstructionKind::DclOutputControlPointCount { count }
        | InstructionKind::DclInputControlPointCount { count } => (count & 0x3F) << 11,
        InstructionKind::DclTessDomain { domain } => {
            let value = match *domain {
                "isoline" => 1u32,
                "tri" => 2,
                "quad" => 3,
                _ => 0,
            };
            value << 11
        }
        InstructionKind::DclTessPartitioning { partitioning } => {
            let value = match *partitioning {
                "integer" => 1u32,
                "pow2" => 2,
                "fractional_odd" => 3,
                "fractional_even" => 4,
                _ => 0,
            };
            value << 11
        }
        InstructionKind::DclTessOutputPrimitive { primitive } => {
            let value = match *primitive {
                "point" => 1u32,
                "line" => 2,
                "triangle_cw" => 3,
                "triangle_ccw" => 4,
                _ => 0,
            };
            value << 11
        }
        InstructionKind::DclUavTyped {
            dimension, flags, ..
        } => (dimension_to_u32(dimension) << 11) | ((flags & 0xFF) << 16),
        InstructionKind::DclUavRaw { flags, .. }
        | InstructionKind::DclUavStructured { flags, .. } => (flags & 0xFF) << 16,
        InstructionKind::DclInterface {
            dynamically_indexed,
            ..
        } => u32::from(*dynamically_indexed) << 11,
        _ => return None,
    })
}

/// Map a resource dimension string back to its token0 integer value.
pub(crate) fn dimension_to_u32(dim: &str) -> u32 {
    match dim {
        "buffer" => 1,
        "texture1d" => 2,
        "texture2d" => 3,
        "texture2dms" => 4,
        "texture3d" => 5,
        "texturecube" => 6,
        "texture1darray" => 7,
        "texture2darray" => 8,
        "texture2dmsarray" => 9,
        "texturecubearray" => 10,
        _ => 0,
    }
}

/// Encode a `customdata` block. `CustomData` has a special two-dword header
/// format where the second dword is the total dword count (not in bits 24-30).
fn encode_custom_data(
    opcode_val: u32,
    subtype: CustomDataType,
    values: &[[f32; 4]],
    raw: &[u32],
    raw_dword_count: usize,
    out: &mut Vec<u32>,
) {
    let subtype_val = match subtype {
        CustomDataType::Comment => 0u32,
        CustomDataType::DebugInfo => 1,
        CustomDataType::Opaque => 2,
        CustomDataType::ImmediateConstantBuffer => 3,
        CustomDataType::Other(v) => v,
    };
    // Token0: opcode in bits [0:10], subtype in bits [11:15]
    out.push((opcode_val & 0x7FF) | (subtype_val << 11));
    if subtype == CustomDataType::ImmediateConstantBuffer {
        // Token1: total = 2 (header) + values.len() * 4, then ICB float rows.
        out.push(wire_u32(2 + values.len() * 4));
        for row in values {
            for val in row {
                out.push(val.to_bits());
            }
        }
    } else {
        // Token1: total dword count, then the preserved raw payload.
        let total = raw_dword_count.max(2 + raw.len());
        out.push(wire_u32(total));
        out.extend_from_slice(raw);
    }
}

/// Encode the instruction-kind-specific payload (operands and trailing data).
fn encode_kind(instr: &Instruction, shader_model_5_1_or_later: bool, out: &mut Vec<u32>) {
    match &instr.kind {
        InstructionKind::Generic {
            operands,
            trailing_tokens,
        } => {
            encode_operands(operands, out);
            out.extend_from_slice(trailing_tokens);
        }
        InstructionKind::DclGlobalFlags { .. }
        | InstructionKind::DclGsInputPrimitive { .. }
        | InstructionKind::DclGsOutputTopology { .. }
        | InstructionKind::DclOutputControlPointCount { .. }
        | InstructionKind::DclInputControlPointCount { .. }
        | InstructionKind::DclTessDomain { .. }
        | InstructionKind::DclTessPartitioning { .. }
        | InstructionKind::DclTessOutputPrimitive { .. }
        | InstructionKind::HsPhase => {}
        InstructionKind::CustomData { .. } => {
            // Already handled in encode_instruction.
            unreachable!();
        }
        _ => encode_declaration_kind(instr, shader_model_5_1_or_later, out),
    }
}

fn encode_declaration_kind(
    instr: &Instruction,
    shader_model_5_1_or_later: bool,
    out: &mut Vec<u32>,
) {
    match &instr.kind {
        InstructionKind::DclInput {
            system_value,
            operands,
            ..
        }
        | InstructionKind::DclOutput {
            system_value,
            operands,
        } => {
            encode_operands(operands, out);
            if let Some(sv) = system_value {
                out.push(sv.to_u32());
            }
        }
        InstructionKind::DclResource {
            return_type,
            operands,
            space,
            ..
        }
        | InstructionKind::DclUavTyped {
            return_type,
            operands,
            space,
            ..
        } => {
            encode_operands(operands, out);
            out.push(encode_return_type_token(return_type));
            if shader_model_5_1_or_later {
                out.push(*space);
            }
        }
        InstructionKind::DclSampler {
            operands, space, ..
        }
        | InstructionKind::DclUavRaw {
            operands, space, ..
        }
        | InstructionKind::DclResourceRaw { operands, space } => {
            encode_operands(operands, out);
            if shader_model_5_1_or_later {
                out.push(*space);
            }
        }
        InstructionKind::DclConstantBuffer {
            operands,
            size,
            space,
            ..
        } => {
            encode_operands(operands, out);
            if shader_model_5_1_or_later {
                out.push(*size);
                out.push(*space);
            }
        }
        InstructionKind::DclUavStructured {
            operands,
            stride,
            space,
            ..
        }
        | InstructionKind::DclResourceStructured {
            stride,
            operands,
            space,
        } => {
            encode_operands(operands, out);
            out.push(*stride);
            if shader_model_5_1_or_later {
                out.push(*space);
            }
        }
        _ => encode_auxiliary_kind(instr, out),
    }
}

fn encode_auxiliary_kind(instr: &Instruction, out: &mut Vec<u32>) {
    match &instr.kind {
        InstructionKind::DclTemps { count }
        | InstructionKind::DclMaxOutputVertexCount { count }
        | InstructionKind::DclGsInstanceCount { count }
        | InstructionKind::DclHsForkPhaseInstanceCount { count }
        | InstructionKind::DclHsJoinPhaseInstanceCount { count } => out.push(*count),
        InstructionKind::DclIndexableTemp {
            reg,
            size,
            components,
        } => {
            out.push(*reg);
            out.push(*size);
            out.push(*components);
        }
        InstructionKind::DclHsMaxTessFactor { value } => {
            out.push(value.to_bits());
        }
        InstructionKind::DclThreadGroup { x, y, z } => {
            out.push(*x);
            out.push(*y);
            out.push(*z);
        }
        InstructionKind::DclThreadGroupSharedMemoryRaw {
            operands,
            byte_count,
        } => {
            encode_operands(operands, out);
            out.push(*byte_count);
        }
        InstructionKind::DclThreadGroupSharedMemoryStructured {
            operands,
            stride,
            count,
        } => {
            encode_operands(operands, out);
            out.push(*stride);
            out.push(*count);
        }
        _ => encode_interface_kind(instr, out),
    }
}

fn encode_interface_kind(instr: &Instruction, out: &mut Vec<u32>) {
    match &instr.kind {
        InstructionKind::DclFunctionBody { index } => {
            out.push(*index);
        }
        InstructionKind::DclFunctionTable {
            table_index,
            body_indices,
        } => {
            out.push(*table_index);
            out.push(wire_u32(body_indices.len()));
            out.extend_from_slice(body_indices);
        }
        InstructionKind::DclInterface {
            interface_index,
            expected_table_size,
            array_length,
            table_indices,
            ..
        } => {
            out.push(*interface_index);
            out.push(*expected_table_size);
            let table_length = u32::from(
                u16::try_from(table_indices.len())
                    .expect("SHEX interface table exceeds 65535 entries"),
            );
            out.push(table_length | (u32::from(*array_length) << 16));
            out.extend_from_slice(table_indices);
        }
        InstructionKind::InterfaceCall {
            function_index,
            operands,
        } => {
            out.push(*function_index);
            encode_operands(operands, out);
        }
        InstructionKind::DclIndexRange { operands, count } => {
            encode_operands(operands, out);
            out.push(*count);
        }
        _ => unreachable!("instruction kind was not classified for encoding"),
    }
}

/// Set declaration-specific bits in token0 that the decoder extracts from
/// the primary instruction token rather than from trailing dwords.
/// Encode a sequence of operands into the dword stream.
fn encode_operands(operands: &[Operand], out: &mut Vec<u32>) {
    for op in operands {
        encode_one_operand(op, out);
    }
}

/// Encode a single operand token and its index/immediate data.
fn encode_one_operand(op: &Operand, out: &mut Vec<u32>) {
    let op_type_val = op.reg_type.to_u32();
    let index_dim = wire_u32(op.indices.len());

    let num_components = op.components.num_components();
    let sel_bits = match &op.components {
        ComponentSelect::ZeroComponent | ComponentSelect::OneComponent => 0u32,
        ComponentSelect::Mask(m) => u32::from(*m) << 4,
        ComponentSelect::Swizzle(s) => {
            let swiz = u32::from(s[0])
                | (u32::from(s[1]) << 2)
                | (u32::from(s[2]) << 4)
                | (u32::from(s[3]) << 6);
            (1u32 << 2) | (swiz << 4)
        }
        ComponentSelect::Scalar(c) => (2u32 << 2) | (u32::from(*c) << 4),
    };

    let has_extended =
        op.negate || op.abs || op.min_precision != MinPrecision::Default || op.non_uniform;

    // Build operand token.
    let mut token = num_components | sel_bits;
    token |= (op_type_val & 0xFF) << 12;
    token |= (index_dim & 0x3) << 20;
    if has_extended {
        token |= 1 << 31;
    }

    // Encode index representation bits for each dimension.
    for (i, idx) in op.indices.iter().enumerate() {
        let repr = match idx {
            OperandIndex::Imm32(_) => 0u32,
            OperandIndex::Imm64(_) => 1,
            OperandIndex::Relative(_) => 2,
            OperandIndex::RelativePlusImm(_, _) => 3,
            OperandIndex::RelativePlusImm64(_, _) => 4,
        };
        token |= repr << (22 + i * 3);
    }

    out.push(token);

    // Extended modifier token.
    if has_extended {
        let mut ext = 1u32; // type = 1 (extended operand modifier)
        let modifier = match (op.negate, op.abs) {
            (false, false) => 0,
            (true, false) => 1,
            (false, true) => 2,
            (true, true) => 3,
        };
        ext |= modifier << 6;
        ext |= op.min_precision.to_u32() << 14;
        if op.non_uniform {
            ext |= 1 << 17;
        }
        out.push(ext);
    }

    // Emit immediates for Immediate32/Immediate64 (index_dim == 0).
    if index_dim == 0
        && (op.reg_type == RegisterType::Immediate32 || op.reg_type == RegisterType::Immediate64)
    {
        for &v in &op.immediate_values {
            out.push(v);
        }
        return;
    }

    // Emit index data.
    for idx in &op.indices {
        encode_index(idx, out);
    }
}

/// Encode a single operand index.
fn encode_index(idx: &OperandIndex, out: &mut Vec<u32>) {
    match idx {
        OperandIndex::Imm32(v) => out.push(*v),
        OperandIndex::Imm64(v) => {
            out.push(u32::try_from(*v & u64::from(u32::MAX)).expect("masked value fits in u32"));
            out.push(u32::try_from(*v >> 32).expect("shifted value fits in u32"));
        }
        OperandIndex::Relative(sub) => encode_one_operand(sub, out),
        OperandIndex::RelativePlusImm(imm, sub) => {
            out.push(*imm);
            encode_one_operand(sub, out);
        }
        OperandIndex::RelativePlusImm64(imm, sub) => {
            out.push(u32::try_from(*imm & u64::from(u32::MAX)).expect("masked value fits in u32"));
            out.push(u32::try_from(*imm >> 32).expect("shifted value fits in u32"));
            encode_one_operand(sub, out);
        }
    }
}

/// Pack four return types into a single dword.
fn encode_return_type_token(rt: &[ReturnType; 4]) -> u32 {
    rt[0].to_u32() | (rt[1].to_u32() << 4) | (rt[2].to_u32() << 8) | (rt[3].to_u32() << 12)
}

/// Convert a system value name string back to its `D3D10_SB_NAME` / `D3D11_SB_NAME` value.
#[must_use]
pub fn system_value_to_u32(name: &str) -> u32 {
    SystemValue::from_name(name).map_or(0, SystemValue::to_u32)
}

/// Build the chain of extended opcode tokens for an instruction.
///
/// Returns an empty vec if no extended tokens are needed. Each token
/// has bit 31 set correctly to indicate whether another follows.
fn build_extended_tokens(instr: &Instruction) -> Vec<u32> {
    let mut tokens: Vec<u32> = Vec::new();

    if let Some(offsets) = &instr.tex_offsets {
        let u = u32::from(offsets[0].to_ne_bytes()[0]) & 0xF;
        let v = u32::from(offsets[1].to_ne_bytes()[0]) & 0xF;
        let w = u32::from(offsets[2].to_ne_bytes()[0]) & 0xF;
        // Type 1 = D3D10_SB_EXTENDED_OPCODE_SAMPLE_CONTROLS
        tokens.push(1 | (u << 9) | (v << 13) | (w << 17));
    }

    if let Some(raw) = instr.resource_dim {
        // Preserve the raw token but clear bit 31 (we'll set it below if needed).
        tokens.push(raw & 0x7FFF_FFFF);
    }

    if let Some(raw) = instr.resource_return_type {
        tokens.push(raw & 0x7FFF_FFFF);
    }

    // Set bit 31 (continuation) on all tokens except the last.
    for i in 0..tokens.len().saturating_sub(1) {
        tokens[i] |= 1 << 31;
    }

    tokens
}

#[cfg(test)]
mod tests {
    use alloc::boxed::Box;

    use smallvec::smallvec;

    use super::*;
    use crate::shex::decode_with_fourcc;

    #[test]
    fn long_function_table_uses_extended_instruction_length() {
        let instruction = Instruction {
            opcode: Opcode::DclFunctionTable,
            saturate: false,
            test_nonzero: false,
            precise_mask: 0,
            resinfo_return_type: None,
            sync_flags: 0,
            tex_offsets: None,
            resource_dim: None,
            resource_return_type: None,
            kind: InstructionKind::DclFunctionTable {
                table_index: 7,
                body_indices: (0..125).collect(),
            },
        };
        let program = Program {
            shader_type: "ps",
            major_version: 5,
            minor_version: 0,
            instructions: alloc::vec![instruction],
            warnings: Vec::new(),
            fourcc: *b"SHEX",
        };

        let bytes = encode(&program);
        let token = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let extended_length = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_ne!(token & (1 << 31), 0);
        assert_eq!((token >> 24) & 0x7f, 0x81 & 0x7f);
        assert_eq!(extended_length, 129);
        assert_eq!(decode_with_fourcc(&bytes, *b"SHEX").unwrap(), program);
    }

    #[test]
    fn immediate64_plus_relative_index_uses_full_three_bit_field() {
        let relative = Operand {
            reg_type: RegisterType::Temp,
            components: ComponentSelect::OneComponent,
            negate: false,
            abs: false,
            min_precision: MinPrecision::Default,
            non_uniform: false,
            indices: smallvec![OperandIndex::Imm32(2)],
            immediate_values: smallvec![],
        };
        let destination = Operand {
            reg_type: RegisterType::Temp,
            components: ComponentSelect::Mask(0xf),
            negate: false,
            abs: false,
            min_precision: MinPrecision::Default,
            non_uniform: false,
            indices: smallvec![OperandIndex::Imm32(0)],
            immediate_values: smallvec![],
        };
        let source = Operand {
            reg_type: RegisterType::ConstantBuffer,
            components: ComponentSelect::Swizzle([0, 1, 2, 3]),
            negate: false,
            abs: false,
            min_precision: MinPrecision::Default,
            non_uniform: false,
            indices: smallvec![OperandIndex::RelativePlusImm64(
                0x1122_3344_5566_7788,
                Box::new(relative),
            )],
            immediate_values: smallvec![],
        };
        let instruction = Instruction {
            opcode: Opcode::Mov,
            saturate: false,
            test_nonzero: false,
            precise_mask: 0,
            resinfo_return_type: None,
            sync_flags: 0,
            tex_offsets: None,
            resource_dim: None,
            resource_return_type: None,
            kind: InstructionKind::Generic {
                operands: smallvec![destination, source],
                trailing_tokens: smallvec![],
            },
        };
        let program = Program {
            shader_type: "ps",
            major_version: 5,
            minor_version: 0,
            instructions: alloc::vec![instruction],
            warnings: Vec::new(),
            fourcc: *b"SHEX",
        };

        let bytes = encode(&program);
        let source_token = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
        assert_eq!((source_token >> 22) & 0x7, 4);
        assert_eq!(decode_with_fourcc(&bytes, *b"SHEX").unwrap(), program);
    }
}
