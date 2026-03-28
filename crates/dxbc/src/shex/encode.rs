//! Encoder: structured IR → raw SHEX/SHDR bytes.

use alloc::vec::Vec;

use super::ir::*;

/// Encode a [`Program`] into raw SHEX/SHDR chunk bytes.
///
/// This is the inverse of [`super::decode::decode`]. The returned byte
/// slice is suitable for embedding directly as a DXBC chunk payload.
pub fn encode(program: &Program) -> Vec<u8> {
    let mut dwords: Vec<u32> = Vec::new();

    // Version token: shader_type << 16 | major << 4 | minor
    let shader_type_val = match program.shader_type {
        "ps" => 0u32,
        "vs" => 1,
        "gs" => 2,
        "hs" => 3,
        "ds" => 4,
        "cs" => 5,
        _ => 0,
    };
    dwords.push((shader_type_val << 16) | (program.major_version << 4) | program.minor_version);

    // Placeholder for length (dword count including header).
    dwords.push(0);

    for instr in &program.instructions {
        encode_instruction(instr, &mut dwords);
    }

    // Fix up total length.
    dwords[1] = dwords.len() as u32;

    // Convert to little-endian bytes.
    dwords.iter().flat_map(|d| d.to_le_bytes()).collect()
}

/// Encode a single instruction into the dword stream.
fn encode_instruction(instr: &Instruction, out: &mut Vec<u32>) {
    let opcode_val = instr.opcode.to_u32();

    if let InstructionKind::CustomData {
        subtype,
        values,
        raw_dword_count,
    } = &instr.kind
    {
        encode_custom_data(opcode_val, subtype, values, *raw_dword_count, out);
        return;
    }

    // Reserve a slot for the primary instruction token; we'll patch it later.
    let token0_idx = out.len();
    out.push(0);

    // Build extended opcode token for tex offsets if present.
    let has_extended = instr.tex_offsets.is_some();
    if let Some(offsets) = &instr.tex_offsets {
        let u = (offsets[0] as u32) & 0xF;
        let v = (offsets[1] as u32) & 0xF;
        let w = (offsets[2] as u32) & 0xF;
        // Type 1 = sample controls, bit 31 = 0 (no further extended tokens)
        out.push(1 | (u << 9) | (v << 13) | (w << 17));
    }

    // Encode instruction-kind-specific payload.
    encode_kind(instr, out);

    // Compute instruction length and build token0.
    let instr_len = (out.len() - token0_idx) as u32;
    let mut token0 = opcode_val & 0x7FF;
    if instr.saturate {
        token0 |= 1 << 13;
    }
    if let Some(rt) = instr.resinfo_return_type {
        token0 |= (rt & 0x3) << 11;
    }
    if has_extended {
        token0 |= 1 << 31;
    }
    token0 |= (instr_len & 0x7F) << 24;

    // Apply declaration-specific bits to token0.
    apply_token0_dcl_bits(instr, &mut token0);

    out[token0_idx] = token0;
}

/// Encode a `customdata` block. CustomData has a special two-dword header
/// format where the second dword is the total dword count (not in bits 24-30).
fn encode_custom_data(
    opcode_val: u32,
    subtype: &CustomDataType,
    values: &[[f32; 4]],
    raw_dword_count: usize,
    out: &mut Vec<u32>,
) {
    let subtype_val = match subtype {
        CustomDataType::Comment => 0u32,
        CustomDataType::DebugInfo => 1,
        CustomDataType::Opaque => 2,
        CustomDataType::ImmediateConstantBuffer => 3,
        CustomDataType::Other(v) => *v,
    };
    // Token0: opcode in bits [0:10], subtype in bits [11:15]
    out.push((opcode_val & 0x7FF) | (subtype_val << 11));
    // Token1: total dword count (header + data).
    // For ICB, the total is 2 (header) + values.len() * 4
    let total = if *subtype == CustomDataType::ImmediateConstantBuffer {
        2 + values.len() * 4
    } else {
        raw_dword_count
    };
    out.push(total as u32);
    // Emit ICB float data.
    for row in values {
        for val in row {
            out.push(val.to_bits());
        }
    }
}

/// Encode the instruction-kind-specific payload (operands and trailing data).
fn encode_kind(instr: &Instruction, out: &mut Vec<u32>) {
    match &instr.kind {
        InstructionKind::Generic { operands } => {
            encode_operands(operands, out);
        }
        InstructionKind::DclGlobalFlags { .. } => {
            // All bits are in token0 (handled by apply_token0_dcl_bits).
        }
        InstructionKind::DclInput {
            system_value,
            operands,
            ..
        } => {
            encode_operands(operands, out);
            if let Some(sv) = system_value {
                out.push(system_value_to_u32(sv));
            }
        }
        InstructionKind::DclOutput {
            system_value,
            operands,
        } => {
            encode_operands(operands, out);
            if let Some(sv) = system_value {
                out.push(system_value_to_u32(sv));
            }
        }
        InstructionKind::DclResource {
            return_type,
            operands,
            ..
        } => {
            encode_operands(operands, out);
            out.push(encode_return_type_token(return_type));
        }
        InstructionKind::DclSampler { operands, .. } => {
            encode_operands(operands, out);
        }
        InstructionKind::DclConstantBuffer { operands, .. } => {
            encode_operands(operands, out);
        }
        InstructionKind::DclTemps { count } => {
            out.push(*count);
        }
        InstructionKind::DclIndexableTemp {
            reg,
            size,
            components,
        } => {
            out.push(*reg);
            out.push(*size);
            out.push(*components);
        }
        InstructionKind::DclGsInputPrimitive { .. }
        | InstructionKind::DclGsOutputTopology { .. }
        | InstructionKind::DclOutputControlPointCount { .. }
        | InstructionKind::DclInputControlPointCount { .. }
        | InstructionKind::DclTessDomain { .. }
        | InstructionKind::DclTessPartitioning { .. }
        | InstructionKind::DclTessOutputPrimitive { .. } => {
            // All bits live in token0.
        }
        InstructionKind::DclMaxOutputVertexCount { count }
        | InstructionKind::DclGsInstanceCount { count }
        | InstructionKind::DclHsForkPhaseInstanceCount { count } => {
            out.push(*count);
        }
        InstructionKind::DclHsMaxTessFactor { value } => {
            out.push(value.to_bits());
        }
        InstructionKind::DclThreadGroup { x, y, z } => {
            out.push(*x);
            out.push(*y);
            out.push(*z);
        }
        InstructionKind::DclUavTyped {
            return_type,
            operands,
            ..
        } => {
            encode_operands(operands, out);
            out.push(encode_return_type_token(return_type));
        }
        InstructionKind::DclUavRaw { operands, .. } => {
            encode_operands(operands, out);
        }
        InstructionKind::DclUavStructured {
            operands, stride, ..
        } => {
            encode_operands(operands, out);
            out.push(*stride);
        }
        InstructionKind::DclResourceRaw { operands } => {
            encode_operands(operands, out);
        }
        InstructionKind::DclResourceStructured { stride, operands } => {
            encode_operands(operands, out);
            out.push(*stride);
        }
        InstructionKind::DclFunctionBody { index } => {
            out.push(*index);
        }
        InstructionKind::DclFunctionTable {
            table_index,
            body_indices,
        } => {
            out.push(*table_index);
            out.push(body_indices.len() as u32);
            out.extend_from_slice(body_indices);
        }
        InstructionKind::DclInterface {
            interface_index,
            num_call_sites,
            table_indices,
        } => {
            out.push(*interface_index);
            out.push(*num_call_sites);
            out.push(table_indices.len() as u32);
            out.extend_from_slice(table_indices);
        }
        InstructionKind::DclIndexRange { operands, count } => {
            encode_operands(operands, out);
            out.push(*count);
        }
        InstructionKind::HsPhase => {
            // No payload — opcode-only instruction.
        }
        InstructionKind::CustomData { .. } => {
            // Already handled in encode_instruction.
            unreachable!();
        }
    }
}

/// Set declaration-specific bits in token0 that the decoder extracts from
/// the primary instruction token rather than from trailing dwords.
fn apply_token0_dcl_bits(instr: &Instruction, token0: &mut u32) {
    match &instr.kind {
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
            ];
            let mut bits = 0u32;
            for flag in flags {
                if let Some(pos) = names.iter().position(|n| n == flag) {
                    bits |= 1 << pos;
                }
            }
            *token0 |= bits << 11;
        }
        InstructionKind::DclInput {
            interpolation: Some(interp),
            ..
        } => {
            let mode = match *interp {
                "undefined" => 0u32,
                "constant" => 1,
                "linear" => 2,
                "linearCentroid" => 3,
                "linearNoperspective" => 4,
                "linearNoperspectiveCentroid" => 5,
                "linearSample" => 6,
                "linearNoperspectiveSample" => 7,
                _ => 0,
            };
            *token0 |= mode << 11;
        }
        InstructionKind::DclInput { .. } => {}
        InstructionKind::DclResource { dimension, .. }
        | InstructionKind::DclUavTyped { dimension, .. } => {
            let dim = match *dimension {
                "buffer" => 1u32,
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
            };
            *token0 |= dim << 11;
        }
        InstructionKind::DclSampler { mode, .. } => {
            let m = match *mode {
                "default" => 0u32,
                "comparison" => 1,
                "mono" => 2,
                _ => 0,
            };
            *token0 |= m << 11;
        }
        InstructionKind::DclConstantBuffer { access, .. } => {
            if *access == "dynamicIndexed" {
                *token0 |= 1 << 11;
            }
        }
        InstructionKind::DclGsInputPrimitive { primitive } => {
            let prim = match *primitive {
                "point" => 1u32,
                "line" => 2,
                "triangle" => 3,
                "lineAdj" => 4,
                "triangleAdj" => 5,
                _ => 0,
            };
            *token0 |= prim << 11;
        }
        InstructionKind::DclGsOutputTopology { topology } => {
            let topo = match *topology {
                "pointlist" => 1u32,
                "linestrip" => 2,
                "trianglestrip" => 3,
                _ => 0,
            };
            *token0 |= topo << 11;
        }
        InstructionKind::DclOutputControlPointCount { count }
        | InstructionKind::DclInputControlPointCount { count } => {
            *token0 |= (count & 0x3F) << 11;
        }
        InstructionKind::DclTessDomain { domain } => {
            let d = match *domain {
                "undefined" => 0u32,
                "isoline" => 1,
                "tri" => 2,
                "quad" => 3,
                _ => 0,
            };
            *token0 |= d << 11;
        }
        InstructionKind::DclTessPartitioning { partitioning } => {
            let p = match *partitioning {
                "undefined" => 0u32,
                "integer" => 1,
                "pow2" => 2,
                "fractional_odd" => 3,
                "fractional_even" => 4,
                _ => 0,
            };
            *token0 |= p << 11;
        }
        InstructionKind::DclTessOutputPrimitive { primitive } => {
            let p = match *primitive {
                "undefined" => 0u32,
                "point" => 1,
                "line" => 2,
                "triangle_cw" => 3,
                "triangle_ccw" => 4,
                _ => 0,
            };
            *token0 |= p << 11;
        }
        InstructionKind::DclUavRaw { flags, .. } => {
            *token0 |= (flags & 0xFF) << 16;
        }
        InstructionKind::DclUavStructured { flags, .. } => {
            *token0 |= (flags & 0xFF) << 16;
        }
        _ => {}
    }
}

/// Encode a sequence of operands into the dword stream.
fn encode_operands(operands: &[Operand], out: &mut Vec<u32>) {
    for op in operands {
        encode_one_operand(op, out);
    }
}

/// Encode a single operand token and its index/immediate data.
fn encode_one_operand(op: &Operand, out: &mut Vec<u32>) {
    let op_type_val = op.reg_type.to_u32();
    let index_dim = op.indices.len() as u32;

    // Determine num_components and selection mode.
    let (num_components, sel_bits) = match &op.components {
        ComponentSelect::None => (0u32, 0u32),
        ComponentSelect::Mask(m) => {
            // N-component with mask mode (sel_mode=0).
            // Use num_components=2 for 4-component registers, 1 for 1-component.
            (2, (*m as u32) << 4)
        }
        ComponentSelect::Swizzle(s) => {
            let swiz =
                (s[0] as u32) | ((s[1] as u32) << 2) | ((s[2] as u32) << 4) | ((s[3] as u32) << 6);
            // sel_mode=1 (swizzle), num_components=2
            (2, (1u32 << 2) | (swiz << 4))
        }
        ComponentSelect::Scalar(c) => {
            // sel_mode=2 (scalar), num_components=2
            (2, (2u32 << 2) | ((*c as u32) << 4))
        }
    };

    let has_extended = op.negate || op.abs;

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
        };
        token |= repr << (22 + i * 3);
    }

    out.push(token);

    // Extended modifier token.
    if has_extended {
        let mut ext = 1u32; // type = 1 (extended operand modifier)
        if op.negate {
            ext |= 1 << 6;
        }
        if op.abs {
            ext |= 1 << 7;
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
            out.push(*v as u32);
            out.push((*v >> 32) as u32);
        }
        OperandIndex::Relative(sub) => encode_one_operand(sub, out),
        OperandIndex::RelativePlusImm(imm, sub) => {
            out.push(*imm);
            encode_one_operand(sub, out);
        }
    }
}

/// Pack four return types into a single dword.
fn encode_return_type_token(rt: &[ReturnType; 4]) -> u32 {
    rt[0].to_u32() | (rt[1].to_u32() << 4) | (rt[2].to_u32() << 8) | (rt[3].to_u32() << 12)
}

/// Convert a system value name string back to its `D3D10_SB_NAME` / `D3D11_SB_NAME` value.
fn system_value_to_u32(name: &str) -> u32 {
    match name {
        "undefined" => 0,
        "position" => 1,
        "clip_distance" => 2,
        "cull_distance" => 3,
        "render_target_array_index" => 4,
        "viewport_array_index" => 5,
        "vertex_id" => 6,
        "primitive_id" => 7,
        "instance_id" => 8,
        "is_front_face" => 9,
        "sample_index" => 10,
        "finalQuadUeq0EdgeTessFactor" => 11,
        "finalQuadVeq0EdgeTessFactor" => 12,
        "finalQuadUeq1EdgeTessFactor" => 13,
        "finalQuadVeq1EdgeTessFactor" => 14,
        "finalQuadUInsideTessFactor" => 15,
        "finalQuadVInsideTessFactor" => 16,
        "finalTriUeq0EdgeTessFactor" => 17,
        "finalTriVeq0EdgeTessFactor" => 18,
        "finalTriWeq0EdgeTessFactor" => 19,
        "finalTriInsideTessFactor" => 20,
        "finalLineDetailTessFactor" => 21,
        "finalLineDensityTessFactor" => 22,
        "target" => 23,
        "depth" => 24,
        "coverage" => 25,
        "depth_greater_equal" => 26,
        "depth_less_equal" => 27,
        "stencil_ref" => 64,
        "inner_coverage" => 65,
        _ => 0,
    }
}
