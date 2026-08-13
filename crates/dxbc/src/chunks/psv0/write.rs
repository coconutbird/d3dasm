//! PSV0 chunk writer — reconstruct PSV0 binary from parsed fields.

use alloc::vec::Vec;

use super::types::{
    PSVRuntimeInfo1Union, PSVSignatureElement, PipelineStateValidation, ShaderStageInfo,
};

fn write_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u16(out: &mut Vec<u8>, v: u16) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn write_u8(out: &mut Vec<u8>, v: u8) {
    out.push(v);
}

fn write_stage_info(out: &mut Vec<u8>, info: &ShaderStageInfo) {
    match info {
        ShaderStageInfo::Vertex {
            output_position_present,
            padding,
        } => {
            write_u8(out, *output_position_present);
            out.extend_from_slice(padding);
        }
        ShaderStageInfo::Hull {
            input_control_point_count,
            output_control_point_count,
            tessellator_domain,
            tessellator_output_primitive,
        } => {
            write_u32(out, *input_control_point_count);
            write_u32(out, *output_control_point_count);
            write_u32(out, *tessellator_domain);
            write_u32(out, *tessellator_output_primitive);
        }
        ShaderStageInfo::Domain {
            input_control_point_count,
            output_position_present,
            field_padding,
            tessellator_domain,
            union_padding,
        } => {
            write_u32(out, *input_control_point_count);
            write_u8(out, *output_position_present);
            out.extend_from_slice(field_padding);
            write_u32(out, *tessellator_domain);
            out.extend_from_slice(union_padding);
        }
        ShaderStageInfo::Geometry {
            input_primitive,
            output_topology,
            output_stream_mask,
            output_position_present,
            padding,
        } => {
            write_u32(out, *input_primitive);
            write_u32(out, *output_topology);
            write_u32(out, *output_stream_mask);
            write_u8(out, *output_position_present);
            out.extend_from_slice(padding);
        }
        ShaderStageInfo::Pixel {
            depth_output,
            sample_frequency,
            padding,
        } => {
            write_u8(out, *depth_output);
            write_u8(out, *sample_frequency);
            out.extend_from_slice(padding);
        }
        ShaderStageInfo::Mesh {
            group_shared_bytes_used,
            group_shared_bytes_dependent_on_view_id,
            payload_size_in_bytes,
            max_output_vertices,
            max_output_primitives,
        } => {
            write_u32(out, *group_shared_bytes_used);
            write_u32(out, *group_shared_bytes_dependent_on_view_id);
            write_u32(out, *payload_size_in_bytes);
            write_u32(
                out,
                u32::from(*max_output_vertices) | (u32::from(*max_output_primitives) << 16),
            );
        }
        ShaderStageInfo::Amplification {
            payload_size_in_bytes,
            padding,
        } => {
            write_u32(out, *payload_size_in_bytes);
            out.extend_from_slice(padding);
        }
        ShaderStageInfo::Compute { raw } => out.extend_from_slice(raw),
        ShaderStageInfo::Other { raw } => {
            out.extend_from_slice(raw);
        }
    }
}

fn write_sig_element(out: &mut Vec<u8>, e: &PSVSignatureElement, stride: u32) {
    let start = out.len();
    write_u32(out, e.semantic_name);
    write_u32(out, e.semantic_indexes);
    write_u8(out, e.rows);
    write_u8(out, e.start_row);
    write_u8(out, e.cols_and_start);
    write_u8(out, e.semantic_kind);
    write_u8(out, e.component_type);
    write_u8(out, e.interpolation_mode);
    write_u8(out, e.dynamic_mask_and_stream);
    write_u8(out, e.reserved);
    out.extend_from_slice(&e.extra);
    let end = start
        .checked_add(usize::try_from(stride).expect("PSV signature stride fits in usize"))
        .expect("PSV signature stride overflows usize");
    assert!(out.len() <= end, "PSV signature element exceeds its stride");
    out.resize(end, 0);
}

fn write_versioned_sections(out: &mut Vec<u8>, psv: &PipelineStateValidation) {
    let Some(ref runtime_info) = psv.runtime_info_1 else {
        return;
    };

    write_u32(
        out,
        u32::try_from(psv.string_table.len()).expect("PSV string table exceeds u32"),
    );
    out.extend_from_slice(&psv.string_table);

    write_u32(
        out,
        u32::try_from(psv.semantic_index_table.len())
            .expect("PSV semantic index table exceeds u32"),
    );
    for &index in &psv.semantic_index_table {
        write_u32(out, index);
    }

    let signature_count = psv.sig_input_elements.len()
        + psv.sig_output_elements.len()
        + psv.sig_patch_const_or_prim_elements.len();
    if signature_count > 0 {
        write_u32(out, psv.sig_element_size);
        for element in &psv.sig_input_elements {
            write_sig_element(out, element, psv.sig_element_size);
        }
        for element in &psv.sig_output_elements {
            write_sig_element(out, element, psv.sig_element_size);
        }
        for element in &psv.sig_patch_const_or_prim_elements {
            write_sig_element(out, element, psv.sig_element_size);
        }
    }

    if runtime_info.uses_view_id != 0 {
        for mask in &psv.view_id_output_masks {
            for &value in mask {
                write_u32(out, value);
            }
        }
        for &value in &psv.view_id_pc_or_prim_output_mask {
            write_u32(out, value);
        }
    }

    for table in &psv.input_to_output_tables {
        for &value in table {
            write_u32(out, value);
        }
    }
    for &value in &psv.input_to_pc_output_table {
        write_u32(out, value);
    }
    for &value in &psv.pc_input_to_output_table {
        write_u32(out, value);
    }
}

/// Reconstruct a PSV0 chunk payload from parsed fields.
///
/// # Panics
///
/// Panics if a collection length or encoded section size exceeds its 32-bit
/// PSV field.
#[must_use]
pub fn write_psv0(psv: &PipelineStateValidation) -> Vec<u8> {
    let mut out = Vec::new();

    // info_size
    write_u32(&mut out, psv.info_size);

    // RuntimeInfo0: union (16 bytes) + wave lane counts (8 bytes) = 24 bytes
    write_stage_info(&mut out, &psv.stage_info);
    write_u32(&mut out, psv.min_wave_lane_count);
    write_u32(&mut out, psv.max_wave_lane_count);

    // RuntimeInfo1 (12 bytes)
    if let Some(ref ri1) = psv.runtime_info_1 {
        write_u8(&mut out, ri1.shader_stage);
        write_u8(&mut out, ri1.uses_view_id);
        let raw_union = match ri1.stage_union {
            PSVRuntimeInfo1Union::MaxVertexCount(value) | PSVRuntimeInfo1Union::Reserved(value) => {
                value
            }
            PSVRuntimeInfo1Union::PatchConstantVectors { vectors, padding } => {
                u16::from(vectors) | (u16::from(padding) << 8)
            }
            PSVRuntimeInfo1Union::Mesh {
                signature_primitive_vectors,
                output_topology,
            } => u16::from(signature_primitive_vectors) | (u16::from(output_topology) << 8),
        };
        write_u16(&mut out, raw_union);
        write_u8(&mut out, ri1.sig_input_elements);
        write_u8(&mut out, ri1.sig_output_elements);
        write_u8(&mut out, ri1.sig_patch_const_or_prim_elements);
        write_u8(&mut out, ri1.sig_input_vectors);
        for &v in &ri1.sig_output_vectors {
            write_u8(&mut out, v);
        }
    }

    // RuntimeInfo2 (12 bytes)
    if let Some(ref ri2) = psv.runtime_info_2 {
        write_u32(&mut out, ri2.num_threads_x);
        write_u32(&mut out, ri2.num_threads_y);
        write_u32(&mut out, ri2.num_threads_z);
    }

    // v3: entry function name
    if let Some(efn) = psv.entry_function_name {
        write_u32(&mut out, efn);
    }

    // v4: group shared memory bytes
    if let Some(gsm) = psv.num_bytes_group_shared_memory {
        write_u32(&mut out, gsm);
    }

    out.extend_from_slice(&psv.runtime_info_extra);
    let runtime_end = 4usize
        .checked_add(usize::try_from(psv.info_size).expect("PSV info size fits in usize"))
        .expect("PSV info size overflows usize");
    assert!(
        out.len() <= runtime_end,
        "PSV runtime info exceeds info_size"
    );
    out.resize(runtime_end, 0);

    // Resource count
    write_u32(
        &mut out,
        u32::try_from(psv.resources.len()).expect("PSV resource count exceeds u32"),
    );
    if !psv.resources.is_empty() {
        write_u32(&mut out, psv.resource_bind_info_size);
        for r in &psv.resources {
            let rstart = out.len();
            write_u32(&mut out, r.res_type);
            write_u32(&mut out, r.space);
            write_u32(&mut out, r.lower_bound);
            write_u32(&mut out, r.upper_bound);
            if let Some(kind) = r.res_kind {
                write_u32(&mut out, kind);
            }
            if let Some(flags) = r.res_flags {
                write_u32(&mut out, flags);
            }
            out.extend_from_slice(&r.extra);
            let rend = rstart
                .checked_add(
                    usize::try_from(psv.resource_bind_info_size)
                        .expect("PSV resource stride fits in usize"),
                )
                .expect("PSV resource stride overflows usize");
            assert!(out.len() <= rend, "PSV resource binding exceeds its stride");
            out.resize(rend, 0);
        }
    }

    write_versioned_sections(&mut out, psv);

    out
}
