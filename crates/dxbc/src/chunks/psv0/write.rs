//! PSV0 chunk writer — reconstruct PSV0 binary from parsed fields.

use alloc::vec::Vec;

use super::types::*;

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
        } => {
            write_u32(out, if *output_position_present { 1 } else { 0 });
            write_u32(out, 0);
            write_u32(out, 0);
            write_u32(out, 0);
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
            tessellator_domain,
        } => {
            write_u32(out, *input_control_point_count);
            write_u32(out, if *output_position_present { 1 } else { 0 });
            write_u32(out, *tessellator_domain);
            write_u32(out, 0);
        }
        ShaderStageInfo::Geometry {
            input_primitive,
            output_topology,
            output_stream_mask,
            output_position_present,
        } => {
            write_u32(out, *input_primitive);
            write_u32(out, *output_topology);
            write_u32(out, *output_stream_mask);
            write_u32(out, if *output_position_present { 1 } else { 0 });
        }
        ShaderStageInfo::Pixel {
            depth_output,
            sample_frequency,
        } => {
            let b = (if *depth_output { 1u32 } else { 0 })
                | (if *sample_frequency { 1u32 } else { 0 }) << 8;
            write_u32(out, b);
            write_u32(out, 0);
            write_u32(out, 0);
            write_u32(out, 0);
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
                (*max_output_vertices as u32) | ((*max_output_primitives as u32) << 16),
            );
        }
        ShaderStageInfo::Amplification {
            payload_size_in_bytes,
        } => {
            write_u32(out, *payload_size_in_bytes);
            write_u32(out, 0);
            write_u32(out, 0);
            write_u32(out, 0);
        }
        ShaderStageInfo::Compute => {
            write_u32(out, 0);
            write_u32(out, 0);
            write_u32(out, 0);
            write_u32(out, 0);
        }
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
    // Pad to stride
    let written = out.len() - start;
    for _ in written..(stride as usize) {
        out.push(0);
    }
}

/// Reconstruct a PSV0 chunk payload from parsed fields.
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
        write_u16(&mut out, ri1.max_vert_or_patch_prim);
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

    // Pad info struct to info_size (in case future versions have more)
    let info_written = out.len() - 4; // subtract the info_size u32 itself
    if (info_written as u32) < psv.info_size {
        let pad = psv.info_size as usize - info_written;
        out.resize(out.len() + pad, 0);
    }

    // Resource count
    write_u32(&mut out, psv.resources.len() as u32);
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
            // Pad to resource_bind_info_size
            let written = out.len() - rstart;
            if written < psv.resource_bind_info_size as usize {
                out.resize(
                    out.len() + (psv.resource_bind_info_size as usize - written),
                    0,
                );
            }
        }
    }

    // v1+ sections
    if let Some(ref ri1) = psv.runtime_info_1 {
        // String table
        write_u32(&mut out, psv.string_table.len() as u32);
        if !psv.string_table.is_empty() {
            out.extend_from_slice(&psv.string_table);
        }

        // Semantic index table
        write_u32(&mut out, psv.semantic_index_table.len() as u32);
        for &idx in &psv.semantic_index_table {
            write_u32(&mut out, idx);
        }

        // Signature elements
        let total_sig = psv.sig_input_elements.len()
            + psv.sig_output_elements.len()
            + psv.sig_patch_const_or_prim_elements.len();
        if total_sig > 0 {
            write_u32(&mut out, psv.sig_element_size);
            for e in &psv.sig_input_elements {
                write_sig_element(&mut out, e, psv.sig_element_size);
            }
            for e in &psv.sig_output_elements {
                write_sig_element(&mut out, e, psv.sig_element_size);
            }
            for e in &psv.sig_patch_const_or_prim_elements {
                write_sig_element(&mut out, e, psv.sig_element_size);
            }
        }

        // ViewID output masks
        if ri1.uses_view_id != 0 {
            for mask in &psv.view_id_output_masks {
                for &v in mask {
                    write_u32(&mut out, v);
                }
            }
            for &v in &psv.view_id_pc_or_prim_output_mask {
                write_u32(&mut out, v);
            }
        }

        // Input-to-output dependency tables
        for table in &psv.input_to_output_tables {
            for &v in table {
                write_u32(&mut out, v);
            }
        }

        // HS: input-to-patch-constant table
        for &v in &psv.input_to_pc_output_table {
            write_u32(&mut out, v);
        }

        // DS: patch-constant-to-output table
        for &v in &psv.pc_input_to_output_table {
            write_u32(&mut out, v);
        }
    }

    out
}
