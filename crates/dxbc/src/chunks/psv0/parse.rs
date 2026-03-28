//! PSV0 chunk parser.

use alloc::vec::Vec;
use nostdio::{ReadLe, Seek, SeekFrom, SliceCursor};

use super::types::*;

const PSV_GS_MAX_STREAMS: usize = 4;

fn mask_dwords_from_vectors(vectors: u32) -> usize {
    ((vectors as usize) + 7) >> 3
}

fn io_table_dwords(input_vectors: u32, output_vectors: u32) -> usize {
    mask_dwords_from_vectors(output_vectors) * input_vectors as usize * 4
}

/// Parse the shader-kind-specific union from the first 16 bytes of RuntimeInfo.
fn parse_stage_info(c: &mut SliceCursor<'_>, shader_kind: u32) -> Option<ShaderStageInfo> {
    // The union occupies bytes 0..16 of PSVRuntimeInfo0, but shader_kind
    // itself is NOT in the union — it's in RuntimeInfo1.ShaderStage (byte 24).
    // For v0, shader_kind is inferred from the DXBC shader type.
    // The union bytes are the first 16 bytes of RuntimeInfo0.
    let b0 = c.read_u32_le().ok()?;
    let b4 = c.read_u32_le().ok()?;
    let b8 = c.read_u32_le().ok()?;
    let b12 = c.read_u32_le().ok()?;

    let kind = PSVShaderKind::from_u8(shader_kind as u8);
    Some(match kind {
        PSVShaderKind::Vertex => ShaderStageInfo::Vertex {
            output_position_present: (b0 & 0xFF) != 0,
        },
        PSVShaderKind::Hull => ShaderStageInfo::Hull {
            input_control_point_count: b0,
            output_control_point_count: b4,
            tessellator_domain: b8,
            tessellator_output_primitive: b12,
        },
        PSVShaderKind::Domain => ShaderStageInfo::Domain {
            input_control_point_count: b0,
            output_position_present: (b4 & 0xFF) != 0,
            tessellator_domain: b8,
        },
        PSVShaderKind::Geometry => ShaderStageInfo::Geometry {
            input_primitive: b0,
            output_topology: b4,
            output_stream_mask: b8,
            output_position_present: (b12 & 0xFF) != 0,
        },
        PSVShaderKind::Pixel => ShaderStageInfo::Pixel {
            depth_output: (b0 & 0xFF) != 0,
            sample_frequency: ((b0 >> 8) & 0xFF) != 0,
        },
        PSVShaderKind::Mesh => ShaderStageInfo::Mesh {
            group_shared_bytes_used: b0,
            group_shared_bytes_dependent_on_view_id: b4,
            payload_size_in_bytes: b8,
            max_output_vertices: (b12 & 0xFFFF) as u16,
            max_output_primitives: (b12 >> 16) as u16,
        },
        PSVShaderKind::Amplification => ShaderStageInfo::Amplification {
            payload_size_in_bytes: b0,
        },
        PSVShaderKind::Compute => ShaderStageInfo::Compute,
        _ => {
            let mut raw = [0u8; 16];
            raw[0..4].copy_from_slice(&b0.to_le_bytes());
            raw[4..8].copy_from_slice(&b4.to_le_bytes());
            raw[8..12].copy_from_slice(&b8.to_le_bytes());
            raw[12..16].copy_from_slice(&b12.to_le_bytes());
            ShaderStageInfo::Other { raw }
        }
    })
}

fn read_u32s(c: &mut SliceCursor<'_>, count: usize) -> Option<Vec<u32>> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(c.read_u32_le().ok()?);
    }

    Some(v)
}

fn parse_sig_element(c: &mut SliceCursor<'_>, stride: u32) -> Option<PSVSignatureElement> {
    let base_pos = c.position();
    let semantic_name = c.read_u32_le().ok()?;
    let semantic_indexes = c.read_u32_le().ok()?;
    let rows = c.read_u8_le().ok()?;
    let start_row = c.read_u8_le().ok()?;
    let cols_and_start = c.read_u8_le().ok()?;
    let semantic_kind = c.read_u8_le().ok()?;
    let component_type = c.read_u8_le().ok()?;
    let interpolation_mode = c.read_u8_le().ok()?;
    let dynamic_mask_and_stream = c.read_u8_le().ok()?;
    let reserved = c.read_u8_le().ok()?;
    // Skip any extra bytes in stride
    let consumed = (c.position() - base_pos) as u32;
    if stride > consumed {
        c.seek(SeekFrom::Current((stride - consumed) as i64)).ok()?;
    }
    Some(PSVSignatureElement {
        semantic_name,
        semantic_indexes,
        rows,
        start_row,
        cols_and_start,
        semantic_kind,
        component_type,
        interpolation_mode,
        dynamic_mask_and_stream,
        reserved,
    })
}

/// Parse a PSV0 chunk from raw bytes.
pub fn parse_psv0(data: &[u8]) -> Option<PipelineStateValidation> {
    if data.len() < 4 {
        return None;
    }
    let mut c = SliceCursor::new(data);

    let info_size = c.read_u32_le().ok()?;
    if data.len() < 4 + info_size as usize || info_size < 24 {
        return None;
    }

    // The runtime info starts at offset 4.
    let info_start = c.position();
    // For v0, the shader kind comes from outside (container); inside RuntimeInfo0
    // it's just the union. For v1+, ShaderStage is at offset 24 within the struct.
    // We'll read the v1 shader_stage later, but we need a "hint" to parse the union.
    // For now, read as unknown and re-parse if we get v1 info.

    // Read the 16-byte union (advances cursor; values parsed via parse_stage_info)
    let _union_b0 = c.read_u32_le().ok()?;
    let _union_b4 = c.read_u32_le().ok()?;
    let _union_b8 = c.read_u32_le().ok()?;
    let _union_b12 = c.read_u32_le().ok()?;

    // MinimumExpectedWaveLaneCount, MaximumExpectedWaveLaneCount
    let min_wave_lane_count = c.read_u32_le().ok()?;
    let max_wave_lane_count = c.read_u32_le().ok()?;
    // We've now consumed 24 bytes of RuntimeInfo0.

    // Parse v1+ fields
    let runtime_info_1 = if info_size >= 36 {
        let shader_stage = c.read_u8_le().ok()?;
        let uses_view_id = c.read_u8_le().ok()?;
        let max_vert_or_patch_prim = c.read_u16_le().ok()?;
        let sig_input_elements = c.read_u8_le().ok()?;
        let sig_output_elements = c.read_u8_le().ok()?;
        let sig_patch_const_or_prim_elements = c.read_u8_le().ok()?;
        let sig_input_vectors = c.read_u8_le().ok()?;
        let mut sig_output_vectors = [0u8; 4];
        for v in &mut sig_output_vectors {
            *v = c.read_u8_le().ok()?;
        }
        Some(PSVRuntimeInfo1 {
            shader_stage,
            uses_view_id,
            max_vert_or_patch_prim,
            sig_input_elements,
            sig_output_elements,
            sig_patch_const_or_prim_elements,
            sig_input_vectors,
            sig_output_vectors,
        })
    } else {
        None
    };

    // Now parse the stage info union using the shader kind from v1 (or 0xFF for v0)
    let shader_kind_hint = runtime_info_1
        .as_ref()
        .map(|ri| ri.shader_stage as u32)
        .unwrap_or(0xFF);

    // Re-interpret the union bytes
    let stage_info = {
        let mut uc = SliceCursor::new(data);
        uc.seek(SeekFrom::Start(info_start as u64)).ok()?;
        parse_stage_info(&mut uc, shader_kind_hint)?
    };

    let runtime_info_2 = if info_size >= 48 {
        let x = c.read_u32_le().ok()?;
        let y = c.read_u32_le().ok()?;
        let z = c.read_u32_le().ok()?;
        Some(PSVRuntimeInfo2 {
            num_threads_x: x,
            num_threads_y: y,
            num_threads_z: z,
        })
    } else {
        None
    };

    let entry_function_name = if info_size >= 52 {
        Some(c.read_u32_le().ok()?)
    } else {
        None
    };

    let num_bytes_group_shared_memory = if info_size >= 56 {
        Some(c.read_u32_le().ok()?)
    } else {
        None
    };

    // Skip any remaining bytes in the info struct we don't know about.
    c.seek(SeekFrom::Start((4 + info_size as usize) as u64))
        .ok()?;

    // Resource count
    let resource_count = c.read_u32_le().ok()? as usize;
    let mut resource_bind_info_size = 0u32;
    let mut resources = Vec::with_capacity(resource_count);
    if resource_count > 0 {
        resource_bind_info_size = c.read_u32_le().ok()?;
        for _ in 0..resource_count {
            let rbase = c.position();
            let res_type = c.read_u32_le().ok()?;
            let space = c.read_u32_le().ok()?;
            let lower_bound = c.read_u32_le().ok()?;
            let upper_bound = c.read_u32_le().ok()?;
            let (res_kind, res_flags) = if resource_bind_info_size >= 24 {
                (Some(c.read_u32_le().ok()?), Some(c.read_u32_le().ok()?))
            } else {
                (None, None)
            };
            // Skip any extra bytes in the bind info struct.
            let consumed = (c.position() - rbase) as u32;
            if resource_bind_info_size > consumed {
                c.seek(SeekFrom::Current(
                    (resource_bind_info_size - consumed) as i64,
                ))
                .ok()?;
            }
            resources.push(PSVResourceBindInfo {
                res_type,
                space,
                lower_bound,
                upper_bound,
                res_kind,
                res_flags,
            });
        }
    }

    // The rest is v1+ only.
    let mut string_table = Vec::new();
    let mut semantic_index_table = Vec::new();
    let mut sig_element_size = 0u32;
    let mut sig_input_elements = Vec::new();
    let mut sig_output_elements = Vec::new();
    let mut sig_patch_const_or_prim_elements = Vec::new();
    let mut view_id_output_masks: Vec<Vec<u32>> = Vec::new();
    let mut view_id_pc_or_prim_output_mask = Vec::new();
    let mut input_to_output_tables: Vec<Vec<u32>> = Vec::new();
    let mut input_to_pc_output_table = Vec::new();
    let mut pc_input_to_output_table = Vec::new();

    if let Some(ref ri1) = runtime_info_1 {
        // String table
        let st_size = c.read_u32_le().ok()? as usize;
        if st_size > 0 {
            let pos = c.position();
            if pos + st_size > data.len() {
                return None;
            }
            string_table = Vec::from(&data[pos..pos + st_size]);
            c.seek(SeekFrom::Current(st_size as i64)).ok()?;
        }

        // Semantic index table
        let si_entries = c.read_u32_le().ok()? as usize;
        if si_entries > 0 {
            semantic_index_table = read_u32s(&mut c, si_entries)?;
        }

        // Signature elements
        let total_sig = ri1.sig_input_elements as usize
            + ri1.sig_output_elements as usize
            + ri1.sig_patch_const_or_prim_elements as usize;
        if total_sig > 0 {
            sig_element_size = c.read_u32_le().ok()?;
            for _ in 0..ri1.sig_input_elements {
                sig_input_elements.push(parse_sig_element(&mut c, sig_element_size)?);
            }
            for _ in 0..ri1.sig_output_elements {
                sig_output_elements.push(parse_sig_element(&mut c, sig_element_size)?);
            }
            for _ in 0..ri1.sig_patch_const_or_prim_elements {
                sig_patch_const_or_prim_elements.push(parse_sig_element(&mut c, sig_element_size)?);
            }
        }

        let shader_kind = PSVShaderKind::from_u8(shader_kind_hint as u8);
        let is_gs = shader_kind == PSVShaderKind::Geometry;
        let is_hs = shader_kind == PSVShaderKind::Hull;
        let is_ds = shader_kind == PSVShaderKind::Domain;
        let is_ms = shader_kind == PSVShaderKind::Mesh;
        let patch_const_vectors = (ri1.max_vert_or_patch_prim & 0xFF) as u32;

        // ViewID output masks
        if ri1.uses_view_id != 0 {
            let stream_count = if is_gs { PSV_GS_MAX_STREAMS } else { 1 };
            for i in 0..stream_count {
                let ov = ri1.sig_output_vectors[i] as u32;
                if ov > 0 {
                    let dwords = mask_dwords_from_vectors(ov);
                    view_id_output_masks.push(read_u32s(&mut c, dwords)?);
                } else {
                    view_id_output_masks.push(Vec::new());
                }
                if !is_gs {
                    break;
                }
            }
            if (is_hs || is_ms) && patch_const_vectors > 0 {
                let dwords = mask_dwords_from_vectors(patch_const_vectors);
                view_id_pc_or_prim_output_mask = read_u32s(&mut c, dwords)?;
            }
        }

        // Input-to-output dependency tables
        let stream_count = if is_gs { PSV_GS_MAX_STREAMS } else { 1 };
        for i in 0..stream_count {
            let ov = ri1.sig_output_vectors[i] as u32;
            let iv = ri1.sig_input_vectors as u32;
            if !is_ms && ov > 0 && iv > 0 {
                let dwords = io_table_dwords(iv, ov);
                input_to_output_tables.push(read_u32s(&mut c, dwords)?);
            } else {
                input_to_output_tables.push(Vec::new());
            }
            if !is_gs {
                break;
            }
        }

        // HS: input-to-patch-constant table
        if is_hs && patch_const_vectors > 0 && ri1.sig_input_vectors > 0 {
            let dwords = io_table_dwords(ri1.sig_input_vectors as u32, patch_const_vectors);
            input_to_pc_output_table = read_u32s(&mut c, dwords)?;
        }

        // DS: patch-constant-to-output table
        if is_ds && ri1.sig_output_vectors[0] > 0 && patch_const_vectors > 0 {
            let dwords = io_table_dwords(patch_const_vectors, ri1.sig_output_vectors[0] as u32);
            pc_input_to_output_table = read_u32s(&mut c, dwords)?;
        }
    }

    Some(PipelineStateValidation {
        info_size,
        stage_info,
        min_wave_lane_count,
        max_wave_lane_count,
        runtime_info_1,
        runtime_info_2,
        entry_function_name,
        num_bytes_group_shared_memory,
        resource_bind_info_size,
        resources,
        string_table,
        semantic_index_table,
        sig_element_size,
        sig_input_elements,
        sig_output_elements,
        sig_patch_const_or_prim_elements,
        view_id_output_masks,
        view_id_pc_or_prim_output_mask,
        input_to_output_tables,
        input_to_pc_output_table,
        pc_input_to_output_table,
    })
}
