//! PSV0 chunk parser.

use alloc::vec::Vec;
use nostdio::{Cursor, ReadLe, Seek, SeekFrom};

use super::types::{
    PSVResourceBindInfo, PSVRuntimeInfo1, PSVRuntimeInfo2, PSVShaderKind, PSVSignatureElement,
    PipelineStateValidation, ShaderStageInfo,
};

const PSV_GS_MAX_STREAMS: usize = 4;

fn mask_dwords_from_vectors(vectors: u32) -> usize {
    ((vectors as usize) + 7) >> 3
}

fn io_table_dwords(input_vectors: u32, output_vectors: u32) -> usize {
    mask_dwords_from_vectors(output_vectors) * input_vectors as usize * 4
}

/// Parse the shader-kind-specific union from the first 16 bytes of `RuntimeInfo`.
fn parse_stage_info(c: &mut Cursor<&[u8]>, shader_kind: u32) -> Option<ShaderStageInfo> {
    // The union occupies bytes 0..16 of PSVRuntimeInfo0, but shader_kind
    // itself is NOT in the union — it's in RuntimeInfo1.ShaderStage (byte 24).
    // For v0, shader_kind is inferred from the DXBC shader type.
    // The union bytes are the first 16 bytes of RuntimeInfo0.
    let b0 = c.read_u32_le().ok()?;
    let b4 = c.read_u32_le().ok()?;
    let b8 = c.read_u32_le().ok()?;
    let b12 = c.read_u32_le().ok()?;

    let kind = PSVShaderKind::from_u8(u8::try_from(shader_kind).ok()?);
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

fn read_u32s(c: &mut Cursor<&[u8]>, count: usize) -> Option<Vec<u32>> {
    let mut v = Vec::with_capacity(count);
    for _ in 0..count {
        v.push(c.read_u32_le().ok()?);
    }

    Some(v)
}

fn parse_sig_element(c: &mut Cursor<&[u8]>, stride: u32) -> Option<PSVSignatureElement> {
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
    let consumed = u32::try_from(c.position().checked_sub(base_pos)?).ok()?;
    if stride > consumed {
        c.seek(SeekFrom::Current(i64::from(stride - consumed)))
            .ok()?;
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

struct RuntimeInfo {
    info_size: u32,
    stage_info: ShaderStageInfo,
    min_wave_lane_count: u32,
    max_wave_lane_count: u32,
    version_1: Option<PSVRuntimeInfo1>,
    version_2: Option<PSVRuntimeInfo2>,
    entry_function_name: Option<u32>,
    num_bytes_group_shared_memory: Option<u32>,
}

struct ResourceInfo {
    bind_info_size: u32,
    resources: Vec<PSVResourceBindInfo>,
}

#[derive(Default)]
struct VersionedSections {
    string_table: Vec<u8>,
    semantic_index_table: Vec<u32>,
    sig_element_size: u32,
    sig_input_elements: Vec<PSVSignatureElement>,
    sig_output_elements: Vec<PSVSignatureElement>,
    sig_patch_const_or_prim_elements: Vec<PSVSignatureElement>,
    view_id_output_masks: Vec<Vec<u32>>,
    view_id_pc_or_prim_output_mask: Vec<u32>,
    input_to_output_tables: Vec<Vec<u32>>,
    input_to_pc_output_table: Vec<u32>,
    pc_input_to_output_table: Vec<u32>,
}

#[derive(Default)]
struct DependencyTables {
    view_id_output_masks: Vec<Vec<u32>>,
    view_id_pc_or_prim_output_mask: Vec<u32>,
    input_to_output_tables: Vec<Vec<u32>>,
    input_to_pc_output_table: Vec<u32>,
    pc_input_to_output_table: Vec<u32>,
}

fn parse_runtime_info_1(cursor: &mut Cursor<&[u8]>) -> Option<PSVRuntimeInfo1> {
    let shader_stage = cursor.read_u8_le().ok()?;
    let uses_view_id = cursor.read_u8_le().ok()?;
    let max_vert_or_patch_prim = cursor.read_u16_le().ok()?;
    let sig_input_elements = cursor.read_u8_le().ok()?;
    let sig_output_elements = cursor.read_u8_le().ok()?;
    let sig_patch_const_or_prim_elements = cursor.read_u8_le().ok()?;
    let sig_input_vectors = cursor.read_u8_le().ok()?;
    let mut sig_output_vectors = [0u8; 4];
    for value in &mut sig_output_vectors {
        *value = cursor.read_u8_le().ok()?;
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
}

fn parse_runtime_info(data: &[u8], cursor: &mut Cursor<&[u8]>) -> Option<RuntimeInfo> {
    let info_size = cursor.read_u32_le().ok()?;
    if data.len() < 4 + info_size as usize || info_size < 24 {
        return None;
    }
    let info_start = cursor.position();
    for _ in 0..4 {
        cursor.read_u32_le().ok()?;
    }
    let min_wave_lane_count = cursor.read_u32_le().ok()?;
    let max_wave_lane_count = cursor.read_u32_le().ok()?;
    let version_1 = if info_size >= 36 {
        Some(parse_runtime_info_1(cursor)?)
    } else {
        None
    };
    let shader_kind = version_1
        .as_ref()
        .map_or(0xFF, |info| u32::from(info.shader_stage));
    let stage_info = {
        let mut union_cursor = Cursor::new(data);
        union_cursor.seek(SeekFrom::Start(info_start)).ok()?;
        parse_stage_info(&mut union_cursor, shader_kind)?
    };
    let version_2 = if info_size >= 48 {
        Some(PSVRuntimeInfo2 {
            num_threads_x: cursor.read_u32_le().ok()?,
            num_threads_y: cursor.read_u32_le().ok()?,
            num_threads_z: cursor.read_u32_le().ok()?,
        })
    } else {
        None
    };
    let entry_function_name = if info_size >= 52 {
        Some(cursor.read_u32_le().ok()?)
    } else {
        None
    };
    let num_bytes_group_shared_memory = if info_size >= 56 {
        Some(cursor.read_u32_le().ok()?)
    } else {
        None
    };
    cursor
        .seek(SeekFrom::Start(u64::from(info_size) + 4))
        .ok()?;
    Some(RuntimeInfo {
        info_size,
        stage_info,
        min_wave_lane_count,
        max_wave_lane_count,
        version_1,
        version_2,
        entry_function_name,
        num_bytes_group_shared_memory,
    })
}

fn parse_resources(cursor: &mut Cursor<&[u8]>) -> Option<ResourceInfo> {
    let resource_count = cursor.read_u32_le().ok()? as usize;
    let mut bind_info_size = 0;
    let mut resources = Vec::with_capacity(resource_count);
    if resource_count > 0 {
        bind_info_size = cursor.read_u32_le().ok()?;
        for _ in 0..resource_count {
            let base_position = cursor.position();
            let res_type = cursor.read_u32_le().ok()?;
            let space = cursor.read_u32_le().ok()?;
            let lower_bound = cursor.read_u32_le().ok()?;
            let upper_bound = cursor.read_u32_le().ok()?;
            let (res_kind, res_flags) = if bind_info_size >= 24 {
                (
                    Some(cursor.read_u32_le().ok()?),
                    Some(cursor.read_u32_le().ok()?),
                )
            } else {
                (None, None)
            };
            let consumed = u32::try_from(cursor.position().checked_sub(base_position)?).ok()?;
            if bind_info_size > consumed {
                cursor
                    .seek(SeekFrom::Current(i64::from(bind_info_size - consumed)))
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
    Some(ResourceInfo {
        bind_info_size,
        resources,
    })
}

fn parse_dependency_tables(
    cursor: &mut Cursor<&[u8]>,
    runtime_info: &PSVRuntimeInfo1,
) -> Option<DependencyTables> {
    let shader_kind = PSVShaderKind::from_u8(runtime_info.shader_stage);
    let is_geometry = shader_kind == PSVShaderKind::Geometry;
    let is_hull = shader_kind == PSVShaderKind::Hull;
    let is_domain = shader_kind == PSVShaderKind::Domain;
    let is_mesh = shader_kind == PSVShaderKind::Mesh;
    let patch_vectors = u32::from(runtime_info.max_vert_or_patch_prim & 0xFF);
    let stream_count = if is_geometry { PSV_GS_MAX_STREAMS } else { 1 };
    let mut tables = DependencyTables::default();

    if runtime_info.uses_view_id != 0 {
        for &output_vectors in &runtime_info.sig_output_vectors[..stream_count] {
            let dword_count = mask_dwords_from_vectors(u32::from(output_vectors));
            tables
                .view_id_output_masks
                .push(read_u32s(cursor, dword_count)?);
        }
        if (is_hull || is_mesh) && patch_vectors > 0 {
            let dword_count = mask_dwords_from_vectors(patch_vectors);
            tables.view_id_pc_or_prim_output_mask = read_u32s(cursor, dword_count)?;
        }
    }

    for &output_vectors in &runtime_info.sig_output_vectors[..stream_count] {
        let input_vectors = u32::from(runtime_info.sig_input_vectors);
        let output_vectors = u32::from(output_vectors);
        if !is_mesh && output_vectors > 0 && input_vectors > 0 {
            let count = io_table_dwords(input_vectors, output_vectors);
            tables
                .input_to_output_tables
                .push(read_u32s(cursor, count)?);
        } else {
            tables.input_to_output_tables.push(Vec::new());
        }
    }
    if is_hull && patch_vectors > 0 && runtime_info.sig_input_vectors > 0 {
        let count = io_table_dwords(u32::from(runtime_info.sig_input_vectors), patch_vectors);
        tables.input_to_pc_output_table = read_u32s(cursor, count)?;
    }
    if is_domain && runtime_info.sig_output_vectors[0] > 0 && patch_vectors > 0 {
        let count = io_table_dwords(patch_vectors, u32::from(runtime_info.sig_output_vectors[0]));
        tables.pc_input_to_output_table = read_u32s(cursor, count)?;
    }
    Some(tables)
}

fn parse_versioned_sections(
    data: &[u8],
    cursor: &mut Cursor<&[u8]>,
    runtime_info: Option<&PSVRuntimeInfo1>,
) -> Option<VersionedSections> {
    let Some(runtime_info) = runtime_info else {
        return Some(VersionedSections::default());
    };
    let string_size = usize::try_from(cursor.read_u32_le().ok()?).ok()?;
    let string_start = usize::try_from(cursor.position()).ok()?;
    let string_end = string_start.checked_add(string_size)?;
    if string_end > data.len() {
        return None;
    }
    let string_table = Vec::from(&data[string_start..string_end]);
    cursor
        .seek(SeekFrom::Current(i64::try_from(string_size).ok()?))
        .ok()?;

    let semantic_count = usize::try_from(cursor.read_u32_le().ok()?).ok()?;
    let semantic_index_table = read_u32s(cursor, semantic_count)?;
    let signature_count = runtime_info.sig_input_elements as usize
        + runtime_info.sig_output_elements as usize
        + runtime_info.sig_patch_const_or_prim_elements as usize;
    let sig_element_size = if signature_count > 0 {
        cursor.read_u32_le().ok()?
    } else {
        0
    };
    let mut sig_input_elements = Vec::with_capacity(runtime_info.sig_input_elements as usize);
    for _ in 0..runtime_info.sig_input_elements {
        sig_input_elements.push(parse_sig_element(cursor, sig_element_size)?);
    }
    let mut sig_output_elements = Vec::with_capacity(runtime_info.sig_output_elements as usize);
    for _ in 0..runtime_info.sig_output_elements {
        sig_output_elements.push(parse_sig_element(cursor, sig_element_size)?);
    }
    let mut sig_patch_const_or_prim_elements =
        Vec::with_capacity(runtime_info.sig_patch_const_or_prim_elements as usize);
    for _ in 0..runtime_info.sig_patch_const_or_prim_elements {
        sig_patch_const_or_prim_elements.push(parse_sig_element(cursor, sig_element_size)?);
    }
    let dependencies = parse_dependency_tables(cursor, runtime_info)?;
    Some(VersionedSections {
        string_table,
        semantic_index_table,
        sig_element_size,
        sig_input_elements,
        sig_output_elements,
        sig_patch_const_or_prim_elements,
        view_id_output_masks: dependencies.view_id_output_masks,
        view_id_pc_or_prim_output_mask: dependencies.view_id_pc_or_prim_output_mask,
        input_to_output_tables: dependencies.input_to_output_tables,
        input_to_pc_output_table: dependencies.input_to_pc_output_table,
        pc_input_to_output_table: dependencies.pc_input_to_output_table,
    })
}

/// Parse a PSV0 chunk from raw bytes.
#[must_use]
pub fn parse_psv0(data: &[u8]) -> Option<PipelineStateValidation> {
    if data.len() < 4 {
        return None;
    }
    let mut cursor = Cursor::new(data);
    let runtime = parse_runtime_info(data, &mut cursor)?;
    let resource_info = parse_resources(&mut cursor)?;
    let sections = parse_versioned_sections(data, &mut cursor, runtime.version_1.as_ref())?;

    Some(PipelineStateValidation {
        info_size: runtime.info_size,
        stage_info: runtime.stage_info,
        min_wave_lane_count: runtime.min_wave_lane_count,
        max_wave_lane_count: runtime.max_wave_lane_count,
        runtime_info_1: runtime.version_1,
        runtime_info_2: runtime.version_2,
        entry_function_name: runtime.entry_function_name,
        num_bytes_group_shared_memory: runtime.num_bytes_group_shared_memory,
        resource_bind_info_size: resource_info.bind_info_size,
        resources: resource_info.resources,
        string_table: sections.string_table,
        semantic_index_table: sections.semantic_index_table,
        sig_element_size: sections.sig_element_size,
        sig_input_elements: sections.sig_input_elements,
        sig_output_elements: sections.sig_output_elements,
        sig_patch_const_or_prim_elements: sections.sig_patch_const_or_prim_elements,
        view_id_output_masks: sections.view_id_output_masks,
        view_id_pc_or_prim_output_mask: sections.view_id_pc_or_prim_output_mask,
        input_to_output_tables: sections.input_to_output_tables,
        input_to_pc_output_table: sections.input_to_pc_output_table,
        pc_input_to_output_table: sections.pc_input_to_output_table,
    })
}
