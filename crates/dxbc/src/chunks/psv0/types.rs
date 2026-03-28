//! PSV0 sub-types — enums, runtime info, resource bindings, signature elements.

use alloc::vec::Vec;

/// Shader kind as encoded in PSVRuntimeInfo1.ShaderStage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PSVShaderKind {
    Pixel = 0,
    Vertex = 1,
    Geometry = 2,
    Hull = 3,
    Domain = 4,
    Compute = 5,
    Library = 6,
    RayGeneration = 7,
    Intersection = 8,
    AnyHit = 9,
    ClosestHit = 10,
    Miss = 11,
    Callable = 12,
    Mesh = 13,
    Amplification = 14,
    Node = 15,
    Invalid = 16,
}

impl PSVShaderKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Pixel,
            1 => Self::Vertex,
            2 => Self::Geometry,
            3 => Self::Hull,
            4 => Self::Domain,
            5 => Self::Compute,
            6 => Self::Library,
            7 => Self::RayGeneration,
            8 => Self::Intersection,
            9 => Self::AnyHit,
            10 => Self::ClosestHit,
            11 => Self::Miss,
            12 => Self::Callable,
            13 => Self::Mesh,
            14 => Self::Amplification,
            15 => Self::Node,
            _ => Self::Invalid,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Pixel => "PS",
            Self::Vertex => "VS",
            Self::Geometry => "GS",
            Self::Hull => "HS",
            Self::Domain => "DS",
            Self::Compute => "CS",
            Self::Library => "Lib",
            Self::Mesh => "MS",
            Self::Amplification => "AS",
            Self::RayGeneration => "RayGen",
            Self::Intersection => "Intersection",
            Self::AnyHit => "AnyHit",
            Self::ClosestHit => "ClosestHit",
            Self::Miss => "Miss",
            Self::Callable => "Callable",
            Self::Node => "Node",
            Self::Invalid => "Invalid",
        }
    }
}

/// Shader-kind-specific data from the PSVRuntimeInfo union.
#[derive(Debug, Clone)]
pub enum ShaderStageInfo {
    Vertex {
        output_position_present: bool,
    },
    Hull {
        input_control_point_count: u32,
        output_control_point_count: u32,
        tessellator_domain: u32,
        tessellator_output_primitive: u32,
    },
    Domain {
        input_control_point_count: u32,
        output_position_present: bool,
        tessellator_domain: u32,
    },
    Geometry {
        input_primitive: u32,
        output_topology: u32,
        output_stream_mask: u32,
        output_position_present: bool,
    },
    Pixel {
        depth_output: bool,
        sample_frequency: bool,
    },
    Mesh {
        group_shared_bytes_used: u32,
        group_shared_bytes_dependent_on_view_id: u32,
        payload_size_in_bytes: u32,
        max_output_vertices: u16,
        max_output_primitives: u16,
    },
    Amplification {
        payload_size_in_bytes: u32,
    },
    Compute,
    Other {
        raw: [u8; 16],
    },
}

/// PSV resource binding (v0: 16 bytes, v2+: 24 bytes).
#[derive(Debug, Clone)]
pub struct PSVResourceBindInfo {
    pub res_type: u32,
    pub space: u32,
    pub lower_bound: u32,
    pub upper_bound: u32,
    /// v2+ only.
    pub res_kind: Option<u32>,
    /// v2+ only.
    pub res_flags: Option<u32>,
}

/// PSV signature element (12 bytes, PSVSignatureElement0).
#[derive(Debug, Clone)]
pub struct PSVSignatureElement {
    pub semantic_name: u32,
    pub semantic_indexes: u32,
    pub rows: u8,
    pub start_row: u8,
    pub cols_and_start: u8,
    pub semantic_kind: u8,
    pub component_type: u8,
    pub interpolation_mode: u8,
    pub dynamic_mask_and_stream: u8,
    pub reserved: u8,
}

/// v1+ fields from PSVRuntimeInfo1.
#[derive(Debug, Clone, Default)]
pub struct PSVRuntimeInfo1 {
    pub shader_stage: u8,
    pub uses_view_id: u8,
    pub max_vert_or_patch_prim: u16,
    pub sig_input_elements: u8,
    pub sig_output_elements: u8,
    pub sig_patch_const_or_prim_elements: u8,
    pub sig_input_vectors: u8,
    pub sig_output_vectors: [u8; 4],
}

/// v2+ fields from PSVRuntimeInfo2.
#[derive(Debug, Clone, Default)]
pub struct PSVRuntimeInfo2 {
    pub num_threads_x: u32,
    pub num_threads_y: u32,
    pub num_threads_z: u32,
}

/// Fully parsed PSV0 chunk.
#[derive(Debug, Clone)]
pub struct PipelineStateValidation {
    /// Size of the runtime info struct (determines PSV version).
    pub info_size: u32,
    /// Shader-kind-specific stage info from the runtime info union.
    pub stage_info: ShaderStageInfo,
    /// Minimum expected wave lane count.
    pub min_wave_lane_count: u32,
    /// Maximum expected wave lane count.
    pub max_wave_lane_count: u32,
    /// v1+ runtime info fields.
    pub runtime_info_1: Option<PSVRuntimeInfo1>,
    /// v2+ runtime info fields.
    pub runtime_info_2: Option<PSVRuntimeInfo2>,
    /// v3: entry function name offset into string table.
    pub entry_function_name: Option<u32>,
    /// v4: group shared memory bytes.
    pub num_bytes_group_shared_memory: Option<u32>,
    /// Size of resource bind info structs (0 if no resources).
    pub resource_bind_info_size: u32,
    /// Resource bindings.
    pub resources: Vec<PSVResourceBindInfo>,
    /// String table bytes (v1+, dword-aligned).
    pub string_table: Vec<u8>,
    /// Semantic index table entries (v1+).
    pub semantic_index_table: Vec<u32>,
    /// Signature element struct size (v1+, 0 if no elements).
    pub sig_element_size: u32,
    /// Input signature elements.
    pub sig_input_elements: Vec<PSVSignatureElement>,
    /// Output signature elements.
    pub sig_output_elements: Vec<PSVSignatureElement>,
    /// Patch constant or primitive signature elements.
    pub sig_patch_const_or_prim_elements: Vec<PSVSignatureElement>,
    /// ViewID output masks per stream (up to 4, each a Vec<u32>).
    pub view_id_output_masks: Vec<Vec<u32>>,
    /// ViewID patch constant / primitive output mask (HS/MS only).
    pub view_id_pc_or_prim_output_mask: Vec<u32>,
    /// Input-to-output dependency tables per stream.
    pub input_to_output_tables: Vec<Vec<u32>>,
    /// Input-to-patch-constant output table (HS only).
    pub input_to_pc_output_table: Vec<u32>,
    /// Patch-constant-to-output table (DS only).
    pub pc_input_to_output_table: Vec<u32>,
}
