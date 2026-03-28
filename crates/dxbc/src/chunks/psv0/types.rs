//! PSV0 sub-types — enums, runtime info, resource bindings, signature elements.

use alloc::vec::Vec;

/// Shader kind as encoded in PSVRuntimeInfo1.ShaderStage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PSVShaderKind {
    /// Pixel shader.
    Pixel = 0,
    /// Vertex shader.
    Vertex = 1,
    /// Geometry shader.
    Geometry = 2,
    /// Hull shader.
    Hull = 3,
    /// Domain shader.
    Domain = 4,
    /// Compute shader.
    Compute = 5,
    /// Shader library (SM6.3+).
    Library = 6,
    /// Ray generation shader.
    RayGeneration = 7,
    /// Ray intersection shader.
    Intersection = 8,
    /// Ray any-hit shader.
    AnyHit = 9,
    /// Ray closest-hit shader.
    ClosestHit = 10,
    /// Ray miss shader.
    Miss = 11,
    /// Ray callable shader.
    Callable = 12,
    /// Mesh shader.
    Mesh = 13,
    /// Amplification shader.
    Amplification = 14,
    /// Work graph node shader.
    Node = 15,
    /// Invalid / unrecognised shader kind.
    Invalid = 16,
}

impl PSVShaderKind {
    /// Converts a raw `u8` shader stage value to the corresponding variant.
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

    /// Returns the short display name (e.g. `"PS"`, `"VS"`, `"CS"`).
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
    /// Vertex shader stage info.
    Vertex {
        /// Whether the shader writes `SV_Position`.
        output_position_present: bool,
    },
    /// Hull shader stage info.
    Hull {
        /// Number of input control points.
        input_control_point_count: u32,
        /// Number of output control points.
        output_control_point_count: u32,
        /// Tessellator domain (tri/quad/isoline).
        tessellator_domain: u32,
        /// Tessellator output primitive type.
        tessellator_output_primitive: u32,
    },
    /// Domain shader stage info.
    Domain {
        /// Number of input control points.
        input_control_point_count: u32,
        /// Whether the shader writes `SV_Position`.
        output_position_present: bool,
        /// Tessellator domain (tri/quad/isoline).
        tessellator_domain: u32,
    },
    /// Geometry shader stage info.
    Geometry {
        /// GS input primitive topology.
        input_primitive: u32,
        /// GS output topology.
        output_topology: u32,
        /// Bitmask of active output streams.
        output_stream_mask: u32,
        /// Whether the shader writes `SV_Position`.
        output_position_present: bool,
    },
    /// Pixel shader stage info.
    Pixel {
        /// Whether the shader writes depth.
        depth_output: bool,
        /// Whether the shader uses sample-frequency execution.
        sample_frequency: bool,
    },
    /// Mesh shader stage info.
    Mesh {
        /// Group-shared memory usage in bytes.
        group_shared_bytes_used: u32,
        /// Group-shared bytes that depend on `SV_ViewID`.
        group_shared_bytes_dependent_on_view_id: u32,
        /// Payload size in bytes for the mesh/amplification handshake.
        payload_size_in_bytes: u32,
        /// Maximum output vertex count.
        max_output_vertices: u16,
        /// Maximum output primitive count.
        max_output_primitives: u16,
    },
    /// Amplification shader stage info.
    Amplification {
        /// Payload size in bytes for the amplification/mesh handshake.
        payload_size_in_bytes: u32,
    },
    /// Compute shader (no extra fields).
    Compute,
    /// Unrecognised shader kind — raw bytes preserved.
    Other {
        /// Raw 16-byte stage info block.
        raw: [u8; 16],
    },
}

/// PSV resource binding (v0: 16 bytes, v2+: 24 bytes).
#[derive(Debug, Clone)]
pub struct PSVResourceBindInfo {
    /// Resource type (`D3D_SHADER_INPUT_TYPE` value).
    pub res_type: u32,
    /// Register space.
    pub space: u32,
    /// Lower bound of the register range.
    pub lower_bound: u32,
    /// Upper bound of the register range.
    pub upper_bound: u32,
    /// Resource kind (v2+ only).
    pub res_kind: Option<u32>,
    /// Resource flags (v2+ only).
    pub res_flags: Option<u32>,
}

/// PSV signature element (12 bytes, PSVSignatureElement0).
#[derive(Debug, Clone)]
pub struct PSVSignatureElement {
    /// Offset into the string table for the semantic name.
    pub semantic_name: u32,
    /// Offset into the semantic index table.
    pub semantic_indexes: u32,
    /// Number of rows occupied by this element.
    pub rows: u8,
    /// Starting row.
    pub start_row: u8,
    /// Packed: columns (low nibble) and start column (high nibble).
    pub cols_and_start: u8,
    /// System-value semantic kind.
    pub semantic_kind: u8,
    /// Component data type.
    pub component_type: u8,
    /// Interpolation mode.
    pub interpolation_mode: u8,
    /// Packed: dynamic index mask (low nibble) and output stream (high nibble).
    pub dynamic_mask_and_stream: u8,
    /// Reserved byte (must be zero).
    pub reserved: u8,
}

/// v1+ fields from PSVRuntimeInfo1.
#[derive(Debug, Clone, Default)]
pub struct PSVRuntimeInfo1 {
    /// Shader stage (`PSVShaderKind` as `u8`).
    pub shader_stage: u8,
    /// Non-zero if the shader uses `SV_ViewID`.
    pub uses_view_id: u8,
    /// GS max vertex count or HS/DS max tess factor (union).
    pub max_vert_or_patch_prim: u16,
    /// Number of input signature elements.
    pub sig_input_elements: u8,
    /// Number of output signature elements.
    pub sig_output_elements: u8,
    /// Number of patch-constant or primitive signature elements.
    pub sig_patch_const_or_prim_elements: u8,
    /// Number of input signature vectors.
    pub sig_input_vectors: u8,
    /// Number of output signature vectors per output stream (up to 4).
    pub sig_output_vectors: [u8; 4],
}

/// v2+ fields from PSVRuntimeInfo2.
#[derive(Debug, Clone, Default)]
pub struct PSVRuntimeInfo2 {
    /// Compute/mesh/amplification thread group X dimension.
    pub num_threads_x: u32,
    /// Compute/mesh/amplification thread group Y dimension.
    pub num_threads_y: u32,
    /// Compute/mesh/amplification thread group Z dimension.
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
    /// ViewID output masks per stream (up to 4, each a `Vec<u32>`).
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
