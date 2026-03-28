//! SFI0 chunk parser — shader feature info flags.
//!
//! The SFI0 chunk is an 8-byte (two u32) bitmask that describes which
//! optional GPU features the shader requires. The first u32 holds the
//! primary feature flags; the second is reserved (must be zero in practice).

use core::fmt;

use super::ChunkParser;
use crate::util::read_u32;

// D3D11/D3D12 shader feature flag constants.
pub const DOUBLES: u64 = 0x0001;
pub const CS_RAW_STRUCTURED_BUFS: u64 = 0x0002;
pub const UAVS_AT_EVERY_STAGE: u64 = 0x0004;
pub const _64_UAVS: u64 = 0x0008;
pub const MINIMUM_PRECISION: u64 = 0x0010;
pub const _11_1_DOUBLE_EXTENSIONS: u64 = 0x0020;
pub const _11_1_SHADER_EXTENSIONS: u64 = 0x0040;
pub const LEVEL_9_COMPARISON_FILTERING: u64 = 0x0080;
pub const TILED_RESOURCES: u64 = 0x0100;
pub const STENCIL_REF: u64 = 0x0200;
pub const INNER_COVERAGE: u64 = 0x0400;
pub const TYPED_UAV_LOAD_ADDITIONAL_FORMATS: u64 = 0x0800;
pub const ROVS: u64 = 0x1000;
pub const VP_RT_ARRAY_INDEX_ANY_SHADER: u64 = 0x2000;
pub const WAVE_OPS: u64 = 0x4000;
pub const INT64_OPS: u64 = 0x8000;
pub const VIEW_ID: u64 = 0x0001_0000;
pub const BARYCENTRICS: u64 = 0x0002_0000;
pub const NATIVE_LOW_PRECISION: u64 = 0x0004_0000;
pub const SHADING_RATE: u64 = 0x0008_0000;
pub const RAYTRACING_TIER_1_1: u64 = 0x0010_0000;
pub const SAMPLER_FEEDBACK: u64 = 0x0020_0000;
pub const ATOMIC_INT64_ON_TYPED_RESOURCE: u64 = 0x0040_0000;
pub const ATOMIC_INT64_ON_GROUP_SHARED: u64 = 0x0080_0000;
pub const DERIVS_IN_MESH_AMP_SHADERS: u64 = 0x0100_0000;
pub const RESOURCE_DESC_HEAP_INDEXING: u64 = 0x0200_0000;
pub const SAMPLER_DESC_HEAP_INDEXING: u64 = 0x0400_0000;
pub const ATOMIC_INT64_ON_DESC_HEAP: u64 = 0x0800_0000;

/// Parsed SFI0 chunk.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShaderFeatureInfo {
    /// Raw flags value (two u32s packed into a u64).
    pub flags: u64,
}

/// Parse an SFI0 chunk.
pub fn parse_sfi0(data: &[u8]) -> Option<ShaderFeatureInfo> {
    let lo = if data.len() >= 4 {
        read_u32(data, 0) as u64
    } else {
        0
    };
    let hi = if data.len() >= 8 {
        read_u32(data, 4) as u64
    } else {
        0
    };
    Some(ShaderFeatureInfo {
        flags: lo | (hi << 32),
    })
}

impl ChunkParser for ShaderFeatureInfo {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_sfi0(data)
    }
}

impl ShaderFeatureInfo {
    /// Returns true if the given flag bit is set.
    pub fn has(&self, flag: u64) -> bool {
        self.flags & flag != 0
    }
}

/// All known flag names in bit order.
const FLAG_NAMES: &[(u64, &str)] = &[
    (DOUBLES, "Doubles"),
    (CS_RAW_STRUCTURED_BUFS, "CS+Raw/StructuredBuffers"),
    (UAVS_AT_EVERY_STAGE, "UAVsAtEveryStage"),
    (_64_UAVS, "64UAVs"),
    (MINIMUM_PRECISION, "MinimumPrecision"),
    (_11_1_DOUBLE_EXTENSIONS, "11.1DoubleExtensions"),
    (_11_1_SHADER_EXTENSIONS, "11.1ShaderExtensions"),
    (LEVEL_9_COMPARISON_FILTERING, "Level9ComparisonFiltering"),
    (TILED_RESOURCES, "TiledResources"),
    (STENCIL_REF, "StencilRef"),
    (INNER_COVERAGE, "InnerCoverage"),
    (
        TYPED_UAV_LOAD_ADDITIONAL_FORMATS,
        "TypedUAVLoadAdditionalFormats",
    ),
    (ROVS, "ROVs"),
    (
        VP_RT_ARRAY_INDEX_ANY_SHADER,
        "VPAndRTArrayIndexFromAnyShader",
    ),
    (WAVE_OPS, "WaveOps"),
    (INT64_OPS, "Int64Ops"),
    (VIEW_ID, "ViewID"),
    (BARYCENTRICS, "Barycentrics"),
    (NATIVE_LOW_PRECISION, "NativeLowPrecision"),
    (SHADING_RATE, "ShadingRate"),
    (RAYTRACING_TIER_1_1, "RaytracingTier1_1"),
    (SAMPLER_FEEDBACK, "SamplerFeedback"),
    (ATOMIC_INT64_ON_TYPED_RESOURCE, "AtomicInt64OnTypedResource"),
    (ATOMIC_INT64_ON_GROUP_SHARED, "AtomicInt64OnGroupShared"),
    (DERIVS_IN_MESH_AMP_SHADERS, "DerivativesInMeshAndAmpShaders"),
    (
        RESOURCE_DESC_HEAP_INDEXING,
        "ResourceDescriptorHeapIndexing",
    ),
    (SAMPLER_DESC_HEAP_INDEXING, "SamplerDescriptorHeapIndexing"),
    (ATOMIC_INT64_ON_DESC_HEAP, "AtomicInt64OnDescriptorHeap"),
];

impl fmt::Display for ShaderFeatureInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.flags == 0 {
            return writeln!(f, "// Shader Feature Info: (none)");
        }
        writeln!(f, "// Shader Feature Info: 0x{:016X}", self.flags)?;
        for &(bit, name) in FLAG_NAMES {
            if self.flags & bit != 0 {
                writeln!(f, "//   {name}")?;
            }
        }
        Ok(())
    }
}
