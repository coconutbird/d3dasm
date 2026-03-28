//! SFI0 chunk parser — shader feature info flags.
//!
//! The SFI0 chunk is an 8-byte (two u32) bitmask that describes which
//! optional GPU features the shader requires. The first u32 holds the
//! primary feature flags; the second is reserved (must be zero in practice).

use core::fmt;

use nostdio::{ReadLe, SliceCursor};

use super::{ChunkParser, ChunkWriter};

/// Shader uses double-precision floating point.
pub const DOUBLES: u64 = 0x0001;
/// CS uses raw and/or structured buffers.
pub const CS_RAW_STRUCTURED_BUFS: u64 = 0x0002;
/// UAVs bound at every shader stage.
pub const UAVS_AT_EVERY_STAGE: u64 = 0x0004;
/// Shader uses 64 or more UAV slots.
pub const _64_UAVS: u64 = 0x0008;
/// Shader uses minimum-precision types.
pub const MINIMUM_PRECISION: u64 = 0x0010;
/// Shader uses DX11.1 double-precision extensions.
pub const _11_1_DOUBLE_EXTENSIONS: u64 = 0x0020;
/// Shader uses DX11.1 shader extensions.
pub const _11_1_SHADER_EXTENSIONS: u64 = 0x0040;
/// Shader requires Level-9 comparison filtering.
pub const LEVEL_9_COMPARISON_FILTERING: u64 = 0x0080;
/// Shader uses tiled resources.
pub const TILED_RESOURCES: u64 = 0x0100;
/// Shader outputs a stencil reference value.
pub const STENCIL_REF: u64 = 0x0200;
/// Shader uses inner coverage (conservative rasterization).
pub const INNER_COVERAGE: u64 = 0x0400;
/// Shader uses typed UAV loads with additional formats.
pub const TYPED_UAV_LOAD_ADDITIONAL_FORMATS: u64 = 0x0800;
/// Shader uses rasterizer-ordered views (ROVs).
pub const ROVS: u64 = 0x1000;
/// Shader uses `SV_RenderTargetArrayIndex` / `SV_ViewportArrayIndex` from any stage.
pub const VP_RT_ARRAY_INDEX_ANY_SHADER: u64 = 0x2000;
/// Shader uses SM6.0 wave intrinsics.
pub const WAVE_OPS: u64 = 0x4000;
/// Shader uses 64-bit integer operations.
pub const INT64_OPS: u64 = 0x8000;
/// Shader uses `SV_ViewID`.
pub const VIEW_ID: u64 = 0x0001_0000;
/// Shader uses barycentric coordinates.
pub const BARYCENTRICS: u64 = 0x0002_0000;
/// Shader uses native 16-bit (low-precision) types.
pub const NATIVE_LOW_PRECISION: u64 = 0x0004_0000;
/// Shader uses variable-rate shading.
pub const SHADING_RATE: u64 = 0x0008_0000;
/// Shader uses DXR tier 1.1 features.
pub const RAYTRACING_TIER_1_1: u64 = 0x0010_0000;
/// Shader uses sampler feedback.
pub const SAMPLER_FEEDBACK: u64 = 0x0020_0000;
/// Shader uses 64-bit atomics on typed resources.
pub const ATOMIC_INT64_ON_TYPED_RESOURCE: u64 = 0x0040_0000;
/// Shader uses 64-bit atomics on group-shared memory.
pub const ATOMIC_INT64_ON_GROUP_SHARED: u64 = 0x0080_0000;
/// Shader uses derivatives in mesh/amplification shaders.
pub const DERIVS_IN_MESH_AMP_SHADERS: u64 = 0x0100_0000;
/// Shader uses resource descriptor heap indexing.
pub const RESOURCE_DESC_HEAP_INDEXING: u64 = 0x0200_0000;
/// Shader uses sampler descriptor heap indexing.
pub const SAMPLER_DESC_HEAP_INDEXING: u64 = 0x0400_0000;
/// Shader uses 64-bit atomics on descriptor heap resources.
pub const ATOMIC_INT64_ON_DESC_HEAP: u64 = 0x0800_0000;

/// Parsed SFI0 chunk.
#[derive(Debug, Clone, Copy, Default)]
pub struct ShaderFeatureInfo {
    /// Raw flags value (two u32s packed into a u64).
    pub flags: u64,
}

/// Parse an SFI0 chunk.
pub fn parse_sfi0(data: &[u8]) -> Option<ShaderFeatureInfo> {
    let mut c = SliceCursor::new(data);
    let lo = if data.len() >= 4 {
        c.read_u32_le().ok()? as u64
    } else {
        0
    };
    let hi = if data.len() >= 8 {
        c.read_u32_le().ok()? as u64
    } else {
        0
    };
    Some(ShaderFeatureInfo {
        flags: lo | (hi << 32),
    })
}

impl ChunkParser<'_> for ShaderFeatureInfo {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_sfi0(data)
    }
}

impl ChunkWriter for ShaderFeatureInfo {
    fn fourcc(&self) -> [u8; 4] {
        *b"SFI0"
    }

    fn write_payload(&self) -> alloc::vec::Vec<u8> {
        let mut buf = alloc::vec::Vec::with_capacity(8);
        buf.extend_from_slice(&(self.flags as u32).to_le_bytes());
        buf.extend_from_slice(&((self.flags >> 32) as u32).to_le_bytes());
        buf
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
