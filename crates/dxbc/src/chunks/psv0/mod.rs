//! PSV0 chunk — Pipeline State Validation data.
//!
//! The PSV0 chunk carries metadata the D3D12 runtime uses to validate pipeline
//! state against the shader's requirements. The format is versioned:
//!   - v0: 24-byte runtime info (no signature data)
//!   - v1: 36-byte runtime info + string table + semantic index table + sig elements + masks
//!   - v2: 48-byte runtime info (adds NumThreads) + v2 resource bind info (24 bytes)
//!   - v3: 52-byte runtime info (adds EntryFunctionName)
//!   - v4: 56-byte runtime info (adds NumBytesGroupSharedMemory)

pub mod parse;
pub mod types;
pub mod write;

use alloc::vec::Vec;
use core::fmt;

use super::ChunkWriter;
pub use types::PipelineStateValidation;

impl super::ChunkParser for PipelineStateValidation {
    fn parse(data: &[u8]) -> Option<Self> {
        parse::parse_psv0(data)
    }
}

impl ChunkWriter for PipelineStateValidation {
    fn fourcc(&self) -> [u8; 4] {
        *b"PSV0"
    }

    fn write_payload(&self) -> Vec<u8> {
        write::write_psv0(self)
    }
}

impl fmt::Display for PipelineStateValidation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ver = match self.info_size {
            24 => "v0",
            36 => "v1",
            48 => "v2",
            52 => "v3",
            56 => "v4",
            _ => "v?",
        };
        let kind_name = if let Some(ref ri1) = self.runtime_info_1 {
            types::PSVShaderKind::from_u8(ri1.shader_stage).name()
        } else {
            match &self.stage_info {
                types::ShaderStageInfo::Vertex { .. } => "VS",
                types::ShaderStageInfo::Pixel { .. } => "PS",
                types::ShaderStageInfo::Geometry { .. } => "GS",
                types::ShaderStageInfo::Hull { .. } => "HS",
                types::ShaderStageInfo::Domain { .. } => "DS",
                types::ShaderStageInfo::Compute => "CS",
                types::ShaderStageInfo::Mesh { .. } => "MS",
                types::ShaderStageInfo::Amplification { .. } => "AS",
                types::ShaderStageInfo::Other { .. } => "Unknown",
            }
        };
        writeln!(
            f,
            "// Pipeline State Validation ({ver}): shader={kind_name}, resources={}, sigs=({},{},{})",
            self.resources.len(),
            self.sig_input_elements.len(),
            self.sig_output_elements.len(),
            self.sig_patch_const_or_prim_elements.len(),
        )?;
        if let Some(ref ri1) = self.runtime_info_1
            && ri1.uses_view_id != 0
        {
            writeln!(f, "//   uses ViewID")?;
        }
        Ok(())
    }
}
