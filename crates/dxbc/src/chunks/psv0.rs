//! PSV0 chunk parser — Pipeline State Validation data.
//!
//! PSV0 carries metadata the runtime uses to validate pipeline state against
//! the shader's requirements. The header varies by PSV version (0–2).
//!
//! Version 0 header (24 bytes):
//!   0x00: u32 — shader kind (VS=0, PS=1, GS=2, HS=3, DS=4, CS=5, …)
//!   0x04: u8  — uses view ID
//!   0x05: u8  — HS max tess factor (encoded)
//!   0x06: u8  — number of sig element groups (DXIL-only)
//!   0x07: u8  — padding
//!   0x08: u32 — GS max vertex count / patch constant sigs / etc (union)
//!   0x0C: u32 — sig input elements
//!   0x10: u32 — sig output elements
//!   0x14: u32 — sig patch constant or prim elements
//!
//! We parse the header and resource/signature counts.

use core::fmt;

use super::ChunkParser;
use crate::util::read_u32;

/// Parsed PSV0 (Pipeline State Validation) chunk.
#[derive(Debug, Clone)]
pub struct PipelineStateValidation {
    /// PSV info size (determines version: 24=v0, 32=v1, 40=v2).
    pub info_size: u32,
    /// Shader kind (0=VS, 1=PS, 2=GS, 3=HS, 4=DS, 5=CS, 6=Lib, 13=MS, 14=AS).
    pub shader_kind: u32,
    /// Whether the shader uses ViewID.
    pub uses_view_id: bool,
    /// Number of input signature elements.
    pub sig_input_elements: u32,
    /// Number of output signature elements.
    pub sig_output_elements: u32,
    /// Number of patch constant / primitive signature elements.
    pub sig_patch_or_prim_elements: u32,
    /// Number of PSV resources.
    pub resource_count: u32,
}

fn shader_kind_name(kind: u32) -> &'static str {
    match kind {
        0 => "VS",
        1 => "PS",
        2 => "GS",
        3 => "HS",
        4 => "DS",
        5 => "CS",
        6 => "Lib",
        13 => "MS",
        14 => "AS",
        _ => "Unknown",
    }
}

/// Parse a PSV0 chunk.
pub fn parse_psv0(data: &[u8]) -> Option<PipelineStateValidation> {
    // First u32 is the info struct size.
    if data.len() < 4 {
        return None;
    }
    let info_size = read_u32(data, 0);
    if data.len() < 4 + info_size as usize {
        return None;
    }
    if info_size < 24 {
        return None;
    }

    let base = 4; // info struct starts after the size field
    let shader_kind = read_u32(data, base);
    let uses_view_id = data[base + 4] != 0;
    let sig_input_elements = read_u32(data, base + 0x0C);
    let sig_output_elements = read_u32(data, base + 0x10);
    let sig_patch_or_prim_elements = read_u32(data, base + 0x14);

    // Resource count follows the info struct.
    let after_info = 4 + info_size as usize;
    let resource_count = if after_info + 4 <= data.len() {
        read_u32(data, after_info)
    } else {
        0
    };

    Some(PipelineStateValidation {
        info_size,
        shader_kind,
        uses_view_id,
        sig_input_elements,
        sig_output_elements,
        sig_patch_or_prim_elements,
        resource_count,
    })
}

impl ChunkParser for PipelineStateValidation {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_psv0(data)
    }
}

impl fmt::Display for PipelineStateValidation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ver = match self.info_size {
            24 => "v0",
            32 => "v1",
            40 => "v2",
            _ => "v?",
        };
        writeln!(
            f,
            "// Pipeline State Validation ({ver}): shader={}, resources={}, sigs=({},{},{})",
            shader_kind_name(self.shader_kind),
            self.resource_count,
            self.sig_input_elements,
            self.sig_output_elements,
            self.sig_patch_or_prim_elements
        )?;
        if self.uses_view_id {
            writeln!(f, "//   uses ViewID")?;
        }
        Ok(())
    }
}
