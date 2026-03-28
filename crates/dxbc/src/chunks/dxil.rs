//! DXIL chunk stub — SM6.0+ shader bytecode (LLVM IR bitcode).
//!
//! The DXIL chunk replaces SHEX/SHDR for Shader Model 6.0 and above.
//! Layout:
//!   0x00: u32 — DXIL four-CC confirmation (usually matches outer chunk)
//!   0x04: u32 — digest or hash
//!   0x08: u16 — major version
//!   0x0A: u16 — minor version
//!   0x0C: u32 — offset to DXIL bitcode within the chunk
//!   0x10: u32 — size of DXIL bitcode in bytes
//!   followed by LLVM bitcode payload
//!
//! We don't decode the LLVM IR — just extract the version and bitcode size.

use core::fmt;

use super::ChunkParser;
use crate::util::read_u32;

/// Parsed DXIL chunk (stub — no LLVM bitcode decoding).
#[derive(Debug, Clone)]
pub struct DxilData {
    /// Shader Model major version (e.g. 6).
    pub major_version: u16,
    /// Shader Model minor version (e.g. 0, 1, 2, …).
    pub minor_version: u16,
    /// Size of the DXIL bitcode payload in bytes.
    pub bitcode_size: u32,
    /// Total chunk data size.
    pub total_size: usize,
}

/// Parse a DXIL chunk.
pub fn parse_dxil(data: &[u8]) -> Option<DxilData> {
    if data.len() < 0x14 {
        return None;
    }

    let major = (data[0x08] as u16) | ((data[0x09] as u16) << 8);
    let minor = (data[0x0A] as u16) | ((data[0x0B] as u16) << 8);
    let bitcode_size = read_u32(data, 0x10);

    Some(DxilData {
        major_version: major,
        minor_version: minor,
        bitcode_size,
        total_size: data.len(),
    })
}

impl ChunkParser for DxilData {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_dxil(data)
    }
}

impl fmt::Display for DxilData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "// DXIL: SM{}.{}, bitcode={} bytes (decoding not implemented)",
            self.major_version, self.minor_version, self.bitcode_size
        )
    }
}
