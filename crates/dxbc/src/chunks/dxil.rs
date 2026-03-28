//! DXIL chunk — SM6.0+ shader bytecode (LLVM IR bitcode).
//!
//! The DXIL chunk replaces SHEX/SHDR for Shader Model 6.0 and above.
//! Binary layout (`DxilProgramHeader`):
//!
//!   0x00: u32 — ProgramVersion: (shader_kind << 16) | (major << 4) | minor
//!   0x04: u32 — SizeInUint32: total size including header, in u32 units
//!   DxilBitcodeHeader:
//!     0x08: u32 — DxilMagic ('DXIL' = 0x4C495844)
//!     0x0C: u32 — DxilVersion
//!     0x10: u32 — BitcodeOffset (from start of DxilBitcodeHeader, typically 16)
//!     0x14: u32 — BitcodeSize
//!   0x18..BitcodeOffset: optional padding
//!   bitcode payload
//!
//! The LLVM bitcode itself is not decoded. The `bitcode` field is exposed as a
//! mutable `Vec<u8>` so users can replace or modify the IL directly. The writer
//! reconstructs the header with correct sizes.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};
use nostdio::{ReadLe, SliceCursor};

/// Magic value for the DxilBitcodeHeader (`'DXIL'` as little-endian u32).
const DXIL_MAGIC: u32 = 0x4C495844;

/// Parsed DXIL chunk — header fields + raw LLVM bitcode.
///
/// The bitcode blob borrows from the input on read (zero-copy). Assign
/// `Cow::Owned(...)` to replace the IL for patching, and the writer will
/// reconstruct the surrounding header with correct sizes.
#[derive(Debug, Clone)]
pub struct DxilData<'a> {
    /// Shader kind from ProgramVersion (e.g. 0=Pixel, 1=Vertex, …).
    pub shader_kind: u16,
    /// Shader Model major version (e.g. 6).
    pub major_version: u8,
    /// Shader Model minor version (e.g. 0, 1, 2, …).
    pub minor_version: u8,
    /// DXIL version field from the bitcode header.
    pub dxil_version: u32,
    /// LLVM bitcode payload. Replace this to swap the IL.
    pub bitcode: Cow<'a, [u8]>,
}

/// Parse a DXIL chunk.
pub fn parse_dxil<'a>(data: &'a [u8]) -> Option<DxilData<'a>> {
    if data.len() < 0x18 {
        return None;
    }
    let mut c = SliceCursor::new(data);

    let program_version = c.read_u32_le().ok()?;
    let _size_in_u32 = c.read_u32_le().ok()?;

    let dxil_magic = c.read_u32_le().ok()?;
    if dxil_magic != DXIL_MAGIC {
        return None;
    }
    let dxil_version = c.read_u32_le().ok()?;
    let bitcode_offset = c.read_u32_le().ok()? as usize;
    let bitcode_size = c.read_u32_le().ok()? as usize;

    // Bitcode starts at &DxilBitcodeHeader + bitcode_offset.
    // DxilBitcodeHeader starts at offset 8 in the chunk data.
    let bc_start = 8 + bitcode_offset;
    let bc_end = bc_start + bitcode_size;
    if bc_end > data.len() {
        return None;
    }

    let shader_kind = ((program_version >> 16) & 0xFFFF) as u16;
    let major = ((program_version >> 4) & 0xF) as u8;
    let minor = (program_version & 0xF) as u8;

    Some(DxilData {
        shader_kind,
        major_version: major,
        minor_version: minor,
        dxil_version,
        bitcode: Cow::Borrowed(&data[bc_start..bc_end]),
    })
}

impl<'a> ChunkParser<'a> for DxilData<'a> {
    fn parse(data: &'a [u8]) -> Option<Self> {
        parse_dxil(data)
    }
}

impl ChunkWriter for DxilData<'_> {
    fn fourcc(&self) -> [u8; 4] {
        *b"DXIL"
    }

    fn write_payload(&self) -> Vec<u8> {
        let bc_size = self.bitcode.len() as u32;
        // BitcodeOffset is sizeof(DxilBitcodeHeader) = 16.
        let bitcode_offset: u32 = 16;
        // Total size = DxilProgramHeader(8) + DxilBitcodeHeader(16) + bitcode
        let total_bytes = 8 + bitcode_offset as usize + self.bitcode.len();
        let size_in_u32 = total_bytes.div_ceil(4) as u32;

        let program_version = ((self.shader_kind as u32) << 16)
            | ((self.major_version as u32) << 4)
            | (self.minor_version as u32);

        let mut out = Vec::with_capacity(total_bytes);
        out.extend_from_slice(&program_version.to_le_bytes());
        out.extend_from_slice(&size_in_u32.to_le_bytes());
        out.extend_from_slice(&DXIL_MAGIC.to_le_bytes());
        out.extend_from_slice(&self.dxil_version.to_le_bytes());
        out.extend_from_slice(&bitcode_offset.to_le_bytes());
        out.extend_from_slice(&bc_size.to_le_bytes());
        out.extend_from_slice(&self.bitcode);
        // Pad to 4-byte alignment.
        while out.len() % 4 != 0 {
            out.push(0);
        }
        out
    }
}

impl fmt::Display for DxilData<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "// DXIL: SM{}.{}, kind={}, bitcode={} bytes",
            self.major_version,
            self.minor_version,
            self.shader_kind,
            self.bitcode.len()
        )
    }
}
