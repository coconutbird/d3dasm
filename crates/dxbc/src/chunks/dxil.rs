//! DXIL chunk — SM6.0+ shader bytecode (LLVM IR bitcode).
//!
//! The DXIL chunk replaces SHEX/SHDR for Shader Model 6.0 and above.
//! Binary layout (`DxilProgramHeader`):
//!
//!   0x00: u32 — `ProgramVersion`: (`shader_kind` << 16) | (major << 4) | minor
//!   0x04: u32 — `SizeInUint32`: total size including header, in u32 units
//!   `DxilBitcodeHeader`:
//!     0x08: u32 — `DxilMagic` ('DXIL' = 0x4C495844)
//!     0x0C: u32 — `DxilVersion`
//!     0x10: u32 — `BitcodeOffset` (from start of `DxilBitcodeHeader`, typically 16)
//!     0x14: u32 — `BitcodeSize`
//!   0x18..BitcodeOffset: optional padding
//!   bitcode payload
//!
//! The LLVM bitcode itself is not decoded. The `bitcode` field is exposed as a
//! `Cow<[u8]>` so users can replace the IL without forcing an allocation when
//! merely inspecting it. The writer reconstructs the header with correct sizes.

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};
use nostdio::{Cursor, ReadLe};

/// Magic value for the `DxilBitcodeHeader` (`'DXIL'` as little-endian u32).
const DXIL_MAGIC: u32 = 0x4C49_5844;

/// Parsed DXIL chunk — header fields + raw LLVM bitcode.
///
/// The bitcode blob borrows from the input on read (zero-copy). Assign
/// `Cow::Owned(...)` to replace the IL for patching, and the writer will
/// reconstruct the surrounding header with correct sizes.
#[derive(Debug, Clone)]
pub struct DxilData<'a> {
    /// Container part type. `DXIL` is executable IL; `STAT` is reflection IL.
    pub fourcc: [u8; 4],
    /// Shader kind from `ProgramVersion` (e.g. 0=Pixel, 1=Vertex, …).
    pub shader_kind: u16,
    /// Shader Model major version (e.g. 6).
    pub major_version: u8,
    /// Shader Model minor version (e.g. 0, 1, 2, …).
    pub minor_version: u8,
    /// Uninterpreted `ProgramVersion` bits 8..15.
    ///
    /// Microsoft's validator does not require these bits to be zero, so they
    /// are retained rather than normalized during a rewrite.
    pub version_reserved: u8,
    /// DXIL version field from the bitcode header.
    pub dxil_version: u32,
    /// Bytes between the fixed bitcode header and the bitcode offset.
    pub bitcode_prefix: Cow<'a, [u8]>,
    /// LLVM bitcode payload. Replace this to swap the IL.
    pub bitcode: Cow<'a, [u8]>,
    /// Bytes after the bitcode but still inside `SizeInUint32`.
    pub program_suffix: Cow<'a, [u8]>,
    /// Any part bytes after the declared DXIL program size.
    pub chunk_suffix: Cow<'a, [u8]>,
}

/// Parse a DXIL chunk.
#[must_use]
pub fn parse_dxil(data: &[u8]) -> Option<DxilData<'_>> {
    parse_dxil_with_fourcc(data, *b"DXIL")
}

/// Parse a `DxilProgramHeader` from either a DXIL or STAT container part.
#[must_use]
pub fn parse_dxil_with_fourcc(data: &[u8], fourcc: [u8; 4]) -> Option<DxilData<'_>> {
    if !matches!(&fourcc, b"DXIL" | b"STAT") {
        return None;
    }
    if data.len() < 0x18 {
        return None;
    }
    let mut c = Cursor::new(data);

    let program_version = c.read_u32_le().ok()?;
    let size_in_u32 = usize::try_from(c.read_u32_le().ok()?).ok()?;
    let program_size = size_in_u32.checked_mul(4)?;
    if program_size < 0x18 || program_size > data.len() {
        return None;
    }

    let dxil_magic = c.read_u32_le().ok()?;
    if dxil_magic != DXIL_MAGIC {
        return None;
    }
    let dxil_version = c.read_u32_le().ok()?;
    let bitcode_offset = usize::try_from(c.read_u32_le().ok()?).ok()?;
    let bitcode_size = usize::try_from(c.read_u32_le().ok()?).ok()?;
    if bitcode_offset < 16 || bitcode_size == 0 {
        return None;
    }

    // Bitcode starts at &DxilBitcodeHeader + bitcode_offset.
    // DxilBitcodeHeader starts at offset 8 in the chunk data.
    let bc_start = 8usize.checked_add(bitcode_offset)?;
    let bc_end = bc_start.checked_add(bitcode_size)?;
    if bc_start < 0x18 || bc_end > program_size {
        return None;
    }

    let shader_kind = ((program_version >> 16) & 0xFFFF) as u16;
    let major = ((program_version >> 4) & 0xF) as u8;
    let minor = (program_version & 0xF) as u8;
    let version_reserved = ((program_version >> 8) & 0xff) as u8;

    Some(DxilData {
        fourcc,
        shader_kind,
        major_version: major,
        minor_version: minor,
        version_reserved,
        dxil_version,
        bitcode_prefix: Cow::Borrowed(&data[0x18..bc_start]),
        bitcode: Cow::Borrowed(&data[bc_start..bc_end]),
        program_suffix: Cow::Borrowed(&data[bc_end..program_size]),
        chunk_suffix: Cow::Borrowed(&data[program_size..]),
    })
}

impl<'a> ChunkParser<'a> for DxilData<'a> {
    fn parse(data: &'a [u8]) -> Option<Self> {
        parse_dxil(data)
    }
}

impl ChunkWriter for DxilData<'_> {
    fn fourcc(&self) -> [u8; 4] {
        self.fourcc
    }

    fn write_payload(&self) -> Vec<u8> {
        assert!(
            self.major_version <= 0x0f,
            "DXIL major version exceeds 4 bits"
        );
        assert!(
            self.minor_version <= 0x0f,
            "DXIL minor version exceeds 4 bits"
        );
        let bc_size = u32::try_from(self.bitcode.len()).expect("DXIL bitcode size exceeds u32");
        let bitcode_offset = 16usize
            .checked_add(self.bitcode_prefix.len())
            .expect("DXIL bitcode offset overflow");
        let bitcode_offset_u32 =
            u32::try_from(bitcode_offset).expect("DXIL bitcode offset exceeds u32");
        let unaligned_program_size = 24usize
            .checked_add(self.bitcode_prefix.len())
            .and_then(|size| size.checked_add(self.bitcode.len()))
            .and_then(|size| size.checked_add(self.program_suffix.len()))
            .expect("DXIL program size overflow");
        let total_bytes = unaligned_program_size
            .checked_add(3)
            .expect("DXIL program size overflow")
            & !3;
        let size_in_u32 = u32::try_from(total_bytes / 4).expect("DXIL program size exceeds u32");

        let program_version = (u32::from(self.shader_kind) << 16)
            | (u32::from(self.version_reserved) << 8)
            | (u32::from(self.major_version) << 4)
            | u32::from(self.minor_version);

        let capacity = total_bytes
            .checked_add(self.chunk_suffix.len())
            .expect("DXIL chunk size overflow");
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(&program_version.to_le_bytes());
        out.extend_from_slice(&size_in_u32.to_le_bytes());
        out.extend_from_slice(&DXIL_MAGIC.to_le_bytes());
        out.extend_from_slice(&self.dxil_version.to_le_bytes());
        out.extend_from_slice(&bitcode_offset_u32.to_le_bytes());
        out.extend_from_slice(&bc_size.to_le_bytes());
        out.extend_from_slice(&self.bitcode_prefix);
        out.extend_from_slice(&self.bitcode);
        out.extend_from_slice(&self.program_suffix);
        while out.len() < total_bytes {
            out.push(0);
        }
        out.extend_from_slice(&self.chunk_suffix);
        out
    }
}

impl fmt::Display for DxilData<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "// {} program: SM{}.{}, kind={}, bitcode={} bytes",
            core::str::from_utf8(&self.fourcc).unwrap_or("????"),
            self.major_version,
            self.minor_version,
            self.shader_kind,
            self.bitcode.len()
        )
    }
}
