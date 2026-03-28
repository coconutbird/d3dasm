//! RDAT chunk — Runtime Data (SM6.1+).
//!
//! The RDAT chunk carries extended resource and function information for
//! DXIL-based shaders. The binary format is a "table of parts":
//!
//!   - **Versioned** (v1.0+): `RuntimeDataHeader { version, part_count }`,
//!     followed by a `u32` offsets array, then each part at its offset.
//!   - **Legacy** (pre-release): just `u32 part_count` followed by sequential
//!     parts.
//!
//! Each part has a `RuntimeDataPartHeader { type, size }` followed by data.
//! Record-based parts (Resource, Function, Subobject, etc.) start with a
//! `RuntimeDataTableHeader { record_count, record_stride }`.

pub mod parse;
pub mod types;
pub mod write;

use alloc::vec::Vec;
use core::fmt;

use super::ChunkWriter;
pub use types::RuntimeData;

impl super::ChunkParser<'_> for RuntimeData {
    fn parse(data: &[u8]) -> Option<Self> {
        parse::parse_rdat(data)
    }
}

impl ChunkWriter for RuntimeData {
    fn fourcc(&self) -> [u8; 4] {
        *b"RDAT"
    }

    fn write_payload(&self) -> Vec<u8> {
        write::write_rdat(self)
    }
}

impl fmt::Display for RuntimeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ver = if self.version >= types::RDAT_VERSION_10 {
            "v1.0"
        } else {
            "legacy"
        };
        writeln!(f, "// Runtime Data ({ver}): {} part(s)", self.parts.len())?;
        for (i, part) in self.parts.iter().enumerate() {
            match part {
                types::RdatPart::StringBuffer(buf) => {
                    writeln!(f, "//   [{i}] StringBuffer ({} bytes)", buf.len())?;
                }
                types::RdatPart::IndexArrays(indices) => {
                    writeln!(f, "//   [{i}] IndexArrays ({} u32s)", indices.len())?;
                }
                types::RdatPart::RawBytes(buf) => {
                    writeln!(f, "//   [{i}] RawBytes ({} bytes)", buf.len())?;
                }
                types::RdatPart::RecordTable {
                    part_type,
                    record_stride,
                    records,
                } => {
                    let name = types::PartType::from_u32(*part_type)
                        .map(|pt| pt.name())
                        .unwrap_or("Unknown");
                    writeln!(
                        f,
                        "//   [{i}] {name} ({} records, stride {})",
                        records.len(),
                        record_stride,
                    )?;
                }
                types::RdatPart::Unknown { part_type, data } => {
                    writeln!(
                        f,
                        "//   [{i}] Unknown(0x{part_type:08X}) ({} bytes)",
                        data.len()
                    )?;
                }
            }
        }
        Ok(())
    }
}
