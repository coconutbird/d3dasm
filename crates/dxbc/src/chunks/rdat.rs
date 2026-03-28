//! RDAT chunk stub — Runtime Data (SM6.1+).
//!
//! The RDAT chunk carries extended resource and function information for
//! DXIL-based shaders. The format is a series of tagged tables:
//!   0x00: u32 — number of tables (parts)
//! For each table:
//!   u32 — table type
//!   u32 — data size
//!   [u8; data_size] — table data
//!
//! We parse the table inventory but don't decode individual table contents.

use alloc::vec::Vec;
use core::fmt;

use super::ChunkParser;
use crate::util::read_u32;

/// Known RDAT table types.
fn table_type_name(ty: u32) -> &'static str {
    match ty {
        0 => "StringBuffer",
        1 => "IndexArray",
        2 => "ResourceTable",
        3 => "FunctionTable",
        4 => "RawBytes",
        5 => "SubobjectTable",
        _ => "Unknown",
    }
}

/// An entry in the RDAT table inventory.
#[derive(Debug, Clone)]
pub struct RdatTable {
    /// Table type ID.
    pub table_type: u32,
    /// Data size in bytes.
    pub data_size: u32,
}

/// Parsed RDAT (Runtime Data) chunk.
#[derive(Debug, Clone)]
pub struct RuntimeData {
    /// Tables found in the RDAT chunk.
    pub tables: Vec<RdatTable>,
}

/// Parse an RDAT chunk.
pub fn parse_rdat(data: &[u8]) -> Option<RuntimeData> {
    if data.len() < 4 {
        return None;
    }
    let num_tables = read_u32(data, 0) as usize;
    let mut tables = Vec::with_capacity(num_tables);
    let mut off = 4;

    for _ in 0..num_tables {
        if off + 8 > data.len() {
            break;
        }
        let table_type = read_u32(data, off);
        let data_size = read_u32(data, off + 4);
        tables.push(RdatTable {
            table_type,
            data_size,
        });
        off += 8 + data_size as usize;
        // Align to 4 bytes.
        off = (off + 3) & !3;
    }

    Some(RuntimeData { tables })
}

impl ChunkParser for RuntimeData {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_rdat(data)
    }
}

impl fmt::Display for RuntimeData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Runtime Data: {} table(s)", self.tables.len())?;
        for (i, t) in self.tables.iter().enumerate() {
            writeln!(
                f,
                "//   [{i}] {} ({} bytes)",
                table_type_name(t.table_type),
                t.data_size
            )?;
        }
        Ok(())
    }
}
