//! RDAT types — Runtime Data (SM6.1+) structures.

use alloc::vec::Vec;

/// RDAT version constant (0x10 = version 1.0).
pub const RDAT_VERSION_10: u32 = 0x10;

/// Known RDAT part types from `RuntimeDataPartType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum PartType {
    /// Invalid / sentinel.
    Invalid = 0,
    /// Shared null-terminated string buffer.
    StringBuffer = 1,
    /// Index arrays (offset → count + indices).
    IndexArrays = 2,
    /// Resource binding table.
    ResourceTable = 3,
    /// Function (entry point) table.
    FunctionTable = 4,
    /// Raw bytes blob.
    RawBytes = 5,
    /// DXR / pipeline subobject table.
    SubobjectTable = 6,
    /// Work-graph node ID table.
    NodeIDTable = 7,
    /// Work-graph node shader I/O attribute table.
    NodeShaderIOAttribTable = 8,
    /// Work-graph node shader function attribute table.
    NodeShaderFuncAttribTable = 9,
    /// Work-graph I/O node table.
    IONodeTable = 10,
    /// Work-graph node shader info table.
    NodeShaderInfoTable = 11,
}

impl PartType {
    /// Converts a raw `u32` part type value to the corresponding variant.
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Invalid,
            1 => Self::StringBuffer,
            2 => Self::IndexArrays,
            3 => Self::ResourceTable,
            4 => Self::FunctionTable,
            5 => Self::RawBytes,
            6 => Self::SubobjectTable,
            7 => Self::NodeIDTable,
            8 => Self::NodeShaderIOAttribTable,
            9 => Self::NodeShaderFuncAttribTable,
            10 => Self::IONodeTable,
            11 => Self::NodeShaderInfoTable,
            _ => return None,
        })
    }

    /// Whether this part type uses the record-table layout
    /// (RuntimeDataTableHeader { count, stride } followed by records).
    pub fn is_record_table(self) -> bool {
        matches!(
            self,
            Self::ResourceTable
                | Self::FunctionTable
                | Self::SubobjectTable
                | Self::NodeIDTable
                | Self::NodeShaderIOAttribTable
                | Self::NodeShaderFuncAttribTable
                | Self::IONodeTable
                | Self::NodeShaderInfoTable
        )
    }

    /// Returns the display name string for this part type.
    pub fn name(self) -> &'static str {
        match self {
            Self::Invalid => "Invalid",
            Self::StringBuffer => "StringBuffer",
            Self::IndexArrays => "IndexArrays",
            Self::ResourceTable => "ResourceTable",
            Self::FunctionTable => "FunctionTable",
            Self::RawBytes => "RawBytes",
            Self::SubobjectTable => "SubobjectTable",
            Self::NodeIDTable => "NodeIDTable",
            Self::NodeShaderIOAttribTable => "NodeShaderIOAttribTable",
            Self::NodeShaderFuncAttribTable => "NodeShaderFuncAttribTable",
            Self::IONodeTable => "IONodeTable",
            Self::NodeShaderInfoTable => "NodeShaderInfoTable",
        }
    }
}

/// A parsed RDAT part.
#[derive(Debug, Clone)]
pub enum RdatPart {
    /// Shared string buffer (null-terminated UTF-8 strings).
    StringBuffer(Vec<u8>),
    /// Index arrays — a sequence of u32 values. Each "row" is accessed via
    /// an offset; at that offset, the first u32 is the count followed by
    /// count u32 index values.
    IndexArrays(Vec<u32>),
    /// Raw bytes blob used for inline data (e.g. root signatures).
    RawBytes(Vec<u8>),
    /// A record table: stride + opaque record blobs.
    RecordTable {
        /// Raw `PartType` discriminant.
        part_type: u32,
        /// Byte stride of each record.
        record_stride: u32,
        /// Opaque record blobs (each `record_stride` bytes).
        records: Vec<Vec<u8>>,
    },
    /// Unknown/unrecognised part — preserve raw for round-trip.
    Unknown {
        /// Raw part type value.
        part_type: u32,
        /// Raw part data.
        data: Vec<u8>,
    },
}

/// Fully parsed RDAT (Runtime Data) chunk.
#[derive(Debug, Clone)]
pub struct RuntimeData {
    /// RDAT version (typically 0x10).
    pub version: u32,
    /// Parsed parts in order.
    pub parts: Vec<RdatPart>,
}
