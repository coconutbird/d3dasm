//! RDAT types — Runtime Data (SM6.1+) structures.

use alloc::vec::Vec;

/// Runtime-data container version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeDataVersion {
    /// Initial versioned RDAT layout.
    V1_0,
    /// A newer additive RDAT version accepted by DXC's reader.
    Newer(u32),
}

impl RuntimeDataVersion {
    /// Decode an RDAT version. DXC rejects versions older than 1.0 and accepts
    /// newer values additively.
    #[must_use]
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0x10 => Some(Self::V1_0),
            0x11.. => Some(Self::Newer(value)),
            _ => None,
        }
    }

    /// Encode the raw version value.
    #[must_use]
    pub fn to_u32(self) -> u32 {
        match self {
            Self::V1_0 => 0x10,
            Self::Newer(value) => value,
        }
    }
}

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
    /// Reserved mesh-node preview record table.
    ReservedMeshNodesPreviewInfoTable = 12,
    /// Experimental signature element table.
    SignatureElementTable = 13,
    /// Experimental vertex-shader info table.
    VSInfoTable = 14,
    /// Experimental pixel-shader info table.
    PSInfoTable = 15,
    /// Experimental hull-shader info table.
    HSInfoTable = 16,
    /// Experimental domain-shader info table.
    DSInfoTable = 17,
    /// Experimental geometry-shader info table.
    GSInfoTable = 18,
    /// Experimental compute-shader info table.
    CSInfoTable = 19,
    /// Experimental mesh-shader info table.
    MSInfoTable = 20,
    /// Experimental amplification-shader info table.
    ASInfoTable = 21,
    /// Grouped PDB information table.
    DxilPdbInfoTable = 0x0001_0001,
    /// Grouped PDB source table.
    DxilPdbInfoSourceTable = 0x0001_0002,
    /// Grouped PDB library table.
    DxilPdbInfoLibraryTable = 0x0001_0003,
}

impl PartType {
    /// Converts a raw `u32` part type value to the corresponding variant.
    #[must_use]
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
            12 => Self::ReservedMeshNodesPreviewInfoTable,
            13 => Self::SignatureElementTable,
            14 => Self::VSInfoTable,
            15 => Self::PSInfoTable,
            16 => Self::HSInfoTable,
            17 => Self::DSInfoTable,
            18 => Self::GSInfoTable,
            19 => Self::CSInfoTable,
            20 => Self::MSInfoTable,
            21 => Self::ASInfoTable,
            0x0001_0001 => Self::DxilPdbInfoTable,
            0x0001_0002 => Self::DxilPdbInfoSourceTable,
            0x0001_0003 => Self::DxilPdbInfoLibraryTable,
            _ => return None,
        })
    }

    /// Whether this part type uses the record-table layout
    /// (`RuntimeDataTableHeader` { count, stride } followed by records).
    #[must_use]
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
                | Self::ReservedMeshNodesPreviewInfoTable
                | Self::SignatureElementTable
                | Self::VSInfoTable
                | Self::PSInfoTable
                | Self::HSInfoTable
                | Self::DSInfoTable
                | Self::GSInfoTable
                | Self::CSInfoTable
                | Self::MSInfoTable
                | Self::ASInfoTable
                | Self::DxilPdbInfoTable
                | Self::DxilPdbInfoSourceTable
                | Self::DxilPdbInfoLibraryTable
        )
    }

    /// Returns the display name string for this part type.
    #[must_use]
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
            Self::ReservedMeshNodesPreviewInfoTable => "ReservedMeshNodesPreviewInfoTable",
            Self::SignatureElementTable => "SignatureElementTable",
            Self::VSInfoTable => "VSInfoTable",
            Self::PSInfoTable => "PSInfoTable",
            Self::HSInfoTable => "HSInfoTable",
            Self::DSInfoTable => "DSInfoTable",
            Self::GSInfoTable => "GSInfoTable",
            Self::CSInfoTable => "CSInfoTable",
            Self::MSInfoTable => "MSInfoTable",
            Self::ASInfoTable => "ASInfoTable",
            Self::DxilPdbInfoTable => "DxilPdbInfoTable",
            Self::DxilPdbInfoSourceTable => "DxilPdbInfoSourceTable",
            Self::DxilPdbInfoLibraryTable => "DxilPdbInfoLibraryTable",
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
    /// RDAT layout version.
    pub version: RuntimeDataVersion,
    /// Parsed parts in order.
    pub parts: Vec<RdatPart>,
    /// Original payload retained for a layout-preserving no-op rewrite.
    #[doc(hidden)]
    pub original_data: Option<Vec<u8>>,
    /// Canonical encoding of the semantic state at parse time.
    #[doc(hidden)]
    pub original_canonical: Option<Vec<u8>>,
}
