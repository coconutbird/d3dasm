use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::util::{read_cstring, read_u32};

/// Which binary layout variant was used in the DXBC chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureVersion {
    /// ISGN / OSGN / PCSG — 24 bytes per element.
    V0,
    /// OSG5 — 28 bytes per element (prepends `stream`).
    V5,
    /// ISG1 / OSG1 / PSG1 — 32 bytes per element (prepends `stream`, appends `min_precision`).
    V1,
}

impl SignatureVersion {
    /// Derive the version from a FourCC string.
    pub fn from_fourcc(fourcc: &str) -> Self {
        match fourcc {
            "OSG5" => Self::V5,
            "ISG1" | "OSG1" | "PSG1" => Self::V1,
            _ => Self::V0,
        }
    }

    fn stride(self) -> usize {
        match self {
            Self::V0 => 24,
            Self::V5 => 28,
            Self::V1 => 32,
        }
    }

    fn has_stream(self) -> bool {
        matches!(self, Self::V5 | Self::V1)
    }
    fn has_min_precision(self) -> bool {
        matches!(self, Self::V1)
    }
}

/// Minimum precision hint (ISG1 / OSG1 / PSG1 only).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinPrecision {
    Default,
    Float16,
    Float2_8,
    Reserved,
    SInt16,
    UInt16,
    Any16,
    Any10,
    Unknown(u32),
}

impl MinPrecision {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => Self::Default,
            1 => Self::Float16,
            2 => Self::Float2_8,
            3 => Self::Reserved,
            4 => Self::SInt16,
            5 => Self::UInt16,
            0xf0 => Self::Any16,
            0xf1 => Self::Any10,
            x => Self::Unknown(x),
        }
    }
}

impl fmt::Display for MinPrecision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Default => Ok(()),
            Self::Float16 => write!(f, " [min16f]"),
            Self::Float2_8 => write!(f, " [min2_8f]"),
            Self::Reserved => write!(f, " [reserved]"),
            Self::SInt16 => write!(f, " [min16i]"),
            Self::UInt16 => write!(f, " [min16u]"),
            Self::Any16 => write!(f, " [any16]"),
            Self::Any10 => write!(f, " [any10]"),
            Self::Unknown(v) => write!(f, " [minprec={}]", v),
        }
    }
}

/// A parsed input or output signature element.
#[derive(Debug)]
pub struct SignatureElement<'a> {
    pub semantic_name: &'a str,
    pub semantic_index: u32,
    pub system_value: u32,
    pub component_type: u32,
    pub register: u32,
    pub mask: u8,
    pub rw_mask: u8,
    /// GS output stream index (OSG5 / ISG1 / OSG1 / PSG1 only).
    pub stream: Option<u32>,
    /// Minimum precision hint (ISG1 / OSG1 / PSG1 only).
    pub min_precision: Option<MinPrecision>,
}

impl SignatureElement<'_> {
    /// The semantic name including the index suffix (e.g. `"TEXCOORD1"`).
    pub fn name_with_index(&self) -> String {
        if self.semantic_index > 0 {
            format!("{}{}", self.semantic_name, self.semantic_index)
        } else {
            String::from(self.semantic_name)
        }
    }

    /// The component type name.
    pub fn component_type_name(&self) -> &'static str {
        match self.component_type {
            1 => "uint",
            2 => "int",
            3 => "float",
            4 => "uint16",
            5 => "int16",
            6 => "float16",
            7 => "uint64",
            8 => "int64",
            9 => "float64",
            _ => "unknown",
        }
    }

    /// Format the columns after the name: type and register.mask.
    pub fn format_columns(&self) -> String {
        let mask_str = format_mask(self.mask);
        format!(
            "{:>7} v{}.{}",
            self.component_type_name(),
            self.register,
            mask_str
        )
    }
}

impl fmt::Display for SignatureElement<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<24} {}",
            self.name_with_index(),
            self.format_columns()
        )?;
        if let Some(s) = self.stream
            && s != 0
        {
            write!(f, " stream={}", s)?;
        }
        if let Some(mp) = self.min_precision {
            write!(f, "{}", mp)?;
        }
        Ok(())
    }
}

fn format_mask(mask: u8) -> String {
    let mut s = String::with_capacity(4);
    if mask & 1 != 0 {
        s.push('x');
    }

    if mask & 2 != 0 {
        s.push('y');
    }

    if mask & 4 != 0 {
        s.push('z');
    }

    if mask & 8 != 0 {
        s.push('w');
    }

    s
}

/// Parse a signature chunk. The `fourcc` string determines the element layout:
/// - `"ISGN"` / `"OSGN"` / `"PCSG"` → 24 bytes (base)
/// - `"OSG5"` → 28 bytes (adds `stream`)
/// - `"ISG1"` / `"OSG1"` / `"PSG1"` → 32 bytes (adds `stream` + `min_precision`)
pub fn parse_signature<'a>(fourcc: &str, data: &'a [u8]) -> Vec<SignatureElement<'a>> {
    if data.len() < 8 {
        return Vec::new();
    }
    let ver = SignatureVersion::from_fourcc(fourcc);
    let stride = ver.stride();
    let count = read_u32(data, 0) as usize;
    let mut elements = Vec::with_capacity(count);

    for i in 0..count {
        let base = 8 + i * stride;
        if base + stride > data.len() {
            break;
        }

        let (stream, ofs) = if ver.has_stream() {
            (Some(read_u32(data, base)), base + 4)
        } else {
            (None, base)
        };

        let name_offset = read_u32(data, ofs) as usize;
        let semantic_index = read_u32(data, ofs + 4);
        let system_value = read_u32(data, ofs + 8);
        let component_type = read_u32(data, ofs + 12);
        let register = read_u32(data, ofs + 16);
        let mask = data[ofs + 20];
        let rw_mask = data[ofs + 21];

        let min_precision = if ver.has_min_precision() {
            Some(MinPrecision::from_u32(read_u32(data, ofs + 24)))
        } else {
            None
        };

        elements.push(SignatureElement {
            semantic_name: read_cstring(data, name_offset),
            semantic_index,
            system_value,
            component_type,
            register,
            mask,
            rw_mask,
            stream,
            min_precision,
        });
    }
    elements
}
