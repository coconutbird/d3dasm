//! Input/output/patch-constant signature parsing (ISGN, OSGN, PCSG, ISG1, OSG1, PSG1).

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::{ChunkWriter, WritableChunk};

use nostdio::{ReadLe, Seek, SeekFrom, SliceCursor};

use crate::util::{StringTableWriter, read_cstring};

/// A parsed signature chunk, preserving its original FourCC for round-tripping.
///
/// Different FourCCs use different binary layouts (element strides, optional
/// fields). Storing the FourCC ensures `ISG1` stays `ISG1` instead of
/// collapsing to `ISGN` on write.
#[derive(Debug)]
pub struct Signature<'a> {
    /// Original FourCC (e.g. `ISGN`, `ISG1`, `OSG5`, `OSGN`, `PCSG`, `PSG1`).
    pub fourcc: [u8; 4],
    /// Parsed signature elements.
    pub elements: Vec<SignatureElement<'a>>,
}

impl fmt::Display for Signature<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for e in &self.elements {
            writeln!(f, "{e}")?;
        }

        Ok(())
    }
}

impl ChunkWriter for Signature<'_> {
    fn fourcc(&self) -> [u8; 4] {
        self.fourcc
    }

    fn write_payload(&self) -> Vec<u8> {
        write_signature(self.fourcc, &self.elements).data
    }
}

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
    /// Default (full) precision.
    Default,
    /// 16-bit float minimum precision.
    Float16,
    /// 2.8 fixed-point float minimum precision.
    Float2_8,
    /// Reserved value.
    Reserved,
    /// 16-bit signed integer minimum precision.
    SInt16,
    /// 16-bit unsigned integer minimum precision.
    UInt16,
    /// Any 16-bit precision.
    Any16,
    /// Any 10-bit precision.
    Any10,
    /// Unrecognised minimum precision (raw value preserved).
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

    fn to_u32(self) -> u32 {
        match self {
            Self::Default => 0,
            Self::Float16 => 1,
            Self::Float2_8 => 2,
            Self::Reserved => 3,
            Self::SInt16 => 4,
            Self::UInt16 => 5,
            Self::Any16 => 0xf0,
            Self::Any10 => 0xf1,
            Self::Unknown(x) => x,
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

/// Component data type for signature elements (D3D_REGISTER_COMPONENT_TYPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ComponentType {
    /// Unknown / unspecified.
    Unknown = 0,
    /// 32-bit unsigned integer.
    UInt = 1,
    /// 32-bit signed integer.
    Int = 2,
    /// 32-bit IEEE float.
    Float = 3,
    /// 16-bit unsigned integer.
    UInt16 = 4,
    /// 16-bit signed integer.
    Int16 = 5,
    /// 16-bit IEEE float (half).
    Float16 = 6,
    /// 64-bit unsigned integer.
    UInt64 = 7,
    /// 64-bit signed integer.
    Int64 = 8,
    /// 64-bit double-precision float.
    Float64 = 9,
}

impl ComponentType {
    /// Converts a raw `D3D_REGISTER_COMPONENT_TYPE` value to the corresponding variant.
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Unknown,
            1 => Self::UInt,
            2 => Self::Int,
            3 => Self::Float,
            4 => Self::UInt16,
            5 => Self::Int16,
            6 => Self::Float16,
            7 => Self::UInt64,
            8 => Self::Int64,
            9 => Self::Float64,
            _ => return None,
        })
    }

    /// Returns the lowercase type name used in disassembly output.
    pub fn name(self) -> &'static str {
        match self {
            Self::Unknown => "unknown",
            Self::UInt => "uint",
            Self::Int => "int",
            Self::Float => "float",
            Self::UInt16 => "uint16",
            Self::Int16 => "int16",
            Self::Float16 => "float16",
            Self::UInt64 => "uint64",
            Self::Int64 => "int64",
            Self::Float64 => "float64",
        }
    }
}

/// A parsed input or output signature element.
#[derive(Debug)]
pub struct SignatureElement<'a> {
    /// Semantic name (e.g. `"POSITION"`, `"TEXCOORD"`).
    pub semantic_name: Cow<'a, str>,
    /// Semantic index (e.g. `0` for `TEXCOORD0`, `1` for `TEXCOORD1`).
    pub semantic_index: u32,
    /// System-value semantic (`D3D_NAME` enum, 0 = none).
    pub system_value: u32,
    /// Component data type (`D3D_REGISTER_COMPONENT_TYPE` value).
    pub component_type: u32,
    /// Register number (`v#` for inputs, `o#` for outputs).
    pub register: u32,
    /// Write mask — which `.xyzw` components are written.
    pub mask: u8,
    /// Read/write mask — which components are actually read or written.
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
            String::from(&*self.semantic_name)
        }
    }

    /// The component type name.
    pub fn component_type_name(&self) -> &'static str {
        match ComponentType::from_u32(self.component_type) {
            Some(ct) => ct.name(),
            None => "unknown",
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
    let mut c = SliceCursor::new(data);
    let count = match c.read_u32_le() {
        Ok(v) => v as usize,
        Err(_) => return Vec::new(),
    };
    let mut elements = Vec::with_capacity(count);

    for i in 0..count {
        let base = 8 + i * stride;
        if base + stride > data.len() {
            break;
        }

        let _ = c.seek(SeekFrom::Start(base as u64));

        let stream = if ver.has_stream() {
            Some(c.read_u32_le().unwrap_or(0))
        } else {
            None
        };

        let name_offset = c.read_u32_le().unwrap_or(0) as usize;
        let semantic_index = c.read_u32_le().unwrap_or(0);
        let system_value = c.read_u32_le().unwrap_or(0);
        let component_type = c.read_u32_le().unwrap_or(0);
        let register = c.read_u32_le().unwrap_or(0);
        let mask = c.read_u8_le().unwrap_or(0);
        let rw_mask = c.read_u8_le().unwrap_or(0);

        let min_precision = if ver.has_min_precision() {
            // Skip 2 bytes padding after mask/rw_mask to reach min_precision at +24
            let _ = c.seek(SeekFrom::Current(2));
            Some(MinPrecision::from_u32(c.read_u32_le().unwrap_or(0)))
        } else {
            None
        };

        elements.push(SignatureElement {
            semantic_name: Cow::Borrowed(read_cstring(data, name_offset)),
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

/// Serialize a signature chunk back to bytes.
///
/// `fourcc` is the 4-byte chunk tag (e.g. `*b"ISGN"`, `*b"OSG1"`).
/// Returns a `WritableChunk` ready for `build_dxbc`.
pub fn write_signature(fourcc: [u8; 4], elements: &[SignatureElement<'_>]) -> WritableChunk {
    let fourcc_str = core::str::from_utf8(&fourcc).unwrap_or("ISGN");
    let ver = SignatureVersion::from_fourcc(fourcc_str);
    let stride = ver.stride();
    let w = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_le_bytes());

    // Build string table: collect unique names and assign offsets.
    // String table starts right after the header + element array.
    let string_table_start = 8 + elements.len() * stride;
    let mut strings = StringTableWriter::new(string_table_start);
    let mut name_offsets = Vec::with_capacity(elements.len());

    for elem in elements {
        name_offsets.push(strings.add(&elem.semantic_name));
    }

    let total = string_table_start + strings.len();
    let mut buf = Vec::with_capacity(total);

    // Header: count + 8 bytes (the second u32 is always 8 in real DXBC files)
    w(&mut buf, elements.len() as u32);
    w(&mut buf, 8);

    // Elements
    for (i, elem) in elements.iter().enumerate() {
        if ver.has_stream() {
            w(&mut buf, elem.stream.unwrap_or(0));
        }

        w(&mut buf, name_offsets[i]);
        w(&mut buf, elem.semantic_index);
        w(&mut buf, elem.system_value);
        w(&mut buf, elem.component_type);
        w(&mut buf, elem.register);
        buf.push(elem.mask);
        buf.push(elem.rw_mask);
        if ver.has_min_precision() {
            buf.extend_from_slice(&[0u8; 2]); // 2 bytes padding
            w(&mut buf, elem.min_precision.map_or(0, |mp| mp.to_u32()));
        } else {
            buf.extend_from_slice(&[0u8; 2]); // 2 bytes padding to fill stride
        }
    }

    // String table
    buf.extend_from_slice(&strings.finish());

    WritableChunk { fourcc, data: buf }
}
