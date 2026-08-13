//! Input/output/patch-constant signature parsing (ISGN, OSGN, PCSG, ISG1, OSG1, PSG1).

use alloc::borrow::Cow;
use alloc::vec::Vec;
use core::fmt;

use super::{ChunkWriter, WritableChunk};

use crate::util::StringTableWriter;

/// A parsed signature chunk, preserving its original `FourCC` for round-tripping.
///
/// Different `FourCCs` use different binary layouts (element strides, optional
/// fields). Storing the `FourCC` ensures `ISG1` stays `ISG1` instead of
/// collapsing to `ISGN` on write.
#[derive(Debug)]
pub struct Signature<'a> {
    /// Original `FourCC` (e.g. `ISGN`, `ISG1`, `OSG5`, `OSGN`, `PCSG`, `PSG1`).
    pub fourcc: [u8; 4],
    /// Parsed signature elements.
    pub elements: Vec<SignatureElement<'a>>,
    /// Opaque bytes after the last referenced semantic string.
    ///
    /// The format has no string-table length field. In practice these bytes
    /// include compiler-specific DWORD padding and may also contain unreferenced
    /// strings, so they must be retained verbatim.
    pub trailing_data: Vec<u8>,
    /// Original payload retained for a layout-preserving no-op rewrite.
    #[doc(hidden)]
    pub original_data: Option<Cow<'a, [u8]>>,
    /// Canonical encoding of the semantic state at parse time.
    #[doc(hidden)]
    pub original_canonical: Option<Vec<u8>>,
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
        let canonical =
            write_signature_with_trailing(self.fourcc, &self.elements, &self.trailing_data).data;
        if self.original_canonical.as_deref() == Some(canonical.as_slice()) {
            if let Some(original) = &self.original_data {
                return original.to_vec();
            }
        }
        canonical
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
    /// Derive the version from a `FourCC` string.
    #[must_use]
    pub fn from_fourcc(fourcc: &str) -> Option<Self> {
        Some(match fourcc {
            "ISGN" | "OSGN" | "PCSG" => Self::V0,
            "OSG5" => Self::V5,
            "ISG1" | "OSG1" | "PSG1" => Self::V1,
            _ => return None,
        })
    }

    fn stride(self) -> usize {
        match self {
            Self::V0 => 24,
            Self::V5 => 28,
            Self::V1 => 32,
        }
    }

    /// Whether this layout carries a per-element GS output `stream`.
    #[must_use]
    pub fn has_stream(self) -> bool {
        matches!(self, Self::V5 | Self::V1)
    }

    /// Whether this layout carries a per-element `min_precision` field.
    #[must_use]
    pub fn has_min_precision(self) -> bool {
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
    /// Converts a raw `D3D_MIN_PRECISION` value to the corresponding variant.
    #[must_use]
    pub fn from_u32(v: u32) -> Self {
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

    /// The raw `D3D_MIN_PRECISION` value for this variant.
    #[must_use]
    pub fn to_u32(self) -> u32 {
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
            Self::Unknown(v) => write!(f, " [minprec={v}]"),
        }
    }
}

/// Component data type for signature elements (`D3D_REGISTER_COMPONENT_TYPE`).
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
    #[must_use]
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
    #[must_use]
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

/// Signature system value (`D3D_NAME` / `DxilProgramSigSemantic`).
///
/// This wire enum is intentionally separate from SHEX's `D3D10_SB_NAME`:
/// signature chunks use aggregate tessellation-factor values 11–16 and the
/// pixel-output range 64–70, while SHEX uses per-edge values 11–22.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureSystemValue {
    /// No system-value interpretation.
    Undefined,
    /// `SV_Position`.
    Position,
    /// `SV_ClipDistance`.
    ClipDistance,
    /// `SV_CullDistance`.
    CullDistance,
    /// `SV_RenderTargetArrayIndex`.
    RenderTargetArrayIndex,
    /// `SV_ViewportArrayIndex`.
    ViewportArrayIndex,
    /// `SV_VertexID`.
    VertexId,
    /// `SV_PrimitiveID`.
    PrimitiveId,
    /// `SV_InstanceID`.
    InstanceId,
    /// `SV_IsFrontFace`.
    IsFrontFace,
    /// `SV_SampleIndex`.
    SampleIndex,
    /// Quad-domain edge tessellation factors.
    FinalQuadEdgeTessFactor,
    /// Quad-domain inside tessellation factors.
    FinalQuadInsideTessFactor,
    /// Triangle-domain edge tessellation factors.
    FinalTriEdgeTessFactor,
    /// Triangle-domain inside tessellation factor.
    FinalTriInsideTessFactor,
    /// Isoline detail tessellation factor.
    FinalLineDetailTessFactor,
    /// Isoline density tessellation factor.
    FinalLineDensityTessFactor,
    /// `SV_Barycentrics`.
    Barycentrics,
    /// `SV_ShadingRate`.
    ShadingRate,
    /// `SV_CullPrimitive`.
    CullPrimitive,
    /// `SV_Target`.
    Target,
    /// `SV_Depth`.
    Depth,
    /// `SV_Coverage`.
    Coverage,
    /// `SV_DepthGreaterEqual`.
    DepthGreaterEqual,
    /// `SV_DepthLessEqual`.
    DepthLessEqual,
    /// `SV_StencilRef`.
    StencilRef,
    /// `SV_InnerCoverage`.
    InnerCoverage,
    /// Unrecognised raw system-value ID.
    Unknown(u32),
}

impl SignatureSystemValue {
    /// Decode a raw signature system-value ID.
    #[must_use]
    pub fn from_u32(value: u32) -> Self {
        match value {
            0 => Self::Undefined,
            1 => Self::Position,
            2 => Self::ClipDistance,
            3 => Self::CullDistance,
            4 => Self::RenderTargetArrayIndex,
            5 => Self::ViewportArrayIndex,
            6 => Self::VertexId,
            7 => Self::PrimitiveId,
            8 => Self::InstanceId,
            9 => Self::IsFrontFace,
            10 => Self::SampleIndex,
            11 => Self::FinalQuadEdgeTessFactor,
            12 => Self::FinalQuadInsideTessFactor,
            13 => Self::FinalTriEdgeTessFactor,
            14 => Self::FinalTriInsideTessFactor,
            15 => Self::FinalLineDetailTessFactor,
            16 => Self::FinalLineDensityTessFactor,
            23 => Self::Barycentrics,
            24 => Self::ShadingRate,
            25 => Self::CullPrimitive,
            64 => Self::Target,
            65 => Self::Depth,
            66 => Self::Coverage,
            67 => Self::DepthGreaterEqual,
            68 => Self::DepthLessEqual,
            69 => Self::StencilRef,
            70 => Self::InnerCoverage,
            value => Self::Unknown(value),
        }
    }

    /// Encode this value to its raw signature system-value ID.
    #[must_use]
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Undefined => 0,
            Self::Position => 1,
            Self::ClipDistance => 2,
            Self::CullDistance => 3,
            Self::RenderTargetArrayIndex => 4,
            Self::ViewportArrayIndex => 5,
            Self::VertexId => 6,
            Self::PrimitiveId => 7,
            Self::InstanceId => 8,
            Self::IsFrontFace => 9,
            Self::SampleIndex => 10,
            Self::FinalQuadEdgeTessFactor => 11,
            Self::FinalQuadInsideTessFactor => 12,
            Self::FinalTriEdgeTessFactor => 13,
            Self::FinalTriInsideTessFactor => 14,
            Self::FinalLineDetailTessFactor => 15,
            Self::FinalLineDensityTessFactor => 16,
            Self::Barycentrics => 23,
            Self::ShadingRate => 24,
            Self::CullPrimitive => 25,
            Self::Target => 64,
            Self::Depth => 65,
            Self::Coverage => 66,
            Self::DepthGreaterEqual => 67,
            Self::DepthLessEqual => 68,
            Self::StencilRef => 69,
            Self::InnerCoverage => 70,
            Self::Unknown(value) => value,
        }
    }

    /// Canonical editable name for a published value.
    #[must_use]
    pub fn name(self) -> Option<&'static str> {
        Some(match self {
            Self::Undefined => "undefined",
            Self::Position => "position",
            Self::ClipDistance => "clip_distance",
            Self::CullDistance => "cull_distance",
            Self::RenderTargetArrayIndex => "render_target_array_index",
            Self::ViewportArrayIndex => "viewport_array_index",
            Self::VertexId => "vertex_id",
            Self::PrimitiveId => "primitive_id",
            Self::InstanceId => "instance_id",
            Self::IsFrontFace => "is_front_face",
            Self::SampleIndex => "sample_index",
            Self::FinalQuadEdgeTessFactor => "finalQuadEdgeTessFactor",
            Self::FinalQuadInsideTessFactor => "finalQuadInsideTessFactor",
            Self::FinalTriEdgeTessFactor => "finalTriEdgeTessFactor",
            Self::FinalTriInsideTessFactor => "finalTriInsideTessFactor",
            Self::FinalLineDetailTessFactor => "finalLineDetailTessFactor",
            Self::FinalLineDensityTessFactor => "finalLineDensityTessFactor",
            Self::Barycentrics => "barycentrics",
            Self::ShadingRate => "shading_rate",
            Self::CullPrimitive => "cull_primitive",
            Self::Target => "target",
            Self::Depth => "depth",
            Self::Coverage => "coverage",
            Self::DepthGreaterEqual => "depth_greater_equal",
            Self::DepthLessEqual => "depth_less_equal",
            Self::StencilRef => "stencil_ref",
            Self::InnerCoverage => "inner_coverage",
            Self::Unknown(_) => return None,
        })
    }

    /// Parse a canonical published name.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        const VALUES: &[u32] = &[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 23, 24, 25, 64, 65, 66, 67,
            68, 69, 70,
        ];
        VALUES
            .iter()
            .copied()
            .map(Self::from_u32)
            .find(|value| value.name() == Some(name))
    }
}

/// A parsed input or output signature element.
#[derive(Debug)]
pub struct SignatureElement<'a> {
    /// Semantic name (e.g. `"POSITION"`, `"TEXCOORD"`).
    pub semantic_name: Cow<'a, str>,
    /// Semantic index (e.g. `0` for `TEXCOORD0`, `1` for `TEXCOORD1`).
    pub semantic_index: u32,
    /// System-value semantic (`D3D_NAME`).
    pub system_value: SignatureSystemValue,
    /// Component data type (`D3D_REGISTER_COMPONENT_TYPE` value).
    pub component_type: u32,
    /// Register number (`v#` for inputs, `o#` for outputs).
    pub register: u32,
    /// Write mask — which `.xyzw` components are written.
    pub mask: u8,
    /// Read/write mask — which components are actually read or written.
    pub rw_mask: u8,
    /// Opaque two-byte alignment padding at the end of the base parameter.
    ///
    /// The Microsoft structure leaves these bytes unspecified. Compiler and
    /// assembler output does not consistently zero them, so preserving the
    /// value is required for byte-exact round trips.
    pub padding: u16,
    /// GS output stream index (OSG5 / ISG1 / OSG1 / PSG1 only).
    pub stream: Option<u32>,
    /// Minimum precision hint (ISG1 / OSG1 / PSG1 only).
    pub min_precision: Option<MinPrecision>,
}

impl SignatureElement<'_> {
    /// The component type name.
    #[must_use]
    pub fn component_type_name(&self) -> &'static str {
        match ComponentType::from_u32(self.component_type) {
            Some(ct) => ct.name(),
            None => "unknown",
        }
    }

    /// Length in characters of the "name with index" string.
    #[must_use]
    pub fn name_with_index_len(&self) -> usize {
        let base = self.semantic_name.len();
        if self.semantic_index > 0 {
            base + digit_count(self.semantic_index)
        } else {
            base
        }
    }

    /// Write the semantic name with index suffix directly to a formatter.
    ///
    /// # Errors
    ///
    /// Returns the formatter's error if any write fails.
    pub fn fmt_name_with_index(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.semantic_name)?;
        if self.semantic_index > 0 {
            write!(f, "{}", self.semantic_index)?;
        }
        Ok(())
    }

    /// Write the columns after the name: type and register.mask.
    ///
    /// # Errors
    ///
    /// Returns the formatter's error if any write fails.
    pub fn fmt_columns(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:>7} v{}.", self.component_type_name(), self.register)?;
        write_mask(f, self.mask)
    }
}

impl fmt::Display for SignatureElement<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use padding via a Display wrapper to avoid allocating a String.
        write!(f, "{:<24} ", DisplayNameWithIndex(self))?;
        self.fmt_columns(f)?;
        if let Some(s) = self.stream.filter(|stream| *stream != 0) {
            write!(f, " stream={s}")?;
        }

        if let Some(mp) = self.min_precision {
            write!(f, "{mp}")?;
        }

        Ok(())
    }
}

/// Display adapter that writes `semantic_name` + optional index suffix.
struct DisplayNameWithIndex<'a, 'b>(&'a SignatureElement<'b>);

impl fmt::Display for DisplayNameWithIndex<'_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_name_with_index(f)
    }
}

fn write_mask(f: &mut fmt::Formatter<'_>, mask: u8) -> fmt::Result {
    if mask & 1 != 0 {
        f.write_str("x")?;
    }
    if mask & 2 != 0 {
        f.write_str("y")?;
    }
    if mask & 4 != 0 {
        f.write_str("z")?;
    }
    if mask & 8 != 0 {
        f.write_str("w")?;
    }
    Ok(())
}

/// Number of decimal digits in a u32.
fn digit_count(mut n: u32) -> usize {
    let mut count = 1;
    while n >= 10 {
        n /= 10;
        count += 1;
    }
    count
}

/// Parse a signature chunk. The `fourcc` string determines the element layout:
/// - `"ISGN"` / `"OSGN"` / `"PCSG"` → 24 bytes (base)
/// - `"OSG5"` → 28 bytes (adds `stream`)
/// - `"ISG1"` / `"OSG1"` / `"PSG1"` → 32 bytes (adds `stream` + `min_precision`)
#[must_use]
pub fn parse_signature<'a>(fourcc: &str, data: &'a [u8]) -> Option<Vec<SignatureElement<'a>>> {
    parse_signature_data(fourcc, data).map(|(elements, _)| elements)
}

/// Parse a complete signature chunk, including opaque trailing string-table
/// data needed for a byte-exact rewrite.
#[must_use]
pub fn parse_signature_chunk(fourcc: [u8; 4], data: &[u8]) -> Option<Signature<'_>> {
    let fourcc_str = core::str::from_utf8(&fourcc).ok()?;
    let (elements, trailing_data) = parse_signature_data(fourcc_str, data)?;
    let mut signature = Signature {
        fourcc,
        elements,
        trailing_data,
        original_data: None,
        original_canonical: None,
    };
    signature.original_canonical = Some(
        write_signature_with_trailing(
            signature.fourcc,
            &signature.elements,
            &signature.trailing_data,
        )
        .data,
    );
    signature.original_data = Some(Cow::Borrowed(data));
    Some(signature)
}

fn parse_signature_data<'a>(
    fourcc: &str,
    data: &'a [u8],
) -> Option<(Vec<SignatureElement<'a>>, Vec<u8>)> {
    if data.len() < 8 {
        return None;
    }

    let ver = SignatureVersion::from_fourcc(fourcc)?;
    let stride = ver.stride();
    let count = usize::try_from(read_u32(data, 0)?).ok()?;
    let parameter_offset = usize::try_from(read_u32(data, 4)?).ok()?;
    if parameter_offset < 8 || parameter_offset % 4 != 0 {
        return None;
    }
    let parameter_bytes = count.checked_mul(stride)?;
    let parameters_end = parameter_offset.checked_add(parameter_bytes)?;
    if parameters_end > data.len() {
        return None;
    }

    let mut elements = Vec::with_capacity(count);
    let mut strings_end = parameters_end;

    for i in 0..count {
        let base = parameter_offset.checked_add(i.checked_mul(stride)?)?;
        let mut field = base;

        let stream = if ver.has_stream() {
            let value = read_u32(data, field)?;
            field = field.checked_add(4)?;
            if value > 3 {
                return None;
            }
            Some(value)
        } else {
            None
        };

        let name_offset = usize::try_from(read_u32(data, field)?).ok()?;
        field = field.checked_add(4)?;
        let semantic_index = read_u32(data, field)?;
        field = field.checked_add(4)?;
        let system_value = SignatureSystemValue::from_u32(read_u32(data, field)?);
        field = field.checked_add(4)?;
        let component_type = read_u32(data, field)?;
        field = field.checked_add(4)?;
        let register = read_u32(data, field)?;
        field = field.checked_add(4)?;
        let mask = *data.get(field)?;
        let rw_mask = *data.get(field.checked_add(1)?)?;
        let padding = u16::from_le_bytes(
            data.get(field.checked_add(2)?..field.checked_add(4)?)?
                .try_into()
                .ok()?,
        );
        if mask & 0xf0 != 0 || rw_mask & 0xf0 != 0 {
            return None;
        }
        field = field.checked_add(4)?;

        let min_precision = if ver.has_min_precision() {
            let value = MinPrecision::from_u32(read_u32(data, field)?);
            Some(value)
        } else {
            None
        };

        if name_offset < parameters_end {
            return None;
        }
        let semantic_name = read_cstring_checked(data, name_offset)?;
        strings_end = strings_end.max(
            name_offset
                .checked_add(semantic_name.len())?
                .checked_add(1)?,
        );

        elements.push(SignatureElement {
            semantic_name: Cow::Borrowed(semantic_name),
            semantic_index,
            system_value,
            component_type,
            register,
            mask,
            rw_mask,
            padding,
            stream,
            min_precision,
        });
    }

    Some((elements, Vec::from(data.get(strings_end..)?)))
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    Some(u32::from_le_bytes(data.get(offset..end)?.try_into().ok()?))
}

fn read_cstring_checked(data: &[u8], offset: usize) -> Option<&str> {
    let tail = data.get(offset..)?;
    let length = tail.iter().position(|&byte| byte == 0)?;
    core::str::from_utf8(&tail[..length]).ok()
}

/// Serialize a signature chunk back to bytes.
///
/// `fourcc` is the 4-byte chunk tag (e.g. `*b"ISGN"`, `*b"OSG1"`).
/// Returns a `WritableChunk` ready for `build_dxbc`.
///
/// # Panics
///
/// Panics if the element count or resulting string-table offset exceeds a
/// 32-bit signature field.
#[must_use]
pub fn write_signature(fourcc: [u8; 4], elements: &[SignatureElement<'_>]) -> WritableChunk {
    write_signature_with_trailing(fourcc, elements, &[])
}

/// Serialize a signature and append preserved bytes after its referenced
/// semantic strings.
///
/// # Panics
///
/// Panics if the `fourcc` is not a supported ASCII signature tag, or if an
/// element count, semantic-name offset, or serialized size exceeds its 32-bit
/// wire field.
#[must_use]
pub fn write_signature_with_trailing(
    fourcc: [u8; 4],
    elements: &[SignatureElement<'_>],
    trailing_data: &[u8],
) -> WritableChunk {
    let fourcc_str = core::str::from_utf8(&fourcc).expect("signature FourCC must be ASCII");
    let ver = SignatureVersion::from_fourcc(fourcc_str).expect("unsupported signature FourCC");
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
    w(
        &mut buf,
        u32::try_from(elements.len()).expect("signature element count exceeds u32"),
    );
    w(&mut buf, 8);

    // Elements
    for (i, elem) in elements.iter().enumerate() {
        if ver.has_stream() {
            w(&mut buf, elem.stream.unwrap_or(0));
        }

        w(&mut buf, name_offsets[i]);
        w(&mut buf, elem.semantic_index);
        w(&mut buf, elem.system_value.to_u32());
        w(&mut buf, elem.component_type);
        w(&mut buf, elem.register);
        buf.push(elem.mask);
        buf.push(elem.rw_mask);
        buf.extend_from_slice(&elem.padding.to_le_bytes());
        if ver.has_min_precision() {
            w(&mut buf, elem.min_precision.map_or(0, MinPrecision::to_u32));
        }
    }

    // String table
    buf.extend_from_slice(&strings.finish());
    buf.extend_from_slice(trailing_data);

    // New signatures use neutral zero padding. Parsed signatures supply their
    // exact compiler-specific padding through `trailing_data` above.
    while buf.len() % 4 != 0 {
        buf.push(0);
    }

    WritableChunk { fourcc, data: buf }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signature_system_values_match_the_container_enum() {
        let cases = [
            (11, SignatureSystemValue::FinalQuadEdgeTessFactor),
            (16, SignatureSystemValue::FinalLineDensityTessFactor),
            (23, SignatureSystemValue::Barycentrics),
            (25, SignatureSystemValue::CullPrimitive),
            (64, SignatureSystemValue::Target),
            (70, SignatureSystemValue::InnerCoverage),
        ];
        for (raw, expected) in cases {
            assert_eq!(SignatureSystemValue::from_u32(raw), expected);
            assert_eq!(expected.to_u32(), raw);
        }
    }

    #[test]
    fn compiler_specific_string_padding_roundtrips() {
        let element = SignatureElement {
            semantic_name: Cow::Borrowed("A"),
            semantic_index: 0,
            system_value: SignatureSystemValue::Undefined,
            component_type: 3,
            register: 0,
            mask: 0xf,
            rw_mask: 0xf,
            padding: 0,
            stream: Some(0),
            min_precision: Some(MinPrecision::Default),
        };
        let mut bytes = write_signature(*b"ISG1", &[element]).data;
        let len = bytes.len();
        bytes[len - 2..].copy_from_slice(&[0xab, 0xcd]);

        let parsed = parse_signature_chunk(*b"ISG1", &bytes).expect("valid signature");
        assert_eq!(parsed.trailing_data, [0xab, 0xcd]);
        assert_eq!(parsed.write_payload(), bytes);
    }

    #[test]
    fn accepted_parameter_offset_is_preserved_until_edited() {
        let element = SignatureElement {
            semantic_name: Cow::Borrowed("A"),
            semantic_index: 0,
            system_value: SignatureSystemValue::Undefined,
            component_type: 3,
            register: 0,
            mask: 0xf,
            rw_mask: 0xf,
            padding: 0,
            stream: None,
            min_precision: None,
        };
        let canonical = write_signature(*b"ISGN", &[element]).data;
        let mut noncanonical = canonical.clone();
        noncanonical[4..8].copy_from_slice(&12u32.to_le_bytes());
        noncanonical.splice(8..8, [0xde, 0xad, 0xbe, 0xef]);
        noncanonical[12..16].copy_from_slice(&36u32.to_le_bytes());

        let mut parsed =
            parse_signature_chunk(*b"ISGN", &noncanonical).expect("valid offset-based signature");
        assert_eq!(parsed.write_payload(), noncanonical);
        parsed.elements[0].semantic_index = 1;
        let edited = parsed.write_payload();
        assert_ne!(edited, noncanonical);
        assert_eq!(
            parse_signature_chunk(*b"ISGN", &edited)
                .expect("canonical edited signature")
                .elements[0]
                .semantic_index,
            1
        );
    }
}
