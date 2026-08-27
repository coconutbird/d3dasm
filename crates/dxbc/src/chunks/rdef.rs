//! RDEF chunk parser — resource definitions.
//!
//! The RDEF chunk describes constant buffers, resource bindings, and the
//! compiler creator string. Layout (28-byte header):
//!   0x00: u32 — constant buffer count
//!   0x04: u32 — constant buffer offset
//!   0x08: u32 — bound resource count
//!   0x0C: u32 — bound resource offset
//!   0x10: u32 — target version (minor | major<<8 | type<<16)
//!   0x14: u32 — compile flags
//!   0x18: u32 — creator string offset

use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::ChunkWriter;

/// Parsed RDEF chunk — constant buffers, resource bindings, and creator string.
#[derive(Debug)]
pub struct ResourceDef<'a> {
    /// Constant buffer definitions (layouts and variables).
    pub constant_buffers: Vec<CBufferDef<'a>>,
    /// Shader resource bindings (textures, samplers, UAVs, etc.).
    pub bindings: Vec<ResourceBinding<'a>>,
    /// Compiler identification string (e.g. `"Microsoft (R) HLSL Shader Compiler 10.1"`).
    pub creator: Cow<'a, str>,
    /// Target version (minor | major<<8 | type<<16).
    pub target_version: u32,
    /// Compile flags.
    pub compile_flags: u32,
    /// SM5 RD11 layout descriptor (32 bytes after the main header). None for SM4.
    pub rd11: Option<Rd11Header>,
    /// Original bytes retained for a layout-preserving no-op rewrite.
    #[doc(hidden)]
    pub original_data: Option<Cow<'a, [u8]>>,
    /// Canonical encoding of the semantic state at parse time.
    #[doc(hidden)]
    pub original_canonical: Option<Vec<u8>>,
}

/// The SM5 RD11 layout descriptor embedded after the 28-byte RDEF header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rd11Header {
    /// Format identifier (`RD11` for SM5.0, `0x2544_1313` for SM5.1).
    pub identifier: u32,
    /// Total RDEF+RD11 header size (currently 60 bytes).
    pub header_size: u32,
    /// Constant-buffer record stride.
    pub constant_buffer_stride: u32,
    /// Resource-binding record stride (32 for SM5.0, 40 for SM5.1).
    pub resource_binding_stride: u32,
    /// Constant-buffer variable record stride.
    pub variable_stride: u32,
    /// Type descriptor stride.
    pub type_stride: u32,
    /// Structure-member record stride.
    pub member_stride: u32,
    /// Number of interface slots reported by reflection.
    pub interface_slot_count: u32,
}

impl Rd11Header {
    /// Standard SM5.0 layout descriptor.
    pub const SM5_0: Self = Self {
        identifier: u32::from_le_bytes(*b"RD11"),
        header_size: 60,
        constant_buffer_stride: 24,
        resource_binding_stride: 32,
        variable_stride: 40,
        type_stride: 36,
        member_stride: 12,
        interface_slot_count: 0,
    };

    /// Return the serialized dwords in order.
    #[must_use]
    pub const fn to_words(self) -> [u32; 8] {
        [
            self.identifier,
            self.header_size,
            self.constant_buffer_stride,
            self.resource_binding_stride,
            self.variable_stride,
            self.type_stride,
            self.member_stride,
            self.interface_slot_count,
        ]
    }

    /// Decode the eight serialized dwords.
    #[must_use]
    pub const fn from_words(words: [u32; 8]) -> Self {
        Self {
            identifier: words[0],
            header_size: words[1],
            constant_buffer_stride: words[2],
            resource_binding_stride: words[3],
            variable_stride: words[4],
            type_stride: words[5],
            member_stride: words[6],
            interface_slot_count: words[7],
        }
    }
}

/// A single constant buffer and its variable layout.
#[derive(Debug)]
pub struct CBufferDef<'a> {
    /// Constant buffer name (e.g. `"$Globals"`, `"cb0"`).
    pub name: Cow<'a, str>,
    /// Variables declared inside the buffer.
    pub variables: Vec<CBufferVariable<'a>>,
    /// Total buffer size in bytes.
    pub size: u32,
    /// Buffer flags.
    pub flags: u32,
    /// Buffer type (0=cbuffer, 1=tbuffer, etc.).
    pub cb_type: u32,
}

/// A variable inside a constant buffer.
#[derive(Debug)]
pub struct CBufferVariable<'a> {
    /// Variable name.
    pub name: Cow<'a, str>,
    /// Byte offset within the constant buffer.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
    /// Variable flags.
    pub flags: u32,
    /// Parsed type descriptor for this variable.
    pub var_type: TypeDesc<'a>,
    /// Default value bytes (empty if none).
    pub default_value: Cow<'a, [u8]>,
    /// SM5 extra: start texture slot (-1 if unused).
    pub texture_start: Option<u32>,
    /// SM5 extra: texture bind count.
    pub texture_size: Option<u32>,
    /// SM5 extra: start sampler slot (-1 if unused).
    pub sampler_start: Option<u32>,
    /// SM5 extra: sampler bind count.
    pub sampler_size: Option<u32>,
}

/// HLSL type descriptor (`D3D11_SHADER_TYPE_DESC`).
///
/// Binary layout (all u32):
///   0: `[class:16 | type:16]`
///   4: `[rows:16 | columns:16]`
///   8: `[elements:16 | members:16]`
///  12: member descriptor offset
/// SM5 adds 4 unknown u32s + 1 name offset u32.
#[derive(Debug, Clone, PartialEq)]
pub struct TypeDesc<'a> {
    /// Variable class (scalar, vector, matrix, object, struct).
    pub class: u16,
    /// Variable type (void, bool, int, float, etc.).
    pub var_type: u16,
    /// Number of rows (1 for scalars/vectors, >1 for matrices).
    pub rows: u16,
    /// Number of columns.
    pub columns: u16,
    /// Array element count (0 if not an array).
    pub elements: u16,
    /// Struct member descriptors (empty if not a struct).
    pub members: Vec<MemberDesc<'a>>,
    /// SM5 extra: 4 unknown u32 values preserved for round-trip.
    pub sm5_extra: Option<[u32; 4]>,
    /// Type name (SM5 interface types only, empty otherwise).
    pub name: Cow<'a, str>,
}

/// A struct member descriptor inside a type.
///
/// Binary layout (12 bytes):
///   0: u32 name offset
///   4: u32 type offset (recursive)
///   8: u32 byte offset within parent
#[derive(Debug, Clone, PartialEq)]
pub struct MemberDesc<'a> {
    /// Member name.
    pub name: Cow<'a, str>,
    /// Parsed type for this member.
    pub member_type: TypeDesc<'a>,
    /// Byte offset within the parent structure.
    pub offset: u32,
}

/// Resource input type (`D3D_SHADER_INPUT_TYPE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ResourceInputType {
    /// Constant buffer.
    CBuffer = 0,
    /// Texture buffer.
    TBuffer = 1,
    /// Texture (SRV).
    Texture = 2,
    /// Sampler state.
    Sampler = 3,
    /// Read-write typed UAV.
    UavRwTyped = 4,
    /// Structured buffer (SRV).
    Structured = 5,
    /// Read-write structured buffer (UAV).
    UavRwStructured = 6,
    /// Byte-address buffer (SRV).
    ByteAddress = 7,
    /// Read-write byte-address buffer (UAV).
    UavRwByteAddress = 8,
    /// Append structured buffer (UAV).
    UavAppendStructured = 9,
    /// Consume structured buffer (UAV).
    UavConsumeStructured = 10,
    /// Read-write structured buffer with hidden counter (UAV).
    UavRwStructuredWithCounter = 11,
    /// Ray-tracing acceleration structure.
    RtAccelerationStructure = 12,
    /// Sampler-feedback texture UAV.
    UavFeedbackTexture = 13,
}

impl ResourceInputType {
    /// Converts a raw `D3D_SHADER_INPUT_TYPE` value to the corresponding variant.
    #[must_use]
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::CBuffer,
            1 => Self::TBuffer,
            2 => Self::Texture,
            3 => Self::Sampler,
            4 => Self::UavRwTyped,
            5 => Self::Structured,
            6 => Self::UavRwStructured,
            7 => Self::ByteAddress,
            8 => Self::UavRwByteAddress,
            9 => Self::UavAppendStructured,
            10 => Self::UavConsumeStructured,
            11 => Self::UavRwStructuredWithCounter,
            12 => Self::RtAccelerationStructure,
            13 => Self::UavFeedbackTexture,
            _ => return None,
        })
    }

    /// Returns the lowercase name used in disassembly output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::CBuffer => "cbuffer",
            Self::TBuffer => "tbuffer",
            Self::Texture => "texture",
            Self::Sampler => "sampler",
            Self::UavRwTyped => "uav_rwtyped",
            Self::Structured => "structured",
            Self::UavRwStructured => "uav_rwstructured",
            Self::ByteAddress => "byteaddress",
            Self::UavRwByteAddress => "uav_rwbyteaddress",
            Self::UavAppendStructured => "uav_append_structured",
            Self::UavConsumeStructured => "uav_consume_structured",
            Self::UavRwStructuredWithCounter => "uav_rwstructured_with_counter",
            Self::RtAccelerationStructure => "rt_acceleration_structure",
            Self::UavFeedbackTexture => "uav_feedback_texture",
        }
    }
}

/// Resource dimension (`D3D_SRV_DIMENSION`).
///
/// NOTE: this enum's numbering is **not** the same as the shader-bytecode
/// resource-dimension enum (`D3D10_SB_RESOURCE_DIMENSION`) used by `dcl_resource`
/// in the SHEX chunk — e.g. SRV `Texture2DArray` is 5 here but 8 there. The RDEF
/// binding's dimension field is an SRV dimension, so it must be decoded with
/// this table; mixing the two mislabels every texture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ResourceDimension {
    /// Dimension is not applicable or unknown.
    Unknown = 0,
    /// Buffer resource.
    Buffer = 1,
    /// 1D texture.
    Texture1D = 2,
    /// 1D texture array.
    Texture1DArray = 3,
    /// 2D texture.
    Texture2D = 4,
    /// 2D texture array.
    Texture2DArray = 5,
    /// 2D multisampled texture.
    Texture2DMS = 6,
    /// 2D multisampled texture array.
    Texture2DMSArray = 7,
    /// 3D (volume) texture.
    Texture3D = 8,
    /// Cube-map texture.
    TextureCube = 9,
    /// Cube-map texture array.
    TextureCubeArray = 10,
    /// Extended buffer dimension.
    BufferEx = 11,
}

impl ResourceDimension {
    /// Converts a raw `D3D_SRV_DIMENSION` value to the corresponding variant.
    #[must_use]
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            0 => Self::Unknown,
            1 => Self::Buffer,
            2 => Self::Texture1D,
            3 => Self::Texture1DArray,
            4 => Self::Texture2D,
            5 => Self::Texture2DArray,
            6 => Self::Texture2DMS,
            7 => Self::Texture2DMSArray,
            8 => Self::Texture3D,
            9 => Self::TextureCube,
            10 => Self::TextureCubeArray,
            11 => Self::BufferEx,
            _ => return None,
        })
    }

    /// Returns the short dimension name used in disassembly output.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Unknown => "NA",
            Self::Buffer => "buf",
            Self::Texture1D => "1d",
            Self::Texture1DArray => "1darray",
            Self::Texture2D => "2d",
            Self::Texture2DArray => "2darray",
            Self::Texture2DMS => "2dMS",
            Self::Texture2DMSArray => "2dMSarray",
            Self::Texture3D => "3d",
            Self::TextureCube => "cube",
            Self::TextureCubeArray => "cubearray",
            Self::BufferEx => "bufex",
        }
    }

    /// The HLSL type stem for this dimension (`Texture2DArray`, etc.). Used when
    /// reconstructing editable resource declarations.
    #[must_use]
    pub fn hlsl_stem(self) -> &'static str {
        match self {
            Self::Unknown => "Unknown",
            Self::Buffer => "Buffer",
            Self::Texture1D => "Texture1D",
            Self::Texture1DArray => "Texture1DArray",
            Self::Texture2D => "Texture2D",
            Self::Texture2DArray => "Texture2DArray",
            Self::Texture2DMS => "Texture2DMS",
            Self::Texture2DMSArray => "Texture2DMSArray",
            Self::Texture3D => "Texture3D",
            Self::TextureCube => "TextureCube",
            Self::TextureCubeArray => "TextureCubeArray",
            Self::BufferEx => "BufferEx",
        }
    }

    /// Inverse of [`hlsl_stem`](Self::hlsl_stem).
    #[must_use]
    pub fn from_hlsl_stem(s: &str) -> Option<Self> {
        Some(match s {
            "Unknown" => Self::Unknown,
            "Buffer" => Self::Buffer,
            "Texture1D" => Self::Texture1D,
            "Texture1DArray" => Self::Texture1DArray,
            "Texture2D" => Self::Texture2D,
            "Texture2DArray" => Self::Texture2DArray,
            "Texture2DMS" => Self::Texture2DMS,
            "Texture2DMSArray" => Self::Texture2DMSArray,
            "Texture3D" => Self::Texture3D,
            "TextureCube" => Self::TextureCube,
            "TextureCubeArray" => Self::TextureCubeArray,
            "BufferEx" => Self::BufferEx,
            _ => return None,
        })
    }
}

// Binding flags.
//
// Two distinct fields in RDEF carry "flags" with *different* bit meanings; keep
// them apart to avoid the historical mix-up.

/// `D3D_SHADER_INPUT_FLAGS` — the flags field of a *resource binding*.
pub const SIF_USER_PACKED: u32 = 0x1;
/// Binding is a comparison sampler.
pub const SIF_COMPARISON_SAMPLER: u32 = 0x2;
/// Texture return-component bit 0. Together with bit 1 encodes (components − 1).
pub const SIF_TEXTURE_COMPONENT_0: u32 = 0x4;
/// Texture return-component bit 1.
pub const SIF_TEXTURE_COMPONENT_1: u32 = 0x8;
/// Both texture-component bits set → a 4-component return type.
pub const SIF_TEX_COMPONENTS: u32 = SIF_TEXTURE_COMPONENT_0 | SIF_TEXTURE_COMPONENT_1;
/// Binding is declared but not used by the shader.
pub const SIF_UNUSED: u32 = 0x10;

/// `D3D_SHADER_VARIABLE_FLAGS::D3D_SVF_USED` — a *cbuffer variable* is read by
/// the shader. This is a different field from the binding flags above (where bit
/// `0x2` instead means `comparisonSampler`).
pub const SVF_USED: u32 = 0x2;

/// Sentinel value for unused texture/sampler start slots.
pub const SLOT_UNUSED: u32 = 0xFFFF_FFFF;

/// A shader resource binding (texture, sampler, cbuffer, UAV, etc.).
#[derive(Debug)]
pub struct ResourceBinding<'a> {
    /// Binding name.
    pub name: Cow<'a, str>,
    /// Resource type (0=cbuffer, 2=texture, 3=sampler, `4=uav_rwtyped`, …).
    pub input_type: u32,
    /// Return type for typed resources.
    pub return_type: u32,
    /// Resource dimension (1=buffer, 2=1d, 3=2d, …).
    pub dimension: u32,
    /// Number of samples (for multisampled resources).
    pub num_samples: u32,
    /// Register slot (e.g. `t0`, `s1`, `b2`).
    pub bind_point: u32,
    /// Number of contiguous registers bound.
    pub bind_count: u32,
    /// Binding flags (`D3D_SHADER_INPUT_FLAGS`: userPacked, comparisonSampler,
    /// texComp0/1, unused).
    pub flags: u32,
    /// Register space (SM5.1 binding records; zero for older layouts).
    pub space: u32,
    /// Resource range ID in shader bytecode (SM5.1; zero for older layouts).
    pub id: u32,
}

impl ResourceBinding<'_> {
    fn type_name(&self) -> &'static str {
        match ResourceInputType::from_u32(self.input_type) {
            Some(t) => t.name(),
            None => "unknown",
        }
    }

    fn dim_name(&self) -> &'static str {
        match ResourceDimension::from_u32(self.dimension) {
            Some(d) => d.name(),
            None => "NA",
        }
    }

    fn write_flags(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.flags == 0 {
            return Ok(());
        }
        let parts: &[(&str, u32)] = &[
            ("userPacked", SIF_USER_PACKED),
            ("comparisonSampler", SIF_COMPARISON_SAMPLER),
            ("texComp0", SIF_TEXTURE_COMPONENT_0),
            ("texComp1", SIF_TEXTURE_COMPONENT_1),
            ("unused", SIF_UNUSED),
        ];
        let mut first = true;
        let mut matched = false;
        for &(name, bit) in parts {
            if self.flags & bit != 0 {
                if !first {
                    f.write_str(";")?;
                }
                f.write_str(name)?;
                first = false;
                matched = true;
            }
        }
        if !matched {
            write!(f, "0x{:x}", self.flags)?;
        }
        Ok(())
    }

    /// Write the type, dimension, slot, bind count, and flags columns.
    ///
    /// # Errors
    ///
    /// Returns the formatter's error if any write fails.
    pub fn fmt_columns(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:<12} {:<8} {:<4} {:<5}",
            self.type_name(),
            self.dim_name(),
            self.bind_point,
            self.bind_count,
        )?;
        if self.flags != 0 {
            f.write_str(" ")?;
            self.write_flags(f)?;
        }
        Ok(())
    }
}

impl fmt::Display for ResourceBinding<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<20} ", self.name)?;
        self.fmt_columns(f)
    }
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    Some(u32::from_le_bytes(data.get(offset..end)?.try_into().ok()?))
}

fn read_cstring_checked(data: &[u8], offset: usize) -> Option<&str> {
    let tail = data.get(offset..)?;
    let end = tail.iter().position(|&byte| byte == 0)?;
    core::str::from_utf8(&tail[..end]).ok()
}

fn record_base(data: &[u8], offset: usize, index: usize, stride: usize) -> Option<usize> {
    let base = offset.checked_add(index.checked_mul(stride)?)?;
    let end = base.checked_add(stride)?;
    (end <= data.len()).then_some(base)
}

#[derive(Clone, Copy)]
struct RdefLayout {
    constant_buffer_stride: usize,
    resource_binding_stride: usize,
    variable_stride: usize,
    type_stride: usize,
    member_stride: usize,
    is_sm5: bool,
}

fn parse_bindings(
    data: &[u8],
    count: usize,
    offset: usize,
    stride: usize,
) -> Option<Vec<ResourceBinding<'_>>> {
    if stride < 32 || stride % 4 != 0 || matches!(stride, 33..=39) {
        return None;
    }
    let mut bindings = Vec::new();
    for index in 0..count {
        let base = record_base(data, offset, index, stride)?;
        let name_offset = usize::try_from(read_u32(data, base)?).ok()?;
        let input_type = read_u32(data, base + 4)?;
        let dimension = read_u32(data, base + 12)?;
        let flags = read_u32(data, base + 28)?;
        if ResourceInputType::from_u32(input_type).is_none()
            || ResourceDimension::from_u32(dimension).is_none()
            || flags & !0x1f != 0
        {
            return None;
        }
        bindings.push(ResourceBinding {
            name: Cow::Borrowed(read_cstring_checked(data, name_offset)?),
            input_type,
            return_type: read_u32(data, base + 8)?,
            dimension,
            num_samples: read_u32(data, base + 16)?,
            bind_point: read_u32(data, base + 20)?,
            bind_count: read_u32(data, base + 24)?,
            flags,
            space: if stride >= 40 {
                read_u32(data, base + 32)?
            } else {
                0
            },
            id: if stride >= 40 {
                read_u32(data, base + 36)?
            } else {
                0
            },
        });
    }
    Some(bindings)
}

fn parse_variables(
    data: &[u8],
    count: usize,
    offset: usize,
    layout: RdefLayout,
) -> Option<Vec<CBufferVariable<'_>>> {
    if layout.variable_stride < if layout.is_sm5 { 40 } else { 24 }
        || layout.variable_stride % 4 != 0
    {
        return None;
    }
    let mut variables = Vec::new();
    for index in 0..count {
        let base = record_base(data, offset, index, layout.variable_stride)?;
        let name_offset = usize::try_from(read_u32(data, base)?).ok()?;
        let variable_offset = read_u32(data, base + 4)?;
        let size = read_u32(data, base + 8)?;
        let flags = read_u32(data, base + 12)?;
        let type_offset = usize::try_from(read_u32(data, base + 16)?).ok()?;
        let default_offset = usize::try_from(read_u32(data, base + 20)?).ok()?;
        let (texture_start, texture_size, sampler_start, sampler_size) = if layout.is_sm5 {
            (
                Some(read_u32(data, base + 24)?),
                Some(read_u32(data, base + 28)?),
                Some(read_u32(data, base + 32)?),
                Some(read_u32(data, base + 36)?),
            )
        } else {
            (None, None, None, None)
        };
        let default_value = if default_offset == 0 {
            Cow::Borrowed(&[][..])
        } else {
            let default_size = usize::try_from(size).ok()?;
            let default_end = default_offset.checked_add(default_size)?;
            Cow::Borrowed(data.get(default_offset..default_end)?)
        };
        let mut path = Vec::new();
        variables.push(CBufferVariable {
            name: Cow::Borrowed(read_cstring_checked(data, name_offset)?),
            offset: variable_offset,
            size,
            flags,
            var_type: parse_type_desc(data, type_offset, layout, &mut path, 0)?,
            default_value,
            texture_start,
            texture_size,
            sampler_start,
            sampler_size,
        });
    }
    Some(variables)
}

fn parse_constant_buffers(
    data: &[u8],
    count: usize,
    offset: usize,
    layout: RdefLayout,
) -> Option<Vec<CBufferDef<'_>>> {
    if layout.constant_buffer_stride < 24 || layout.constant_buffer_stride % 4 != 0 {
        return None;
    }
    let mut buffers = Vec::new();
    for index in 0..count {
        let base = record_base(data, offset, index, layout.constant_buffer_stride)?;
        let name_offset = usize::try_from(read_u32(data, base)?).ok()?;
        let variable_count = usize::try_from(read_u32(data, base + 4)?).ok()?;
        let variable_offset = usize::try_from(read_u32(data, base + 8)?).ok()?;
        let size = read_u32(data, base + 12)?;
        let flags = read_u32(data, base + 16)?;
        let buffer_type = read_u32(data, base + 20)?;
        let variables = parse_variables(data, variable_count, variable_offset, layout)?;
        buffers.push(CBufferDef {
            name: Cow::Borrowed(read_cstring_checked(data, name_offset)?),
            variables,
            size,
            flags,
            cb_type: buffer_type,
        });
    }
    Some(buffers)
}

/// Parse an RDEF chunk.
///
/// RDEF header layout (28 bytes):
///   0: u32 `constant_buffer_count`
///   4: u32 `constant_buffer_offset`
///   8: u32 `bound_resource_count`
///  12: u32 `bound_resource_offset`
///  16: u32 `target_version`  (minor | major<<8 | type<<16)
///  20: u32 flags
///  24: u32 `creator_offset`
#[must_use]
pub fn parse_rdef(data: &[u8]) -> Option<ResourceDef<'_>> {
    if data.len() < 28 {
        return None;
    }

    let cb_count = usize::try_from(read_u32(data, 0)?).ok()?;
    let cb_offset = usize::try_from(read_u32(data, 4)?).ok()?;
    let binding_count = usize::try_from(read_u32(data, 8)?).ok()?;
    let binding_offset = usize::try_from(read_u32(data, 12)?).ok()?;
    let target_version = read_u32(data, 16)?;
    let compile_flags = read_u32(data, 20)?;

    // SM5 uses 40-byte variable descriptors; SM4 uses 24-byte.
    let major_version = (target_version >> 8) & 0xFF;
    let is_sm5 = major_version == 5;
    if !matches!(major_version, 4 | 5) {
        return None;
    }

    // Creator string (at offset 24 in the RDEF header)
    let creator_off = usize::try_from(read_u32(data, 24)?).ok()?;
    let creator = Cow::Borrowed(read_cstring_checked(data, creator_off)?);

    // SM5 has an RD11 sub-header (8 u32s = 32 bytes) after the main header.
    let rd11 = if is_sm5 {
        if data.len() < 60 {
            return None;
        }
        let mut words = [0u32; 8];
        for (index, word) in words.iter_mut().enumerate() {
            *word = read_u32(data, 28 + index * 4)?;
        }
        let header = Rd11Header::from_words(words);
        if !matches!(header.identifier, 0x3131_4452 | 0x2544_1313)
            || header.header_size != 60
            || header.constant_buffer_stride != 24
            || !matches!(header.resource_binding_stride, 32 | 40)
            || header.variable_stride != 40
            || header.type_stride != 36
            || header.member_stride != 12
        {
            return None;
        }
        Some(header)
    } else {
        None
    };

    let layout = match rd11 {
        None => RdefLayout {
            constant_buffer_stride: 24,
            resource_binding_stride: 32,
            variable_stride: 24,
            type_stride: 16,
            member_stride: 12,
            is_sm5: false,
        },
        Some(header) => RdefLayout {
            constant_buffer_stride: usize::try_from(header.constant_buffer_stride).ok()?,
            resource_binding_stride: usize::try_from(header.resource_binding_stride).ok()?,
            variable_stride: usize::try_from(header.variable_stride).ok()?,
            type_stride: usize::try_from(header.type_stride).ok()?,
            member_stride: usize::try_from(header.member_stride).ok()?,
            is_sm5: true,
        },
    };

    let bindings = parse_bindings(
        data,
        binding_count,
        binding_offset,
        layout.resource_binding_stride,
    )?;
    let constant_buffers = parse_constant_buffers(data, cb_count, cb_offset, layout)?;

    let mut resource_def = ResourceDef {
        constant_buffers,
        bindings,
        creator,
        target_version,
        compile_flags,
        rd11,
        original_data: None,
        original_canonical: None,
    };
    resource_def.original_canonical = Some(write_rdef_canonical(&resource_def));
    resource_def.original_data = Some(Cow::Borrowed(data));
    Some(resource_def)
}

/// Parse a type descriptor at `offset` within the RDEF chunk data.
fn parse_type_desc<'a>(
    data: &'a [u8],
    offset: usize,
    layout: RdefLayout,
    path: &mut Vec<usize>,
    depth: usize,
) -> Option<TypeDesc<'a>> {
    if depth > 64 || path.contains(&offset) {
        return None;
    }
    let end = offset.checked_add(layout.type_stride)?;
    if end > data.len() {
        return None;
    }
    path.push(offset);
    let class_type = read_u32(data, offset)?;
    let rows_cols = read_u32(data, offset + 4)?;
    let elems_members = read_u32(data, offset + 8)?;
    let member_offset = usize::try_from(read_u32(data, offset + 12)?).ok()?;

    let class = (class_type & 0xFFFF) as u16;
    let var_type = (class_type >> 16) as u16;
    let rows = (rows_cols & 0xFFFF) as u16;
    let columns = (rows_cols >> 16) as u16;
    let elements = (elems_members & 0xFFFF) as u16;
    let member_count = (elems_members >> 16) as u16;

    let sm5_extra = if layout.is_sm5 {
        Some([
            read_u32(data, offset + 16)?,
            read_u32(data, offset + 20)?,
            read_u32(data, offset + 24)?,
            read_u32(data, offset + 28)?,
        ])
    } else {
        None
    };

    // Parse member descriptors (12 bytes each: name_off, type_off, byte_offset)
    let mut members = Vec::new();
    if member_count > 0 {
        for member_index in 0..usize::from(member_count) {
            let mbase = record_base(data, member_offset, member_index, layout.member_stride)?;
            let mname_off = usize::try_from(read_u32(data, mbase)?).ok()?;
            let mtype_off = usize::try_from(read_u32(data, mbase + 4)?).ok()?;
            let moffset = read_u32(data, mbase + 8)?;
            members.push(MemberDesc {
                name: Cow::Borrowed(read_cstring_checked(data, mname_off)?),
                member_type: parse_type_desc(data, mtype_off, layout, path, depth + 1)?,
                offset: moffset,
            });
        }
    }

    // SM5: name offset comes after the 4 unknowns in the type descriptor body
    let name = if layout.is_sm5 {
        let name_offset = usize::try_from(read_u32(data, offset + 32)?).ok()?;
        if name_offset == 0 {
            Cow::Borrowed("")
        } else {
            Cow::Borrowed(read_cstring_checked(data, name_offset)?)
        }
    } else {
        Cow::Borrowed("")
    };

    let result = TypeDesc {
        class,
        var_type,
        rows,
        columns,
        elements,
        members,
        sm5_extra,
        name,
    };
    path.pop();
    Some(result)
}

impl ResourceDef<'_> {
    fn is_sm5(&self) -> bool {
        ((self.target_version >> 8) & 0xFF) >= 5
    }
}

// Low-level helpers for the byte-exact RDEF writer.

fn push_u32(out: &mut Vec<u8>, v: u32) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn wire_u32(value: usize) -> u32 {
    u32::try_from(value).expect("RDEF size or offset exceeds u32")
}

fn patch_u32(out: &mut [u8], pos: usize, v: u32) {
    out[pos..pos + 4].copy_from_slice(&v.to_le_bytes());
}

/// Pad `out` to a 4-byte boundary with fxc's 0xAB fill byte.
fn pad4(out: &mut Vec<u8>) {
    while out.len() % 4 != 0 {
        out.push(0xAB);
    }
}

/// Intern a null-terminated string into a single content-deduped pool,
/// appending it at the current end of `out` on first use.
fn intern(
    out: &mut Vec<u8>,
    pool: &mut alloc::collections::BTreeMap<alloc::string::String, u32>,
    s: &str,
) -> u32 {
    if let Some(&off) = pool.get(s) {
        return off;
    }
    let off = wire_u32(out.len());
    out.extend_from_slice(s.as_bytes());
    out.push(0);
    pool.insert(alloc::string::String::from(s), off);
    off
}

/// `D3D_SHADER_VARIABLE_TYPE` → HLSL scalar base-type name.
fn base_type_name(t: u16) -> Option<&'static str> {
    Some(match t {
        1 => "bool",
        2 => "int",
        3 => "float",
        19 => "uint",
        39 => "double",
        _ => return None,
    })
}

/// HLSL scalar base-type name → `D3D_SHADER_VARIABLE_TYPE`.
fn base_type_from(s: &str) -> Option<u16> {
    Some(match s {
        "bool" => 1,
        "int" => 2,
        "float" => 3,
        "uint" => 19,
        "double" => 39,
        _ => return None,
    })
}

/// Canonical HLSL type spelling for a [`TypeDesc`] (without any `row_major`
/// qualifier or array suffix handling beyond `[N]`). Returns `None` for types
/// outside the standard scalar/vector/matrix/struct families so the caller can
/// fall back to the explicit numeric form.
#[must_use]
pub fn hlsl_type_name(td: &TypeDesc<'_>) -> Option<alloc::string::String> {
    use alloc::string::String;
    // Struct / object types are named directly.
    if !td.members.is_empty() || td.class == 5 {
        if td.name.is_empty() {
            return None;
        }
        return Some(String::from(td.name.as_ref()));
    }
    let base = base_type_name(td.var_type)?;
    let core = match td.class {
        0 => String::from(base),                                     // scalar
        1 => alloc::format!("{base}{}", td.columns),                 // vector
        2 | 3 => alloc::format!("{base}{}x{}", td.rows, td.columns), // matrix
        _ => return None,
    };
    Some(if td.elements > 0 {
        alloc::format!("{core}[{}]", td.elements)
    } else {
        core
    })
}

/// Parse a canonical HLSL type spelling into `(class, base, rows, cols,
/// elements)`. Matrices default to column-major (`class = 3`); a `row_major`
/// qualifier is handled by the caller. Returns `None` for unrecognised spellings.
#[must_use]
pub fn parse_hlsl_type(s: &str) -> Option<(u16, u16, u16, u16, u16)> {
    // Optional [N] array suffix.
    let (core, elements) = match s.split_once('[') {
        Some((c, rest)) => (c, rest.strip_suffix(']')?.parse().ok()?),
        None => (s, 0u16),
    };
    // Split leading alphabetic base from the trailing dimension digits.
    let split = core
        .find(|c: char| c.is_ascii_digit())
        .unwrap_or(core.len());
    let base = base_type_from(&core[..split])?;
    let dims = &core[split..];
    let (class, rows, cols) = if dims.is_empty() {
        (0u16, 1u16, 1u16) // scalar
    } else if let Some((r, c)) = dims.split_once('x') {
        (3u16, r.parse().ok()?, c.parse().ok()?) // matrix (column-major default)
    } else {
        (1u16, 1u16, dims.parse().ok()?) // vector
    };
    Some((class, base, rows, cols, elements))
}

/// Resolve a variable's type to a descriptor offset, reproducing fxc's sharing:
/// non-array types (`elements == 0`) are de-duplicated by value; array types
/// always get a fresh descriptor. Returns the assigned offset, placing the
/// type's name string + descriptor at the current cursor on first emission.
///
/// Struct members are not laid out here, so types with members won't reproduce
/// exactly — those RDEFs fall back to raw hex via the emit-time verification.
fn place_type<'a>(
    out: &mut Vec<u8>,
    pool: &mut alloc::collections::BTreeMap<alloc::string::String, u32>,
    placed: &mut Vec<(TypeDesc<'a>, u32)>,
    member_pool: &mut Vec<(TypeDesc<'a>, u32)>,
    td: &TypeDesc<'a>,
    is_sm5: bool,
    dedup: bool,
) -> u32 {
    if dedup && td.elements == 0 {
        if let Some((_, off)) = placed.iter().find(|(t, _)| t == td) {
            return *off;
        }
    }
    // Type name first (struct/object types reference an interned name string).
    let name_off = if td.name.is_empty() {
        0
    } else {
        intern(out, pool, &td.name)
    };
    // For struct types: lay out member names + member type descriptors, then
    // the member-descriptor array, before the struct descriptor itself.
    let member_off = if td.members.is_empty() {
        0u32
    } else {
        // Member type descriptors dedup in a pool shared across all structs,
        // but separate from top-level types (fxc never shares the two).
        let mut child_offsets = Vec::with_capacity(td.members.len());
        for m in &td.members {
            intern(out, pool, &m.name);
            let mut nested: Vec<(TypeDesc<'_>, u32)> = Vec::new();
            child_offsets.push(place_type(
                out,
                pool,
                member_pool,
                &mut nested,
                &m.member_type,
                is_sm5,
                true,
            ));
        }
        pad4(out);
        let moff = wire_u32(out.len());
        for (i, m) in td.members.iter().enumerate() {
            let mn = intern(out, pool, &m.name);
            push_u32(out, mn);
            push_u32(out, child_offsets[i]);
            push_u32(out, m.offset);
        }
        moff
    };
    pad4(out);
    let off = wire_u32(out.len());
    push_u32(out, u32::from(td.class) | (u32::from(td.var_type) << 16));
    push_u32(out, u32::from(td.rows) | (u32::from(td.columns) << 16));
    push_u32(
        out,
        u32::from(td.elements)
            | (u32::from(u16::try_from(td.members.len()).expect("RDEF member count exceeds u16"))
                << 16),
    );
    push_u32(out, member_off);
    if is_sm5 {
        match &td.sm5_extra {
            Some(e) => {
                for x in e {
                    push_u32(out, *x);
                }
            }
            None => out.extend_from_slice(&[0u8; 16]),
        }
        push_u32(out, name_off);
    }
    if dedup {
        placed.push((td.clone(), off));
    }
    off
}

#[derive(Default)]
struct RdefPools<'a> {
    binding_names: BTreeMap<String, u32>,
    variable_names: BTreeMap<String, u32>,
    default_values: BTreeMap<Vec<u8>, u32>,
    placed_types: Vec<(TypeDesc<'a>, u32)>,
    member_types: Vec<(TypeDesc<'a>, u32)>,
}

fn write_bindings(
    out: &mut Vec<u8>,
    names: &mut BTreeMap<String, u32>,
    bindings: &[ResourceBinding<'_>],
    stride: usize,
) {
    let mut name_positions = Vec::with_capacity(bindings.len());
    for binding in bindings {
        name_positions.push(out.len());
        push_u32(out, 0);
        push_u32(out, binding.input_type);
        push_u32(out, binding.return_type);
        push_u32(out, binding.dimension);
        push_u32(out, binding.num_samples);
        push_u32(out, binding.bind_point);
        push_u32(out, binding.bind_count);
        push_u32(out, binding.flags);
        if stride == 40 {
            push_u32(out, binding.space);
            push_u32(out, binding.id);
        } else {
            assert_eq!(
                binding.space, 0,
                "32-byte RDEF binding has no register space"
            );
            assert_eq!(binding.id, 0, "32-byte RDEF binding has no range ID");
        }
    }
    for (position, binding) in name_positions.into_iter().zip(bindings) {
        let offset = intern(out, names, &binding.name);
        patch_u32(out, position, offset);
    }
}

fn write_constant_buffer_headers(
    out: &mut Vec<u8>,
    names: &mut BTreeMap<String, u32>,
    buffers: &[CBufferDef<'_>],
    section_position: usize,
) -> Vec<usize> {
    if !buffers.is_empty() {
        pad4(out);
    }
    let section_offset = if buffers.is_empty() {
        0
    } else {
        wire_u32(out.len())
    };
    patch_u32(out, section_position, section_offset);

    let mut name_positions = Vec::with_capacity(buffers.len());
    let mut variable_positions = Vec::with_capacity(buffers.len());
    for buffer in buffers {
        name_positions.push(out.len());
        push_u32(out, 0);
        push_u32(out, wire_u32(buffer.variables.len()));
        variable_positions.push(out.len());
        push_u32(out, 0);
        push_u32(out, buffer.size);
        push_u32(out, buffer.flags);
        push_u32(out, buffer.cb_type);
    }
    for (position, buffer) in name_positions.into_iter().zip(buffers) {
        let offset = intern(out, names, &buffer.name);
        patch_u32(out, position, offset);
    }
    variable_positions
}

fn write_variable_sections<'a>(
    out: &mut Vec<u8>,
    pools: &mut RdefPools<'a>,
    buffers: &[CBufferDef<'a>],
    section_positions: &[usize],
    is_sm5: bool,
) {
    for (buffer, &section_position) in buffers.iter().zip(section_positions) {
        pad4(out);
        let section_offset = wire_u32(out.len());
        patch_u32(out, section_position, section_offset);

        let mut name_positions = Vec::with_capacity(buffer.variables.len());
        let mut type_positions = Vec::with_capacity(buffer.variables.len());
        let mut default_positions = Vec::with_capacity(buffer.variables.len());
        for variable in &buffer.variables {
            name_positions.push(out.len());
            push_u32(out, 0);
            push_u32(out, variable.offset);
            push_u32(out, variable.size);
            push_u32(out, variable.flags);
            type_positions.push(out.len());
            push_u32(out, 0);
            default_positions.push(out.len());
            push_u32(out, 0);
            if is_sm5 {
                push_u32(out, variable.texture_start.unwrap_or(SLOT_UNUSED));
                push_u32(out, variable.texture_size.unwrap_or(0));
                push_u32(out, variable.sampler_start.unwrap_or(SLOT_UNUSED));
                push_u32(out, variable.sampler_size.unwrap_or(0));
            }
        }
        write_variable_data(
            out,
            pools,
            buffer,
            &name_positions,
            &type_positions,
            &default_positions,
            is_sm5,
        );
    }
}

fn write_variable_data<'a>(
    out: &mut Vec<u8>,
    pools: &mut RdefPools<'a>,
    buffer: &CBufferDef<'a>,
    name_positions: &[usize],
    type_positions: &[usize],
    default_positions: &[usize],
    is_sm5: bool,
) {
    for (index, variable) in buffer.variables.iter().enumerate() {
        let name_offset = intern(out, &mut pools.variable_names, &variable.name);
        patch_u32(out, name_positions[index], name_offset);
        let type_offset = place_type(
            out,
            &mut pools.variable_names,
            &mut pools.placed_types,
            &mut pools.member_types,
            &variable.var_type,
            is_sm5,
            buffer.cb_type == 0,
        );
        patch_u32(out, type_positions[index], type_offset);
        let default_position = default_positions[index];
        if variable.default_value.is_empty() {
            patch_u32(out, default_position, 0);
        } else if let Some(&offset) = pools.default_values.get(variable.default_value.as_ref()) {
            patch_u32(out, default_position, offset);
        } else {
            pad4(out);
            let offset = wire_u32(out.len());
            out.extend_from_slice(&variable.default_value);
            pools
                .default_values
                .insert(variable.default_value.to_vec(), offset);
            patch_u32(out, default_position, offset);
        }
    }
}

fn write_rdef_canonical(rdef: &ResourceDef<'_>) -> Vec<u8> {
    let is_sm5 = rdef.is_sm5();
    assert_eq!(
        is_sm5,
        rdef.rd11.is_some(),
        "SM5 RDEF requires an RD11 header"
    );
    let rd11_size: u32 = if rdef.rd11.is_some() { 32 } else { 0 };
    let binding_stride = rdef.rd11.map_or(32, |header| {
        assert_eq!(header.header_size, 60, "unsupported RD11 header size");
        assert_eq!(header.constant_buffer_stride, 24);
        assert!(matches!(header.resource_binding_stride, 32 | 40));
        assert_eq!(header.variable_stride, 40);
        assert_eq!(header.type_stride, 36);
        assert_eq!(header.member_stride, 12);
        usize::try_from(header.resource_binding_stride).expect("u32 fits in usize")
    });
    let mut out: Vec<u8> = Vec::new();
    let mut pools = RdefPools::default();

    push_u32(&mut out, wire_u32(rdef.constant_buffers.len()));
    let buffer_section_position = out.len();
    push_u32(&mut out, 0);
    push_u32(&mut out, wire_u32(rdef.bindings.len()));
    push_u32(&mut out, 28 + rd11_size);
    push_u32(&mut out, rdef.target_version);
    push_u32(&mut out, rdef.compile_flags);
    let creator_position = out.len();
    push_u32(&mut out, 0);
    if let Some(rd11) = rdef.rd11 {
        for value in rd11.to_words() {
            push_u32(&mut out, value);
        }
    }

    write_bindings(
        &mut out,
        &mut pools.binding_names,
        &rdef.bindings,
        binding_stride,
    );
    let variable_positions = write_constant_buffer_headers(
        &mut out,
        &mut pools.binding_names,
        &rdef.constant_buffers,
        buffer_section_position,
    );
    write_variable_sections(
        &mut out,
        &mut pools,
        &rdef.constant_buffers,
        &variable_positions,
        is_sm5,
    );
    let creator_offset = intern(&mut out, &mut pools.variable_names, &rdef.creator);
    patch_u32(&mut out, creator_position, creator_offset);
    pad4(&mut out);
    out
}

impl ChunkWriter for ResourceDef<'_> {
    fn fourcc(&self) -> [u8; 4] {
        *b"RDEF"
    }

    fn write_payload(&self) -> Vec<u8> {
        let canonical = write_rdef_canonical(self);
        if self.original_canonical.as_deref() == Some(canonical.as_slice()) {
            if let Some(original) = &self.original_data {
                return original.to_vec();
            }
        }
        canonical
    }
}

// Editable text codec

/// Serialize an RDEF to editable `key=value` text. Returns `None` for RDEFs
/// with struct members, whose byte-exact layout is not yet reproduced (those
/// stay raw hex via the container's emit-time verification).
#[cfg(test)]
mod tests {
    use super::*;

    fn binding(dimension: u32, flags: u32) -> ResourceBinding<'static> {
        ResourceBinding {
            name: Cow::Borrowed("t"),
            input_type: 2, // texture
            return_type: 5,
            dimension,
            num_samples: 0,
            bind_point: 0,
            bind_count: 1,
            flags,
            space: 0,
            id: 0,
        }
    }

    #[test]
    fn parsed_layout_is_reused_only_until_semantics_change() {
        let source = ResourceDef {
            constant_buffers: Vec::new(),
            bindings: Vec::new(),
            creator: Cow::Borrowed("x"),
            target_version: 0xffff_0400,
            compile_flags: 0,
            rd11: None,
            original_data: None,
            original_canonical: None,
        };
        let mut noncanonical = write_rdef_canonical(&source);
        assert_eq!(&noncanonical[noncanonical.len() - 2..], &[0xab, 0xab]);
        let len = noncanonical.len();
        noncanonical[len - 2..].fill(0);

        let mut parsed = parse_rdef(&noncanonical).expect("valid RDEF layout");
        assert_eq!(parsed.write_payload(), noncanonical);

        parsed.creator = Cow::Borrowed("y");
        let edited = parsed.write_payload();
        assert_ne!(edited, noncanonical);
        assert_eq!(
            parse_rdef(&edited).expect("canonical edited RDEF").creator,
            "y"
        );
    }

    #[test]
    fn srv_dimension_numbering() {
        // The RDEF binding dimension is a D3D_SRV_DIMENSION, which numbers
        // Texture2D=4, Texture2DArray=5, Texture2DMS=6, Texture3D=8 — distinct
        // from the SHEX (SB) enum. Regression for the SB/SRV mix-up.
        assert_eq!(ResourceDimension::from_u32(4).unwrap().name(), "2d");
        assert_eq!(ResourceDimension::from_u32(5).unwrap().name(), "2darray");
        assert_eq!(ResourceDimension::from_u32(6).unwrap().name(), "2dMS");
        assert_eq!(ResourceDimension::from_u32(8).unwrap().name(), "3d");
    }

    #[test]
    fn hlsl_stem_roundtrips() {
        for v in 1..=10 {
            let d = ResourceDimension::from_u32(v).unwrap();
            assert_eq!(ResourceDimension::from_hlsl_stem(d.hlsl_stem()), Some(d));
            assert_eq!(d as u32, v);
        }
        assert_eq!(
            ResourceDimension::from_u32(5).unwrap().hlsl_stem(),
            "Texture2DArray"
        );
        assert_eq!(
            ResourceDimension::from_u32(8).unwrap().hlsl_stem(),
            "Texture3D"
        );
    }

    #[test]
    fn binding_flags_are_input_flags() {
        // A plain float4 texture has flags 0xC = TEXTURE_COMPONENT_0|_1. It must
        // render as the component bits, never as `comparisonSampler` (the old
        // table had a bogus `used` bit that shifted everything up one).
        let s = alloc::format!("{}", binding(4, SIF_TEX_COMPONENTS));
        assert!(s.contains("texComp0;texComp1"), "got: {s}");
        assert!(!s.contains("comparisonSampler"), "got: {s}");

        let cmp = alloc::format!("{}", binding(4, SIF_COMPARISON_SAMPLER));
        assert!(cmp.contains("comparisonSampler"), "got: {cmp}");
        assert!(!cmp.contains("texComp"), "got: {cmp}");
    }
}
