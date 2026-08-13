//! RTS0 — serialized D3D12 versioned root signatures.

use alloc::vec::Vec;
use core::fmt;

use super::{ChunkParser, ChunkWriter};

const HEADER_SIZE: usize = 24;
const PARAMETER_SIZE: usize = 12;
const ROOT_SIGNATURE_VALID_FLAGS: u32 = 0x8000_0fff;
const DESCRIPTOR_RANGE_VALID_FLAGS: u32 = 0x0001_000f;
const SAMPLER_RANGE_VALID_FLAGS: u32 = 0x1;
const ROOT_DESCRIPTOR_VALID_FLAGS: u32 = 0xe;
const STATIC_SAMPLER_VALID_FLAGS: u32 = 0x3;

/// Serialized root-signature version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum RootSignatureVersion {
    /// D3D12 root signature 1.0.
    V1_0 = 1,
    /// D3D12 root signature 1.1 (descriptor flags).
    V1_1 = 2,
    /// D3D12 root signature 1.2 (static sampler flags).
    V1_2 = 3,
}

impl RootSignatureVersion {
    fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            1 => Self::V1_0,
            2 => Self::V1_1,
            3 => Self::V1_2,
            _ => return None,
        })
    }

    const fn has_descriptor_flags(self) -> bool {
        matches!(self, Self::V1_1 | Self::V1_2)
    }

    const fn has_static_sampler_flags(self) -> bool {
        matches!(self, Self::V1_2)
    }

    const fn label(self) -> &'static str {
        match self {
            Self::V1_0 => "1.0",
            Self::V1_1 => "1.1",
            Self::V1_2 => "1.2",
        }
    }
}

/// Parsed RTS0 payload.
#[derive(Debug)]
pub struct RootSignature {
    /// Version controlling the descriptor and sampler record strides.
    pub version: RootSignatureVersion,
    /// `D3D12_ROOT_SIGNATURE_FLAGS`.
    pub flags: u32,
    /// Root parameters.
    pub parameters: Vec<RootParameter>,
    /// Static samplers.
    pub static_samplers: Vec<StaticSampler>,
    /// Original payload retained for a layout-preserving no-op rewrite.
    #[doc(hidden)]
    pub original_data: Option<Vec<u8>>,
    /// Canonical encoding of the semantic state at parse time.
    #[doc(hidden)]
    pub original_canonical: Option<Vec<u8>>,
}

/// One root parameter and its shader visibility.
#[derive(Debug)]
pub struct RootParameter {
    /// Parameter payload.
    pub param_type: RootParameterType,
    /// Shader stages that can access the parameter.
    pub visibility: ShaderVisibility,
}

/// Root parameter payload (`D3D12_ROOT_PARAMETER_TYPE`).
#[derive(Debug)]
pub enum RootParameterType {
    /// Descriptor table.
    DescriptorTable {
        /// Ordered descriptor ranges.
        ranges: Vec<DescriptorRange>,
    },
    /// Inline 32-bit constants.
    Constants32Bit {
        /// Base constant-buffer register.
        register: u32,
        /// Register space.
        space: u32,
        /// Number of 32-bit constants.
        num_values: u32,
    },
    /// Inline constant-buffer descriptor.
    Cbv {
        /// Shader register.
        register: u32,
        /// Register space.
        space: u32,
        /// `D3D12_ROOT_DESCRIPTOR_FLAGS` (zero in version 1.0).
        flags: u32,
    },
    /// Inline shader-resource descriptor.
    Srv {
        /// Shader register.
        register: u32,
        /// Register space.
        space: u32,
        /// `D3D12_ROOT_DESCRIPTOR_FLAGS` (zero in version 1.0).
        flags: u32,
    },
    /// Inline unordered-access descriptor.
    Uav {
        /// Shader register.
        register: u32,
        /// Register space.
        space: u32,
        /// `D3D12_ROOT_DESCRIPTOR_FLAGS` (zero in version 1.0).
        flags: u32,
    },
}

/// One serialized descriptor range.
#[derive(Debug)]
pub struct DescriptorRange {
    /// SRV, UAV, CBV, or sampler range.
    pub range_type: DescriptorRangeType,
    /// Number of descriptors (`u32::MAX` means unbounded).
    pub num_descriptors: u32,
    /// First shader register.
    pub base_shader_register: u32,
    /// Register space.
    pub register_space: u32,
    /// `D3D12_DESCRIPTOR_RANGE_FLAGS` (zero in version 1.0).
    pub flags: u32,
    /// Table offset (`u32::MAX` means append).
    pub offset_in_descriptors_from_table_start: u32,
}

/// `D3D12_DESCRIPTOR_RANGE_TYPE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DescriptorRangeType {
    /// Shader resource views.
    Srv = 0,
    /// Unordered access views.
    Uav = 1,
    /// Constant buffer views.
    Cbv = 2,
    /// Samplers.
    Sampler = 3,
}

impl DescriptorRangeType {
    fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::Srv,
            1 => Self::Uav,
            2 => Self::Cbv,
            3 => Self::Sampler,
            _ => return None,
        })
    }

    /// HLSL register prefix for this range.
    #[must_use]
    pub const fn prefix(self) -> char {
        match self {
            Self::Srv => 't',
            Self::Uav => 'u',
            Self::Cbv => 'b',
            Self::Sampler => 's',
        }
    }
}

/// `D3D12_SHADER_VISIBILITY`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ShaderVisibility {
    /// All stages.
    All = 0,
    /// Vertex stage.
    Vertex = 1,
    /// Hull stage.
    Hull = 2,
    /// Domain stage.
    Domain = 3,
    /// Geometry stage.
    Geometry = 4,
    /// Pixel stage.
    Pixel = 5,
    /// Amplification stage.
    Amplification = 6,
    /// Mesh stage.
    Mesh = 7,
}

impl ShaderVisibility {
    fn from_u32(value: u32) -> Option<Self> {
        Some(match value {
            0 => Self::All,
            1 => Self::Vertex,
            2 => Self::Hull,
            3 => Self::Domain,
            4 => Self::Geometry,
            5 => Self::Pixel,
            6 => Self::Amplification,
            7 => Self::Mesh,
            _ => return None,
        })
    }
}

/// Static sampler record. Version 1.2 appends `flags` to the older 52-byte form.
#[derive(Debug)]
pub struct StaticSampler {
    /// `D3D12_FILTER`.
    pub filter: u32,
    /// U address mode.
    pub address_u: u32,
    /// V address mode.
    pub address_v: u32,
    /// W address mode.
    pub address_w: u32,
    /// Mip LOD bias.
    pub mip_lod_bias: f32,
    /// Maximum anisotropy.
    pub max_anisotropy: u32,
    /// `D3D12_COMPARISON_FUNC`.
    pub comparison_func: u32,
    /// `D3D12_STATIC_BORDER_COLOR`.
    pub border_color: u32,
    /// Minimum LOD.
    pub min_lod: f32,
    /// Maximum LOD.
    pub max_lod: f32,
    /// Sampler shader register.
    pub shader_register: u32,
    /// Register space.
    pub register_space: u32,
    /// Shader visibility.
    pub visibility: ShaderVisibility,
    /// `D3D12_SAMPLER_FLAGS` (serialized only in version 1.2).
    pub flags: u32,
}

fn read_u32(data: &[u8], offset: usize) -> Option<u32> {
    let end = offset.checked_add(4)?;
    Some(u32::from_le_bytes(data.get(offset..end)?.try_into().ok()?))
}

fn read_f32(data: &[u8], offset: usize) -> Option<f32> {
    Some(f32::from_bits(read_u32(data, offset)?))
}

fn checked_record_offset(base: usize, index: usize, stride: usize, len: usize) -> Option<usize> {
    let offset = base.checked_add(index.checked_mul(stride)?)?;
    let end = offset.checked_add(stride)?;
    (end <= len).then_some(offset)
}

fn valid_descriptor_flags(range_type: DescriptorRangeType, flags: u32) -> bool {
    let mask = if range_type == DescriptorRangeType::Sampler {
        SAMPLER_RANGE_VALID_FLAGS
    } else {
        DESCRIPTOR_RANGE_VALID_FLAGS
    };
    flags & !mask == 0
}

/// Return whether a serialized `D3D12_FILTER` is one of the values accepted by
/// DXC's root-signature validator, and whether it is a comparison filter.
fn filter_is_comparison(filter: u32) -> Option<bool> {
    Some(match filter {
        0x00 | 0x01 | 0x04 | 0x05 | 0x10 | 0x11 | 0x14 | 0x15 | 0x55 | 0x100 | 0x101 | 0x104
        | 0x105 | 0x110 | 0x111 | 0x114 | 0x115 | 0x155 | 0x180 | 0x181 | 0x184 | 0x185 | 0x190
        | 0x191 | 0x194 | 0x195 | 0x1d5 => false,
        0x80 | 0x81 | 0x84 | 0x85 | 0x90 | 0x91 | 0x94 | 0x95 | 0xd5 => true,
        _ => return None,
    })
}

fn valid_static_sampler(sampler: &StaticSampler) -> bool {
    let Some(is_comparison) = filter_is_comparison(sampler.filter) else {
        return false;
    };
    (1..=5).contains(&sampler.address_u)
        && (1..=5).contains(&sampler.address_v)
        && (1..=5).contains(&sampler.address_w)
        && !sampler.mip_lod_bias.is_nan()
        && (-16.0..=15.99).contains(&sampler.mip_lod_bias)
        && sampler.max_anisotropy <= 16
        && (!is_comparison || (1..=8).contains(&sampler.comparison_func))
        && sampler.border_color <= 4
        // DXC explicitly permits both infinities for the LOD bounds.
        && !sampler.min_lod.is_nan()
        && !sampler.max_lod.is_nan()
        && sampler.flags & !STATIC_SAMPLER_VALID_FLAGS == 0
}

fn parse_static_samplers(
    data: &[u8],
    version: RootSignatureVersion,
    count: usize,
    offset: usize,
) -> Option<Vec<StaticSampler>> {
    let stride = if version.has_static_sampler_flags() {
        56
    } else {
        52
    };
    let mut samplers = Vec::new();
    for index in 0..count {
        let base = checked_record_offset(offset, index, stride, data.len())?;
        let visibility = ShaderVisibility::from_u32(read_u32(data, base + 48)?)?;
        let flags = if version.has_static_sampler_flags() {
            read_u32(data, base + 52)?
        } else {
            0
        };
        let sampler = StaticSampler {
            filter: read_u32(data, base)?,
            address_u: read_u32(data, base + 4)?,
            address_v: read_u32(data, base + 8)?,
            address_w: read_u32(data, base + 12)?,
            mip_lod_bias: read_f32(data, base + 16)?,
            max_anisotropy: read_u32(data, base + 20)?,
            comparison_func: read_u32(data, base + 24)?,
            border_color: read_u32(data, base + 28)?,
            min_lod: read_f32(data, base + 32)?,
            max_lod: read_f32(data, base + 36)?,
            shader_register: read_u32(data, base + 40)?,
            register_space: read_u32(data, base + 44)?,
            visibility,
            flags,
        };
        if !valid_static_sampler(&sampler) {
            return None;
        }
        samplers.push(sampler);
    }
    Some(samplers)
}

struct RootSignatureHeader {
    version: RootSignatureVersion,
    parameter_count: usize,
    parameters_offset: usize,
    sampler_count: usize,
    samplers_offset: usize,
    flags: u32,
}

fn parse_header(data: &[u8]) -> Option<RootSignatureHeader> {
    if data.len() < HEADER_SIZE || data.len() % 4 != 0 {
        return None;
    }
    let header = RootSignatureHeader {
        version: RootSignatureVersion::from_u32(read_u32(data, 0)?)?,
        parameter_count: usize::try_from(read_u32(data, 4)?).ok()?,
        parameters_offset: usize::try_from(read_u32(data, 8)?).ok()?,
        sampler_count: usize::try_from(read_u32(data, 12)?).ok()?,
        samplers_offset: usize::try_from(read_u32(data, 16)?).ok()?,
        flags: read_u32(data, 20)?,
    };
    let parameter_bytes = header.parameter_count.checked_mul(PARAMETER_SIZE)?;
    if header.flags & !ROOT_SIGNATURE_VALID_FLAGS != 0
        || header.parameters_offset % 4 != 0
        || header.samplers_offset % 4 != 0
        || header.parameters_offset.checked_add(parameter_bytes)? > data.len()
    {
        return None;
    }
    Some(header)
}

fn parse_descriptor_ranges(
    data: &[u8],
    version: RootSignatureVersion,
    payload_offset: usize,
) -> Option<Vec<DescriptorRange>> {
    let range_count = usize::try_from(read_u32(data, payload_offset)?).ok()?;
    let ranges_offset = usize::try_from(read_u32(data, payload_offset.checked_add(4)?)?).ok()?;
    if ranges_offset % 4 != 0 {
        return None;
    }
    let stride = if version.has_descriptor_flags() {
        24
    } else {
        20
    };
    let mut ranges = Vec::new();
    for range_index in 0..range_count {
        let base = checked_record_offset(ranges_offset, range_index, stride, data.len())?;
        let range_type = DescriptorRangeType::from_u32(read_u32(data, base)?)?;
        let flags = if version.has_descriptor_flags() {
            read_u32(data, base + 16)?
        } else {
            0
        };
        if !valid_descriptor_flags(range_type, flags) {
            return None;
        }
        ranges.push(DescriptorRange {
            range_type,
            num_descriptors: read_u32(data, base + 4)?,
            base_shader_register: read_u32(data, base + 8)?,
            register_space: read_u32(data, base + 12)?,
            flags,
            offset_in_descriptors_from_table_start: read_u32(
                data,
                base + if version.has_descriptor_flags() {
                    20
                } else {
                    16
                },
            )?,
        });
    }
    Some(ranges)
}

fn parse_parameter_type(
    data: &[u8],
    version: RootSignatureVersion,
    parameter_type: u32,
    payload_offset: usize,
) -> Option<RootParameterType> {
    Some(match parameter_type {
        0 => RootParameterType::DescriptorTable {
            ranges: parse_descriptor_ranges(data, version, payload_offset)?,
        },
        1 => RootParameterType::Constants32Bit {
            register: read_u32(data, payload_offset)?,
            space: read_u32(data, payload_offset.checked_add(4)?)?,
            num_values: read_u32(data, payload_offset.checked_add(8)?)?,
        },
        descriptor_type @ 2..=4 => {
            let register = read_u32(data, payload_offset)?;
            let space = read_u32(data, payload_offset.checked_add(4)?)?;
            let flags = if version.has_descriptor_flags() {
                read_u32(data, payload_offset.checked_add(8)?)?
            } else {
                0
            };
            if flags & !ROOT_DESCRIPTOR_VALID_FLAGS != 0 {
                return None;
            }
            match descriptor_type {
                2 => RootParameterType::Cbv {
                    register,
                    space,
                    flags,
                },
                3 => RootParameterType::Srv {
                    register,
                    space,
                    flags,
                },
                _ => RootParameterType::Uav {
                    register,
                    space,
                    flags,
                },
            }
        }
        _ => return None,
    })
}

fn parse_parameters(
    data: &[u8],
    version: RootSignatureVersion,
    count: usize,
    offset: usize,
) -> Option<Vec<RootParameter>> {
    let mut parameters = Vec::with_capacity(count);
    for index in 0..count {
        let base = checked_record_offset(offset, index, PARAMETER_SIZE, data.len())?;
        let visibility = ShaderVisibility::from_u32(read_u32(data, base + 4)?)?;
        let payload_offset = usize::try_from(read_u32(data, base + 8)?).ok()?;
        if payload_offset % 4 != 0 {
            return None;
        }
        parameters.push(RootParameter {
            param_type: parse_parameter_type(data, version, read_u32(data, base)?, payload_offset)?,
            visibility,
        });
    }
    Some(parameters)
}

/// Parse an RTS0 payload.
#[must_use]
pub fn parse_rts0(data: &[u8]) -> Option<RootSignature> {
    let header = parse_header(data)?;
    let parameters = parse_parameters(
        data,
        header.version,
        header.parameter_count,
        header.parameters_offset,
    )?;
    let static_samplers = parse_static_samplers(
        data,
        header.version,
        header.sampler_count,
        header.samplers_offset,
    )?;
    let mut root = RootSignature {
        version: header.version,
        flags: header.flags,
        parameters,
        static_samplers,
        original_data: None,
        original_canonical: None,
    };
    root.original_canonical = Some(root.write_canonical());
    root.original_data = Some(data.to_vec());
    Some(root)
}

impl ChunkParser<'_> for RootSignature {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_rts0(data)
    }
}

fn wire_u32(value: usize) -> u32 {
    u32::try_from(value).expect("root-signature size or offset exceeds u32")
}

fn checked_add(left: usize, right: usize) -> usize {
    left.checked_add(right)
        .expect("root-signature size overflow")
}

fn push_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}

struct RootSignatureLayout {
    payload_offsets: Vec<usize>,
    samplers_offset: usize,
    total_size: usize,
}

fn serialized_layout(root: &RootSignature) -> RootSignatureLayout {
    let range_stride = if root.version.has_descriptor_flags() {
        24
    } else {
        20
    };
    let descriptor_stride = if root.version.has_descriptor_flags() {
        12
    } else {
        8
    };
    let payloads_start = checked_add(
        HEADER_SIZE,
        root.parameters
            .len()
            .checked_mul(PARAMETER_SIZE)
            .expect("root-parameter table overflow"),
    );
    let mut payload_offsets = Vec::with_capacity(root.parameters.len());
    let mut next_offset = payloads_start;
    for parameter in &root.parameters {
        payload_offsets.push(next_offset);
        let payload_size = match &parameter.param_type {
            RootParameterType::DescriptorTable { ranges } => checked_add(
                8,
                ranges
                    .len()
                    .checked_mul(range_stride)
                    .expect("descriptor-range table overflow"),
            ),
            RootParameterType::Constants32Bit { .. } => 12,
            RootParameterType::Cbv { .. }
            | RootParameterType::Srv { .. }
            | RootParameterType::Uav { .. } => descriptor_stride,
        };
        next_offset = checked_add(next_offset, payload_size);
    }
    let sampler_stride = if root.version.has_static_sampler_flags() {
        56
    } else {
        52
    };
    let total_size = checked_add(
        next_offset,
        root.static_samplers
            .len()
            .checked_mul(sampler_stride)
            .expect("static-sampler table overflow"),
    );
    RootSignatureLayout {
        payload_offsets,
        samplers_offset: next_offset,
        total_size,
    }
}

fn parameter_type_tag(parameter: &RootParameterType) -> u32 {
    match parameter {
        RootParameterType::DescriptorTable { .. } => 0,
        RootParameterType::Constants32Bit { .. } => 1,
        RootParameterType::Cbv { .. } => 2,
        RootParameterType::Srv { .. } => 3,
        RootParameterType::Uav { .. } => 4,
    }
}

fn write_parameter_headers(root: &RootSignature, layout: &RootSignatureLayout, out: &mut Vec<u8>) {
    for (parameter, &payload_offset) in root.parameters.iter().zip(&layout.payload_offsets) {
        push_u32(out, parameter_type_tag(&parameter.param_type));
        push_u32(out, parameter.visibility as u32);
        push_u32(out, wire_u32(payload_offset));
    }
}

fn write_descriptor_ranges(
    version: RootSignatureVersion,
    ranges: &[DescriptorRange],
    out: &mut Vec<u8>,
) {
    push_u32(out, wire_u32(ranges.len()));
    let ranges_offset = checked_add(out.len(), 4);
    push_u32(out, wire_u32(ranges_offset));
    for range in ranges {
        assert!(
            valid_descriptor_flags(range.range_type, range.flags),
            "invalid descriptor-range flags"
        );
        if !version.has_descriptor_flags() {
            assert_eq!(range.flags, 0, "version 1.0 cannot encode range flags");
        }
        push_u32(out, range.range_type as u32);
        push_u32(out, range.num_descriptors);
        push_u32(out, range.base_shader_register);
        push_u32(out, range.register_space);
        if version.has_descriptor_flags() {
            push_u32(out, range.flags);
        }
        push_u32(out, range.offset_in_descriptors_from_table_start);
    }
}

fn write_parameter_payloads(root: &RootSignature, out: &mut Vec<u8>) {
    for parameter in &root.parameters {
        match &parameter.param_type {
            RootParameterType::DescriptorTable { ranges } => {
                write_descriptor_ranges(root.version, ranges, out);
            }
            RootParameterType::Constants32Bit {
                register,
                space,
                num_values,
            } => {
                push_u32(out, *register);
                push_u32(out, *space);
                push_u32(out, *num_values);
            }
            RootParameterType::Cbv {
                register,
                space,
                flags,
            }
            | RootParameterType::Srv {
                register,
                space,
                flags,
            }
            | RootParameterType::Uav {
                register,
                space,
                flags,
            } => {
                assert_eq!(
                    flags & !ROOT_DESCRIPTOR_VALID_FLAGS,
                    0,
                    "invalid root-descriptor flags"
                );
                if !root.version.has_descriptor_flags() {
                    assert_eq!(*flags, 0, "version 1.0 cannot encode descriptor flags");
                }
                push_u32(out, *register);
                push_u32(out, *space);
                if root.version.has_descriptor_flags() {
                    push_u32(out, *flags);
                }
            }
        }
    }
}

fn write_static_samplers(root: &RootSignature, out: &mut Vec<u8>) {
    for sampler in &root.static_samplers {
        assert!(valid_static_sampler(sampler), "invalid static sampler");
        if !root.version.has_static_sampler_flags() {
            assert_eq!(sampler.flags, 0, "only version 1.2 encodes sampler flags");
        }
        push_u32(out, sampler.filter);
        push_u32(out, sampler.address_u);
        push_u32(out, sampler.address_v);
        push_u32(out, sampler.address_w);
        push_u32(out, sampler.mip_lod_bias.to_bits());
        push_u32(out, sampler.max_anisotropy);
        push_u32(out, sampler.comparison_func);
        push_u32(out, sampler.border_color);
        push_u32(out, sampler.min_lod.to_bits());
        push_u32(out, sampler.max_lod.to_bits());
        push_u32(out, sampler.shader_register);
        push_u32(out, sampler.register_space);
        push_u32(out, sampler.visibility as u32);
        if root.version.has_static_sampler_flags() {
            push_u32(out, sampler.flags);
        }
    }
}

impl RootSignature {
    fn write_canonical(&self) -> Vec<u8> {
        assert_eq!(
            self.flags & !ROOT_SIGNATURE_VALID_FLAGS,
            0,
            "invalid root-signature flags"
        );
        let layout = serialized_layout(self);
        let mut out = Vec::with_capacity(layout.total_size);
        push_u32(&mut out, self.version as u32);
        push_u32(&mut out, wire_u32(self.parameters.len()));
        push_u32(&mut out, wire_u32(HEADER_SIZE));
        push_u32(&mut out, wire_u32(self.static_samplers.len()));
        push_u32(&mut out, wire_u32(layout.samplers_offset));
        push_u32(&mut out, self.flags);
        write_parameter_headers(self, &layout, &mut out);
        write_parameter_payloads(self, &mut out);
        write_static_samplers(self, &mut out);
        debug_assert_eq!(out.len(), layout.total_size);
        out
    }
}

impl ChunkWriter for RootSignature {
    fn fourcc(&self) -> [u8; 4] {
        *b"RTS0"
    }

    fn write_payload(&self) -> Vec<u8> {
        let canonical = self.write_canonical();
        if self.original_canonical.as_deref() == Some(canonical.as_slice()) {
            if let Some(original) = &self.original_data {
                return original.clone();
            }
        }
        canonical
    }
}

impl fmt::Display for ShaderVisibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::All => "ALL",
            Self::Vertex => "VS",
            Self::Hull => "HS",
            Self::Domain => "DS",
            Self::Geometry => "GS",
            Self::Pixel => "PS",
            Self::Amplification => "AS",
            Self::Mesh => "MS",
        })
    }
}

impl fmt::Display for DescriptorRangeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Srv => "SRV",
            Self::Uav => "UAV",
            Self::Cbv => "CBV",
            Self::Sampler => "SAMPLER",
        })
    }
}

impl fmt::Display for DescriptorRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = if self.num_descriptors == u32::MAX {
            alloc::string::String::from("unbounded")
        } else {
            alloc::format!("{}", self.num_descriptors)
        };
        let offset = if self.offset_in_descriptors_from_table_start == u32::MAX {
            alloc::string::String::from("APPEND")
        } else {
            alloc::format!("{}", self.offset_in_descriptors_from_table_start)
        };
        write!(
            f,
            "{}({count}) {}{} space={} offset={offset} flags=0x{:X}",
            self.range_type,
            self.range_type.prefix(),
            self.base_shader_register,
            self.register_space,
            self.flags
        )
    }
}

impl fmt::Display for RootParameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.param_type {
            RootParameterType::DescriptorTable { ranges } => {
                write!(f, "DescriptorTable vis={}", self.visibility)?;
                for range in ranges {
                    write!(f, "\n  {range}")?;
                }
                Ok(())
            }
            RootParameterType::Constants32Bit {
                register,
                space,
                num_values,
            } => write!(
                f,
                "32BitConstants vis={} b{register} space={space} num32BitValues={num_values}",
                self.visibility
            ),
            RootParameterType::Cbv {
                register,
                space,
                flags,
            } => write!(
                f,
                "CBV vis={} b{register} space={space} flags=0x{flags:X}",
                self.visibility
            ),
            RootParameterType::Srv {
                register,
                space,
                flags,
            } => write!(
                f,
                "SRV vis={} t{register} space={space} flags=0x{flags:X}",
                self.visibility
            ),
            RootParameterType::Uav {
                register,
                space,
                flags,
            } => write!(
                f,
                "UAV vis={} u{register} space={space} flags=0x{flags:X}",
                self.visibility
            ),
        }
    }
}

impl fmt::Display for StaticSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StaticSampler s{} space={} vis={} filter={} addr=({},{},{}) lod=[{},{}] flags=0x{:X}",
            self.shader_register,
            self.register_space,
            self.visibility,
            self.filter,
            self.address_u,
            self.address_v,
            self.address_w,
            self.min_lod,
            self.max_lod,
            self.flags
        )
    }
}

impl fmt::Display for RootSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "// Root Signature v{} — {} parameter(s), {} static sampler(s), flags=0x{:X}",
            self.version.label(),
            self.parameters.len(),
            self.static_samplers.len(),
            self.flags
        )?;
        for (index, parameter) in self.parameters.iter().enumerate() {
            writeln!(f, "//   [{index:2}] {parameter}")?;
        }
        for (index, sampler) in self.static_samplers.iter().enumerate() {
            writeln!(f, "//   StaticSampler[{index}] {sampler}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_1_1_uses_flagged_descriptor_strides() {
        let root = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 1,
            parameters: alloc::vec![
                RootParameter {
                    visibility: ShaderVisibility::Pixel,
                    param_type: RootParameterType::DescriptorTable {
                        ranges: alloc::vec![DescriptorRange {
                            range_type: DescriptorRangeType::Srv,
                            num_descriptors: 4,
                            base_shader_register: 2,
                            register_space: 3,
                            flags: 1,
                            offset_in_descriptors_from_table_start: u32::MAX,
                        }],
                    },
                },
                RootParameter {
                    visibility: ShaderVisibility::All,
                    param_type: RootParameterType::Cbv {
                        register: 1,
                        space: 2,
                        flags: 4,
                    },
                },
            ],
            static_samplers: Vec::new(),
            original_data: None,
            original_canonical: None,
        };
        let bytes = root.write_payload();
        let parsed = parse_rts0(&bytes).expect("valid root signature 1.1");
        let RootParameterType::DescriptorTable { ranges } = &parsed.parameters[0].param_type else {
            panic!("descriptor table expected");
        };
        assert_eq!(ranges[0].flags, 1);
        assert_eq!(ranges[0].offset_in_descriptors_from_table_start, u32::MAX);
        let RootParameterType::Cbv { flags, .. } = parsed.parameters[1].param_type else {
            panic!("CBV expected");
        };
        assert_eq!(flags, 4);
        assert_eq!(parsed.write_payload(), bytes);
    }

    #[test]
    fn version_1_2_appends_static_sampler_flags() {
        let root = RootSignature {
            version: RootSignatureVersion::V1_2,
            flags: 0,
            parameters: Vec::new(),
            static_samplers: alloc::vec![StaticSampler {
                filter: 0,
                address_u: 1,
                address_v: 1,
                address_w: 1,
                mip_lod_bias: 0.0,
                max_anisotropy: 1,
                comparison_func: 0,
                border_color: 0,
                min_lod: 0.0,
                max_lod: 16.0,
                shader_register: 0,
                register_space: 0,
                visibility: ShaderVisibility::Mesh,
                flags: 2,
            }],
            original_data: None,
            original_canonical: None,
        };
        let bytes = root.write_payload();
        assert_eq!(bytes.len(), HEADER_SIZE + 56);
        let parsed = parse_rts0(&bytes).expect("valid root signature 1.2");
        assert_eq!(parsed.static_samplers[0].flags, 2);
        assert_eq!(parsed.write_payload(), bytes);
    }

    #[test]
    fn static_sampler_validation_matches_dxc_lod_rules() {
        let sampler = StaticSampler {
            filter: 0,
            address_u: 1,
            address_v: 1,
            address_w: 1,
            mip_lod_bias: -16.0,
            max_anisotropy: 16,
            // Ignored for a non-comparison filter, as in DXC's validator.
            comparison_func: u32::MAX,
            border_color: 4,
            min_lod: f32::NEG_INFINITY,
            max_lod: f32::INFINITY,
            shader_register: 0,
            register_space: 0,
            visibility: ShaderVisibility::All,
            flags: 0,
        };
        assert!(valid_static_sampler(&sampler));

        let comparison = StaticSampler {
            filter: 0x80,
            comparison_func: 0,
            ..sampler
        };
        assert!(!valid_static_sampler(&comparison));
    }

    #[test]
    fn accepted_trailing_layout_is_preserved_until_edited() {
        let root = RootSignature {
            version: RootSignatureVersion::V1_1,
            flags: 0,
            parameters: Vec::new(),
            static_samplers: Vec::new(),
            original_data: None,
            original_canonical: None,
        };
        let mut noncanonical = root.write_payload();
        noncanonical.extend_from_slice(&[0xde, 0xad, 0xbe, 0xef]);

        let mut parsed =
            parse_rts0(&noncanonical).expect("valid root signature with trailing data");
        assert_eq!(parsed.write_payload(), noncanonical);
        parsed.flags = 1;
        let edited = parsed.write_payload();
        assert_ne!(edited, noncanonical);
        assert_eq!(
            parse_rts0(&edited)
                .expect("canonical edited root signature")
                .flags,
            1
        );
    }
}
