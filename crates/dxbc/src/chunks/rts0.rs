//! RTS0 chunk parser — D3D12 serialized Root Signature (version 1.0).

use alloc::vec::Vec;
use core::fmt;

use super::ChunkParser;
use crate::util::read_u32;

/// Parsed RTS0 (root signature) chunk.
#[derive(Debug)]
pub struct RootSignature {
    /// Root signature version (1 = v1.0, 2 = v1.1).
    pub version: u32,
    /// Root signature flags (e.g. ALLOW_INPUT_ASSEMBLER, DENY_*_SHADER_ROOT_ACCESS).
    pub flags: u32,
    /// Root parameters (descriptor tables, constants, root descriptors).
    pub parameters: Vec<RootParameter>,
    /// Static sampler definitions baked into the root signature.
    pub static_samplers: Vec<StaticSampler>,
}

/// A single root parameter entry.
#[derive(Debug)]
pub struct RootParameter {
    /// The parameter payload (table, constants, or inline descriptor).
    pub param_type: RootParameterType,
    /// Which shader stage(s) can see this parameter.
    pub visibility: ShaderVisibility,
}

/// Root parameter payload variants.
#[derive(Debug)]
pub enum RootParameterType {
    DescriptorTable {
        ranges: Vec<DescriptorRange>,
    },
    Constants32Bit {
        register: u32,
        space: u32,
        num_values: u32,
    },
    Cbv {
        register: u32,
        space: u32,
    },
    Srv {
        register: u32,
        space: u32,
    },
    Uav {
        register: u32,
        space: u32,
    },
}

/// A descriptor range within a descriptor table.
#[derive(Debug)]
pub struct DescriptorRange {
    /// SRV, UAV, CBV, or Sampler.
    pub range_type: DescriptorRangeType,
    /// Number of descriptors (0xFFFFFFFF = unbounded).
    pub num_descriptors: u32,
    /// First shader register in this range.
    pub base_shader_register: u32,
    /// Register space.
    pub register_space: u32,
    /// Offset from the start of the table (0xFFFFFFFF = append).
    pub offset_in_descriptors_from_table_start: u32,
}

/// Descriptor range type.
#[derive(Debug, Clone, Copy)]
pub enum DescriptorRangeType {
    Srv,
    Uav,
    Cbv,
    Sampler,
}

/// Which shader stage(s) a root parameter or static sampler is visible to.
#[derive(Debug, Clone, Copy)]
pub enum ShaderVisibility {
    All,
    Vertex,
    Hull,
    Domain,
    Geometry,
    Pixel,
}

/// A static sampler baked into the root signature.
#[derive(Debug)]
pub struct StaticSampler {
    pub filter: u32,
    pub address_u: u32,
    pub address_v: u32,
    pub address_w: u32,
    pub mip_lod_bias: f32,
    pub max_anisotropy: u32,
    pub comparison_func: u32,
    pub border_color: u32,
    pub min_lod: f32,
    pub max_lod: f32,
    pub shader_register: u32,
    pub register_space: u32,
    pub visibility: ShaderVisibility,
}

fn read_f32(data: &[u8], off: usize) -> f32 {
    f32::from_le_bytes(data[off..off + 4].try_into().unwrap())
}

fn parse_vis(v: u32) -> ShaderVisibility {
    match v {
        1 => ShaderVisibility::Vertex,
        2 => ShaderVisibility::Hull,
        3 => ShaderVisibility::Domain,
        4 => ShaderVisibility::Geometry,
        5 => ShaderVisibility::Pixel,
        _ => ShaderVisibility::All,
    }
}

fn parse_rt(v: u32) -> DescriptorRangeType {
    match v {
        1 => DescriptorRangeType::Uav,
        2 => DescriptorRangeType::Cbv,
        3 => DescriptorRangeType::Sampler,
        _ => DescriptorRangeType::Srv,
    }
}

/// Parse RTS0 chunk data (bytes *after* the 8-byte fourcc+size header).
pub fn parse_rts0(data: &[u8]) -> Option<RootSignature> {
    if data.len() < 24 {
        return None;
    }
    let version = read_u32(data, 0);
    let np = read_u32(data, 4) as usize;
    let p_off = read_u32(data, 8) as usize;
    let ns = read_u32(data, 12) as usize;
    let s_off = read_u32(data, 16) as usize;
    let flags = read_u32(data, 20);

    let mut parameters = Vec::with_capacity(np);
    for i in 0..np {
        let b = p_off + i * 12;
        if b + 12 > data.len() {
            break;
        }
        let pt = read_u32(data, b);
        let vis = parse_vis(read_u32(data, b + 4));
        let po = read_u32(data, b + 8) as usize;
        let param_type = match pt {
            0 => {
                let nr = read_u32(data, po) as usize;
                let ro = read_u32(data, po + 4) as usize;
                let mut ranges = Vec::with_capacity(nr);
                for j in 0..nr {
                    let r = ro + j * 20;
                    if r + 20 > data.len() {
                        break;
                    }
                    ranges.push(DescriptorRange {
                        range_type: parse_rt(read_u32(data, r)),
                        num_descriptors: read_u32(data, r + 4),
                        base_shader_register: read_u32(data, r + 8),
                        register_space: read_u32(data, r + 12),
                        offset_in_descriptors_from_table_start: read_u32(data, r + 16),
                    });
                }
                RootParameterType::DescriptorTable { ranges }
            }
            1 => RootParameterType::Constants32Bit {
                register: read_u32(data, po),
                space: read_u32(data, po + 4),
                num_values: read_u32(data, po + 8),
            },
            2 => RootParameterType::Cbv {
                register: read_u32(data, po),
                space: read_u32(data, po + 4),
            },
            3 => RootParameterType::Srv {
                register: read_u32(data, po),
                space: read_u32(data, po + 4),
            },
            _ => RootParameterType::Uav {
                register: read_u32(data, po),
                space: read_u32(data, po + 4),
            },
        };
        parameters.push(RootParameter {
            param_type,
            visibility: vis,
        });
    }

    let mut static_samplers = Vec::with_capacity(ns);
    for i in 0..ns {
        let s = s_off + i * 52;
        if s + 52 > data.len() {
            break;
        }
        static_samplers.push(StaticSampler {
            filter: read_u32(data, s),
            address_u: read_u32(data, s + 4),
            address_v: read_u32(data, s + 8),
            address_w: read_u32(data, s + 12),
            mip_lod_bias: read_f32(data, s + 16),
            max_anisotropy: read_u32(data, s + 20),
            comparison_func: read_u32(data, s + 24),
            border_color: read_u32(data, s + 28),
            min_lod: read_f32(data, s + 32),
            max_lod: read_f32(data, s + 36),
            shader_register: read_u32(data, s + 40),
            register_space: read_u32(data, s + 44),
            visibility: parse_vis(read_u32(data, s + 48)),
        });
    }
    Some(RootSignature {
        version,
        flags,
        parameters,
        static_samplers,
    })
}

impl fmt::Display for ShaderVisibility {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::All => f.write_str("ALL"),
            Self::Vertex => f.write_str("VS"),
            Self::Hull => f.write_str("HS"),
            Self::Domain => f.write_str("DS"),
            Self::Geometry => f.write_str("GS"),
            Self::Pixel => f.write_str("PS"),
        }
    }
}

impl fmt::Display for DescriptorRangeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Srv => f.write_str("SRV"),
            Self::Uav => f.write_str("UAV"),
            Self::Cbv => f.write_str("CBV"),
            Self::Sampler => f.write_str("SAMPLER"),
        }
    }
}

impl DescriptorRangeType {
    pub fn prefix(&self) -> char {
        match self {
            Self::Srv => 't',
            Self::Uav => 'u',
            Self::Cbv => 'b',
            Self::Sampler => 's',
        }
    }
}

impl ChunkParser for RootSignature {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_rts0(data)
    }
}

impl fmt::Display for RootSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "// Root Signature v1.{} \u{2014} {} parameter(s), {} static sampler(s), flags=0x{:X}",
            if self.version >= 2 { "1" } else { "0" },
            self.parameters.len(),
            self.static_samplers.len(),
            self.flags
        )?;
        for (i, p) in self.parameters.iter().enumerate() {
            match &p.param_type {
                RootParameterType::DescriptorTable { ranges } => {
                    writeln!(f, "//   [{i:2}] DescriptorTable  vis={}", p.visibility)?;
                    for r in ranges {
                        let cnt = if r.num_descriptors == 0xFFFFFFFF {
                            alloc::string::String::from("unbounded")
                        } else {
                            alloc::format!("{}", r.num_descriptors)
                        };
                        let off = if r.offset_in_descriptors_from_table_start == 0xFFFFFFFF {
                            alloc::string::String::from("APPEND")
                        } else {
                            alloc::format!("{}", r.offset_in_descriptors_from_table_start)
                        };
                        writeln!(
                            f,
                            "//        {}({}) {}{} space={} offset={}",
                            r.range_type,
                            cnt,
                            r.range_type.prefix(),
                            r.base_shader_register,
                            r.register_space,
                            off
                        )?;
                    }
                }
                RootParameterType::Constants32Bit {
                    register,
                    space,
                    num_values,
                } => writeln!(
                    f,
                    "//   [{i:2}] 32BitConstants   vis={}  b{register} space={space} num32BitValues={num_values}",
                    p.visibility
                )?,
                RootParameterType::Cbv { register, space } => writeln!(
                    f,
                    "//   [{i:2}] CBV              vis={}  b{register} space={space}",
                    p.visibility
                )?,
                RootParameterType::Srv { register, space } => writeln!(
                    f,
                    "//   [{i:2}] SRV              vis={}  t{register} space={space}",
                    p.visibility
                )?,
                RootParameterType::Uav { register, space } => writeln!(
                    f,
                    "//   [{i:2}] UAV              vis={}  u{register} space={space}",
                    p.visibility
                )?,
            }
        }
        for (i, s) in self.static_samplers.iter().enumerate() {
            writeln!(
                f,
                "//   StaticSampler[{i}] s{} space={} vis={} filter={} addr=({},{},{}) lod=[{},{}]",
                s.shader_register,
                s.register_space,
                s.visibility,
                s.filter,
                s.address_u,
                s.address_v,
                s.address_w,
                s.min_lod,
                s.max_lod
            )?;
        }
        Ok(())
    }
}
