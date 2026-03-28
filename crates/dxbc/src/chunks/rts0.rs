//! RTS0 chunk parser — D3D12 serialized Root Signature (version 1.0).

use alloc::vec::Vec;
use core::fmt;

use nostdio::{ReadLe, Seek, SeekFrom, SliceCursor};

use super::{ChunkParser, ChunkWriter};

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
    /// A descriptor table containing one or more descriptor ranges.
    DescriptorTable {
        /// Ordered list of descriptor ranges in this table.
        ranges: Vec<DescriptorRange>,
    },
    /// Inline 32-bit root constants.
    Constants32Bit {
        /// Base shader register (`b` register).
        register: u32,
        /// Register space.
        space: u32,
        /// Number of 32-bit values.
        num_values: u32,
    },
    /// Inline CBV root descriptor.
    Cbv {
        /// Shader register (`b` register).
        register: u32,
        /// Register space.
        space: u32,
    },
    /// Inline SRV root descriptor.
    Srv {
        /// Shader register (`t` register).
        register: u32,
        /// Register space.
        space: u32,
    },
    /// Inline UAV root descriptor.
    Uav {
        /// Shader register (`u` register).
        register: u32,
        /// Register space.
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
    /// D3D12 filter mode (raw `D3D12_FILTER` value).
    pub filter: u32,
    /// Texture address mode for U axis (raw `D3D12_TEXTURE_ADDRESS_MODE`).
    pub address_u: u32,
    /// Texture address mode for V axis.
    pub address_v: u32,
    /// Texture address mode for W axis.
    pub address_w: u32,
    /// Mip LOD bias.
    pub mip_lod_bias: f32,
    /// Maximum anisotropy (1–16).
    pub max_anisotropy: u32,
    /// Comparison function (raw `D3D12_COMPARISON_FUNC`).
    pub comparison_func: u32,
    /// Border color (raw `D3D12_STATIC_BORDER_COLOR`).
    pub border_color: u32,
    /// Minimum LOD clamp.
    pub min_lod: f32,
    /// Maximum LOD clamp.
    pub max_lod: f32,
    /// Sampler shader register (`s` register).
    pub shader_register: u32,
    /// Register space.
    pub register_space: u32,
    /// Which shader stage(s) can see this sampler.
    pub visibility: ShaderVisibility,
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
    let mut c = SliceCursor::new(data);
    let version = c.read_u32_le().ok()?;
    let np = c.read_u32_le().ok()? as usize;
    let p_off = c.read_u32_le().ok()? as usize;
    let ns = c.read_u32_le().ok()? as usize;
    let s_off = c.read_u32_le().ok()? as usize;
    let flags = c.read_u32_le().ok()?;

    let mut parameters = Vec::with_capacity(np);
    for i in 0..np {
        let b = p_off + i * 12;
        if b + 12 > data.len() {
            break;
        }
        c.seek(SeekFrom::Start(b as u64)).ok()?;
        let pt = c.read_u32_le().ok()?;
        let vis = parse_vis(c.read_u32_le().ok()?);
        let po = c.read_u32_le().ok()? as usize;
        let param_type = match pt {
            0 => {
                c.seek(SeekFrom::Start(po as u64)).ok()?;
                let nr = c.read_u32_le().ok()? as usize;
                let ro = c.read_u32_le().ok()? as usize;
                let mut ranges = Vec::with_capacity(nr);
                for j in 0..nr {
                    let r = ro + j * 20;
                    if r + 20 > data.len() {
                        break;
                    }
                    c.seek(SeekFrom::Start(r as u64)).ok()?;
                    ranges.push(DescriptorRange {
                        range_type: parse_rt(c.read_u32_le().ok()?),
                        num_descriptors: c.read_u32_le().ok()?,
                        base_shader_register: c.read_u32_le().ok()?,
                        register_space: c.read_u32_le().ok()?,
                        offset_in_descriptors_from_table_start: c.read_u32_le().ok()?,
                    });
                }
                RootParameterType::DescriptorTable { ranges }
            }
            1 => {
                c.seek(SeekFrom::Start(po as u64)).ok()?;
                RootParameterType::Constants32Bit {
                    register: c.read_u32_le().ok()?,
                    space: c.read_u32_le().ok()?,
                    num_values: c.read_u32_le().ok()?,
                }
            }
            2 => {
                c.seek(SeekFrom::Start(po as u64)).ok()?;
                RootParameterType::Cbv {
                    register: c.read_u32_le().ok()?,
                    space: c.read_u32_le().ok()?,
                }
            }
            3 => {
                c.seek(SeekFrom::Start(po as u64)).ok()?;
                RootParameterType::Srv {
                    register: c.read_u32_le().ok()?,
                    space: c.read_u32_le().ok()?,
                }
            }
            _ => {
                c.seek(SeekFrom::Start(po as u64)).ok()?;
                RootParameterType::Uav {
                    register: c.read_u32_le().ok()?,
                    space: c.read_u32_le().ok()?,
                }
            }
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
        c.seek(SeekFrom::Start(s as u64)).ok()?;
        static_samplers.push(StaticSampler {
            filter: c.read_u32_le().ok()?,
            address_u: c.read_u32_le().ok()?,
            address_v: c.read_u32_le().ok()?,
            address_w: c.read_u32_le().ok()?,
            mip_lod_bias: c.read_f32_le().ok()?,
            max_anisotropy: c.read_u32_le().ok()?,
            comparison_func: c.read_u32_le().ok()?,
            border_color: c.read_u32_le().ok()?,
            min_lod: c.read_f32_le().ok()?,
            max_lod: c.read_f32_le().ok()?,
            shader_register: c.read_u32_le().ok()?,
            register_space: c.read_u32_le().ok()?,
            visibility: parse_vis(c.read_u32_le().ok()?),
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
    /// Returns the HLSL register prefix character for this range type.
    pub fn prefix(&self) -> char {
        match self {
            Self::Srv => 't',
            Self::Uav => 'u',
            Self::Cbv => 'b',
            Self::Sampler => 's',
        }
    }
}

impl fmt::Display for DescriptorRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cnt = if self.num_descriptors == 0xFFFFFFFF {
            alloc::string::String::from("unbounded")
        } else {
            alloc::format!("{}", self.num_descriptors)
        };

        let off = if self.offset_in_descriptors_from_table_start == 0xFFFFFFFF {
            alloc::string::String::from("APPEND")
        } else {
            alloc::format!("{}", self.offset_in_descriptors_from_table_start)
        };

        write!(
            f,
            "{}({}) {}{} space={} offset={}",
            self.range_type,
            cnt,
            self.range_type.prefix(),
            self.base_shader_register,
            self.register_space,
            off
        )
    }
}

impl fmt::Display for RootParameter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.param_type {
            RootParameterType::DescriptorTable { ranges } => {
                write!(f, "DescriptorTable vis={}", self.visibility)?;
                for r in ranges {
                    write!(f, "\n  {r}")?;
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
            RootParameterType::Cbv { register, space } => {
                write!(f, "CBV vis={} b{register} space={space}", self.visibility)
            }
            RootParameterType::Srv { register, space } => {
                write!(f, "SRV vis={} t{register} space={space}", self.visibility)
            }
            RootParameterType::Uav { register, space } => {
                write!(f, "UAV vis={} u{register} space={space}", self.visibility)
            }
        }
    }
}

impl fmt::Display for StaticSampler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "StaticSampler s{} space={} vis={} filter={} addr=({},{},{}) lod=[{},{}]",
            self.shader_register,
            self.register_space,
            self.visibility,
            self.filter,
            self.address_u,
            self.address_v,
            self.address_w,
            self.min_lod,
            self.max_lod
        )
    }
}

impl ChunkParser for RootSignature {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_rts0(data)
    }
}

fn vis_to_u32(v: &ShaderVisibility) -> u32 {
    match v {
        ShaderVisibility::All => 0,
        ShaderVisibility::Vertex => 1,
        ShaderVisibility::Hull => 2,
        ShaderVisibility::Domain => 3,
        ShaderVisibility::Geometry => 4,
        ShaderVisibility::Pixel => 5,
    }
}

fn rt_to_u32(r: &DescriptorRangeType) -> u32 {
    match r {
        DescriptorRangeType::Srv => 0,
        DescriptorRangeType::Uav => 1,
        DescriptorRangeType::Cbv => 2,
        DescriptorRangeType::Sampler => 3,
    }
}

impl ChunkWriter for RootSignature {
    fn fourcc(&self) -> [u8; 4] {
        *b"RTS0"
    }

    fn write_payload(&self) -> Vec<u8> {
        let w = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_le_bytes());
        let wf = |buf: &mut Vec<u8>, v: f32| buf.extend_from_slice(&v.to_le_bytes());

        let np = self.parameters.len();
        let ns = self.static_samplers.len();

        // Compute payload sizes for each parameter
        let mut param_payload_sizes = Vec::with_capacity(np);
        for p in &self.parameters {
            let size = match &p.param_type {
                RootParameterType::DescriptorTable { ranges } => 8 + ranges.len() * 20,
                RootParameterType::Constants32Bit { .. } => 12,
                RootParameterType::Cbv { .. }
                | RootParameterType::Srv { .. }
                | RootParameterType::Uav { .. } => 8,
            };
            param_payload_sizes.push(size);
        }

        // Layout:
        // Header: 24 bytes (version, np, p_off, ns, s_off, flags)
        // Parameter entries: np * 12 bytes
        // Parameter payloads: variable
        // Static samplers: ns * 52 bytes
        let params_offset = 24usize;
        let payloads_start = params_offset + np * 12;
        let mut payload_offsets = Vec::with_capacity(np);
        let mut off = payloads_start;
        for &sz in &param_payload_sizes {
            payload_offsets.push(off);
            off += sz;
        }
        let samplers_offset = off;

        let total = samplers_offset + ns * 52;
        let mut buf = Vec::with_capacity(total);

        // Header
        w(&mut buf, self.version);
        w(&mut buf, np as u32);
        w(&mut buf, params_offset as u32);
        w(&mut buf, ns as u32);
        w(&mut buf, samplers_offset as u32);
        w(&mut buf, self.flags);

        // Parameter entries
        for (i, p) in self.parameters.iter().enumerate() {
            let pt = match &p.param_type {
                RootParameterType::DescriptorTable { .. } => 0u32,
                RootParameterType::Constants32Bit { .. } => 1,
                RootParameterType::Cbv { .. } => 2,
                RootParameterType::Srv { .. } => 3,
                RootParameterType::Uav { .. } => 4,
            };
            w(&mut buf, pt);
            w(&mut buf, vis_to_u32(&p.visibility));
            w(&mut buf, payload_offsets[i] as u32);
        }

        // Parameter payloads
        for p in &self.parameters {
            match &p.param_type {
                RootParameterType::DescriptorTable { ranges } => {
                    let nr = ranges.len();
                    w(&mut buf, nr as u32);
                    // Ranges are written right after the (count, ranges_offset) pair.
                    // The ranges_offset points to absolute position within chunk data.
                    let ro = buf.len() + 4; // after the u32 we're about to write
                    w(&mut buf, ro as u32);
                    for r in ranges {
                        w(&mut buf, rt_to_u32(&r.range_type));
                        w(&mut buf, r.num_descriptors);
                        w(&mut buf, r.base_shader_register);
                        w(&mut buf, r.register_space);
                        w(&mut buf, r.offset_in_descriptors_from_table_start);
                    }
                }
                RootParameterType::Constants32Bit {
                    register,
                    space,
                    num_values,
                } => {
                    w(&mut buf, *register);
                    w(&mut buf, *space);
                    w(&mut buf, *num_values);
                }
                RootParameterType::Cbv { register, space }
                | RootParameterType::Srv { register, space }
                | RootParameterType::Uav { register, space } => {
                    w(&mut buf, *register);
                    w(&mut buf, *space);
                }
            }
        }

        // Static samplers
        for s in &self.static_samplers {
            w(&mut buf, s.filter);
            w(&mut buf, s.address_u);
            w(&mut buf, s.address_v);
            w(&mut buf, s.address_w);
            wf(&mut buf, s.mip_lod_bias);
            w(&mut buf, s.max_anisotropy);
            w(&mut buf, s.comparison_func);
            w(&mut buf, s.border_color);
            wf(&mut buf, s.min_lod);
            wf(&mut buf, s.max_lod);
            w(&mut buf, s.shader_register);
            w(&mut buf, s.register_space);
            w(&mut buf, vis_to_u32(&s.visibility));
        }

        buf
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
