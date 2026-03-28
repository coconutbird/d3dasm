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

use alloc::vec::Vec;
use core::fmt;

use nostdio::{ReadLe, Seek, SeekFrom, SliceCursor};

use super::ChunkWriter;
use crate::util::{StringTableWriter, read_cstring};

/// Parsed RDEF chunk — constant buffers, resource bindings, and creator string.
#[derive(Debug)]
pub struct ResourceDef<'a> {
    /// Constant buffer definitions (layouts and variables).
    pub constant_buffers: Vec<CBufferDef<'a>>,
    /// Shader resource bindings (textures, samplers, UAVs, etc.).
    pub bindings: Vec<ResourceBinding<'a>>,
    /// Compiler identification string (e.g. `"Microsoft (R) HLSL Shader Compiler 10.1"`).
    pub creator: &'a str,
    /// Target version (minor | major<<8 | type<<16).
    pub target_version: u32,
    /// Compile flags.
    pub compile_flags: u32,
    /// SM5 RD11 sub-header (32 bytes after the main header). None for SM4.
    pub rd11_extra: Option<[u32; 8]>,
}

/// A single constant buffer and its variable layout.
#[derive(Debug)]
pub struct CBufferDef<'a> {
    /// Constant buffer name (e.g. `"$Globals"`, `"cb0"`).
    pub name: &'a str,
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
    pub name: &'a str,
    /// Byte offset within the constant buffer.
    pub offset: u32,
    /// Size in bytes.
    pub size: u32,
    /// Variable flags.
    pub flags: u32,
    /// Parsed type descriptor for this variable.
    pub var_type: TypeDesc<'a>,
    /// Default value bytes (empty if none).
    pub default_value: Vec<u8>,
    /// SM5 extra: start texture slot (-1 if unused).
    pub texture_start: Option<u32>,
    /// SM5 extra: texture bind count.
    pub texture_size: Option<u32>,
    /// SM5 extra: start sampler slot (-1 if unused).
    pub sampler_start: Option<u32>,
    /// SM5 extra: sampler bind count.
    pub sampler_size: Option<u32>,
}

/// HLSL type descriptor (D3D11_SHADER_TYPE_DESC).
///
/// Binary layout (all u32):
///   0: `[class:16 | type:16]`
///   4: `[rows:16 | columns:16]`
///   8: `[elements:16 | members:16]`
///  12: member descriptor offset
/// SM5 adds 4 unknown u32s + 1 name offset u32.
#[derive(Debug, Clone)]
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
    pub name: &'a str,
}

/// A struct member descriptor inside a type.
///
/// Binary layout (12 bytes):
///   0: u32 name offset
///   4: u32 type offset (recursive)
///   8: u32 byte offset within parent
#[derive(Debug, Clone)]
pub struct MemberDesc<'a> {
    /// Member name.
    pub name: &'a str,
    /// Parsed type for this member.
    pub member_type: TypeDesc<'a>,
    /// Byte offset within the parent structure.
    pub offset: u32,
}

/// Resource input type (D3D_SHADER_INPUT_TYPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ResourceInputType {
    CBuffer = 0,
    TBuffer = 1,
    Texture = 2,
    Sampler = 3,
    UavRwTyped = 4,
    Structured = 5,
    UavRwStructured = 6,
    ByteAddress = 7,
    UavRwByteAddress = 8,
    UavAppendStructured = 9,
    UavConsumeStructured = 10,
    UavRwStructuredWithCounter = 11,
}

impl ResourceInputType {
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
            _ => return None,
        })
    }

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
        }
    }
}

/// Resource dimension (D3D_SRV_DIMENSION).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ResourceDimension {
    Buffer = 1,
    Texture1D = 2,
    Texture2D = 3,
    Texture2DMS = 4,
    Texture3D = 5,
    TextureCube = 6,
    Texture1DArray = 7,
    Texture2DArray = 8,
    Texture2DMSArray = 9,
    TextureCubeArray = 10,
}

impl ResourceDimension {
    pub fn from_u32(v: u32) -> Option<Self> {
        Some(match v {
            1 => Self::Buffer,
            2 => Self::Texture1D,
            3 => Self::Texture2D,
            4 => Self::Texture2DMS,
            5 => Self::Texture3D,
            6 => Self::TextureCube,
            7 => Self::Texture1DArray,
            8 => Self::Texture2DArray,
            9 => Self::Texture2DMSArray,
            10 => Self::TextureCubeArray,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Buffer => "buf",
            Self::Texture1D => "1d",
            Self::Texture2D => "2d",
            Self::Texture2DMS => "2dMS",
            Self::Texture3D => "3d",
            Self::TextureCube => "cube",
            Self::Texture1DArray => "1darray",
            Self::Texture2DArray => "2darray",
            Self::Texture2DMSArray => "2dMSarray",
            Self::TextureCubeArray => "cubearray",
        }
    }
}

/// Resource binding flags (D3D_SHADER_CBUFFER_FLAGS / input bind flags).
pub const BIND_FLAG_USER_PACKED: u32 = 0x1;
pub const BIND_FLAG_USED: u32 = 0x2;
pub const BIND_FLAG_COMPARISON_SAMPLER: u32 = 0x4;
pub const BIND_FLAG_TEX_COMP_0: u32 = 0x8;
pub const BIND_FLAG_TEX_COMP_1: u32 = 0x10;

/// Sentinel value for unused texture/sampler start slots.
pub const SLOT_UNUSED: u32 = 0xFFFFFFFF;

/// A shader resource binding (texture, sampler, cbuffer, UAV, etc.).
#[derive(Debug)]
pub struct ResourceBinding<'a> {
    /// Binding name.
    pub name: &'a str,
    /// Resource type (0=cbuffer, 2=texture, 3=sampler, 4=uav_rwtyped, …).
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
    /// Binding flags (userPacked, used, comparisonSampler, …).
    pub flags: u32,
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

    fn flags_str(&self) -> alloc::string::String {
        if self.flags == 0 {
            return alloc::string::String::new();
        }
        let mut parts = alloc::vec::Vec::new();
        if self.flags & BIND_FLAG_USER_PACKED != 0 {
            parts.push("userPacked");
        }
        if self.flags & BIND_FLAG_USED != 0 {
            parts.push("used");
        }
        if self.flags & BIND_FLAG_COMPARISON_SAMPLER != 0 {
            parts.push("comparisonSampler");
        }
        if self.flags & BIND_FLAG_TEX_COMP_0 != 0 {
            parts.push("texComp0");
        }
        if self.flags & BIND_FLAG_TEX_COMP_1 != 0 {
            parts.push("texComp1");
        }
        if parts.is_empty() {
            alloc::format!("0x{:x}", self.flags)
        } else {
            let mut s = alloc::string::String::new();
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    s.push(';');
                }
                s.push_str(p);
            }
            s
        }
    }

    /// Format the type, dimension, slot, bind count, and flags columns.
    pub fn format_columns(&self) -> alloc::string::String {
        let flags = self.flags_str();
        if flags.is_empty() {
            alloc::format!(
                "{:<12} {:<8} {:<4} {}",
                self.type_name(),
                self.dim_name(),
                self.bind_point,
                self.bind_count
            )
        } else {
            alloc::format!(
                "{:<12} {:<8} {:<4} {:<5} {}",
                self.type_name(),
                self.dim_name(),
                self.bind_point,
                self.bind_count,
                flags
            )
        }
    }
}

impl fmt::Display for ResourceBinding<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:<20} {}", self.name, self.format_columns())
    }
}

/// Parse an RDEF chunk.
///
/// RDEF header layout (28 bytes):
///   0: u32 constant_buffer_count
///   4: u32 constant_buffer_offset
///   8: u32 bound_resource_count
///  12: u32 bound_resource_offset
///  16: u32 target_version  (minor | major<<8 | type<<16)
///  20: u32 flags
///  24: u32 creator_offset
pub fn parse_rdef(data: &[u8]) -> Option<ResourceDef<'_>> {
    if data.len() < 28 {
        return None;
    }

    let mut c = SliceCursor::new(data);
    let cb_count = c.read_u32_le().ok()? as usize;
    let cb_offset = c.read_u32_le().ok()? as usize;
    let binding_count = c.read_u32_le().ok()? as usize;
    let binding_offset = c.read_u32_le().ok()? as usize;
    let target_version = c.read_u32_le().ok()?;
    let compile_flags = c.read_u32_le().ok()?;

    // SM5 uses 40-byte variable descriptors; SM4 uses 24-byte.
    let major_version = (target_version >> 8) & 0xFF;
    let is_sm5 = major_version >= 5;
    let var_stride: usize = if is_sm5 { 40 } else { 24 };

    // Creator string (at offset 24 in the RDEF header)
    let creator_off = c.read_u32_le().ok()? as usize;
    let creator = if creator_off < data.len() {
        read_cstring(data, creator_off)
    } else {
        ""
    };

    // SM5 has an RD11 sub-header (8 u32s = 32 bytes) after the main header.
    let rd11_extra = if is_sm5 && data.len() >= 60 {
        Some([
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
            c.read_u32_le().ok()?,
        ])
    } else {
        None
    };

    // Parse resource bindings
    let mut bindings = Vec::with_capacity(binding_count);
    for i in 0..binding_count {
        let base = binding_offset + i * 32;
        if base + 32 > data.len() {
            break;
        }
        c.seek(SeekFrom::Start(base as u64)).ok()?;
        let name_off = c.read_u32_le().ok()? as usize;
        let input_type = c.read_u32_le().ok()?;
        let return_type = c.read_u32_le().ok()?;
        let dimension = c.read_u32_le().ok()?;
        let num_samples = c.read_u32_le().ok()?;
        let bind_point = c.read_u32_le().ok()?;
        let bind_count = c.read_u32_le().ok()?;
        let flags = c.read_u32_le().ok()?;
        bindings.push(ResourceBinding {
            name: read_cstring(data, name_off),
            input_type,
            return_type,
            dimension,
            num_samples,
            bind_point,
            bind_count,
            flags,
        });
    }

    // Parse constant buffers
    let mut constant_buffers = Vec::with_capacity(cb_count);
    for i in 0..cb_count {
        let base = cb_offset + i * 24;
        if base + 24 > data.len() {
            break;
        }
        c.seek(SeekFrom::Start(base as u64)).ok()?;
        let name_off = c.read_u32_le().ok()? as usize;
        let var_count = c.read_u32_le().ok()? as usize;
        let var_offset = c.read_u32_le().ok()? as usize;
        let cb_size = c.read_u32_le().ok()?;
        let cb_flags = c.read_u32_le().ok()?;
        let cb_type = c.read_u32_le().ok()?;

        let mut variables = Vec::with_capacity(var_count);
        for j in 0..var_count {
            let vbase = var_offset + j * var_stride;
            if vbase + var_stride > data.len() {
                break;
            }
            c.seek(SeekFrom::Start(vbase as u64)).ok()?;
            let vname_off = c.read_u32_le().ok()? as usize;
            let v_offset = c.read_u32_le().ok()?;
            let v_size = c.read_u32_le().ok()?;
            let v_flags = c.read_u32_le().ok()?;
            let v_type_offset = c.read_u32_le().ok()? as usize;
            let v_default_value_offset = c.read_u32_le().ok()? as usize;
            let (tex_start, tex_size, samp_start, samp_size) = if is_sm5 {
                (
                    Some(c.read_u32_le().ok()?),
                    Some(c.read_u32_le().ok()?),
                    Some(c.read_u32_le().ok()?),
                    Some(c.read_u32_le().ok()?),
                )
            } else {
                (None, None, None, None)
            };

            let var_type = parse_type_desc(data, v_type_offset, is_sm5);
            let default_value = if v_default_value_offset != 0 && v_size > 0 {
                let end = v_default_value_offset + v_size as usize;
                if end <= data.len() {
                    Vec::from(&data[v_default_value_offset..end])
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            variables.push(CBufferVariable {
                name: read_cstring(data, vname_off),
                offset: v_offset,
                size: v_size,
                flags: v_flags,
                var_type,
                default_value,
                texture_start: tex_start,
                texture_size: tex_size,
                sampler_start: samp_start,
                sampler_size: samp_size,
            });
        }

        constant_buffers.push(CBufferDef {
            name: read_cstring(data, name_off),
            variables,
            size: cb_size,
            flags: cb_flags,
            cb_type,
        });
    }

    Some(ResourceDef {
        constant_buffers,
        bindings,
        creator,
        target_version,
        compile_flags,
        rd11_extra,
    })
}

/// Parse a type descriptor at `offset` within the RDEF chunk data.
fn parse_type_desc<'a>(data: &'a [u8], offset: usize, is_sm5: bool) -> TypeDesc<'a> {
    let empty = TypeDesc {
        class: 0,
        var_type: 0,
        rows: 0,
        columns: 0,
        elements: 0,
        members: Vec::new(),
        sm5_extra: None,
        name: "",
    };
    if offset + 16 > data.len() {
        return empty;
    }
    let mut c = SliceCursor::new(data);
    if c.seek(SeekFrom::Start(offset as u64)).is_err() {
        return empty;
    }
    let class_type = match c.read_u32_le() {
        Ok(v) => v,
        Err(_) => return empty,
    };
    let rows_cols = match c.read_u32_le() {
        Ok(v) => v,
        Err(_) => return empty,
    };
    let elems_members = match c.read_u32_le() {
        Ok(v) => v,
        Err(_) => return empty,
    };
    let member_offset = match c.read_u32_le() {
        Ok(v) => v as usize,
        Err(_) => return empty,
    };

    let class = (class_type & 0xFFFF) as u16;
    let var_type = (class_type >> 16) as u16;
    let rows = (rows_cols & 0xFFFF) as u16;
    let columns = (rows_cols >> 16) as u16;
    let elements = (elems_members & 0xFFFF) as u16;
    let member_count = (elems_members >> 16) as u16;

    let sm5_extra = if is_sm5 {
        let a = c.read_u32_le().unwrap_or(0);
        let b = c.read_u32_le().unwrap_or(0);
        let d = c.read_u32_le().unwrap_or(0);
        let e = c.read_u32_le().unwrap_or(0);
        Some([a, b, d, e])
    } else {
        None
    };

    // Parse member descriptors (12 bytes each: name_off, type_off, byte_offset)
    let mut members = Vec::new();
    if member_count > 0 && member_offset + (member_count as usize) * 12 <= data.len() {
        members.reserve(member_count as usize);
        for k in 0..member_count as usize {
            let mbase = member_offset + k * 12;
            if c.seek(SeekFrom::Start(mbase as u64)).is_err() {
                break;
            }
            let mname_off = c.read_u32_le().unwrap_or(0) as usize;
            let mtype_off = c.read_u32_le().unwrap_or(0) as usize;
            let moffset = c.read_u32_le().unwrap_or(0);
            members.push(MemberDesc {
                name: read_cstring(data, mname_off),
                member_type: parse_type_desc(data, mtype_off, is_sm5),
                offset: moffset,
            });
        }
    }

    // SM5: name offset comes after the 4 unknowns in the type descriptor body
    let name = if is_sm5 {
        // Name offset is at: base + 16 (fixed) + 16 (4 unknowns) = base + 32
        let name_pos = offset + 32;
        if name_pos + 4 <= data.len() {
            if c.seek(SeekFrom::Start(name_pos as u64)).is_ok() {
                let noff = c.read_u32_le().unwrap_or(0) as usize;
                if noff > 0 && noff < data.len() {
                    read_cstring(data, noff)
                } else {
                    ""
                }
            } else {
                ""
            }
        } else {
            ""
        }
    } else {
        ""
    };

    TypeDesc {
        class,
        var_type,
        rows,
        columns,
        elements,
        members,
        sm5_extra,
        name,
    }
}

impl ResourceDef<'_> {
    fn is_sm5(&self) -> bool {
        ((self.target_version >> 8) & 0xFF) >= 5
    }

    fn var_stride(&self) -> usize {
        if self.is_sm5() { 40 } else { 24 }
    }

    /// Collect all type descriptors from the variable tree, returning
    /// a de-duplicated list in encounter order. Each entry is an index
    /// into the returned Vec; the index is stored alongside each variable.
    fn collect_types(&self) -> (Vec<&TypeDesc<'_>>, Vec<Vec<usize>>) {
        // types[i] = &TypeDesc, cb_var_type_idx[cb][var] = index into types
        let mut types: Vec<*const TypeDesc<'_>> = Vec::new();
        let mut cb_var_idx: Vec<Vec<usize>> = Vec::new();

        fn register<'a>(td: &'a TypeDesc<'a>, types: &mut Vec<*const TypeDesc<'a>>) -> usize {
            let ptr = td as *const TypeDesc<'a>;
            // Check for pointer-identity dedup
            if let Some(pos) = types.iter().position(|&p| core::ptr::eq(p, ptr)) {
                return pos;
            }
            let idx = types.len();
            types.push(ptr);
            for m in &td.members {
                register(&m.member_type, types);
            }
            idx
        }

        for cb in &self.constant_buffers {
            let mut var_idxs = Vec::with_capacity(cb.variables.len());
            for v in &cb.variables {
                let idx = register(&v.var_type, &mut types);
                var_idxs.push(idx);
            }
            cb_var_idx.push(var_idxs);
        }

        // Safety: we only use these as shared references within this function's
        // caller scope, and the lifetimes are tied to &self.
        let types_ref: Vec<&TypeDesc<'_>> = types.iter().map(|&p| unsafe { &*p }).collect();
        (types_ref, cb_var_idx)
    }
}

impl ChunkWriter for ResourceDef<'_> {
    fn fourcc(&self) -> [u8; 4] {
        *b"RDEF"
    }

    fn write_payload(&self) -> Vec<u8> {
        let is_sm5 = self.is_sm5();
        let var_stride = self.var_stride();

        // Pass 1: calculate section sizes and offsets.
        let header_size: usize = 28;
        let rd11_size: usize = if self.rd11_extra.is_some() { 32 } else { 0 };
        let bindings_size = self.bindings.len() * 32;
        let cb_desc_size = self.constant_buffers.len() * 24;
        let total_vars: usize = self
            .constant_buffers
            .iter()
            .map(|cb| cb.variables.len())
            .sum();
        let vars_size = total_vars * var_stride;

        // Collect types for writing
        let (types, cb_var_type_idx) = self.collect_types();
        let type_desc_stride: usize = if is_sm5 { 36 } else { 16 };
        // Count member descriptors
        let total_members: usize = types.iter().map(|t| t.members.len()).sum();
        let types_size = types.len() * type_desc_stride;
        let members_size = total_members * 12;

        // Default value blobs
        let mut default_blobs: Vec<(&[u8], usize)> = Vec::new(); // (data, absolute_offset)
        let default_values_start = header_size
            + rd11_size
            + bindings_size
            + cb_desc_size
            + vars_size
            + types_size
            + members_size;
        let mut dv_cursor = default_values_start;
        for cb in &self.constant_buffers {
            for v in &cb.variables {
                if !v.default_value.is_empty() {
                    default_blobs.push((&v.default_value, dv_cursor));
                    dv_cursor += v.default_value.len();
                }
            }
        }
        let default_values_size = dv_cursor - default_values_start;

        let string_table_base = header_size
            + rd11_size
            + bindings_size
            + cb_desc_size
            + vars_size
            + types_size
            + members_size
            + default_values_size;
        let mut st = StringTableWriter::new(string_table_base);

        // Pass 2: register all strings.
        st.add(self.creator);
        for b in &self.bindings {
            st.add(b.name);
        }
        for cb in &self.constant_buffers {
            st.add(cb.name);
            for v in &cb.variables {
                st.add(v.name);
            }
        }
        for td in &types {
            if !td.name.is_empty() {
                st.add(td.name);
            }
            for m in &td.members {
                st.add(m.name);
            }
        }

        let total_size = string_table_base + st.len();
        let mut out = Vec::with_capacity(total_size);

        // Offsets for header
        let binding_offset = header_size + rd11_size;
        let cb_offset = binding_offset + bindings_size;

        // Write header (28 bytes)
        out.extend_from_slice(&(self.constant_buffers.len() as u32).to_le_bytes());
        out.extend_from_slice(&(cb_offset as u32).to_le_bytes());
        out.extend_from_slice(&(self.bindings.len() as u32).to_le_bytes());
        out.extend_from_slice(&(binding_offset as u32).to_le_bytes());
        out.extend_from_slice(&self.target_version.to_le_bytes());
        out.extend_from_slice(&self.compile_flags.to_le_bytes());
        out.extend_from_slice(&st.add(self.creator).to_le_bytes());

        // Write RD11 sub-header if present
        if let Some(rd11) = &self.rd11_extra {
            for v in rd11 {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }

        // Write resource bindings (32 bytes each)
        for b in &self.bindings {
            out.extend_from_slice(&st.add(b.name).to_le_bytes());
            out.extend_from_slice(&b.input_type.to_le_bytes());
            out.extend_from_slice(&b.return_type.to_le_bytes());
            out.extend_from_slice(&b.dimension.to_le_bytes());
            out.extend_from_slice(&b.num_samples.to_le_bytes());
            out.extend_from_slice(&b.bind_point.to_le_bytes());
            out.extend_from_slice(&b.bind_count.to_le_bytes());
            out.extend_from_slice(&b.flags.to_le_bytes());
        }

        // Write CBuffer descriptors (24 bytes each)
        // We need to know where each cbuffer's variables start.
        let vars_base = cb_offset + cb_desc_size;
        let mut var_cursor = vars_base;
        for cb in &self.constant_buffers {
            out.extend_from_slice(&st.add(cb.name).to_le_bytes());
            out.extend_from_slice(&(cb.variables.len() as u32).to_le_bytes());
            out.extend_from_slice(&(var_cursor as u32).to_le_bytes());
            out.extend_from_slice(&cb.size.to_le_bytes());
            out.extend_from_slice(&cb.flags.to_le_bytes());
            out.extend_from_slice(&cb.cb_type.to_le_bytes());
            var_cursor += cb.variables.len() * var_stride;
        }

        // Pre-compute type descriptor offsets and member descriptor offsets.
        let types_base = vars_base + vars_size;
        let mut type_offsets: Vec<usize> = Vec::with_capacity(types.len());
        for i in 0..types.len() {
            type_offsets.push(types_base + i * type_desc_stride);
        }
        // Member descriptors come right after all type descriptors
        let members_base = types_base + types_size;
        let mut member_offsets: Vec<usize> = Vec::with_capacity(types.len());
        let mut mcursor = members_base;
        for td in &types {
            member_offsets.push(mcursor);
            mcursor += td.members.len() * 12;
        }

        // Write variable descriptors
        let mut dv_write_cursor = default_values_start;
        for (cb_idx, cb) in self.constant_buffers.iter().enumerate() {
            for (v_idx, v) in cb.variables.iter().enumerate() {
                out.extend_from_slice(&st.add(v.name).to_le_bytes());
                out.extend_from_slice(&v.offset.to_le_bytes());
                out.extend_from_slice(&v.size.to_le_bytes());
                out.extend_from_slice(&v.flags.to_le_bytes());
                let tidx = cb_var_type_idx[cb_idx][v_idx];
                out.extend_from_slice(&(type_offsets[tidx] as u32).to_le_bytes());
                if v.default_value.is_empty() {
                    out.extend_from_slice(&0u32.to_le_bytes());
                } else {
                    out.extend_from_slice(&(dv_write_cursor as u32).to_le_bytes());
                    dv_write_cursor += v.default_value.len();
                }
                if is_sm5 {
                    out.extend_from_slice(&v.texture_start.unwrap_or(SLOT_UNUSED).to_le_bytes());
                    out.extend_from_slice(&v.texture_size.unwrap_or(0).to_le_bytes());
                    out.extend_from_slice(&v.sampler_start.unwrap_or(SLOT_UNUSED).to_le_bytes());
                    out.extend_from_slice(&v.sampler_size.unwrap_or(0).to_le_bytes());
                }
            }
        }

        // Write type descriptors
        for (i, td) in types.iter().enumerate() {
            let class_type = (td.class as u32) | ((td.var_type as u32) << 16);
            let rows_cols = (td.rows as u32) | ((td.columns as u32) << 16);
            let elems_members = (td.elements as u32) | ((td.members.len() as u32) << 16);
            let moff = if td.members.is_empty() {
                0u32
            } else {
                member_offsets[i] as u32
            };
            out.extend_from_slice(&class_type.to_le_bytes());
            out.extend_from_slice(&rows_cols.to_le_bytes());
            out.extend_from_slice(&elems_members.to_le_bytes());
            out.extend_from_slice(&moff.to_le_bytes());
            if is_sm5 {
                if let Some(extra) = &td.sm5_extra {
                    for v in extra {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                } else {
                    for _ in 0..4 {
                        out.extend_from_slice(&0u32.to_le_bytes());
                    }
                }
                let name_off = if td.name.is_empty() {
                    0u32
                } else {
                    st.add(td.name)
                };
                out.extend_from_slice(&name_off.to_le_bytes());
            }
        }

        // Write member descriptors
        for (i, td) in types.iter().enumerate() {
            for m in &td.members {
                out.extend_from_slice(&st.add(m.name).to_le_bytes());
                // Find the type index for this member's type
                let mtype_ptr = &m.member_type as *const TypeDesc<'_>;
                let mtype_idx = types
                    .iter()
                    .position(|t| core::ptr::eq(*t, mtype_ptr))
                    .unwrap_or(0);
                out.extend_from_slice(&(type_offsets[mtype_idx] as u32).to_le_bytes());
                out.extend_from_slice(&m.offset.to_le_bytes());
            }
            let _ = i;
        }

        // Write default value blobs
        for cb in &self.constant_buffers {
            for v in &cb.variables {
                if !v.default_value.is_empty() {
                    out.extend_from_slice(&v.default_value);
                }
            }
        }

        // Write string table
        out.extend_from_slice(&st.finish());

        out
    }
}
