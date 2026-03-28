//! Parser for the STAT (statistics) chunk in DXBC shader bytecode.

use core::fmt;

use nostdio::{ReadLe, Seek, SeekFrom, SliceCursor};

use super::{ChunkParser, ChunkWriter};

/// Shader statistics extracted from the STAT chunk.
///
/// Fields after `is_sample_frequency` are SM5-extended and only present
/// in STAT chunks larger than 120 bytes.  They are preserved for
/// round-trip fidelity; missing fields default to zero.
#[derive(Debug, Clone, Default)]
pub struct ShaderStats {
    /// Total number of instructions in the shader.
    pub instruction_count: u32,
    /// Number of temporary registers used.
    pub temp_register_count: u32,
    /// Number of `#define` constants (SM4 legacy, typically zero for SM5).
    pub define_count: u32,
    /// Number of declaration instructions.
    pub declaration_count: u32,
    /// Number of floating-point arithmetic instructions.
    pub float_instruction_count: u32,
    /// Number of signed-integer arithmetic instructions.
    pub int_instruction_count: u32,
    /// Number of unsigned-integer arithmetic instructions.
    pub uint_instruction_count: u32,
    /// Number of static flow-control instructions (`if`/`else`/`switch`).
    pub static_flow_control_count: u32,
    /// Number of dynamic flow-control instructions (`breakc`/`continuec`).
    pub dynamic_flow_control_count: u32,
    /// Macro instruction count (legacy, usually zero).
    pub macro_instruction_count: u32,
    /// Number of indexable temporary arrays.
    pub temp_array_count: u32,
    /// Number of array-indexed instructions.
    pub array_instruction_count: u32,
    /// Number of `cut` instructions (geometry shader).
    pub cut_instruction_count: u32,
    /// Number of `emit` instructions (geometry shader).
    pub emit_instruction_count: u32,
    /// Number of normal texture-sampling instructions.
    pub texture_normal_instructions: u32,
    /// Number of texture load instructions.
    pub texture_load_instructions: u32,
    /// Number of texture comparison instructions.
    pub texture_comp_instructions: u32,
    /// Number of texture bias instructions.
    pub texture_bias_instructions: u32,
    /// Number of texture gradient instructions.
    pub texture_gradient_instructions: u32,
    /// Number of `mov` instructions.
    pub mov_instruction_count: u32,
    /// Number of `movc` (conditional move) instructions.
    pub movc_instruction_count: u32,
    /// Number of type-conversion instructions.
    pub conversion_instruction_count: u32,
    /// Geometry shader input primitive type (raw enum value).
    pub gs_input_primitive: u32,
    /// Geometry shader output topology (raw enum value).
    pub gs_output_topology: u32,
    /// Maximum number of vertices a GS invocation may emit.
    pub gs_max_output_vertex_count: u32,
    /// Whether the pixel shader runs at sample frequency.
    pub is_sample_frequency: bool,

    // SM5 extended fields (offsets 120+)
    /// GS instance count, or HS/DS/CS control-point count depending on
    /// shader type.
    pub gs_instance_count: u32,
    /// Number of control points for hull shaders.
    pub hs_control_points: u32,
    /// HS output primitive topology (raw enum value).
    pub hs_output_primitive: u32,
    /// HS partitioning mode (raw enum value).
    pub hs_partitioning: u32,
    /// DS tessellator domain (raw enum value).
    pub ds_tessellator_domain: u32,
    /// Number of barrier instructions.
    pub barrier_instructions: u32,
    /// Number of interlocked (atomic) instructions.
    pub interlocked_instructions: u32,
    /// Number of texture store instructions.
    pub texture_store_instructions: u32,

    /// Original chunk size in bytes, used during round-trip writing so we
    /// emit exactly the same number of bytes as we read.
    pub raw_size: usize,
}

/// Parse a STAT chunk into [`ShaderStats`].
///
/// Returns `None` if the data is too short to contain a valid STAT chunk.
pub fn parse_stat(data: &[u8]) -> Option<ShaderStats> {
    if data.len() < 116 {
        return None;
    }

    let mut c = SliceCursor::new(data);
    let r = &mut c;

    let mut s = ShaderStats {
        instruction_count: r.read_u32_le().ok()?,
        temp_register_count: r.read_u32_le().ok()?,
        define_count: r.read_u32_le().ok()?,
        declaration_count: r.read_u32_le().ok()?,
        float_instruction_count: r.read_u32_le().ok()?,
        int_instruction_count: r.read_u32_le().ok()?,
        uint_instruction_count: r.read_u32_le().ok()?,
        static_flow_control_count: r.read_u32_le().ok()?,
        dynamic_flow_control_count: r.read_u32_le().ok()?,
        macro_instruction_count: r.read_u32_le().ok()?,
        temp_array_count: r.read_u32_le().ok()?,
        array_instruction_count: r.read_u32_le().ok()?,
        cut_instruction_count: r.read_u32_le().ok()?,
        emit_instruction_count: r.read_u32_le().ok()?,
        texture_normal_instructions: r.read_u32_le().ok()?,
        texture_load_instructions: r.read_u32_le().ok()?,
        texture_comp_instructions: r.read_u32_le().ok()?,
        texture_bias_instructions: r.read_u32_le().ok()?,
        texture_gradient_instructions: r.read_u32_le().ok()?,
        mov_instruction_count: r.read_u32_le().ok()?,
        movc_instruction_count: r.read_u32_le().ok()?,
        conversion_instruction_count: r.read_u32_le().ok()?,
        // Offsets 88 and 92 are unknown fields — skip 2 u32s
        gs_input_primitive: {
            r.seek(SeekFrom::Current(8)).ok()?;
            r.read_u32_le().ok()?
        },
        gs_output_topology: r.read_u32_le().ok()?,
        gs_max_output_vertex_count: r.read_u32_le().ok()?,
        // Offsets 108 and 112 are unknown fields
        is_sample_frequency: {
            if data.len() >= 120 {
                r.seek(SeekFrom::Start(116)).ok()?;
                r.read_u32_le().ok()? != 0
            } else {
                false
            }
        },
        raw_size: data.len(),
        ..Default::default()
    };

    // SM5 extended fields start at offset 120
    if data.len() >= 152 {
        r.seek(SeekFrom::Start(120)).ok()?;
        s.gs_instance_count = r.read_u32_le().ok()?;
        s.hs_control_points = r.read_u32_le().ok()?;
        s.hs_output_primitive = r.read_u32_le().ok()?;
        s.hs_partitioning = r.read_u32_le().ok()?;
        s.ds_tessellator_domain = r.read_u32_le().ok()?;
        s.barrier_instructions = r.read_u32_le().ok()?;
        s.interlocked_instructions = r.read_u32_le().ok()?;
        s.texture_store_instructions = r.read_u32_le().ok()?;
    }

    Some(s)
}

impl ChunkParser for ShaderStats {
    fn parse(data: &[u8]) -> Option<Self> {
        parse_stat(data)
    }
}

impl ChunkWriter for ShaderStats {
    fn fourcc(&self) -> [u8; 4] {
        *b"STAT"
    }

    fn write_payload(&self) -> alloc::vec::Vec<u8> {
        let target_size = if self.raw_size > 0 {
            self.raw_size
        } else {
            120
        };
        let mut buf = alloc::vec::Vec::with_capacity(target_size);
        let w = |buf: &mut alloc::vec::Vec<u8>, v: u32| buf.extend_from_slice(&v.to_le_bytes());
        w(&mut buf, self.instruction_count);
        w(&mut buf, self.temp_register_count);
        w(&mut buf, self.define_count);
        w(&mut buf, self.declaration_count);
        w(&mut buf, self.float_instruction_count);
        w(&mut buf, self.int_instruction_count);
        w(&mut buf, self.uint_instruction_count);
        w(&mut buf, self.static_flow_control_count);
        w(&mut buf, self.dynamic_flow_control_count);
        w(&mut buf, self.macro_instruction_count);
        w(&mut buf, self.temp_array_count);
        w(&mut buf, self.array_instruction_count);
        w(&mut buf, self.cut_instruction_count);
        w(&mut buf, self.emit_instruction_count);
        w(&mut buf, self.texture_normal_instructions);
        w(&mut buf, self.texture_load_instructions);
        w(&mut buf, self.texture_comp_instructions);
        w(&mut buf, self.texture_bias_instructions);
        w(&mut buf, self.texture_gradient_instructions);
        w(&mut buf, self.mov_instruction_count);
        w(&mut buf, self.movc_instruction_count);
        w(&mut buf, self.conversion_instruction_count);
        w(&mut buf, 0); // unknown at offset 88
        w(&mut buf, 0); // unknown at offset 92
        w(&mut buf, self.gs_input_primitive);
        w(&mut buf, self.gs_output_topology);
        w(&mut buf, self.gs_max_output_vertex_count);
        w(&mut buf, 0); // unknown at offset 108
        w(&mut buf, 0); // unknown at offset 112
        w(&mut buf, self.is_sample_frequency as u32);

        // SM5 extended fields (offset 120+)
        if target_size >= 152 {
            w(&mut buf, self.gs_instance_count);
            w(&mut buf, self.hs_control_points);
            w(&mut buf, self.hs_output_primitive);
            w(&mut buf, self.hs_partitioning);
            w(&mut buf, self.ds_tessellator_domain);
            w(&mut buf, self.barrier_instructions);
            w(&mut buf, self.interlocked_instructions);
            w(&mut buf, self.texture_store_instructions);
        }

        // Pad to original size if there were additional trailing bytes
        buf.resize(target_size, 0);
        buf
    }
}

impl fmt::Display for ShaderStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "// Statistics:")?;
        writeln!(f, "//   {} instruction(s)", self.instruction_count)?;
        writeln!(f, "//   {} temp register(s)", self.temp_register_count)?;
        if self.declaration_count > 0 {
            writeln!(f, "//   {} declaration(s)", self.declaration_count)?;
        }
        if self.float_instruction_count > 0 {
            writeln!(
                f,
                "//   {} float instruction(s)",
                self.float_instruction_count
            )?;
        }
        if self.int_instruction_count > 0 {
            writeln!(f, "//   {} int instruction(s)", self.int_instruction_count)?;
        }
        if self.uint_instruction_count > 0 {
            writeln!(
                f,
                "//   {} uint instruction(s)",
                self.uint_instruction_count
            )?;
        }
        if self.texture_normal_instructions > 0 {
            writeln!(
                f,
                "//   {} texture normal instruction(s)",
                self.texture_normal_instructions
            )?;
        }
        if self.texture_load_instructions > 0 {
            writeln!(
                f,
                "//   {} texture load instruction(s)",
                self.texture_load_instructions
            )?;
        }
        if self.static_flow_control_count > 0 {
            writeln!(
                f,
                "//   {} static flow control(s)",
                self.static_flow_control_count
            )?;
        }
        if self.dynamic_flow_control_count > 0 {
            writeln!(
                f,
                "//   {} dynamic flow control(s)",
                self.dynamic_flow_control_count
            )?;
        }
        if self.cut_instruction_count > 0 {
            writeln!(f, "//   {} cut instruction(s)", self.cut_instruction_count)?;
        }
        if self.emit_instruction_count > 0 {
            writeln!(
                f,
                "//   {} emit instruction(s)",
                self.emit_instruction_count
            )?;
        }
        if self.is_sample_frequency {
            writeln!(f, "//   sample-frequency execution")?;
        }
        Ok(())
    }
}
