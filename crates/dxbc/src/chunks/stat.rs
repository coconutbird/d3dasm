//! Parser for the STAT (statistics) chunk in DXBC shader bytecode.

use core::fmt;

use nostdio::{Cursor, ReadLe};

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

    /// Reserved/unknown dwords at payload offsets 88, 92, 108, 112 — preserved
    /// verbatim for byte-exact round-trips.
    pub reserved: [u32; 4],

    /// Original chunk size in bytes, used during round-trip writing so we
    /// emit exactly the same number of bytes as we read.
    pub raw_size: usize,
}

/// Parse a STAT chunk into [`ShaderStats`].
///
/// Returns `None` if the data is too short to contain a valid STAT chunk.
#[must_use]
pub fn parse_stat(data: &[u8]) -> Option<ShaderStats> {
    if data.len() < 116 {
        return None;
    }

    // Read every present dword; missing trailing fields default to 0.
    let mut c = Cursor::new(data);
    let n = data.len() / 4;
    let mut d = alloc::vec::Vec::with_capacity(n);
    for _ in 0..n {
        d.push(c.read_u32_le().ok()?);
    }
    let g = |i: usize| d.get(i).copied().unwrap_or(0);

    Some(ShaderStats {
        instruction_count: g(0),
        temp_register_count: g(1),
        define_count: g(2),
        declaration_count: g(3),
        float_instruction_count: g(4),
        int_instruction_count: g(5),
        uint_instruction_count: g(6),
        static_flow_control_count: g(7),
        dynamic_flow_control_count: g(8),
        macro_instruction_count: g(9),
        temp_array_count: g(10),
        array_instruction_count: g(11),
        cut_instruction_count: g(12),
        emit_instruction_count: g(13),
        texture_normal_instructions: g(14),
        texture_load_instructions: g(15),
        texture_comp_instructions: g(16),
        texture_bias_instructions: g(17),
        texture_gradient_instructions: g(18),
        mov_instruction_count: g(19),
        movc_instruction_count: g(20),
        conversion_instruction_count: g(21),
        gs_input_primitive: g(24),
        gs_output_topology: g(25),
        gs_max_output_vertex_count: g(26),
        is_sample_frequency: g(29) != 0,
        gs_instance_count: g(30),
        hs_control_points: g(31),
        hs_output_primitive: g(32),
        hs_partitioning: g(33),
        ds_tessellator_domain: g(34),
        barrier_instructions: g(35),
        interlocked_instructions: g(36),
        texture_store_instructions: g(37),
        reserved: [g(22), g(23), g(27), g(28)],
        raw_size: data.len(),
    })
}

impl ChunkParser<'_> for ShaderStats {
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
        // The full dword layout, in order. Only the first `target_size / 4`
        // are emitted so short STAT chunks reproduce exactly.
        let dwords = [
            self.instruction_count,
            self.temp_register_count,
            self.define_count,
            self.declaration_count,
            self.float_instruction_count,
            self.int_instruction_count,
            self.uint_instruction_count,
            self.static_flow_control_count,
            self.dynamic_flow_control_count,
            self.macro_instruction_count,
            self.temp_array_count,
            self.array_instruction_count,
            self.cut_instruction_count,
            self.emit_instruction_count,
            self.texture_normal_instructions,
            self.texture_load_instructions,
            self.texture_comp_instructions,
            self.texture_bias_instructions,
            self.texture_gradient_instructions,
            self.mov_instruction_count,
            self.movc_instruction_count,
            self.conversion_instruction_count,
            self.reserved[0],
            self.reserved[1],
            self.gs_input_primitive,
            self.gs_output_topology,
            self.gs_max_output_vertex_count,
            self.reserved[2],
            self.reserved[3],
            u32::from(self.is_sample_frequency),
            self.gs_instance_count,
            self.hs_control_points,
            self.hs_output_primitive,
            self.hs_partitioning,
            self.ds_tessellator_domain,
            self.barrier_instructions,
            self.interlocked_instructions,
            self.texture_store_instructions,
        ];

        let mut buf = alloc::vec::Vec::with_capacity(target_size);
        for &v in dwords.iter().take(target_size / 4) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
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
