//! `.d3dasm` text codec for the STAT (statistics) chunk.
//!
//! The byte parser ([`dxbc::chunks::stat::parse_stat`]) and the forensic
//! `Display` live in `dxbc`; this is just the editable `key value` text form.

use alloc::string::String;
use core::fmt::Write as _;

use dxbc::chunks::stat::ShaderStats;

/// Named `(key, value)` view of every STAT counter, in payload order.
fn stat_fields(s: &ShaderStats) -> [(&'static str, u32); 33] {
    [
        ("instructions", s.instruction_count),
        ("temps", s.temp_register_count),
        ("defines", s.define_count),
        ("declarations", s.declaration_count),
        ("float_ops", s.float_instruction_count),
        ("int_ops", s.int_instruction_count),
        ("uint_ops", s.uint_instruction_count),
        ("static_flow", s.static_flow_control_count),
        ("dynamic_flow", s.dynamic_flow_control_count),
        ("macros", s.macro_instruction_count),
        ("temp_arrays", s.temp_array_count),
        ("array_ops", s.array_instruction_count),
        ("cuts", s.cut_instruction_count),
        ("emits", s.emit_instruction_count),
        ("tex_normal", s.texture_normal_instructions),
        ("tex_load", s.texture_load_instructions),
        ("tex_comp", s.texture_comp_instructions),
        ("tex_bias", s.texture_bias_instructions),
        ("tex_gradient", s.texture_gradient_instructions),
        ("movs", s.mov_instruction_count),
        ("movcs", s.movc_instruction_count),
        ("conversions", s.conversion_instruction_count),
        ("gs_input_prim", s.gs_input_primitive),
        ("gs_output_topo", s.gs_output_topology),
        ("gs_max_verts", s.gs_max_output_vertex_count),
        ("gs_instances", s.gs_instance_count),
        ("hs_control_points", s.hs_control_points),
        ("hs_output_prim", s.hs_output_primitive),
        ("hs_partitioning", s.hs_partitioning),
        ("ds_domain", s.ds_tessellator_domain),
        ("barriers", s.barrier_instructions),
        ("interlocked", s.interlocked_instructions),
        ("tex_store", s.texture_store_instructions),
    ]
}

/// Assign a STAT field by its text key. Returns `false` for unknown keys.
fn set_field(s: &mut ShaderStats, key: &str, v: u32) -> bool {
    match key {
        "instructions" => s.instruction_count = v,
        "temps" => s.temp_register_count = v,
        "defines" => s.define_count = v,
        "declarations" => s.declaration_count = v,
        "float_ops" => s.float_instruction_count = v,
        "int_ops" => s.int_instruction_count = v,
        "uint_ops" => s.uint_instruction_count = v,
        "static_flow" => s.static_flow_control_count = v,
        "dynamic_flow" => s.dynamic_flow_control_count = v,
        "macros" => s.macro_instruction_count = v,
        "temp_arrays" => s.temp_array_count = v,
        "array_ops" => s.array_instruction_count = v,
        "cuts" => s.cut_instruction_count = v,
        "emits" => s.emit_instruction_count = v,
        "tex_normal" => s.texture_normal_instructions = v,
        "tex_load" => s.texture_load_instructions = v,
        "tex_comp" => s.texture_comp_instructions = v,
        "tex_bias" => s.texture_bias_instructions = v,
        "tex_gradient" => s.texture_gradient_instructions = v,
        "movs" => s.mov_instruction_count = v,
        "movcs" => s.movc_instruction_count = v,
        "conversions" => s.conversion_instruction_count = v,
        "gs_input_prim" => s.gs_input_primitive = v,
        "gs_output_topo" => s.gs_output_topology = v,
        "gs_max_verts" => s.gs_max_output_vertex_count = v,
        "gs_instances" => s.gs_instance_count = v,
        "hs_control_points" => s.hs_control_points = v,
        "hs_output_prim" => s.hs_output_primitive = v,
        "hs_partitioning" => s.hs_partitioning = v,
        "ds_domain" => s.ds_tessellator_domain = v,
        "barriers" => s.barrier_instructions = v,
        "interlocked" => s.interlocked_instructions = v,
        "tex_store" => s.texture_store_instructions = v,
        _ => return false,
    }
    true
}

/// Serialize STAT to editable `key=value` lines (plus `size` and `reserved`,
/// needed to reproduce the exact byte length).
#[must_use]
pub fn stat_to_text(s: &ShaderStats) -> String {
    let mut o = String::new();
    let _ = writeln!(o, "size={}", s.raw_size);
    let _ = writeln!(
        o,
        "sample_frequency={}",
        if s.is_sample_frequency {
            "true"
        } else {
            "false"
        }
    );
    let _ = writeln!(
        o,
        "reserved={},{},{},{}",
        s.reserved[0], s.reserved[1], s.reserved[2], s.reserved[3]
    );
    for (k, v) in stat_fields(s) {
        let _ = writeln!(o, "{k}={v}");
    }
    o
}

/// Parse the editable text form produced by [`stat_to_text`]. Whitespace around
/// the `=` and list commas is ignored; an unknown key is an error.
#[must_use]
pub fn stat_from_text(text: &str) -> Option<ShaderStats> {
    let mut s = ShaderStats::default();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let (key, val) = line.split_once('=')?;
        let (key, val) = (key.trim(), val.trim());
        match key {
            "size" => s.raw_size = val.parse().ok()?,
            "sample_frequency" => s.is_sample_frequency = parse_bool(val)?,
            "reserved" => {
                let mut it = val.split(',');
                for slot in &mut s.reserved {
                    *slot = it.next()?.trim().parse().ok()?;
                }
            }
            other => {
                let v = val.parse().ok()?;
                if !set_field(&mut s, other, v) {
                    return None;
                }
            }
        }
    }
    Some(s)
}

/// Parse a `true`/`false` boolean literal.
fn parse_bool(s: &str) -> Option<bool> {
    match s {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}
