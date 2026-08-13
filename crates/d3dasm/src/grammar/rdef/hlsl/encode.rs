//! Emit an RDEF as editable HLSL.

use alloc::string::String;

use super::*;
use dxbc::chunks::ChunkWriter;
use dxbc::chunks::rdef::{
    CBufferDef, CBufferVariable, ResourceBinding, ResourceDef, SLOT_UNUSED, SVF_USED,
};

fn emit_binding(o: &mut String, b: &ResourceBinding<'_>, rd: &ResourceDef<'_>) -> Option<()> {
    let (spec, reg) = binding_typespec(b, rd)?;
    let _ = write!(o, "{spec} {} : register({reg}{})", b.name, b.bind_point);
    if b.bind_count != 1 {
        let _ = write!(o, "[{}]", b.bind_count);
    }
    // Annotations for fields the typespec does not capture. Structured buffers
    // always carry dimension=Buffer(1) and return=mixed(6), and textures default
    // to the -1 "not multisampled" sentinel — all derived on parse.
    let structured = matches!(b.input_type, 5 | 6 | 9 | 10 | 11);
    let dim_default = u32::from(structured);
    let ret_default = if structured { 6 } else { 0 };
    if !is_texture(b.input_type) && b.dimension != dim_default {
        let _ = write!(o, " dim={}", b.dimension);
    }
    if !is_texture(b.input_type) && b.return_type != ret_default {
        let _ = write!(o, " ret={}", b.return_type);
    }
    if structured {
        // stride == sizeof(element); derived from the def on parse.
    } else if is_texture(b.input_type) {
        if b.num_samples != 0xFFFF_FFFF {
            let _ = write!(o, " samples={}", b.num_samples);
        }
    } else if b.num_samples != 0 {
        let _ = write!(o, " samples={}", b.num_samples);
    }
    // Tag flags only when they differ from what the declaration implies.
    if b.flags != derived_binding_flags(b.input_type, b.return_type) {
        let _ = write!(o, " flags={}", bind_flags_str(b.flags));
    }
    o.push_str(";\n");
    Some(())
}

fn emit_var(o: &mut String, v: &CBufferVariable<'_>, indent: &str, sm5: bool) -> Option<()> {
    let t = &v.var_type;
    let tref = type_ref(t)?;
    let (stem, arr) = split_array(&tref);
    o.push_str(indent);
    if t.class == 2 {
        o.push_str("row_major ");
    }
    let _ = write!(o, "{stem} {}", v.name);
    if let Some(n) = arr {
        let _ = write!(o, "[{n}]");
    }
    let po = packoffset_str(v.offset)?;
    let _ = write!(o, " : packoffset({po})");
    // size: derived from the type (with array packing), so tag only on mismatch.
    if derived_var_size(t) != Some(v.size) {
        let _ = write!(o, " size={}", v.size);
    }
    // Always emit the `used` bit explicitly so RDEF parses without the SHEX
    // program. Any other variable-flag bits ride along as `var_flags=`.
    let _ = write!(o, " used={}", (v.flags & SVF_USED) != 0);
    let other = v.flags & !SVF_USED;
    if other != 0 {
        let _ = write!(o, " var_flags={other:x}");
    }
    if let Some(e) = t.sm5_extra.as_ref().filter(|extra| **extra != [0; 4]) {
        let _ = write!(o, " sm5={:x},{:x},{:x},{:x}", e[0], e[1], e[2], e[3]);
    }
    if let Some(ts) = v
        .texture_start
        .filter(|start| *start != SLOT_UNUSED || v.texture_size.unwrap_or(0) != 0)
    {
        let start = i32::from_ne_bytes(ts.to_ne_bytes());
        let _ = write!(o, " tex={start},{}", v.texture_size.unwrap_or(0));
    }
    if let Some(ss) = v
        .sampler_start
        .filter(|start| *start != SLOT_UNUSED || v.sampler_size.unwrap_or(0) != 0)
    {
        let start = i32::from_ne_bytes(ss.to_ne_bytes());
        let _ = write!(o, " samp={start},{}", v.sampler_size.unwrap_or(0));
    }
    if !v.default_value.is_empty() {
        if let Some(init) = render_default(t, &v.default_value) {
            let _ = write!(o, " = {init}");
        } else {
            o.push_str(" default=");
            for byte in v.default_value.iter() {
                let _ = write!(o, "{byte:02x}");
            }
        }
    }
    // Track whether this is an SM5 var (tex/samp Options present) for parse.
    let _ = sm5;
    o.push_str(";\n");
    Some(())
}

/// Serialize an RDEF to editable HLSL. Returns `None` for anything this codec
/// cannot model losslessly (caller falls back to the `key=value` form).
#[must_use]
pub fn rdef_to_hlsl(rd: &ResourceDef<'_>) -> Option<String> {
    let sm5 = is_sm5(rd.target_version);
    let mut o = String::new();
    let _ = writeln!(o, "target=0x{:08x}", rd.target_version);
    let _ = writeln!(o, "flags=0x{:x}", rd.compile_flags);
    let _ = writeln!(o, "creator={}", rd.creator);
    if let Some(rd11) = &rd.rd11_extra {
        o.push_str("rd11=");
        for (i, x) in rd11.iter().enumerate() {
            let _ = write!(o, "{}{x:x}", if i == 0 { "" } else { "," });
        }
        o.push('\n');
    }
    o.push('\n');

    // Struct definitions: each distinct named struct type, once.
    let mut seen_structs: Vec<&str> = Vec::new();
    for cb in &rd.constant_buffers {
        for v in &cb.variables {
            let t = &v.var_type;
            if t.members.is_empty() {
                continue;
            }
            let name = t.name.as_ref();
            if name.is_empty() || seen_structs.contains(&name) {
                continue;
            }
            seen_structs.push(name);
            // A struct type descriptor stores rows=1 and cols=size/4; both are
            // derived on parse, so only deviations need annotating.
            let struct_size: u32 = t
                .members
                .iter()
                .filter_map(|m| natural_size(&m.member_type).map(|s| m.offset + s))
                .max()
                .unwrap_or(0);
            let _ = write!(o, "struct {name}");
            if t.class != 5 {
                let _ = write!(o, " class={}", t.class);
            }
            if t.var_type != 0 {
                let _ = write!(o, " vtype={}", t.var_type);
            }
            if t.rows != 1 {
                let _ = write!(o, " rows={}", t.rows);
            }
            if u32::from(t.columns) != struct_size / 4 {
                let _ = write!(o, " cols={}", t.columns);
            }
            if let Some(e) = t.sm5_extra.as_ref().filter(|extra| **extra != [0; 4]) {
                let _ = write!(o, " sm5={:x},{:x},{:x},{:x}", e[0], e[1], e[2], e[3]);
            }
            o.push_str(" {\n");
            // Members are tightly packed; emit `+offset` only when one breaks
            // the running natural-size layout.
            let mut running = 0u32;
            for m in &t.members {
                let mref = type_ref(&m.member_type)?;
                let (stem, arr) = split_array(&mref);
                o.push_str("    ");
                if m.member_type.class == 2 {
                    o.push_str("row_major ");
                }
                let _ = write!(o, "{stem} {}", m.name);
                if let Some(n) = arr {
                    let _ = write!(o, "[{n}]");
                }
                if m.offset != running {
                    let _ = write!(o, " +{}", m.offset);
                }
                if let Some(e) = m
                    .member_type
                    .sm5_extra
                    .as_ref()
                    .filter(|extra| **extra != [0; 4])
                {
                    let _ = write!(o, " sm5={:x},{:x},{:x},{:x}", e[0], e[1], e[2], e[3]);
                }
                o.push_str(";\n");
                running = m.offset + natural_size(&m.member_type).unwrap_or(0);
            }
            o.push_str("};\n");
        }
    }
    if !seen_structs.is_empty() {
        o.push('\n');
    }

    // Body: prefer merged single-declaration cbuffers; fall back to the
    // two-section form when the array orderings don't allow a faithful merge.
    if let Some(body) = emit_merged_body(rd, sm5) {
        let full = alloc::format!("{o}{body}");
        if rdef_from_hlsl(&full)
            .is_some_and(|roundtrip| roundtrip.to_writable().data == rd.to_writable().data)
        {
            return Some(full);
        }
    }
    let body = emit_twosection_body(rd, sm5)?;
    Some(alloc::format!("{o}{body}"))
}

fn emit_cbuffer_block(
    o: &mut String,
    cb: &CBufferDef<'_>,
    sm5: bool,
    register: Option<u32>,
    bind_flags: u32,
) -> Option<()> {
    let _ = write!(o, "cbuffer {}", cb.name);
    if let Some(slot) = register {
        let _ = write!(o, " : register(b{slot})");
    }
    if bind_flags != 0 {
        let _ = write!(o, " flags={}", bind_flags_str(bind_flags));
    }
    if cb.cb_type != 0 {
        let _ = write!(o, " kind={}", cb.cb_type);
    }
    if cb.flags != 0 {
        let _ = write!(o, " cbflags={:x}", cb.flags);
    }
    o.push_str(" {\n");
    for v in &cb.variables {
        emit_var(o, v, "    ", sm5)?;
    }
    o.push_str("};\n");
    Some(())
}

/// Two-section body: every binding as a declaration, then every cbuffer layout.
fn emit_twosection_body(rd: &ResourceDef<'_>, sm5: bool) -> Option<String> {
    let mut o = String::new();
    for b in &rd.bindings {
        emit_binding(&mut o, b, rd)?;
    }
    if !rd.bindings.is_empty() {
        o.push('\n');
    }
    for cb in &rd.constant_buffers {
        emit_cbuffer_block(&mut o, cb, sm5, None, 0)?;
    }
    Some(o)
}

/// Merged body: cbuffer bindings become single `cbuffer N : register(bN) {..}`
/// blocks and resource cbuffer defs are reconstructed from their bindings.
/// Returns None when binding order can't reproduce the cbuffer-def order.
fn emit_merged_body(rd: &ResourceDef<'_>, sm5: bool) -> Option<String> {
    // Reconstruction places kind-0 defs (in cbuffer-binding order) first, then
    // resource defs (in structured-binding order); mergeable only when that
    // reproduces the actual cbuffer-def order. Regular cbuffers become
    // `cbuffer N : register(bN) {..}` blocks; resource defs are rebuilt from
    // their `StructuredBuffer<T>` declarations on parse (no kind=3 block).
    let mut expected: Vec<&str> = rd
        .bindings
        .iter()
        .filter(|b| b.input_type == 0)
        .map(|b| b.name.as_ref())
        .collect();
    expected.extend(
        rd.bindings
            .iter()
            .filter(|b| matches!(b.input_type, 5 | 6 | 9 | 10 | 11))
            .map(|b| b.name.as_ref()),
    );
    let dnames: Vec<&str> = rd
        .constant_buffers
        .iter()
        .map(|c| c.name.as_ref())
        .collect();
    // The reconstructed order must contain exactly the cbuffer-def names.
    let mut es = expected.clone();
    es.sort_unstable();
    let mut ds = dnames.clone();
    ds.sort_unstable();
    if es != ds {
        return None;
    }
    let mut o = String::new();
    // When fxc's def order differs from the reconstructed order, record it.
    if expected != dnames {
        o.push_str("cborder");
        for n in &dnames {
            let _ = write!(o, " {n}");
        }
        o.push('\n');
    }
    for b in &rd.bindings {
        if b.input_type == 0 {
            let cb = rd.constant_buffers.iter().find(|c| c.name == b.name)?;
            emit_cbuffer_block(&mut o, cb, sm5, Some(b.bind_point), b.flags)?;
        } else {
            emit_binding(&mut o, b, rd)?;
        }
    }
    Some(o)
}

// ---------------------------------------------------------------------------
// Parse
// ---------------------------------------------------------------------------
