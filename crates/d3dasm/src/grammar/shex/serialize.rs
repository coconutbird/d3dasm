//! Lossless serializer: [`Program`] -> `.d3dasm` text.
//!
//! Inverse of [`super::parse::parse`]. See the [module docs](super) for the
//! grammar. Every field the `dxbc::shex` encoder reads is emitted;
//! instruction-level modifier fields are emitted only for `Generic`/`HsPhase`
//! kinds, the only kinds whose modifiers the encoder consumes.

use alloc::string::String;
use core::fmt::Write;

use super::{AXES, DIMENSIONS, name_of};
use dxbc::shex::{
    ComponentSelect, CustomDataType, GsPrimitive, Immediates, Indices, Instruction,
    InstructionKind, Opcode, Operand, OperandIndex, Operands, Program, RegisterType, ReturnType,
    SmallU32Vec,
};
use dxbc::shex::{ImmediateType, opcode_imm_type};

/// Serialize a [`Program`] into lossless `.d3dasm` text.
#[must_use]
pub fn serialize(program: &Program) -> String {
    let mut s = String::new();
    let _ = write_program(&mut s, program);
    s
}

/// Render a single operand in `.d3dasm` form (e.g. `cb0[21].w`, `cb0[r0.y+11].x`,
/// `t0.yzxw`). Used to label resource operands in symbolication comments.
#[must_use]
pub fn operand_string(op: &Operand) -> String {
    let mut s = String::new();
    write_operand(&mut s, op, ImmediateType::Float);
    s
}

fn write_program(w: &mut String, p: &Program) -> core::fmt::Result {
    write!(
        w,
        "profile={}_{}_{}",
        p.shader_type, p.major_version, p.minor_version
    )?;
    // Preserve the chunk FourCC when it is not the default `SHEX`.
    if &p.fourcc != b"SHEX" {
        let fourcc = core::str::from_utf8(&p.fourcc).unwrap_or("SHEX");
        write!(w, " fourcc={fourcc}")?;
    }
    w.push('\n');
    for instr in &p.instructions {
        write_instruction(w, instr)?;
        w.push('\n');
    }
    Ok(())
}

fn write_instruction(w: &mut String, instr: &Instruction) -> core::fmt::Result {
    match &instr.kind {
        InstructionKind::Generic { operands } => {
            write_mnemonic(w, instr.opcode);
            write_modifiers(w, instr);
            let imm_type = opcode_imm_type(instr.opcode);
            for op in operands {
                w.push(' ');
                write_operand(w, op, imm_type);
            }
            Ok(())
        }
        InstructionKind::HsPhase => {
            write_mnemonic(w, instr.opcode);
            write_modifiers(w, instr);
            Ok(())
        }
        InstructionKind::CustomData {
            subtype,
            values,
            raw,
            ..
        } => {
            write_custom_data(w, instr.opcode.name(), *subtype, values, raw);
            Ok(())
        }
        _ => write_declaration(w, instr),
    }
}

/// Emit the opcode mnemonic, preserving the raw value of unknown opcodes
/// (whose `name()` would otherwise discard it) as `op<value>`.
fn write_mnemonic(w: &mut String, opcode: Opcode) {
    match opcode {
        Opcode::Unknown(v) => {
            let _ = write!(w, "op{v}");
        }
        other => w.push_str(other.name()),
    }
}

/// Emit non-default instruction modifier suffixes (Generic/HsPhase only).
fn write_modifiers(w: &mut String, instr: &Instruction) {
    if let Some(n) = instr.resinfo_return_type {
        let _ = write!(w, "_ri{n}");
    }
    if instr.saturate {
        w.push_str("_sat");
    }
    if instr.test_nonzero {
        w.push_str("_nz");
    }
    if instr.precise_mask != 0 {
        let _ = write!(w, "_pm{:x}", instr.precise_mask);
    }
    if instr.sync_flags != 0 {
        let _ = write!(w, "_sf{:x}", instr.sync_flags);
    }
    if let Some([u, v, x]) = instr.tex_offsets {
        let _ = write!(w, "_off({u},{v},{x})");
    }
    if let Some(r) = instr.resource_dim {
        write_resource_dim(w, r);
    }
    if let Some(r) = instr.resource_return_type {
        write_resource_rt(w, r);
    }
}

/// Render a resource-dimension extended token (type 2) readably, e.g.
/// `_res(texture2d)` or `_res(structured,stride=16)`, falling back to raw hex
/// `_rd<dword>` when the bits aren't cleanly reconstructible.
fn write_resource_dim(w: &mut String, rd: u32) {
    let dim = (rd >> 6) & 0x1F;
    let stride = (rd >> 11) & 0xF_FFFF;
    let recon = 2 | (dim << 6) | (stride << 11);
    if rd & 0x3F == 2 && recon == rd {
        if let Some(name) = name_of(DIMENSIONS, dim) {
            if stride == 0 {
                let _ = write!(w, "_res({name})");
            } else {
                let _ = write!(w, "_res({name},stride={stride})");
            }
            return;
        }
    }
    let _ = write!(w, "_rd{rd:08x}");
}

/// Render a resource-return-type extended token (type 3) readably, e.g.
/// `_rt(float,float,float,float)`, falling back to raw hex `_rr<dword>`.
fn write_resource_rt(w: &mut String, rr: u32) {
    let parts = [
        (rr >> 6) & 0xF,
        (rr >> 10) & 0xF,
        (rr >> 14) & 0xF,
        (rr >> 18) & 0xF,
    ];
    let recon = 3 | (parts[0] << 6) | (parts[1] << 10) | (parts[2] << 14) | (parts[3] << 18);
    if rr & 0x3F == 3 && recon == rr {
        w.push_str("_rt(");
        for (i, &v) in parts.iter().enumerate() {
            if i > 0 {
                w.push(',');
            }
            match ReturnType::from_u32(v) {
                ReturnType::Unknown(u) => {
                    let _ = write!(w, "unknown{u}");
                }
                t => w.push_str(t.name()),
            }
        }
        w.push(')');
        return;
    }
    let _ = write!(w, "_rr{rr:08x}");
}

// ---------------------------------------------------------------------------
// Operands
// ---------------------------------------------------------------------------

fn write_operand(w: &mut String, op: &Operand, imm_type: ImmediateType) {
    if op.negate {
        w.push('-');
    }
    if op.abs {
        w.push('|');
    }
    // Inline immediates `l(...)` / `d(...)` only when un-indexed. An immediate
    // that carries indices (a degenerate but decodable state) is emitted
    // register-style (prefix + indices) by the `_` arm below.
    let inline_imm = matches!(
        op.reg_type,
        RegisterType::Immediate32 | RegisterType::Immediate64
    ) && op.indices.is_empty();
    match op.reg_type {
        RegisterType::Immediate32 if op.indices.is_empty() => {
            write_immediates(w, 'l', &op.immediate_values, imm_type);
        }
        // 64-bit immediates stay as raw dword hex (no f64 prettifying).
        RegisterType::Immediate64 if op.indices.is_empty() => {
            write_immediates(w, 'd', &op.immediate_values, ImmediateType::Uint);
        }
        RegisterType::Unknown(v) => {
            // `prefix()` returns "?reg" and drops the value; preserve it.
            let _ = write!(w, "?reg({v})");
            write_indices(w, &op.indices);
        }
        _ => {
            w.push_str(op.reg_type.prefix());
            write_indices(w, &op.indices);
        }
    }
    // An inline immediate's component selection is implied by its value count:
    // scalar literals are 1-component (`.1`), vector literals are 4-component
    // mask-mode with an empty mask (`:`). Both are redundant, so omit them — the
    // parser restores the right one from the value count.
    let redundant_imm_select = inline_imm
        && matches!(
            op.components,
            ComponentSelect::OneComponent | ComponentSelect::Mask(0)
        );
    if !redundant_imm_select {
        write_components(w, &op.components);
    }
    if op.abs {
        w.push('|');
    }
}

fn write_immediates(w: &mut String, sigil: char, values: &Immediates, imm_type: ImmediateType) {
    w.push(sigil);
    w.push('(');
    for (i, v) in values.iter().enumerate() {
        if i > 0 {
            w.push(',');
        }
        write_immediate_value(w, *v, imm_type);
    }
    w.push(')');
}

/// Write one immediate value readably, with a hex fallback whenever the pretty
/// form would not re-parse to the exact same bits (NaN, ambiguous, etc.).
fn write_immediate_value(w: &mut String, v: u32, imm_type: ImmediateType) {
    match imm_type {
        ImmediateType::Float => {
            let f = f32::from_bits(v);
            if !f.is_nan() {
                // `{:?}` is the shortest round-trip form and always carries a
                // '.'/'e'/"inf" so the parser treats it as a float, not an int.
                let s = alloc::format!("{f:?}");
                if s.parse::<f32>().map(f32::to_bits) == Ok(v) {
                    w.push_str(&s);
                    return;
                }
            }
            let _ = write!(w, "0x{v:08x}");
        }
        ImmediateType::Int => {
            let _ = write!(w, "{}", i32::from_ne_bytes(v.to_ne_bytes()));
        }
        ImmediateType::Uint => {
            let _ = write!(w, "0x{v:08x}");
        }
    }
}

fn write_indices(w: &mut String, indices: &Indices) {
    for (i, idx) in indices.iter().enumerate() {
        match idx {
            // The first immediate index renders unbracketed (e.g. `r0`, `cb0`).
            OperandIndex::Imm32(v) if i == 0 => {
                let _ = write!(w, "{v}");
            }
            OperandIndex::Imm64(v) if i == 0 => {
                let _ = write!(w, "{v}L");
            }
            _ => {
                w.push('[');
                write_index(w, idx);
                w.push(']');
            }
        }
    }
}

fn write_index(w: &mut String, idx: &OperandIndex) {
    match idx {
        OperandIndex::Imm32(v) => {
            let _ = write!(w, "{v}");
        }
        OperandIndex::Imm64(v) => {
            let _ = write!(w, "{v}L");
        }
        OperandIndex::Relative(sub) => write_operand(w, sub, ImmediateType::Uint),
        OperandIndex::RelativePlusImm(imm, sub) => {
            write_operand(w, sub, ImmediateType::Uint);
            let _ = write!(w, "+{imm}");
        }
    }
}

fn write_components(w: &mut String, comp: &ComponentSelect) {
    match comp {
        ComponentSelect::ZeroComponent => {}
        ComponentSelect::OneComponent => w.push_str(".1"),
        ComponentSelect::Scalar(c) => {
            w.push('.');
            w.push(AXES[*c as usize & 3]);
        }
        ComponentSelect::Swizzle(s) => {
            w.push('.');
            for &c in s {
                w.push(AXES[c as usize & 3]);
            }
        }
        ComponentSelect::Mask(m) => {
            // Write-mask: `:` distinguishes it from a `.` read-swizzle.
            w.push(':');
            for (i, axis) in AXES.iter().enumerate() {
                if m & (1 << i) != 0 {
                    w.push(*axis);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Declarations
// ---------------------------------------------------------------------------

fn write_declaration(w: &mut String, instr: &Instruction) -> core::fmt::Result {
    let name = instr.opcode.name();
    w.push_str(name);
    match &instr.kind {
        InstructionKind::DclGlobalFlags { flags } => {
            for (i, f) in flags.iter().enumerate() {
                w.push_str(if i == 0 { " " } else { "|" });
                w.push_str(f);
            }
        }
        InstructionKind::DclInput {
            interpolation,
            system_value,
            operands,
        } => {
            for op in operands {
                w.push(' ');
                write_operand(w, op, ImmediateType::Uint);
            }
            if let Some(interp) = interpolation {
                write!(w, " interp={interp}")?;
            }
            if let Some(sv) = system_value {
                write!(w, " sv={sv}")?;
            }
        }
        InstructionKind::DclOutput {
            system_value,
            operands,
        } => {
            for op in operands {
                w.push(' ');
                write_operand(w, op, ImmediateType::Uint);
            }
            if let Some(sv) = system_value {
                write!(w, " sv={sv}")?;
            }
        }
        InstructionKind::DclResource {
            dimension,
            sample_count,
            return_type,
            operands,
        } => {
            write_op0(w, operands);
            write!(w, " dim={dimension} rt=")?;
            write_return_types(w, return_type);
            // Always emitted so the declaration is self-describing.
            write!(w, " samples={sample_count}")?;
        }
        InstructionKind::DclSampler { mode, operands } => {
            write_op0(w, operands);
            write!(w, " mode={mode}")?;
        }
        InstructionKind::DclConstantBuffer { access, operands } => {
            write_op0(w, operands);
            write!(w, " access={access}")?;
        }
        _ => write_special_declaration(w, instr)?,
    }
    Ok(())
}

fn write_special_declaration(w: &mut String, instr: &Instruction) -> core::fmt::Result {
    match &instr.kind {
        InstructionKind::DclTemps { count }
        | InstructionKind::DclMaxOutputVertexCount { count }
        | InstructionKind::DclGsInstanceCount { count }
        | InstructionKind::DclOutputControlPointCount { count }
        | InstructionKind::DclInputControlPointCount { count }
        | InstructionKind::DclHsForkPhaseInstanceCount { count } => {
            write!(w, " {count}")?;
        }
        InstructionKind::DclIndexableTemp {
            reg,
            size,
            components,
        } => {
            write!(w, " {reg} {size} {components}")?;
        }
        InstructionKind::DclGsInputPrimitive { primitive } => {
            w.push(' ');
            match primitive {
                GsPrimitive::ControlPointPatch(count) => write!(w, "patchlist({count})")?,
                other => w.push_str(other.name()),
            }
        }
        InstructionKind::DclGsOutputTopology { topology } => {
            write!(w, " {}", topology.name())?;
        }
        InstructionKind::DclTessDomain { domain } => write!(w, " {domain}")?,
        InstructionKind::DclTessPartitioning { partitioning } => write!(w, " {partitioning}")?,
        InstructionKind::DclTessOutputPrimitive { primitive } => write!(w, " {primitive}")?,
        InstructionKind::DclHsMaxTessFactor { value } => {
            write!(w, " 0x{:08x}", value.to_bits())?;
        }
        InstructionKind::DclThreadGroup { x, y, z } => {
            write!(w, " {x} {y} {z}")?;
        }
        InstructionKind::DclUavTyped {
            dimension,
            flags,
            return_type,
            operands,
        } => {
            write_op0(w, operands);
            write!(w, " dim={dimension} rt=")?;
            write_return_types(w, return_type);
            write!(w, " flags=0x{flags:x}")?;
        }
        InstructionKind::DclUavRaw { flags, operands } => {
            write_op0(w, operands);
            write!(w, " flags=0x{flags:x}")?;
        }
        InstructionKind::DclUavStructured {
            flags,
            stride,
            operands,
        } => {
            write_op0(w, operands);
            write!(w, " stride={stride} flags=0x{flags:x}")?;
        }
        InstructionKind::DclResourceRaw { operands } => {
            write_op0(w, operands);
        }
        InstructionKind::DclResourceStructured { stride, operands } => {
            write_op0(w, operands);
            write!(w, " stride={stride}")?;
        }
        InstructionKind::DclFunctionBody { index } => {
            write!(w, " {index}")?;
        }
        InstructionKind::DclFunctionTable {
            table_index,
            body_indices,
        } => {
            write!(w, " {table_index} ")?;
            write_u32_list(w, body_indices);
        }
        InstructionKind::DclInterface {
            interface_index,
            num_call_sites,
            table_indices,
        } => {
            write!(w, " {interface_index} {num_call_sites} ")?;
            write_u32_list(w, table_indices);
        }
        InstructionKind::DclIndexRange { operands, count } => {
            write_op0(w, operands);
            write!(w, " {count}")?;
        }
        _ => {}
    }
    Ok(())
}

/// Write the leading operand of a declaration (space-prefixed), if any.
fn write_op0(w: &mut String, operands: &Operands) {
    for op in operands {
        w.push(' ');
        write_operand(w, op, ImmediateType::Uint);
    }
}

/// Write the four resource return types as a bare `t,t,t,t` list (the value of
/// an `rt=` tag).
fn write_return_types(w: &mut String, rt: &[ReturnType; 4]) {
    for (i, t) in rt.iter().enumerate() {
        if i > 0 {
            w.push(',');
        }
        match t {
            ReturnType::Unknown(v) => {
                let _ = write!(w, "unknown{v}");
            }
            other => w.push_str(other.name()),
        }
    }
}

fn write_u32_list(w: &mut String, list: &SmallU32Vec) {
    w.push('{');
    for (i, v) in list.iter().enumerate() {
        if i > 0 {
            w.push_str(", ");
        }
        let _ = write!(w, "{v}");
    }
    w.push('}');
}

fn write_custom_data(
    w: &mut String,
    keyword: &str,
    subtype: CustomDataType,
    values: &[[f32; 4]],
    raw: &[u32],
) {
    w.push_str(keyword);
    match subtype {
        CustomDataType::ImmediateConstantBuffer => {
            w.push_str(" icb {");
            for (i, row) in values.iter().enumerate() {
                if i > 0 {
                    w.push(',');
                }
                let _ = write!(
                    w,
                    " 0x{:08x} 0x{:08x} 0x{:08x} 0x{:08x}",
                    row[0].to_bits(),
                    row[1].to_bits(),
                    row[2].to_bits(),
                    row[3].to_bits()
                );
            }
            w.push_str(" }");
        }
        CustomDataType::Comment => write_raw_block(w, "comment", raw),
        CustomDataType::DebugInfo => write_raw_block(w, "debuginfo", raw),
        CustomDataType::Opaque => write_raw_block(w, "opaque", raw),
        CustomDataType::Other(v) => {
            let mut tag = String::from("other(");
            let _ = write!(tag, "{v})");
            write_raw_block(w, &tag, raw);
        }
    }
}

/// Write a non-ICB customdata block: ` <tag> { 0x.. 0x.. ... }`.
fn write_raw_block(w: &mut String, tag: &str, raw: &[u32]) {
    w.push(' ');
    w.push_str(tag);
    w.push_str(" {");
    for v in raw {
        let _ = write!(w, " 0x{v:08x}");
    }
    w.push_str(" }");
}
