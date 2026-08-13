//! Resolve cbuffer / texture / sampler / UAV register operands to their source
//! names (from RDEF) and append them as `//` comments on SHEX instruction lines.
//!
//! This is an **analysis layer** over the faithful disassembly: the ground-truth
//! operand (`cb0[21].w`) is kept verbatim, and the resolved name rides alongside
//! as a comment (`// cb0[21].w = gDirLightColor.w`). Comments are stripped on
//! parse, so the lossless round-trip is entirely unaffected.
//!
//! The bytecode addresses constants by register + component because HLSL packs
//! several variables into one 16-byte register, so a single operand can read
//! across more than one variable — those cross-variable accesses have no single
//! name and are left unannotated (the faithful operand still stands).

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write as _;

use dxbc::chunks::rdef::{CBufferDef, CBufferVariable, ResourceDef};
use dxbc::shex::{
    ComponentSelect, Instruction, InstructionKind, Operand, OperandIndex, Program, RegisterType,
};

const AXES: [char; 4] = ['x', 'y', 'z', 'w'];

/// SRV (`t#`) resource input types in RDEF.
const SRV_INPUT_TYPES: &[u32] = &[1, 2, 5, 7];
/// UAV (`u#`) resource input types in RDEF.
const UAV_INPUT_TYPES: &[u32] = &[4, 6, 8, 9, 10, 11];

/// Annotate a serialized SHEX body (`profile=…` line followed by one line per
/// instruction) with resolved cbuffer/resource names. Returns the text unchanged
/// if its line count doesn't line up with the program (defensive — never risk
/// misaligning a comment onto the wrong instruction).
#[must_use]
pub fn annotate_shex(shex_text: &str, program: &Program, rdef: &ResourceDef<'_>) -> String {
    let lines: Vec<&str> = shex_text.lines().collect();
    if lines.len() != program.instructions.len() + 1 {
        return shex_text.to_string();
    }
    let mut out = String::new();
    let _ = writeln!(out, "{}", lines[0]);
    for (i, instr) in program.instructions.iter().enumerate() {
        out.push_str(lines[i + 1]);
        if let Some(comment) = resolve_instruction(instr, rdef) {
            let _ = write!(out, "  // {comment}");
        }
        out.push('\n');
    }
    out
}

/// Build the `addr = name[, addr = name]` comment for one instruction, or `None`
/// when it references no resolvable resource. Only computational (`Generic`)
/// instructions are annotated — declarations index by size/slot, not by access.
fn resolve_instruction(instr: &Instruction, rdef: &ResourceDef<'_>) -> Option<String> {
    if !matches!(instr.kind, InstructionKind::Generic { .. }) {
        return None;
    }
    let mut parts: Vec<String> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for op in instr.operands() {
        if let Some((addr, name)) = resolve_operand(op, rdef) {
            if seen.iter().any(|s| s == &addr) {
                continue;
            }
            seen.push(addr.clone());
            parts.push(format!("{addr} = {name}"));
        }
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(", "))
    }
}

/// `(faithful operand text, resolved name)` for a resource operand, or `None`.
fn resolve_operand(op: &Operand, rdef: &ResourceDef<'_>) -> Option<(String, String)> {
    let name = match op.reg_type {
        RegisterType::ConstantBuffer => return resolve_cb(op, rdef),
        RegisterType::Resource => binding_name(rdef, slot(op)?, SRV_INPUT_TYPES)?,
        RegisterType::Sampler => binding_name(rdef, slot(op)?, &[3])?,
        RegisterType::Uav => binding_name(rdef, slot(op)?, UAV_INPUT_TYPES)?,
        _ => return None,
    };
    Some((crate::grammar::shex::operand_string(op), name))
}

/// The first (unbracketed) immediate index of an operand — the register slot.
fn slot(op: &Operand) -> Option<u32> {
    match op.indices.first()? {
        OperandIndex::Imm32(v) => Some(*v),
        _ => None,
    }
}

/// Binding name for a register slot among the given resource input types.
fn binding_name(rdef: &ResourceDef<'_>, reg: u32, input_types: &[u32]) -> Option<String> {
    rdef.bindings
        .iter()
        .find(|b| input_types.contains(&b.input_type) && b.bind_point == reg)
        .map(|b| b.name.to_string())
}

fn resolve_cb(op: &Operand, rdef: &ResourceDef<'_>) -> Option<(String, String)> {
    let slot = match op.indices.first()? {
        OperandIndex::Imm32(v) => *v,
        _ => return None,
    };
    let binding = rdef
        .bindings
        .iter()
        .find(|b| b.input_type == 0 && b.bind_point == slot)?;
    let cb = rdef
        .constant_buffers
        .iter()
        .find(|c| c.name == binding.name)?;

    let name = match op.indices.get(1)? {
        OperandIndex::Imm32(reg) => resolve_cb_static(cb, *reg, &op.components)?,
        OperandIndex::RelativePlusImm(base, sub) => {
            resolve_cb_dynamic(cb, *base, sub, &op.components)?
        }
        OperandIndex::RelativePlusImm64(_, _) => return None,
        OperandIndex::Relative(sub) => resolve_cb_dynamic(cb, 0, sub, &op.components)?,
        OperandIndex::Imm64(_) => return None,
    };
    Some((crate::grammar::shex::operand_string(op), name))
}

/// Resolve `cb#[reg].<comps>` to a named variable access, or `None` if it spans
/// more than one variable (HLSL packing) or can't be modelled.
fn resolve_cb_static(cb: &CBufferDef<'_>, reg: u32, comps: &ComponentSelect) -> Option<String> {
    let comps = component_list(comps);
    // All accessed components must fall within a single variable.
    let mut var: Option<&CBufferVariable<'_>> = None;
    for &c in &comps {
        let byte = reg * 16 + u32::from(c) * 4;
        let v = cb
            .variables
            .iter()
            .find(|v| byte >= v.offset && byte < v.offset + v.size)?;
        match var {
            None => var = Some(v),
            Some(prev) if prev.name == v.name => {}
            Some(_) => return None, // cross-variable — no single name
        }
    }
    let v = var?;
    Some(render_var_access(v, reg, &comps))
}

/// Resolve `cb#[<sub>+base].<comps>` to `arrayName[<sub>]` when `base` is an
/// array's start register (the common case); otherwise `None`.
fn resolve_cb_dynamic(
    cb: &CBufferDef<'_>,
    base: u32,
    sub: &Operand,
    comps: &ComponentSelect,
) -> Option<String> {
    let byte = base * 16;
    let v = cb
        .variables
        .iter()
        .find(|v| byte >= v.offset && byte < v.offset + v.size)?;
    if v.var_type.elements == 0 {
        return None; // a dynamic index into a non-array isn't nameable
    }
    let elem_regs = elem_registers(v);
    if elem_regs != 1 {
        return None; // multi-register elements: index arithmetic not modelled
    }
    let var_reg = v.offset / 16;
    let dynamic = dxbc::shex::format_operand(sub);
    let index = if base == var_reg {
        dynamic
    } else {
        format!("{dynamic}+{}", base - var_reg)
    };
    Some(format!(
        "{}[{index}]{}",
        v.name,
        swizzle(&component_list(comps), 0)
    ))
}

fn render_var_access(v: &CBufferVariable<'_>, reg: u32, comps: &[u8]) -> String {
    let var_reg = v.offset / 16;
    if v.var_type.elements > 0 {
        // Array element. Index by the per-element register stride.
        let elem = (reg - var_reg) / elem_registers(v).max(1);
        return format!("{}[{}]{}", v.name, elem, swizzle(comps, 0));
    }
    if matches!(v.var_type.class, 2 | 3) {
        // Matrix: each register is a row (row-major) / column (column-major).
        return format!("{}[{}]{}", v.name, reg - var_reg, swizzle(comps, 0));
    }
    // Scalar / vector that may begin mid-register (packoffset cN.y); shift the
    // operand swizzle into the variable's local component space.
    let shift = ((v.offset % 16) / 4) as u8;
    let is_scalar = v.var_type.class == 0 && v.var_type.columns <= 1;
    if is_scalar {
        v.name.to_string()
    } else {
        format!("{}{}", v.name, swizzle(comps, shift))
    }
}

/// Per-element register count of an array variable (HLSL packs each element to a
/// 16-byte boundary).
fn elem_registers(v: &CBufferVariable<'_>) -> u32 {
    let elements = u32::from(v.var_type.elements.max(1));
    let per_elem = v.size.div_ceil(elements);
    per_elem.div_ceil(16).max(1)
}

/// Selected source components as indices (0=x … 3=w).
fn component_list(c: &ComponentSelect) -> Vec<u8> {
    match c {
        ComponentSelect::Scalar(c) => alloc::vec![*c & 3],
        ComponentSelect::Swizzle(s) => s.iter().map(|c| c & 3).collect(),
        ComponentSelect::Mask(m) => (0u8..4).filter(|i| m & (1 << i) != 0).collect(),
        ComponentSelect::OneComponent | ComponentSelect::ZeroComponent => alloc::vec![0],
    }
}

/// Render component indices as `.xyzw`, shifted into a variable's local space.
fn swizzle(comps: &[u8], shift: u8) -> String {
    let mut s = String::from(".");
    for &c in comps {
        let local = c.saturating_sub(shift);
        s.push(AXES[(local & 3) as usize]);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::borrow::Cow;
    use alloc::boxed::Box;
    use dxbc::chunks::rdef::TypeDesc;
    use dxbc::chunks::rdef::{CBufferDef, ResourceBinding};
    use smallvec::smallvec;

    fn td(class: u16, columns: u16, elements: u16) -> TypeDesc<'static> {
        TypeDesc {
            class,
            var_type: 3, // float
            rows: if class >= 2 { columns } else { 1 },
            columns,
            elements,
            members: Vec::new(),
            sm5_extra: None,
            name: Cow::Borrowed(""),
        }
    }
    fn var(
        name: &'static str,
        offset: u32,
        size: u32,
        t: TypeDesc<'static>,
    ) -> CBufferVariable<'static> {
        CBufferVariable {
            name: Cow::Borrowed(name),
            offset,
            size,
            flags: 0,
            var_type: t,
            default_value: Cow::Owned(Vec::new()),
            texture_start: None,
            texture_size: None,
            sampler_start: None,
            sampler_size: None,
        }
    }
    fn rdef() -> ResourceDef<'static> {
        ResourceDef {
            bindings: alloc::vec![ResourceBinding {
                name: Cow::Borrowed("cb"),
                input_type: 0,
                return_type: 0,
                dimension: 0,
                num_samples: 0,
                bind_point: 0,
                bind_count: 1,
                flags: 0,
                space: 0,
                id: 0,
            }],
            constant_buffers: alloc::vec![CBufferDef {
                name: Cow::Borrowed("cb"),
                variables: alloc::vec![
                    var("gColor", 0, 16, td(1, 4, 0)), // c0  float4
                    var("gMat", 16, 64, td(2, 4, 0)),  // c1  float4x4
                    var("gArr", 80, 64, td(1, 4, 4)),  // c5  float4[4]
                    var("gFlag", 144, 4, td(0, 1, 0)), // c9.x bool
                    var("gVec", 148, 12, td(1, 3, 0)), // c9.y float3
                ],
                size: 160,
                flags: 0,
                cb_type: 0,
            }],
            creator: Cow::Borrowed(""),
            target_version: 0xffff_0500,
            compile_flags: 0,
            rd11: None,
            original_data: None,
            original_canonical: None,
        }
    }
    fn cb(reg: OperandIndex, comps: ComponentSelect) -> Operand {
        Operand {
            reg_type: RegisterType::ConstantBuffer,
            components: comps,
            negate: false,
            abs: false,
            min_precision: Default::default(),
            non_uniform: false,
            indices: smallvec![OperandIndex::Imm32(0), reg],
            immediate_values: smallvec![],
        }
    }
    fn resolve(reg: OperandIndex, comps: ComponentSelect) -> Option<String> {
        resolve_operand(&cb(reg, comps), &rdef()).map(|(_, n)| n)
    }

    #[test]
    fn vector_matrix_array_shift_and_crossvar() {
        use ComponentSelect::*;
        // float4 + single component
        assert_eq!(
            resolve(OperandIndex::Imm32(0), Scalar(3)).as_deref(),
            Some("gColor.w")
        );
        // matrix row (c2 = row 1 of gMat @ c1)
        assert_eq!(
            resolve(OperandIndex::Imm32(2), Swizzle([0, 1, 2, 3])).as_deref(),
            Some("gMat[1].xyzw")
        );
        // array element (c6 = gArr[1])
        assert_eq!(
            resolve(OperandIndex::Imm32(6), Scalar(0)).as_deref(),
            Some("gArr[1].x")
        );
        // component shift: gVec @ c9.y, operand .w -> .z
        assert_eq!(
            resolve(OperandIndex::Imm32(9), Scalar(3)).as_deref(),
            Some("gVec.z")
        );
        // cross-variable: c9 .xyz spans gFlag(.x) + gVec(.yz) -> no single name
        assert_eq!(resolve(OperandIndex::Imm32(9), Swizzle([0, 1, 2, 0])), None);
    }

    #[test]
    fn dynamic_array_index() {
        let sub = Operand {
            reg_type: RegisterType::Temp,
            components: ComponentSelect::Scalar(0),
            negate: false,
            abs: false,
            min_precision: Default::default(),
            non_uniform: false,
            indices: smallvec![OperandIndex::Imm32(0)],
            immediate_values: smallvec![],
        };
        // cb0[r0.x + 5].x  ->  gArr[r0.x].x
        let n = resolve(
            OperandIndex::RelativePlusImm(5, Box::new(sub)),
            ComponentSelect::Scalar(0),
        );
        assert_eq!(n.as_deref(), Some("gArr[r0.x].x"));
    }
}
