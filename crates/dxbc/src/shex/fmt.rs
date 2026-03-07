//! Formatter: IR → human-readable disassembly text.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt::Write;

use super::ir::*;
use super::opcodes::Opcode;

/// Format a decoded [`Program`] into disassembly text.
pub fn format_program(program: &Program) -> String {
    let mut out = String::new();
    let _ = writeln!(
        out,
        "{}_{}_{}",
        program.shader_type, program.major_version, program.minor_version
    );

    let mut indent = 0u32;

    for w in &program.warnings {
        let _ = writeln!(out, "// WARNING: {w}");
    }

    for instr in &program.instructions {
        // Pre-indent decrease for closing control flow
        match instr.opcode {
            Opcode::Else | Opcode::EndIf | Opcode::EndLoop | Opcode::EndSwitch => {
                indent = indent.saturating_sub(1);
            }
            _ => {}
        }

        let line = format_instruction(instr);
        let pad = "  ".repeat(indent as usize);
        let _ = writeln!(out, "{pad}{line}");

        // Post-indent increase for opening control flow
        match instr.opcode {
            Opcode::If | Opcode::Else | Opcode::Loop | Opcode::Switch => {
                indent += 1;
            }
            _ => {}
        }
    }

    out
}

/// Format a single [`Instruction`] into its disassembly text.
///
/// Returns the full line, e.g. `"mad_sat r0.x, r1.y, r2.z, r3.w"` or
/// `"dcl_constantbuffer cb4[1].xyzw, immediateIndexed"`.
pub fn format_instruction(instr: &Instruction) -> String {
    match &instr.kind {
        InstructionKind::Generic { operands } => format_generic(instr, operands),
        InstructionKind::HsPhase => format_mnemonic(instr),
        InstructionKind::CustomData {
            subtype,
            values,
            raw_dword_count,
        } => format_custom_data(subtype, values, *raw_dword_count),
        _ => format_declaration(instr),
    }
}

/// Format the bare opcode name.
///
/// Examples: `"mov"`, `"mad"`, `"sample"`, `"dcl_constantbuffer"`.
pub fn format_opcode(opcode: &Opcode) -> &'static str {
    opcode.name()
}

/// Format the full instruction mnemonic, including any modifier suffixes
/// (saturate, resinfo return type, texture offsets).
///
/// Examples: `"mad_sat"`, `"resinfo_uint"`, `"sample(1, 2, 0)"`, `"mov"`.
pub fn format_mnemonic(instr: &Instruction) -> String {
    let name = instr.opcode.name();
    let sat = if instr.saturate { "_sat" } else { "" };

    let suffix = match instr.resinfo_return_type {
        Some(1) => "_rcpFloat",
        Some(2) => "_uint",
        _ => "",
    };

    let offsets = match instr.tex_offsets {
        Some([u, v, w]) if u != 0 || v != 0 || w != 0 => {
            format!("({u}, {v}, {w})")
        }
        _ => String::new(),
    };

    format!("{name}{suffix}{sat}{offsets}")
}

/// Format a single [`Operand`] into its disassembly text.
///
/// Examples: `"r0.xy"`, `"cb5[r0.x].yyyz"`, `"l(1.000000, 0.000000)"`,
/// `"-|r1.x|"`.
pub fn format_operand(op: &Operand) -> String {
    format_operand_core(op, None)
}

/// Format a generic ALU / flow-control / sample instruction.
fn format_generic(instr: &Instruction, operands: &[Operand]) -> String {
    let opcode = format_mnemonic(instr);

    if operands.is_empty() {
        return opcode;
    }

    // Determine active component count from the destination (first operand) mask.
    let dest_width = dest_component_count(operands.first());
    let src_width = source_width_override(instr.opcode, dest_width);
    let mut ops = Vec::with_capacity(operands.len());
    for (i, op) in operands.iter().enumerate() {
        if i == 0 {
            ops.push(format_operand(op));
        } else {
            ops.push(format_operand_with_width(op, src_width));
        }
    }

    format!("{opcode} {}", ops.join(", "))
}

/// Format a declaration instruction (`dcl_*` variants).
fn format_declaration(instr: &Instruction) -> String {
    let name = instr.opcode.name();
    match &instr.kind {
        InstructionKind::DclGlobalFlags { flags } => {
            format!("dcl_globalFlags {}", flags.join("|"))
        }
        InstructionKind::DclInput {
            interpolation,
            system_value,
            operands,
        } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            let sv = system_value
                .filter(|s| *s != "undefined")
                .map(|s| format!(", {s}"))
                .unwrap_or_default();
            if ops.is_empty() {
                if let Some(interp) = interpolation {
                    format!("{name} {interp}{sv}")
                } else {
                    format!("{name}{sv}")
                }
            } else if let Some(interp) = interpolation {
                format!("{name} {interp}, {}{sv}", ops.join(", "))
            } else {
                format!("{name} {}{sv}", ops.join(", "))
            }
        }
        InstructionKind::DclOutput {
            system_value,
            operands,
        } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            let sv = system_value
                .filter(|s| *s != "undefined")
                .map(|s| format!(", {s}"))
                .unwrap_or_default();
            if ops.is_empty() {
                format!("{name}{sv}")
            } else {
                format!("{name} {}{sv}", ops.join(", "))
            }
        }
        InstructionKind::DclResource {
            dimension,
            return_type,
            operands,
        } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            let rt: Vec<&str> = return_type.iter().map(|r| r.name()).collect();
            format!(
                "dcl_resource_{dimension} ({}) {}",
                rt.join(","),
                ops.join(", ")
            )
        }
        InstructionKind::DclSampler { mode, operands } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            format!("dcl_sampler {}, mode_{mode}", ops.join(", "))
        }
        InstructionKind::DclConstantBuffer { access, operands } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            format!("dcl_constantbuffer {}, {access}", ops.join(", "))
        }
        InstructionKind::DclTemps { count } => format!("dcl_temps {count}"),
        InstructionKind::DclIndexableTemp {
            reg,
            size,
            components,
        } => {
            format!("dcl_indexableTemp x{reg}[{size}], {components}")
        }
        InstructionKind::DclGsInputPrimitive { primitive } => {
            format!("dcl_inputPrimitive {primitive}")
        }
        InstructionKind::DclGsOutputTopology { topology } => {
            format!("dcl_outputTopology {topology}")
        }
        InstructionKind::DclMaxOutputVertexCount { count } => {
            format!("dcl_maxOutputVertexCount {count}")
        }
        InstructionKind::DclGsInstanceCount { count } => {
            format!("dcl_gsInstanceCount {count}")
        }
        InstructionKind::DclOutputControlPointCount { count } => {
            format!("dcl_outputControlPointCount {count}")
        }
        InstructionKind::DclInputControlPointCount { count } => {
            format!("dcl_inputControlPointCount {count}")
        }
        InstructionKind::DclTessDomain { domain } => format!("dcl_tessDomain {domain}"),
        InstructionKind::DclTessPartitioning { partitioning } => {
            format!("dcl_tessPartitioning {partitioning}")
        }
        InstructionKind::DclTessOutputPrimitive { primitive } => {
            format!("dcl_tessOutputPrimitive {primitive}")
        }
        InstructionKind::DclHsMaxTessFactor { value } => {
            format!("dcl_hsMaxTessFactor {value}")
        }
        InstructionKind::DclHsForkPhaseInstanceCount { count } => {
            format!("dcl_hsForkPhaseInstanceCount {count}")
        }
        InstructionKind::DclThreadGroup { x, y, z } => {
            format!("dcl_thread_group {x}, {y}, {z}")
        }
        InstructionKind::DclUavTyped {
            dimension,
            return_type,
            operands,
        } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            let rt: Vec<&str> = return_type.iter().map(|r| r.name()).collect();
            format!(
                "dcl_uav_typed_{dimension} ({}) {}",
                rt.join(","),
                ops.join(", ")
            )
        }
        InstructionKind::DclUavRaw { flags, operands } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            let flag_str = format_uav_flags(*flags);
            format!("dcl_uav_raw{flag_str} {}", ops.join(", "))
        }
        InstructionKind::DclUavStructured {
            flags,
            stride,
            operands,
        } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            let flag_str = format_uav_flags(*flags);
            format!("dcl_uav_structured{flag_str} {}, {stride}", ops.join(", "))
        }
        InstructionKind::DclResourceRaw { operands } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            format!("dcl_resource_raw {}", ops.join(", "))
        }
        InstructionKind::DclResourceStructured { stride, operands } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            format!("dcl_resource_structured {}, {stride}", ops.join(", "))
        }

        InstructionKind::DclIndexRange { operands, count } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            format!("dcl_indexRange {}, {count}", ops.join(", "))
        }
        // Generic, HsPhase, and CustomData are handled by format_instruction
        // before calling this function.
        _ => name.to_string(),
    }
}

fn format_uav_flags(flags: u32) -> String {
    // Bit 0 = globally coherent, bit 1 = rasterizer ordered (ROV)
    let mut parts = Vec::new();
    if flags & 0x1 != 0 {
        parts.push("_glc");
    }
    if flags & 0x2 != 0 {
        parts.push("_opc");
    }
    parts.join("")
}

fn format_custom_data(
    subtype: &CustomDataType,
    values: &[[f32; 4]],
    raw_dword_count: usize,
) -> String {
    let ty = match subtype {
        CustomDataType::Comment => "comment",
        CustomDataType::DebugInfo => "debuginfo",
        CustomDataType::Opaque => "opaque",
        CustomDataType::ImmediateConstantBuffer => "dcl_immediateConstantBuffer",
        CustomDataType::Other(v) => {
            return format!("customdata // subtype={v}, {raw_dword_count} dwords");
        }
    };
    if *subtype == CustomDataType::ImmediateConstantBuffer && !values.is_empty() {
        let mut s = format!("{ty} {{");
        for v in values {
            s.push_str(&format!(
                "\n  {{ {}, {}, {}, {} }}",
                format_immediate(v[0].to_bits()),
                format_immediate(v[1].to_bits()),
                format_immediate(v[2].to_bits()),
                format_immediate(v[3].to_bits()),
            ));
        }
        s.push_str("\n}");
        s
    } else {
        format!("{ty} // {raw_dword_count} dwords")
    }
}

/// Determine the number of swizzle components needed for a destination mask.
///
/// This is the position of the highest set bit + 1, NOT the popcount.
/// For `.xy` (bits 0,1) → 2, for `.zw` (bits 2,3) → 4, for `.xw` (bits 0,3) → 4.
/// Source swizzles can be safely truncated to this length since higher
/// components are never read.
fn dest_component_count(op: Option<&Operand>) -> Option<u8> {
    let op = op?;
    match &op.components {
        ComponentSelect::Mask(mask) => {
            if *mask == 0 {
                None
            } else {
                // Highest set bit position + 1
                Some((8 - mask.leading_zeros()) as u8)
            }
        }
        _ => None,
    }
}

/// Override source width for instructions that read more components than
/// the destination mask implies (e.g., dp3 always reads 3 source components,
/// texture sampling always reads 4-component swizzles).
fn source_width_override(opcode: Opcode, dest_width: Option<u8>) -> Option<u8> {
    match opcode {
        Opcode::Dp2 => Some(2),
        Opcode::Dp3 => Some(3),
        Opcode::Dp4 => Some(4),
        // Texture/resource instructions: source swizzles are independent of
        // the destination mask — don't truncate them.
        Opcode::Sample
        | Opcode::SampleB
        | Opcode::SampleC
        | Opcode::SampleCLz
        | Opcode::SampleD
        | Opcode::SampleL
        | Opcode::Ld
        | Opcode::LdMs
        | Opcode::LdUavTyped
        | Opcode::Gather4
        | Opcode::Gather4C
        | Opcode::Gather4Po
        | Opcode::Gather4PoC
        | Opcode::Resinfo
        | Opcode::Lod
        | Opcode::SamplePos
        | Opcode::SampleInfo
        | Opcode::BufInfo
        // Store/atomic instructions: operand widths are dictated by the
        // resource, not the destination mask.
        | Opcode::StoreUavTyped
        | Opcode::StoreRaw
        | Opcode::StoreStructured
        | Opcode::AtomicAnd
        | Opcode::AtomicOr
        | Opcode::AtomicXor
        | Opcode::AtomicCmpStore
        | Opcode::AtomicIAdd
        | Opcode::AtomicIMax
        | Opcode::AtomicIMin
        | Opcode::AtomicUMax
        | Opcode::AtomicUMin
        | Opcode::ImmAtomicAlloc
        | Opcode::ImmAtomicConsume
        | Opcode::ImmAtomicIAdd
        | Opcode::ImmAtomicAnd
        | Opcode::ImmAtomicOr
        | Opcode::ImmAtomicXor
        | Opcode::ImmAtomicExch
        | Opcode::ImmAtomicCmpExch
        | Opcode::ImmAtomicIMax
        | Opcode::ImmAtomicIMin
        | Opcode::ImmAtomicUMax
        | Opcode::ImmAtomicUMin
        | Opcode::LdRaw
        | Opcode::LdStructured => None,
        _ => dest_width,
    }
}

/// Format a source operand, truncating its swizzle to `width` components.
///
/// Truncation is applied at the component level (before modifiers like abs/negate)
/// to avoid breaking closing delimiters like `|`.
fn format_operand_with_width(op: &Operand, width: Option<u8>) -> String {
    format_operand_core(op, width.map(|w| w as usize))
}

fn format_operand_core(op: &Operand, swizzle_width: Option<usize>) -> String {
    let prefix = op.reg_type.prefix();

    // Immediates
    if op.reg_type == RegisterType::Immediate32 {
        let vals: Vec<String> = op
            .immediate_values
            .iter()
            .map(|&v| format_immediate(v))
            .collect();
        return format!("l({})", vals.join(", "));
    }
    if op.reg_type == RegisterType::Immediate64 {
        // Each 64-bit value is stored as two consecutive u32s (lo, hi)
        let mut vals = Vec::new();
        let mut i = 0;
        while i + 1 < op.immediate_values.len() {
            let lo = op.immediate_values[i] as u64;
            let hi = op.immediate_values[i + 1] as u64;
            let bits = lo | (hi << 32);
            let f = f64::from_bits(bits);
            vals.push(format!("{f:?}"));
            i += 2;
        }
        return format!("d({})", vals.join(", "));
    }

    // Build register name with indices
    let mut name = String::from(prefix);
    match op.indices.len() {
        0 => {}
        1 => {
            // 1D: simple registers like r0, v1 use bare number; others use brackets
            let idx_str = format_index(&op.indices[0]);
            if matches!(
                &op.indices[0],
                OperandIndex::Relative(_) | OperandIndex::RelativePlusImm(_, _)
            ) {
                name.push_str(&format!("[{idx_str}]"));
            } else {
                name.push_str(&idx_str);
            }
        }
        2 => {
            name.push_str(&format_index(&op.indices[0]));
            name.push_str(&format!("[{}]", format_index(&op.indices[1])));
        }
        _ => {
            name.push_str(&format_index(&op.indices[0]));
            for idx in &op.indices[1..] {
                name.push_str(&format!("[{}]", format_index(idx)));
            }
        }
    }

    // Append swizzle/mask — truncate swizzle to width if specified
    let swizzle = format_components_with_width(&op.components, swizzle_width);
    if !swizzle.is_empty() {
        name.push('.');
        name.push_str(&swizzle);
    }

    // Apply modifiers (AFTER building name, so delimiters like | are not affected by truncation)
    if op.negate && op.abs {
        format!("-|{name}|")
    } else if op.negate {
        format!("-{name}")
    } else if op.abs {
        format!("|{name}|")
    } else {
        name
    }
}

fn format_index(idx: &OperandIndex) -> String {
    match idx {
        OperandIndex::Imm32(v) => format!("{v}"),
        OperandIndex::Imm64(v) => format!("{v}"),
        OperandIndex::Relative(sub) => format_operand(sub),
        OperandIndex::RelativePlusImm(imm, sub) => format!("{} + {imm}", format_operand(sub)),
    }
}

fn format_components_with_width(comp: &ComponentSelect, width: Option<usize>) -> String {
    match comp {
        ComponentSelect::None => String::new(),
        ComponentSelect::Mask(mask) => format_mask(*mask),
        ComponentSelect::Swizzle(s) => {
            let comps = ['x', 'y', 'z', 'w'];
            let count = width.unwrap_or(4).min(4);
            let full: String = s[..count].iter().map(|&c| comps[c as usize]).collect();
            trim_swizzle(&full)
        }
        ComponentSelect::Scalar(c) => ['x', 'y', 'z', 'w'][*c as usize].to_string(),
    }
}

fn format_mask(mask: u8) -> String {
    let mut s = String::with_capacity(4);
    if mask & 1 != 0 {
        s.push('x');
    }
    if mask & 2 != 0 {
        s.push('y');
    }
    if mask & 4 != 0 {
        s.push('z');
    }
    if mask & 8 != 0 {
        s.push('w');
    }
    s
}

fn trim_swizzle(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= 1 {
        return s.to_string();
    }
    if chars.iter().all(|&c| c == chars[0]) {
        return chars[0].to_string();
    }
    let mut end = chars.len();
    while end > 1 && chars[end - 1] == chars[end - 2] {
        end -= 1;
    }
    chars[..end].iter().collect()
}

fn format_immediate(val: u32) -> String {
    let f = f32::from_bits(val);
    if f == 0.0 || f == 1.0 || f == -1.0 || f == 0.5 || f == -0.5 || f == 2.0 {
        format!("{f:.6}")
    } else if val == 0 {
        "0".to_string()
    } else if f.is_finite() && f.abs() > 0.0001 && f.abs() < 1_000_000.0 {
        format!("{f:.6}")
    } else {
        format!("0x{val:08X}")
    }
}

// ---------------------------------------------------------------------------
// Display impls — delegate to the public format_* functions
// ---------------------------------------------------------------------------

impl core::fmt::Display for Instruction {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&format_instruction(self))
    }
}

impl core::fmt::Display for Operand {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&format_operand(self))
    }
}

impl core::fmt::Display for Opcode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(format_opcode(self))
    }
}

impl core::fmt::Display for Program {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&format_program(self))
    }
}
