//! Formatter: IR → human-readable disassembly text.

use std::fmt::Write;

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

fn format_instruction(instr: &Instruction) -> String {
    let name = instr.opcode.name();
    let sat = if instr.saturate { "_sat" } else { "" };

    match &instr.kind {
        InstructionKind::Generic { operands } => {
            if operands.is_empty() {
                format!("{name}{sat}")
            } else {
                let ops: Vec<String> = operands.iter().map(format_operand).collect();
                format!("{name}{sat} {}", ops.join(", "))
            }
        }
        InstructionKind::DclGlobalFlags { flags } => {
            format!("dcl_globalFlags {}", flags.join("|"))
        }
        InstructionKind::DclInput {
            interpolation,
            system_value,
            operands,
        } => {
            let ops: Vec<String> = operands.iter().map(format_operand).collect();
            let interp = interpolation.map(|i| format!(" {i}")).unwrap_or_default();
            let sv = system_value
                .filter(|s| *s != "undefined")
                .map(|s| format!(", {s}"))
                .unwrap_or_default();
            if ops.is_empty() {
                format!("{name}{interp}{sv}")
            } else {
                format!("{name}{interp}, {}{sv}", ops.join(", "))
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
        InstructionKind::HsPhase => name.to_string(),
        InstructionKind::CustomData {
            subtype,
            values,
            raw_dword_count,
        } => format_custom_data(subtype, values, *raw_dword_count),
    }
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
                "\n    {{ {:e}, {:e}, {:e}, {:e} }}",
                v[0], v[1], v[2], v[3]
            ));
        }
        s.push_str("\n}");
        s
    } else {
        format!("{ty} // {raw_dword_count} dwords")
    }
}

fn format_operand(op: &Operand) -> String {
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

    // Build register name with indices
    let mut name = String::from(prefix);
    match op.indices.len() {
        0 => {}
        1 => {
            name.push_str(&format_index(&op.indices[0]));
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

    // Append swizzle/mask
    let swizzle = format_components(&op.components);
    if !swizzle.is_empty() {
        name.push('.');
        name.push_str(&swizzle);
    }

    // Apply modifiers
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
        OperandIndex::RelativePlusImm(imm, sub) => format!("{imm} + {}", format_operand(sub)),
    }
}

fn format_components(comp: &ComponentSelect) -> String {
    match comp {
        ComponentSelect::None => String::new(),
        ComponentSelect::Mask(mask) => format_mask(*mask),
        ComponentSelect::Swizzle(s) => {
            let comps = ['x', 'y', 'z', 'w'];
            let full: String = s.iter().map(|&c| comps[c as usize]).collect();
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
