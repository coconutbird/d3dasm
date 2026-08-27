//! Declaration collection for the rendering header.

use dxbc::shex::{
    ComponentSelect, Instruction, InstructionKind, Operand, OperandIndex, RegisterType,
};

use crate::dialect::{RegFile, Register};

use super::{Declarations, GeometryLayout, InputRow, operand_register};

/// Records declaration instructions for the rendering header.
#[expect(clippy::too_many_lines, reason = "one arm per declaration kind")]
pub(super) fn collect_declaration(
    instruction: &Instruction,
    declarations: &mut Declarations,
    collect_outputs: bool,
) {
    match &instruction.kind {
        InstructionKind::DclSampler { mode, operands, .. } => {
            if let Some(operand) = operands.first()
                && let Some(OperandIndex::Imm32(index)) = operand.indices.first()
            {
                declarations.samplers.push((*index, mode));
            }
        }
        InstructionKind::DclResource {
            dimension,
            return_type,
            operands,
            ..
        } => {
            if let Some(operand) = operands.first()
                && let Some(OperandIndex::Imm32(index)) = operand.indices.first()
            {
                declarations
                    .resources
                    .push((*index, dimension, *return_type));
            }
        }
        InstructionKind::DclConstantBuffer { operands, size, .. } => {
            if let Some(operand) = operands.first()
                && let Some(OperandIndex::Imm32(register)) = operand.indices.first()
            {
                declarations.constant_buffers.push((*register, *size));
            }
        }
        InstructionKind::DclInput {
            interpolation,
            system_value,
            operands,
        } => {
            if let Some(operand) = operands.first() {
                // A geometry-shader input declaration carries the vertex
                // count first: v[count][register].
                let register =
                    if operand.reg_type == RegisterType::Input && operand.indices.len() == 2 {
                        match operand.indices.get(1) {
                            Some(OperandIndex::Imm32(index)) => Some(Register {
                                file: RegFile::Input,
                                index: *index,
                            }),
                            _ => None,
                        }
                    } else {
                        operand_register(operand).ok()
                    };
                if let Some(register) = register
                    && !declarations
                        .inputs
                        .iter()
                        .any(|row| row.register == register)
                {
                    declarations.inputs.push(InputRow {
                        register,
                        components: operand.components.clone(),
                        system_value: system_value.and_then(dxbc::shex::SystemValue::name),
                        interpolation: *interpolation,
                    });
                }
            }
        }
        InstructionKind::DclOutput {
            system_value,
            operands,
        } => {
            if collect_outputs {
                collect_io(&mut declarations.outputs, operands, *system_value);
            }
        }
        InstructionKind::DclResourceStructured {
            stride, operands, ..
        } => {
            if let Some(operand) = operands.first()
                && let Some(OperandIndex::Imm32(index)) = operand.indices.first()
            {
                declarations.structured.push((*index, *stride));
            }
        }
        InstructionKind::DclThreadGroup { x, y, z } => {
            declarations.thread_group = Some((*x, *y, *z));
        }
        InstructionKind::DclTessDomain { domain } => {
            declarations.tessellation.get_or_insert_default().domain = domain;
        }
        InstructionKind::DclTessPartitioning { partitioning } => {
            declarations
                .tessellation
                .get_or_insert_default()
                .partitioning = Some(partitioning);
        }
        InstructionKind::DclTessOutputPrimitive { primitive } => {
            declarations
                .tessellation
                .get_or_insert_default()
                .output_primitive = Some(primitive);
        }
        InstructionKind::DclInputControlPointCount { count } => {
            declarations
                .tessellation
                .get_or_insert_default()
                .input_control_points = *count;
        }
        InstructionKind::DclOutputControlPointCount { count } => {
            declarations
                .tessellation
                .get_or_insert_default()
                .output_control_points = *count;
        }
        InstructionKind::DclHsMaxTessFactor { value } => {
            declarations.tessellation.get_or_insert_default().max_factor = Some(*value);
        }
        InstructionKind::DclGsInputPrimitive { primitive } => {
            let layout = match primitive {
                dxbc::shex::GsPrimitive::Point => Some(("point", 1)),
                dxbc::shex::GsPrimitive::Line => Some(("line", 2)),
                dxbc::shex::GsPrimitive::Triangle => Some(("triangle", 3)),
                dxbc::shex::GsPrimitive::LineAdj => Some(("lineadj", 4)),
                dxbc::shex::GsPrimitive::TriangleAdj => Some(("triangleadj", 6)),
                _ => None,
            };
            if let Some((keyword, vertices)) = layout {
                let geometry = declarations.geometry.get_or_insert(GeometryLayout {
                    primitive: keyword,
                    vertices,
                    max_vertices: 0,
                });
                geometry.primitive = keyword;
                geometry.vertices = vertices;
            }
        }
        InstructionKind::DclMaxOutputVertexCount { count } => {
            let geometry = declarations.geometry.get_or_insert(GeometryLayout {
                primitive: "point",
                vertices: 1,
                max_vertices: 0,
            });
            geometry.max_vertices = *count;
        }
        InstructionKind::DclUavTyped {
            dimension,
            return_type,
            operands,
            ..
        } => {
            if let Some(operand) = operands.first()
                && let Some(OperandIndex::Imm32(index)) = operand.indices.first()
            {
                declarations.uavs.push((*index, dimension, *return_type));
            }
        }
        InstructionKind::CustomData {
            subtype: dxbc::shex::CustomDataType::ImmediateConstantBuffer,
            values,
            ..
        } => {
            declarations.immediate_constants.clone_from(values);
        }
        InstructionKind::DclIndexableTemp {
            reg,
            size,
            components,
        } => {
            declarations
                .indexable_temps
                .push((*reg, *size, *components));
        }
        _ => {}
    }
}

/// Records one input or output declaration row, deduplicating split
/// component declarations of the same register.
fn collect_io(
    rows: &mut Vec<(Register, ComponentSelect, Option<&'static str>)>,
    operands: &[Operand],
    system_value: Option<dxbc::shex::SystemValue>,
) {
    if let Some(operand) = operands.first()
        && let Ok(register) = operand_register(operand)
        && !rows.iter().any(|(declared, ..)| *declared == register)
    {
        rows.push((
            register,
            operand.components.clone(),
            system_value.and_then(dxbc::shex::SystemValue::name),
        ));
    }
}
