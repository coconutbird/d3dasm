//! SM4/SM5 shader bytecode → HLSL decompiler.
//!
//! Built on cfglib's IR stack: the decoded instruction stream (the
//! language's own low-level form) lifts into shader-dialect RTL — typed
//! parallel transfers over raw registers — cfglib's web lifting recovers
//! typed MLIL variables from register lanes, the HLIL lift recovers
//! structured statements with single-use expressions inlined into their
//! consumers, and the renderer spells the result as compilable HLSL with
//! honest types and RDEF-derived names.

pub mod cfg;
mod dialect;
pub mod interp;
mod lift;
mod names;
mod render;

pub use dialect::ShaderDialect;
pub use lift::{Declarations, LiftedShader, Unsupported, lift_program};
pub use names::{CbField, CbNames, ResourceNames};
pub use render::{PatchRow, Renderer, SemanticNames, SemanticRow};

use dxbc::shex::Program;

/// Decompiles one decoded shader program to HLSL source.
///
/// # Errors
///
/// Returns [`Unsupported`] for shader shapes outside the modeled subset;
/// the message names the first unsupported construct.
pub fn decompile_program(program: &Program) -> Result<String, Unsupported> {
    decompile_with_semantics(program, &SemanticNames::default(), ResourceNames::default())
}

/// Decompiles one shader using its signature chunks for the interface
/// semantic names, so the recompiled signatures carry the original
/// contract.
///
/// # Errors
///
/// Returns [`Unsupported`] for shader shapes outside the modeled subset.
pub fn decompile_shader(shader: &d3dasm::Shader<'_>) -> Result<String, Unsupported> {
    let program = shader
        .program()
        .ok_or_else(|| Unsupported("shader has no program chunk".into()))?;
    let mut semantics = SemanticNames::default();
    let row = |element: &dxbc::chunks::SignatureElement<'_>| render::SemanticRow {
        text: format!("{}{}", element.semantic_name, element.semantic_index),
        // Component types 1 and 2 are D3D's uint32 and sint32.
        integral: matches!(element.component_type, 1 | 2),
        mask: element.mask,
    };
    if let Some(signature) = shader.input_signature() {
        for element in &signature.elements {
            semantics
                .inputs
                .entry(element.register)
                .or_insert_with(|| row(element));
        }
    }
    if let Some(signature) = shader.output_signature() {
        for element in &signature.elements {
            semantics
                .outputs
                .entry(element.register)
                .or_insert_with(|| row(element));
        }
    }
    if let Some(signature) = shader.patch_constant_signature() {
        for element in &signature.elements {
            semantics.patch.push(render::PatchRow {
                register: element.register,
                mask: element.mask,
                text: format!("{}{}", element.semantic_name, element.semantic_index),
                integral: matches!(element.component_type, 1 | 2),
            });
        }
    }
    let resources = shader
        .resource_def()
        .map_or_else(ResourceNames::default, ResourceNames::from_rdef);
    decompile_with_semantics(program, &semantics, resources)
}

fn decompile_with_semantics(
    program: &Program,
    semantics: &SemanticNames,
    resources: ResourceNames,
) -> Result<String, Unsupported> {
    let lifted = lift_program(program)?;
    let structured = |function: &cfglib::ir::rtl::Function<ShaderDialect>| {
        let lifting = cfglib::ir::rtl::lift(function, &())
            .map_err(|error| Unsupported(format!("web lift: {error}")))?;
        let webs = lifting.webs;
        let function = lifting
            .builder
            .finish()
            .map_err(|error| Unsupported(format!("verification: {error}")))?;
        let function = function
            .materialize_constants()
            .map_err(|error| Unsupported(format!("constant materialization: {error}")))?
            .function;
        let hlil = cfglib::ir::hlil::lift_function(&function)
            .map_err(|error| Unsupported(format!("HLIL lift: {error}")))?;
        let hlil = hlil
            .with_recovered_structure()
            .map_err(|error| Unsupported(format!("HLIL recovery: {error}")))?;
        if !hlil.report.is_fully_structured() {
            return Err(Unsupported(format!(
                "control flow did not structure: {:?}",
                hlil.report
            )));
        }
        Ok((hlil, webs))
    };
    let (hlil, webs) = structured(&lifted.function)?;
    let phases = lifted
        .hull
        .as_ref()
        .map(|hull| {
            hull.constants
                .iter()
                .map(|(instances, function)| {
                    structured(function).map(|(hlil, webs)| (*instances, hlil, webs))
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let render = |resources: ResourceNames| {
        let renderer = Renderer::new(&hlil, &lifted.declarations, program, &webs)
            .with_semantics(semantics.clone())
            .with_resource_names(resources);
        match &phases {
            Some(phases) => renderer.render_hull(phases),
            None => renderer.render(),
        }
    };
    // A constant-buffer access outside the RDEF-named fields falls back
    // to the anonymous register-array form.
    match render(resources.clone()) {
        Err(error) if error.0 == render::CB_NAMING_MISS => render(resources.without_cb_fields()),
        result => result,
    }
}

/// Decompiles every shader found in a byte stream, pairing each with its
/// result.
#[must_use]
pub fn decompile(data: &[u8]) -> Vec<Result<String, Unsupported>> {
    d3dasm::parse(data).iter().map(decompile_shader).collect()
}

#[cfg(test)]
mod tests {
    use dxbc::shex::{
        ComponentSelect, Immediates, Indices, Instruction, InstructionKind, MinPrecision, Opcode,
        Operand, OperandIndex, Program, RegisterType, SmallU32Vec,
    };

    use super::decompile_program;

    fn register(register_type: RegisterType, index: u32, components: ComponentSelect) -> Operand {
        Operand {
            reg_type: register_type,
            components,
            negate: false,
            abs: false,
            min_precision: MinPrecision::Default,
            non_uniform: false,
            indices: [OperandIndex::Imm32(index)]
                .into_iter()
                .collect::<Indices>(),
            immediate_values: Immediates::new(),
        }
    }

    fn immediate(values: impl IntoIterator<Item = u32>, components: ComponentSelect) -> Operand {
        Operand {
            reg_type: RegisterType::Immediate32,
            components,
            negate: false,
            abs: false,
            min_precision: MinPrecision::Default,
            non_uniform: false,
            indices: Indices::new(),
            immediate_values: values.into_iter().collect::<Immediates>(),
        }
    }

    fn instruction(opcode: Opcode, operands: impl IntoIterator<Item = Operand>) -> Instruction {
        Instruction {
            opcode,
            saturate: false,
            test_nonzero: false,
            precise_mask: 0,
            resinfo_return_type: None,
            sync_flags: 0,
            tex_offsets: None,
            resource_dim: None,
            resource_return_type: None,
            kind: InstructionKind::Generic {
                operands: operands.into_iter().collect(),
                trailing_tokens: SmallU32Vec::new(),
            },
        }
    }

    fn full_mask() -> ComponentSelect {
        ComponentSelect::Mask(0b1111)
    }

    fn full_swizzle() -> ComponentSelect {
        ComponentSelect::Swizzle([0, 1, 2, 3])
    }

    fn counting_loop_declarations() -> Vec<Instruction> {
        vec![
            Instruction {
                opcode: Opcode::DclInputPs,
                kind: InstructionKind::DclInput {
                    interpolation: Some("linear"),
                    system_value: None,
                    operands: [register(RegisterType::Input, 0, full_mask())]
                        .into_iter()
                        .collect(),
                },
                ..instruction(Opcode::DclInputPs, [])
            },
            Instruction {
                opcode: Opcode::DclOutput,
                kind: InstructionKind::DclOutput {
                    system_value: None,
                    operands: [register(RegisterType::Output, 0, full_mask())]
                        .into_iter()
                        .collect(),
                },
                ..instruction(Opcode::DclOutput, [])
            },
            Instruction {
                opcode: Opcode::DclTemps,
                kind: InstructionKind::DclTemps { count: 3 },
                ..instruction(Opcode::DclTemps, [])
            },
        ]
    }

    fn counting_loop_instructions() -> Vec<Instruction> {
        let scalar = ComponentSelect::Scalar;
        let break_on_zero =
            instruction(Opcode::Breakc, [register(RegisterType::Temp, 1, scalar(2))]);
        vec![
            instruction(
                Opcode::Mov,
                [
                    register(RegisterType::Temp, 0, full_mask()),
                    immediate([0, 0, 0, 0], full_swizzle()),
                ],
            ),
            instruction(
                Opcode::Mov,
                [
                    register(RegisterType::Temp, 1, ComponentSelect::Mask(0b0001)),
                    immediate([0], ComponentSelect::OneComponent),
                ],
            ),
            instruction(
                Opcode::Mov,
                [
                    register(RegisterType::Temp, 2, full_mask()),
                    register(RegisterType::Temp, 0, full_swizzle()),
                ],
            ),
            instruction(
                Opcode::Mov,
                [
                    register(RegisterType::Temp, 1, ComponentSelect::Mask(0b0010)),
                    register(RegisterType::Temp, 1, scalar(0)),
                ],
            ),
            instruction(Opcode::Loop, []),
            instruction(
                Opcode::ILt,
                [
                    register(RegisterType::Temp, 1, ComponentSelect::Mask(0b0100)),
                    register(RegisterType::Temp, 1, scalar(1)),
                    immediate([4], ComponentSelect::OneComponent),
                ],
            ),
            break_on_zero,
            instruction(
                Opcode::Add,
                [
                    register(RegisterType::Temp, 2, full_mask()),
                    register(RegisterType::Temp, 2, full_swizzle()),
                    register(RegisterType::Input, 0, full_swizzle()),
                ],
            ),
            instruction(
                Opcode::Iadd,
                [
                    register(RegisterType::Temp, 1, ComponentSelect::Mask(0b0010)),
                    register(RegisterType::Temp, 1, scalar(1)),
                    immediate([1], ComponentSelect::OneComponent),
                ],
            ),
            instruction(Opcode::EndLoop, []),
            instruction(
                Opcode::Mov,
                [
                    register(RegisterType::Output, 0, full_mask()),
                    register(RegisterType::Temp, 2, full_swizzle()),
                ],
            ),
            instruction(Opcode::Ret, []),
        ]
    }

    fn counting_loop_program() -> Program {
        let mut instructions = counting_loop_declarations();
        instructions.extend(counting_loop_instructions());
        Program {
            shader_type: "ps",
            major_version: 5,
            minor_version: 0,
            instructions,
            warnings: Vec::new(),
            fourcc: *b"SHDR",
        }
    }

    fn exact_integer_program() -> Program {
        let declarations = [
            Instruction {
                opcode: Opcode::DclOutput,
                kind: InstructionKind::DclOutput {
                    system_value: None,
                    operands: [register(RegisterType::Output, 0, full_mask())]
                        .into_iter()
                        .collect(),
                },
                ..instruction(Opcode::DclOutput, [])
            },
            Instruction {
                opcode: Opcode::DclTemps,
                kind: InstructionKind::DclTemps { count: 2 },
                ..instruction(Opcode::DclTemps, [])
            },
        ];
        let instructions = [
            instruction(
                Opcode::Mov,
                [
                    register(RegisterType::Temp, 0, full_mask()),
                    immediate([2, 2, 2, 2], full_swizzle()),
                ],
            ),
            instruction(
                Opcode::Iadd,
                [
                    register(RegisterType::Temp, 1, full_mask()),
                    register(RegisterType::Temp, 0, full_swizzle()),
                    immediate([3, 3, 3, 3], full_swizzle()),
                ],
            ),
            instruction(
                Opcode::Mov,
                [
                    register(RegisterType::Output, 0, full_mask()),
                    register(RegisterType::Temp, 1, full_swizzle()),
                ],
            ),
            instruction(Opcode::Ret, []),
        ];
        Program {
            shader_type: "ps",
            major_version: 5,
            minor_version: 0,
            instructions: declarations.into_iter().chain(instructions).collect(),
            warnings: Vec::new(),
            fourcc: *b"SHDR",
        }
    }

    #[test]
    fn decompilation_materializes_exact_integer_constants() {
        let source = decompile_program(&exact_integer_program()).unwrap();

        assert!(!source.contains(" + "), "{source}");
        assert!(source.contains("asfloat(0x00000005u)"), "{source}");
        assert!(
            !source
                .lines()
                .any(|line| line.trim_start().starts_with("precise ")),
            "{source}"
        );
    }

    #[test]
    fn decompilation_recovers_a_legible_counting_loop() {
        let program = counting_loop_program();
        let source = decompile_program(&program).unwrap();
        let loop_header = source
            .lines()
            .find(|line| line.trim_start().starts_with("for ("))
            .expect("the counted loop should recover as a for statement");
        let induction_name = loop_header
            .trim_start()
            .strip_prefix("for (")
            .and_then(|clause| clause.split_once(" = "))
            .map(|(name, _)| name)
            .expect("the counted loop should begin with its initializer");

        assert!(loop_header.contains(" < 4;"), "{source}");
        assert!(loop_header.contains("r1 += 1)"), "{source}");
        assert!(!source.contains("while ("), "{source}");
        assert!(
            loop_header.trim_start().starts_with("for (r1 = 0;"),
            "{source}"
        );
        assert!(source.contains("r2 += v0;"), "{source}");
        assert!(!source.contains("v0.xyzw"), "{source}");
        assert!(
            source.contains(&format!("precise int {induction_name};")),
            "{source}"
        );
        let induction_declaration = source
            .find(&format!("precise int {induction_name};"))
            .expect("the induction variable should be declared");
        let loop_offset = source
            .find(loop_header)
            .expect("the recovered loop header belongs to the source");
        let accumulator_initialization = source
            .find("r2 = float4(0.0, 0.0, 0.0, 0.0);")
            .expect("the accumulator should be initialized");
        assert!(accumulator_initialization < induction_declaration);
        assert!(induction_declaration < loop_offset);
        assert!(source.contains("r2 = float4(0.0, 0.0, 0.0, 0.0);"));
    }
}
