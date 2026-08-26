//! SM4/SM5 shader bytecode → HLSL decompiler.
//!
//! Built on cfglib's IR stack: the decoded instruction stream lifts into
//! a shader-dialect MLIL function (structured markers become blocks and
//! edges, non-register operand reads materialize as pure value
//! producers), cfglib's HLIL lift recovers structured statements with
//! single-use expressions inlined into their consumers, and the renderer
//! spells the result as compilable HLSL.

pub mod cfg;
mod dialect;
pub mod interp;
mod lift;
mod render;

pub use dialect::ShaderDialect;
pub use lift::{Declarations, LiftedShader, Unsupported, lift_program};
pub use render::{Renderer, SemanticNames, SemanticRow};

use dxbc::shex::Program;

/// Decompiles one decoded shader program to HLSL source.
///
/// # Errors
///
/// Returns [`Unsupported`] for shader shapes outside the modeled subset;
/// the message names the first unsupported construct.
pub fn decompile_program(program: &Program) -> Result<String, Unsupported> {
    decompile_with_semantics(program, SemanticNames::default())
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
    decompile_with_semantics(program, semantics)
}

fn decompile_with_semantics(
    program: &Program,
    semantics: SemanticNames,
) -> Result<String, Unsupported> {
    let lifted = lift_program(program)?;
    let hlil = cfglib::ir::hlil::lift_function(&lifted.function)
        .map_err(|error| Unsupported(format!("HLIL lift: {error}")))?;
    if !hlil.report.is_fully_structured() {
        return Err(Unsupported(format!(
            "control flow did not structure: {:?}",
            hlil.report
        )));
    }
    Renderer::new(&hlil, &lifted.declarations, program)
        .with_semantics(semantics)
        .render()
}

/// Decompiles every shader found in a byte stream, pairing each with its
/// result.
#[must_use]
pub fn decompile(data: &[u8]) -> Vec<Result<String, Unsupported>> {
    d3dasm::parse(data).iter().map(decompile_shader).collect()
}
