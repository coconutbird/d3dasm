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
        let lifting = cfglib::ir::rtl::lift(function)
            .map_err(|error| Unsupported(format!("web lift: {error}")))?;
        let webs = lifting.webs;
        let function = lifting
            .builder
            .finish()
            .map_err(|error| Unsupported(format!("verification: {error}")))?;
        let hlil = cfglib::ir::hlil::lift_function(&function)
            .map_err(|error| Unsupported(format!("HLIL lift: {error}")))?;
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
