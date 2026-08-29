//! Typed-web HLSL rendering of structured shader functions.
//!
//! Every MLIL variable is a lifted *web* with an inferred scalar type, so
//! locals declare as `float3`/`int`/`uint4` and reinterpretation casts
//! appear only where a web genuinely mixes representations (a `Bits` web)
//! or an operation imposes a different interpretation than the web's.
//! Expression trees ([`VarExpr`]) carry storage reads in deterministic
//! order, positionally aligned with the HLIL operand list, so inlined
//! producers substitute into their exact read sites.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use cfglib::ir::hlil::{
    Function as HlilFunction, LiftedFunction, VariableId as HlilVariableId, VariablePlacements,
};
use cfglib::ir::mlil::VariableId as MlilVariableId;
use cfglib::ir::rtl::{LiftedStatement, ScalarType, ValueShape, VarExpr, WebInfo, referenced_webs};
use dxbc::shex::{Opcode, Program};

use crate::dialect::{AluHead, RegFile, ShaderDialect, SystemInput};
use crate::lift::{Declarations, Unsupported};
use crate::names::ResourceNames;

/// The sentinel error for a constant-buffer access that resolves outside
/// the RDEF-named fields; the caller retries without field naming.
pub(crate) const CB_NAMING_MISS: &str = "cb naming: access outside named fields";

type Hlil = HlilFunction<ShaderDialect>;
type Expr = VarExpr<ShaderDialect>;
type Op = LiftedStatement<ShaderDialect>;
type Web = WebInfo<ShaderDialect>;
type Webs = cfglib::ir::rtl::Webs<ShaderDialect>;
type Reads<'a> = cfglib::ir::rtl::ReadResolver<'a, ShaderDialect, ShaderDialect>;

mod interface;
mod statement;
mod value;

const LANES: [char; 4] = ['x', 'y', 'z', 'w'];

fn suffix(lanes: &[u8]) -> String {
    lanes.iter().map(|&lane| LANES[usize::from(lane)]).collect()
}

/// One signature-derived interface row.
#[derive(Debug, Clone)]
pub struct SemanticRow {
    /// Semantic string (`name` + index suffix).
    pub text: String,
    /// Whether the signature types the row as an integer register —
    /// system values like `SV_VertexID` demand scalar `uint` parameters.
    pub integral: bool,
    /// The row's component mask.
    pub mask: u8,
}

/// One patch-constant signature row.
#[derive(Debug, Clone)]
pub struct PatchRow {
    /// Patch-constant register.
    pub register: u32,
    /// Component mask of the row.
    pub mask: u8,
    /// Semantic string (`name` + index suffix).
    pub text: String,
    /// Whether the row is integer-typed.
    pub integral: bool,
}

/// Signature-derived semantic rows by register, per direction.
#[derive(Debug, Clone, Default)]
pub struct SemanticNames {
    /// Input register → interface row.
    pub inputs: BTreeMap<u32, SemanticRow>,
    /// Output register → interface row.
    pub outputs: BTreeMap<u32, SemanticRow>,
    /// Patch-constant rows (tessellation stages).
    pub patch: Vec<PatchRow>,
}

/// One rendered value: its text, produced interpretation, and whether
/// the text binds tighter than any operator (no wrapping needed).
struct Rendered {
    text: String,
    scalar: ScalarType,
    atomic: bool,
}

/// Whether a comparison is rendered directly or as its exact logical
/// complement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComparisonPolarity {
    Direct,
    Negated,
}

impl Rendered {
    fn atom(text: String, scalar: ScalarType) -> Self {
        Self {
            text,
            scalar,
            atomic: true,
        }
    }

    fn composite(text: String, scalar: ScalarType) -> Self {
        Self {
            text,
            scalar,
            atomic: false,
        }
    }

    /// The text, parenthesized when composite.
    fn wrapped(&self) -> String {
        if self.atomic {
            self.text.clone()
        } else {
            format!("({})", self.text)
        }
    }

    /// The text adapted to `wanted` bits, parenthesized when nested.
    fn cast(&self, wanted: ScalarType) -> String {
        if normalize(self.scalar) == normalize(wanted) {
            self.wrapped()
        } else {
            format!("{}({})", cast_name(wanted), self.text)
        }
    }

    /// The text adapted to `wanted` as a full right-hand side.
    fn cast_bare(&self, wanted: ScalarType) -> String {
        if normalize(self.scalar) == normalize(wanted) {
            self.text.clone()
        } else {
            format!("{}({})", cast_name(wanted), self.text)
        }
    }
}

/// `Bits` webs store as raw `uint`, and booleans never persist.
fn normalize(scalar: ScalarType) -> ScalarType {
    match scalar {
        ScalarType::Bits => ScalarType::U32,
        other => other,
    }
}

fn cast_name(wanted: ScalarType) -> &'static str {
    match normalize(wanted) {
        ScalarType::F32 | ScalarType::F64 => "asfloat",
        ScalarType::I32 | ScalarType::I64 => "asint",
        _ => "asuint",
    }
}

fn scalar_keyword(scalar: ScalarType) -> &'static str {
    match normalize(scalar) {
        ScalarType::F32 | ScalarType::F64 => "float",
        ScalarType::I32 | ScalarType::I64 => "int",
        ScalarType::Bool => "bool",
        _ => "uint",
    }
}

/// The HLSL type of one shape.
fn type_name(shape: ValueShape) -> String {
    let keyword = scalar_keyword(shape.scalar);
    if shape.lanes <= 1 {
        keyword.to_owned()
    } else {
        format!("{keyword}{}", shape.lanes)
    }
}

/// Which hull-shader body the renderer is emitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HullMode {
    /// Not a hull shader.
    None,
    /// The control-point entry (`main`).
    ControlPoint,
    /// The patch-constant function.
    Constants,
}

/// Renders one decompiled shader body and its declarations.
pub struct Renderer<'a> {
    function: &'a Hlil,
    declarations: &'a Declarations,
    program: &'a Program,
    webs: &'a Webs,
    semantics: SemanticNames,
    resources: ResourceNames,
    referenced: BTreeSet<MlilVariableId>,
    placements: VariablePlacements,
    web_names: Vec<Option<String>>,
    name_prefix: String,
    hull_mode: HullMode,
    fork_instance: u32,
    input_preambles: Vec<String>,
    out: String,
    indent: usize,
}

impl<'a> Renderer<'a> {
    /// Creates a renderer over one lifted shader.
    #[must_use]
    pub fn new(
        lifted: &'a LiftedFunction<ShaderDialect>,
        declarations: &'a Declarations,
        program: &'a Program,
        webs: &'a Webs,
    ) -> Self {
        let referenced = referenced_webs(&lifted.function);
        let placements = VariablePlacements::compute(&lifted.function);
        Self {
            function: &lifted.function,
            declarations,
            program,
            webs,
            semantics: SemanticNames::default(),
            resources: ResourceNames::default(),
            referenced,
            placements,
            web_names: Vec::new(),
            name_prefix: String::new(),
            hull_mode: HullMode::None,
            fork_instance: 0,
            input_preambles: Vec::new(),
            out: String::new(),
            indent: 0,
        }
    }

    /// Renders a hull shader: the control-point entry plus a
    /// patch-constant function concatenating the fork and join phases.
    ///
    /// # Errors
    ///
    /// Returns [`Unsupported`] for constructs outside the modeled subset.
    pub fn render_hull(
        mut self,
        phases: &'a [(u32, LiftedFunction<ShaderDialect>, Webs)],
    ) -> Result<String, Unsupported> {
        let declarations = self.declarations;
        let program = self.program;
        let semantics = self.semantics.clone();
        let resources = self.resources.clone();
        let points = declarations
            .tessellation
            .as_ref()
            .map_or(1, |tessellation| tessellation.input_control_points.max(1));
        self.hull_mode = HullMode::ControlPoint;
        let mut out = self.render()?;
        let _ = write!(
            out,
            "HsConstants constants(InputPatch<HsInput, {points}> patch, \
             uint vPrim_raw : SV_PrimitiveID)\n{{\n"
        );
        out.push_str("    uint4 vPrim = uint4(vPrim_raw, 0u, 0u, 0u);\n");
        out.push_str("    HsConstants _patch = (HsConstants)0;\n");
        // Staging arrays let computed register indices select control
        // points uniformly, integer rows carried as raw bits.
        let registers = semantics
            .inputs
            .keys()
            .copied()
            .max()
            .map_or(0, |register| register + 1);
        for point in 0..points {
            if registers == 0 {
                break;
            }
            let mut elements = Vec::new();
            for register in 0..registers {
                match semantics.inputs.get(&register) {
                    Some(row) if row.integral => {
                        elements.push(format!("asfloat(patch[{point}].v{register})"));
                    }
                    Some(_) => elements.push(format!("patch[{point}].v{register}")),
                    None => elements.push("float4(0.0, 0.0, 0.0, 0.0)".to_owned()),
                }
            }
            let _ = writeln!(
                out,
                "    float4 _icp{point}[{registers}] = {{ {} }};",
                elements.join(", ")
            );
        }
        for (position, (instances, lifted, webs)) in phases.iter().enumerate() {
            // Instanced phases unroll: one body per instance id.
            for instance in 0..(*instances).max(1) {
                let mut sub = Renderer::new(lifted, declarations, program, webs)
                    .with_semantics(semantics.clone())
                    .with_resource_names(resources.clone());
                sub.hull_mode = HullMode::Constants;
                sub.name_prefix = format!("p{position}i{instance}_");
                sub.fork_instance = instance;
                sub.indent = 1;
                out.push_str(&sub.constants_body()?);
            }
        }
        out.push_str("    return _patch;\n}\n");
        Ok(out)
    }

    /// Renders one patch-constant phase body.
    fn constants_body(mut self) -> Result<String, Unsupported> {
        self.assign_web_names();
        self.entry_web_declarations()?;
        let body = self.function.body();
        self.statements(body)?;
        Ok(self.out)
    }

    /// Uses signature-derived semantic strings for inputs and outputs,
    /// so recompiled signatures carry the original names.
    #[must_use]
    pub fn with_semantics(mut self, semantics: SemanticNames) -> Self {
        self.semantics = semantics;
        self
    }

    /// Uses RDEF-derived names for constant buffers and bindings, so the
    /// decompiled interface carries the original identifiers.
    #[must_use]
    pub fn with_resource_names(mut self, resources: ResourceNames) -> Self {
        self.resources = resources;
        self
    }

    /// Renders the full HLSL translation unit.
    ///
    /// # Errors
    ///
    /// Returns [`Unsupported`] for constructs outside the modeled subset.
    pub fn render(mut self) -> Result<String, Unsupported> {
        self.assign_web_names();
        self.header();
        self.signature();
        self.line("{");
        self.indent += 1;
        for preamble in std::mem::take(&mut self.input_preambles) {
            self.line(&preamble);
        }
        self.icb_declaration();
        self.indexable_declarations();
        self.entry_web_declarations()?;
        let body = self.function.body();
        self.statements(body)?;
        self.indent -= 1;
        self.line("}");
        Ok(self.out)
    }

    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.out.push_str("    ");
        }
        self.out.push_str(text);
        self.out.push('\n');
    }

    /// Names every referenced web: inputs keep their register spelling,
    /// temporaries take the register name with a dense disambiguating suffix,
    /// and synthetic temporaries get fresh names. Webs inlined out of the
    /// final HLIL consume no visible name.
    fn assign_web_names(&mut self) {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        let mut names = Vec::with_capacity(self.webs.len());
        for web in self.webs {
            if !self.referenced.contains(&web.variable) {
                names.push(None);
                continue;
            }
            let name = match &web.storage {
                Some(register) if matches!(register.file, RegFile::Input | RegFile::System(_)) => {
                    Self::register_name(*register)
                }
                Some(register) => {
                    let base = format!("{}{}", self.name_prefix, Self::register_name(*register));
                    let count = counts.entry(base.clone()).or_insert(0);
                    *count += 1;
                    if *count == 1 {
                        base
                    } else {
                        format!("{base}_{count}")
                    }
                }
                None => {
                    let count = counts.entry("tmp".to_owned()).or_insert(0);
                    *count += 1;
                    format!("{}tmp{count}", self.name_prefix)
                }
            };
            names.push(Some(name));
        }
        self.web_names = names;
    }

    /// The rendered name of one referenced web.
    fn web_name(&self, index: usize) -> Result<&str, Unsupported> {
        self.web_names
            .get(index)
            .and_then(Option::as_deref)
            .ok_or_else(|| Unsupported("referenced web without a rendered name".into()))
    }

    /// The web behind one lifted (HLIL) variable.
    fn hlil_web(
        &self,
        variable: cfglib::ir::hlil::VariableId,
    ) -> Result<(usize, &'a Web), Unsupported> {
        self.webs
            .of_lifted(variable)
            .map(|web| (variable.index(), web))
            .ok_or_else(|| Unsupported("variable without a web".into()))
    }

    /// Whether a web renders as an interface register rather than a
    /// declared local.
    fn is_input_web(web: &Web) -> bool {
        matches!(
            web.storage.map(|register| register.file),
            Some(RegFile::Input | RegFile::System(_) | RegFile::PatchConstant)
        )
    }

    /// The storage interpretation behind one web's rendered name.
    fn web_holder(&self, web: &Web) -> ScalarType {
        if Self::is_input_web(web) {
            let integral = web.storage.is_some_and(|register| match register.file {
                RegFile::System(SystemInput::DomainPoint) => false,
                RegFile::System(_) => true,
                RegFile::Input => self
                    .semantics
                    .inputs
                    .get(&register.index)
                    .is_some_and(|row| row.integral),
                RegFile::PatchConstant => web
                    .lanes
                    .first()
                    .and_then(|&lane| self.patch_row(register.index, lane))
                    .is_some_and(|row| row.integral),
                _ => false,
            });
            if integral {
                ScalarType::U32
            } else {
                ScalarType::F32
            }
        } else {
            normalize(web.shape.scalar)
        }
    }

    /// Declares live-in webs at function entry and verifies that every other
    /// rendered local has a structural declaration anchor.
    fn entry_web_declarations(&mut self) -> Result<(), Unsupported> {
        for web in self.webs {
            if Self::is_input_web(web) || !self.referenced.contains(&web.variable) {
                continue;
            }
            if web.live_in {
                let declaration = self.web_declaration(web)?;
                self.line(&declaration);
                continue;
            }
            let variable = HlilVariableId::from_raw(web.variable.raw());
            if self.placements.placement(variable).is_none() {
                return Err(Unsupported(
                    "referenced web without a declaration placement".into(),
                ));
            }
        }
        Ok(())
    }

    /// Declares non-live-in locals at the narrowest structural anchor that
    /// encloses all of their occurrences.
    fn web_declarations_before(
        &mut self,
        statement: cfglib::ir::hlil::StatementId,
    ) -> Result<(), Unsupported> {
        for index in 0..self.placements.before(statement).len() {
            let variable = self.placements.before(statement)[index];
            let (_, web) = self.hlil_web(variable)?;
            if web.live_in || Self::is_input_web(web) || !self.referenced.contains(&web.variable) {
                continue;
            }
            let declaration = self.web_declaration(web)?;
            self.line(&declaration);
        }
        Ok(())
    }

    fn web_declaration(&self, web: &Web) -> Result<String, Unsupported> {
        let shape = ValueShape {
            scalar: normalize(web.shape.scalar),
            lanes: web.shape.lanes,
        };
        let declared = type_name(shape);
        let name = self.web_name(web.variable.index())?;
        // `precise` pins evaluation to the written expression order: the
        // source mirrors the original instruction order, and fxc must not
        // reassociate it (even `/Gis` reorders constant additions otherwise).
        if web.live_in {
            // Version zero can reach this web, so retain the reference
            // interpreter's zero-initialized incoming storage.
            let zero = zero_value(shape);
            Ok(format!("precise {declared} {name} = {zero};"))
        } else {
            // SSA proves every read is reached by a source definition; an
            // initializer would only add a dead synthetic write.
            Ok(format!("precise {declared} {name};"))
        }
    }

    fn indexable_declarations(&mut self) {
        for (register, size, _) in &self.declarations.indexable_temps {
            self.line(&format!("uint4 x{register}[{size}];"));
        }
    }

    fn icb_declaration(&mut self) {
        if self.declarations.immediate_constants.is_empty() {
            return;
        }
        self.line(&format!(
            "const uint4 icb[{}] =",
            self.declarations.immediate_constants.len()
        ));
        self.line("{");
        for row in &self.declarations.immediate_constants {
            let lanes: Vec<String> = row
                .iter()
                .map(|value| scalar_literal(value.to_bits(), ScalarType::U32))
                .collect();
            self.line(&format!("    uint4({}),", lanes.join(", ")));
        }
        self.line("};");
    }
}

fn binary(left: &Rendered, symbol: &str, right: &Rendered, produced: ScalarType) -> Rendered {
    Rendered::composite(
        format!("{} {symbol} {}", left.wrapped(), right.wrapped()),
        produced,
    )
}

fn call1(name: &str, value: &Rendered, produced: ScalarType) -> Rendered {
    Rendered::atom(format!("{name}({})", value.text), produced)
}

fn call2(name: &str, a: &Rendered, b: &Rendered, produced: ScalarType) -> Rendered {
    Rendered::atom(format!("{name}({}, {})", a.text, b.text), produced)
}

fn width_text(width: u8) -> String {
    if width <= 1 {
        String::new()
    } else {
        width.to_string()
    }
}

fn splat(value: &str, count: u8) -> String {
    if count <= 1 {
        value.to_owned()
    } else {
        let parts = vec![value; usize::from(count)];
        format!("uint{count}({})", parts.join(", "))
    }
}

/// The zero initializer of one shape.
fn zero_value(shape: ValueShape) -> String {
    let scalar = match normalize(shape.scalar) {
        ScalarType::F32 | ScalarType::F64 => "0.0",
        ScalarType::I32 | ScalarType::I64 => "0",
        ScalarType::Bool => "false",
        _ => "0u",
    };
    if shape.lanes <= 1 {
        scalar.to_owned()
    } else {
        let parts = vec![scalar; usize::from(shape.lanes)];
        format!("{}({})", type_name(shape), parts.join(", "))
    }
}

/// The optional immediate texel-offset argument of a sample or load.
fn texel_offsets(head: &AluHead, dimensions: usize) -> String {
    let Some(offsets) = head.tex_offsets else {
        return String::new();
    };
    let rendered: Vec<String> = offsets
        .iter()
        .take(dimensions)
        .map(std::string::ToString::to_string)
        .collect();
    if rendered.len() == 1 {
        format!(", {}", rendered[0])
    } else {
        format!(", int{}({})", rendered.len(), rendered.join(", "))
    }
}

/// Selects lanes from a four-lane value, omitting the identity swizzle.
fn select_vector_lanes(value: String, lanes: &[u8]) -> String {
    if lanes == [0, 1, 2, 3] {
        value
    } else {
        format!("{value}.{}", suffix(lanes))
    }
}

/// The HLSL interpolation keywords of one declared mode.
fn interpolation_keywords(mode: Option<&'static str>) -> &'static str {
    match mode {
        Some("constant") => "nointerpolation ",
        Some("linearCentroid") => "centroid ",
        Some("linearNoperspective") => "noperspective ",
        Some("linearNoperspectiveCentroid") => "noperspective centroid ",
        Some("linearSample") => "sample ",
        Some("linearNoperspectiveSample") => "noperspective sample ",
        _ => "",
    }
}

/// The HLSL comparison operator of a comparison opcode.
fn comparison_symbol(opcode: Opcode) -> Option<&'static str> {
    match opcode {
        Opcode::Lt | Opcode::ILt | Opcode::ULt => Some("<"),
        Opcode::Ge | Opcode::IGe | Opcode::UGe => Some(">="),
        Opcode::Eq | Opcode::IEq => Some("=="),
        Opcode::Ne | Opcode::INe => Some("!="),
        _ => None,
    }
}

/// The exact complement of a comparison operator when HLSL can state it
/// directly. Ordered floating `<`/`>=` deliberately stay wrapped in `!`:
/// NaN makes the opposite relation different from logical negation.
fn negated_comparison_symbol(opcode: Opcode) -> Option<&'static str> {
    match opcode {
        Opcode::ILt | Opcode::ULt => Some(">="),
        Opcode::IGe | Opcode::UGe => Some("<"),
        Opcode::Eq | Opcode::IEq => Some("!="),
        Opcode::Ne | Opcode::INe => Some("=="),
        _ => None,
    }
}

/// Formats an immediate's lanes in the consumer's interpretation.
fn literal(bits: &[u64], wanted: ScalarType) -> String {
    let rendered: Vec<String> = bits
        .iter()
        .map(|&lane| scalar_literal(u32::try_from(lane & 0xffff_ffff).unwrap_or(0), wanted))
        .collect();
    if rendered.len() == 1 {
        rendered.into_iter().next().expect("one lane renders")
    } else {
        format!(
            "{}{}({})",
            scalar_keyword(wanted),
            rendered.len(),
            rendered.join(", ")
        )
    }
}

/// One 32-bit lane as an exact HLSL literal.
fn scalar_literal(bits: u32, wanted: ScalarType) -> String {
    match normalize(wanted) {
        ScalarType::F32 | ScalarType::F64 => {
            let value = f32::from_bits(bits);
            // Denormals flush to zero inside fxc's constant folding, and
            // NaN/infinity have no literal spelling: only normal values
            // (and exact zero) round-trip as decimals.
            if bits.trailing_zeros() >= 31 {
                if bits == 0 {
                    "0.0".to_owned()
                } else {
                    "-0.0".to_owned()
                }
            } else if value.is_normal() {
                let mut text = format!("{value}");
                if !text.contains('.') && !text.contains('e') {
                    text.push_str(".0");
                }
                text
            } else {
                format!("asfloat(0x{bits:08x}u)")
            }
        }
        ScalarType::I32 | ScalarType::I64 => {
            #[allow(
                clippy::cast_possible_wrap,
                reason = "the wrap is the reinterpretation"
            )]
            let value = i64::from(bits as i32);
            value.to_string()
        }
        ScalarType::Bool => if bits == 0 { "false" } else { "true" }.to_owned(),
        _ => format!("{bits}u"),
    }
}

/// The struct-field name of one patch-constant row.
fn patch_field_name(row: &PatchRow) -> String {
    let first = (0u8..4)
        .find(|lane| row.mask & (1 << lane) != 0)
        .unwrap_or(0);
    format!("pc{}_{}", row.register, LANES[usize::from(first)])
}

/// One patch-constant field: a run of consecutive scalar rows sharing a
/// semantic name becomes an array (HLSL array semantics auto-increment),
/// so computed register indices stay expressible.
#[derive(Debug, Clone)]
struct PatchGroup {
    field: String,
    semantic: String,
    first_register: u32,
    count: u32,
    lane: u8,
    width: u32,
    integral: bool,
}

fn semantic_parts(text: &str) -> (&str, u32) {
    let digits = text.len() - text.trim_end_matches(|c: char| c.is_ascii_digit()).len();
    let (name, index) = text.split_at(text.len() - digits);
    (name, index.parse().unwrap_or(0))
}

fn patch_groups(rows: &[PatchRow]) -> Vec<PatchGroup> {
    let mut groups: Vec<PatchGroup> = Vec::new();
    for row in rows {
        let (name, index) = semantic_parts(&row.text);
        let lane = (0u8..4)
            .find(|lane| row.mask & (1 << lane) != 0)
            .unwrap_or(0);
        let width = row.mask.count_ones().clamp(1, 4);
        if let Some(group) = groups.last_mut()
            && group.count > 0
            && row.register == group.first_register + group.count
            && group.lane == lane
            && group.integral == row.integral
            && group.width == width
        {
            let (previous_name, previous_index) = semantic_parts(&group.semantic);
            if previous_name == name && index == previous_index + group.count {
                group.count += 1;
                continue;
            }
        }
        groups.push(PatchGroup {
            field: patch_field_name(row),
            semantic: row.text.clone(),
            first_register: row.register,
            count: 1,
            lane,
            width,
            integral: row.integral,
        });
    }
    groups
}

#[cfg(test)]
mod tests {
    use dxbc::shex::Opcode;

    use super::{negated_comparison_symbol, select_vector_lanes};

    #[test]
    fn comparison_negation_preserves_unordered_float_semantics() {
        assert_eq!(negated_comparison_symbol(Opcode::IGe), Some("<"));
        assert_eq!(negated_comparison_symbol(Opcode::UGe), Some("<"));
        assert_eq!(negated_comparison_symbol(Opcode::Ge), None);
        assert_eq!(negated_comparison_symbol(Opcode::Lt), None);
    }

    #[test]
    fn identity_vector_selection_omits_its_swizzle() {
        assert_eq!(
            select_vector_lanes("vertices[0].v0".into(), &[0, 1, 2, 3]),
            "vertices[0].v0"
        );
        assert_eq!(
            select_vector_lanes("vertices[0].v0".into(), &[0, 1]),
            "vertices[0].v0.xy"
        );
    }
}
