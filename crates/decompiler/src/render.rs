//! HLIL → HLSL source rendering.
//!
//! Registers render as `float4` locals; each operation knows the lane
//! interpretation of its operands, so integer operations cast through
//! `asint`/`asuint` while immediate operands format natively in the
//! consumer's lane type — bit-exact without a type inference pass.

use core::fmt::Write as _;

use cfglib::ir::hlil::{ExpressionId, ExpressionKind, LiftedFunction, StatementId, StatementKind};
use dxbc::shex::{ComponentSelect, Opcode, Program};

use crate::dialect::{
    AluOp, DestAccess, Immediate, MemRef, Op, RegFile, Register, ShaderDialect, SourceAccess,
    SystemInput,
};
use crate::lift::{Declarations, Unsupported};

type HlilFunction = cfglib::ir::hlil::Function<ShaderDialect>;

/// Lane interpretation an operation imposes on an operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Lane {
    Float,
    Int,
    Uint,
}

const LANES: [char; 4] = ['x', 'y', 'z', 'w'];

fn mask_lanes(mask: u8) -> Vec<u8> {
    (0u8..4).filter(|lane| mask & (1 << lane) != 0).collect()
}

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

/// Signature-derived semantic rows by register, per direction.
#[derive(Debug, Clone, Default)]
pub struct SemanticNames {
    /// Input register → interface row.
    pub inputs: std::collections::BTreeMap<u32, SemanticRow>,
    /// Output register → interface row.
    pub outputs: std::collections::BTreeMap<u32, SemanticRow>,
}

/// Renders one decompiled shader body and its declarations.
pub struct Renderer<'a> {
    function: &'a HlilFunction,
    declarations: &'a Declarations,
    program: &'a Program,
    semantics: SemanticNames,
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
    ) -> Self {
        Self {
            function: &lifted.function,
            declarations,
            program,
            semantics: SemanticNames::default(),
            input_preambles: Vec::new(),
            out: String::new(),
            indent: 0,
        }
    }

    /// Uses signature-derived semantic strings for inputs and outputs,
    /// so recompiled signatures carry the original names.
    #[must_use]
    pub fn with_semantics(mut self, semantics: SemanticNames) -> Self {
        self.semantics = semantics;
        self
    }

    /// Renders the full HLSL translation unit.
    ///
    /// # Errors
    ///
    /// Returns [`Unsupported`] for constructs outside the modeled subset.
    pub fn render(mut self) -> Result<String, Unsupported> {
        self.header();
        self.signature();
        self.line("{");
        self.indent += 1;
        for preamble in self.input_preambles.clone() {
            self.line(&preamble);
        }
        self.icb_declaration();
        self.temp_declarations();
        let body = self.function.body().to_vec();
        self.statements(&body)?;
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

    fn header(&mut self) {
        self.line("// Decompiled by d3dasm-decompiler.");
        for (index, size) in &self.declarations.constant_buffers {
            self.line(&format!("cbuffer cb{index}_buffer : register(b{index})"));
            self.line("{");
            self.line(&format!("    float4 cb{index}[{size}];"));
            self.line("};");
        }
        for (index, _mode) in &self.declarations.samplers {
            self.line(&format!("SamplerState s{index} : register(s{index});"));
        }
        for (index, dimension, return_type) in &self.declarations.resources {
            let texture_type = match *dimension {
                "buffer" => "Buffer",
                "texture1d" => "Texture1D",
                "texture3d" => "Texture3D",
                "texturecube" => "TextureCube",
                "texture1darray" => "Texture1DArray",
                "texture2darray" => "Texture2DArray",
                "texturecubearray" => "TextureCubeArray",
                "texture2dms" => "Texture2DMS",
                "texture2dmsarray" => "Texture2DMSArray",
                _ => "Texture2D",
            };
            // The element type is semantic: an integer-typed resource
            // read through a float view reinterprets its bits.
            let element = match return_type[0] {
                dxbc::shex::ReturnType::Uint => "uint4",
                dxbc::shex::ReturnType::Sint => "int4",
                _ => "float4",
            };
            self.line(&format!(
                "{texture_type}<{element}> t{index} : register(t{index});"
            ));
        }
        for (index, stride) in &self.declarations.structured {
            let dwords = stride.div_ceil(4);
            self.line(&format!(
                "struct t{index}_element {{ uint value[{dwords}]; }};"
            ));
            self.line(&format!(
                "StructuredBuffer<t{index}_element> t{index} : register(t{index});"
            ));
        }
        if !self.declarations.constant_buffers.is_empty()
            || !self.declarations.samplers.is_empty()
            || !self.declarations.resources.is_empty()
            || !self.declarations.structured.is_empty()
        {
            self.line("");
        }
    }

    /// The register's HLSL spelling.
    fn register_name(register: Register) -> String {
        match register.file {
            RegFile::Temp => format!("r{}", register.index),
            RegFile::Input => format!("v{}", register.index),
            RegFile::Output => format!("o{}", register.index),
            RegFile::OutputDepth => "oDepth".to_owned(),
            RegFile::OutputCoverage => "oMask".to_owned(),
            RegFile::IndexableTemp => format!("x{}", register.index),
            RegFile::ConstantBuffer => format!("cb{}", register.index),
            RegFile::ImmediateConstantBuffer => "icb".to_owned(),
            RegFile::Resource => format!("t{}", register.index),
            RegFile::Sampler => format!("s{}", register.index),
            RegFile::Uav => format!("u{}", register.index),
            RegFile::System(system) => match system {
                SystemInput::ThreadId => "vThreadID",
                SystemInput::ThreadGroupId => "vThreadGroupID",
                SystemInput::ThreadIdInGroup => "vThreadIDInGroup",
                SystemInput::Coverage => "vCoverage",
                SystemInput::PrimitiveId => "vPrim",
            }
            .to_owned(),
        }
    }

    /// The semantic string of one declared input or output.
    fn semantic(
        &self,
        register: Register,
        system_value: Option<&'static str>,
        output: bool,
    ) -> String {
        // The signature chunk's own semantic names take precedence: they
        // are the original interface contract.
        if matches!(register.file, RegFile::Input | RegFile::Output) {
            let table = if output {
                &self.semantics.outputs
            } else {
                &self.semantics.inputs
            };
            if let Some(row) = table.get(&register.index) {
                return row.text.clone();
            }
        }
        if let Some(name) = system_value {
            return name.to_owned();
        }
        match register.file {
            RegFile::OutputDepth => "SV_Depth".to_owned(),
            RegFile::OutputCoverage | RegFile::System(SystemInput::Coverage) => {
                "SV_Coverage".to_owned()
            }
            RegFile::System(SystemInput::ThreadId) => "SV_DispatchThreadID".to_owned(),
            RegFile::System(SystemInput::ThreadGroupId) => "SV_GroupID".to_owned(),
            RegFile::System(SystemInput::ThreadIdInGroup) => "SV_GroupThreadID".to_owned(),
            RegFile::System(SystemInput::PrimitiveId) => "SV_PrimitiveID".to_owned(),
            _ if output && self.program.shader_type == "ps" => {
                format!("SV_Target{}", register.index)
            }
            _ => format!("TEXCOORD{}", register.index),
        }
    }

    fn signature(&mut self) {
        let mut parameters = Vec::new();
        for row in &self.declarations.inputs {
            let semantic = self.semantic(row.register, row.system_value, false);
            let interpolation = interpolation_keywords(row.interpolation);
            let name = Self::register_name(row.register);
            // Integer-typed signature rows (SV_VertexID, SV_InstanceID)
            // must be scalar/vector uint parameters; a preamble local
            // restores the register's raw-bits float4 view.
            let integral = row.register.file == RegFile::Input
                && self
                    .semantics
                    .inputs
                    .get(&row.register.index)
                    .is_some_and(|row| row.integral);
            if integral {
                let mask = self
                    .semantics
                    .inputs
                    .get(&row.register.index)
                    .map_or(0b0001, |row| row.mask);
                let width = usize::try_from(mask.count_ones()).unwrap_or(4).clamp(1, 4);
                let declared = if width == 1 {
                    "uint".to_owned()
                } else {
                    format!("uint{width}")
                };
                parameters.push(format!("{declared} {name}_raw : {semantic}"));
                let mut lanes: Vec<String> = (0..width)
                    .map(|lane| {
                        if width == 1 {
                            format!("{name}_raw")
                        } else {
                            format!("{name}_raw.{}", LANES[lane])
                        }
                    })
                    .collect();
                while lanes.len() < 4 {
                    lanes.push("0u".to_owned());
                }
                self.input_preambles
                    .push(format!("uint4 {name} = uint4({});", lanes.join(", ")));
                continue;
            }
            parameters.push(format!("{interpolation}float4 {name} : {semantic}"));
        }
        for (register, _, system_value) in &self.declarations.outputs {
            let semantic = self.semantic(*register, *system_value, true);
            let declared = if register.file == RegFile::OutputDepth {
                format!("out float {} : {semantic}", Self::register_name(*register))
            } else {
                format!("out float4 {} : {semantic}", Self::register_name(*register))
            };
            parameters.push(declared);
        }
        if let Some((x, y, z)) = self.declarations.thread_group {
            self.line(&format!("[numthreads({x}, {y}, {z})]"));
        }
        let mut signature = String::from("void main(");
        signature.push_str(&parameters.join(", "));
        signature.push(')');
        self.line(&signature);
    }

    /// Output registers in declaration order — the order `ret` uses
    /// them, and so the order of return values.
    fn output_registers(&self) -> Vec<Register> {
        self.function
            .variables()
            .iter()
            .filter(|variable| matches!(variable.role, crate::dialect::Role::Output))
            .filter_map(|variable| variable.native)
            .collect()
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
        for row in &self.declarations.immediate_constants.clone() {
            let lanes: Vec<String> = row
                .iter()
                .map(|value| scalar_literal(value.to_bits(), Lane::Uint))
                .collect();
            self.line(&format!("    uint4({}),", lanes.join(", ")));
        }
        self.line("};");
    }

    fn temp_declarations(&mut self) {
        let mut synthetic: Vec<usize> = self
            .function
            .expressions()
            .iter()
            .filter_map(|expression| match expression.kind() {
                ExpressionKind::Variable(variable) => Some(*variable),
                _ => None,
            })
            .filter(|variable| {
                self.function
                    .variables()
                    .get(variable.index())
                    .is_some_and(|declared| declared.native.is_none())
            })
            .map(cfglib::ir::hlil::VariableId::index)
            .collect();
        synthetic.sort_unstable();
        synthetic.dedup();
        for index in synthetic {
            self.line(&format!("uint4 tmp{index} = uint4(0u, 0u, 0u, 0u);"));
        }
        let mut temps: Vec<u32> = self
            .function
            .variables()
            .iter()
            .filter_map(|variable| match variable.native {
                Some(Register {
                    file: RegFile::Temp,
                    index,
                }) => Some(index),
                _ => None,
            })
            .collect();
        temps.sort_unstable();
        temps.dedup();
        for index in temps {
            self.line(&format!("uint4 r{index} = uint4(0u, 0u, 0u, 0u);"));
        }
        for (register, size, _) in &self.declarations.indexable_temps {
            self.line(&format!("uint4 x{register}[{size}];"));
        }
    }

    fn statements(&mut self, ids: &[StatementId]) -> Result<(), Unsupported> {
        for &id in ids {
            self.statement(id)?;
        }
        Ok(())
    }

    fn statement_kind(
        &self,
        id: StatementId,
    ) -> Result<&'a StatementKind<ShaderDialect>, Unsupported> {
        self.function
            .statement(id)
            .map(cfglib::ir::hlil::Statement::kind)
            .ok_or_else(|| Unsupported("unresolvable statement identity".into()))
    }

    fn expression(
        &self,
        id: ExpressionId,
    ) -> Result<&'a cfglib::ir::hlil::Expression<ShaderDialect>, Unsupported> {
        self.function
            .expression(id)
            .ok_or_else(|| Unsupported("unresolvable expression identity".into()))
    }

    fn statement(&mut self, id: StatementId) -> Result<(), Unsupported> {
        match self.statement_kind(id)? {
            StatementKind::Assign { target, value } => self.assign(*target, *value),
            StatementKind::Expression(expression) => self.effect(*expression),
            StatementKind::If {
                condition,
                then_body,
                else_body,
            } => {
                let condition = self.condition(*condition, false)?;
                self.line(&format!("if ({condition})"));
                self.block(&then_body.clone())?;
                if !else_body.is_empty() {
                    self.line("else");
                    self.block(&else_body.clone())?;
                }
                Ok(())
            }
            StatementKind::While { condition, body } => {
                let condition = self.condition(*condition, false)?;
                self.line(&format!("while ({condition})"));
                self.block(&body.clone())
            }
            StatementKind::DoWhile { body, condition } => {
                let body = body.clone();
                self.line("do");
                self.block(&body)?;
                let condition = self.condition(*condition, false)?;
                let last = self.out.pop();
                debug_assert_eq!(last, Some('\n'));
                writeln!(self.out, " while ({condition});").expect("string writes are infallible");
                Ok(())
            }
            StatementKind::Loop { body } => {
                self.line("while (true)");
                self.block(&body.clone())
            }
            StatementKind::Break { label: None } => {
                self.line("break;");
                Ok(())
            }
            StatementKind::Continue { label: None } => {
                self.line("continue;");
                Ok(())
            }
            StatementKind::Return { values } => {
                let outputs = self.output_registers();
                for (&value, register) in values.clone().iter().zip(outputs) {
                    let name = Self::register_name(register);
                    // A bare variable read means the output was already
                    // materialized by an earlier statement.
                    if let ExpressionKind::Variable(variable) = self.expression(value)?.kind()
                        && self.register_of(*variable).ok() == Some(register)
                    {
                        continue;
                    }
                    let assign_value = self.expression(value)?;
                    if let ExpressionKind::Operation {
                        operation: Op::Alu(alu),
                        operands,
                    } = assign_value.kind()
                    {
                        let mask = alu.dest.clone().unwrap_or(DestAccess { mask: 0b1111 });
                        let lanes = mask_lanes(mask.mask);
                        let holder = self.register_holder(register);
                        let (rendered, produced) = self.alu(alu, operands, &lanes, holder)?;
                        let rendered = converted(rendered, produced, holder);
                        if mask.mask == 0b1111 || register.file == RegFile::OutputDepth {
                            self.line(&format!("{name} = {rendered};"));
                        } else {
                            self.line(&format!("{name}.{} = {rendered};", suffix(&lanes)));
                        }
                    } else {
                        let holder = self.register_holder(register);
                        let rendered = self.source_value(value, &[0, 1, 2, 3], holder, None)?;
                        self.line(&format!("{name} = {rendered};"));
                    }
                }
                self.line("return;");
                Ok(())
            }
            other => Err(Unsupported(format!("statement {other:?}"))),
        }
    }

    fn block(&mut self, ids: &[StatementId]) -> Result<(), Unsupported> {
        self.line("{");
        self.indent += 1;
        self.statements(ids)?;
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    /// An expression standing as a statement (`discard`, stores).
    fn effect(&mut self, id: ExpressionId) -> Result<(), Unsupported> {
        let expression = self.expression(id)?;
        let ExpressionKind::Operation {
            operation,
            operands,
        } = expression.kind()
        else {
            return Err(Unsupported("value expression as statement".into()));
        };
        match operation {
            Op::Discard {
                nonzero,
                components,
            } => {
                let tested = self.tested(operands, components, *nonzero)?;
                self.line(&format!("if ({tested}) discard;"));
                Ok(())
            }
            Op::Ret => {
                self.line("return;");
                Ok(())
            }
            Op::Alu(alu) => {
                // A destination-less retained op (e.g. `sync`).
                let (rendered, _) = self.alu(alu, operands, &[0, 1, 2, 3], Lane::Float)?;
                self.line(&format!("{rendered};"));
                Ok(())
            }
            other => Err(Unsupported(format!("effect {}", other.mnemonic()))),
        }
    }

    fn assign(&mut self, target: ExpressionId, value: ExpressionId) -> Result<(), Unsupported> {
        let target = self.expression(target)?;
        if let ExpressionKind::Operation {
            operation: Op::Load(reference),
            operands,
        } = target.kind()
        {
            // The HLIL store shape: the target is the place expression.
            let holder = if reference.file == RegFile::IndexableTemp {
                Lane::Uint
            } else {
                Lane::Float
            };
            let place = self.memory(reference, operands.first().copied())?;
            let lanes = match &reference.components {
                ComponentSelect::Mask(mask) => mask_lanes(*mask),
                _ => vec![0, 1, 2, 3],
            };
            let rendered = self.source_value(value, &lanes, holder, None)?;
            if lanes.len() == 4 {
                self.line(&format!("{place} = {rendered};"));
            } else {
                self.line(&format!("{place}.{} = {rendered};", suffix(&lanes)));
            }
            return Ok(());
        }
        let ExpressionKind::Variable(variable) = target.kind() else {
            return Err(Unsupported("assignment to a non-register place".into()));
        };
        if let ExpressionKind::Operation {
            operation: Op::Store(reference),
            operands,
        } = self.expression(value)?.kind()
        {
            let reference = reference.clone();
            let operands = operands.clone();
            return self.store(&reference, &operands);
        }
        let name = self.variable_name(*variable);
        self.emit_assignment(&name, *variable, value)
    }

    /// Emits one register assignment, unrolling a chain of captured
    /// partial writes back into their statement sequence.
    fn emit_assignment(
        &mut self,
        name: &str,
        variable: cfglib::ir::hlil::VariableId,
        value: ExpressionId,
    ) -> Result<(), Unsupported> {
        let register = self.register_of(variable).ok();
        match self.expression(value)?.kind() {
            ExpressionKind::Operation {
                operation: Op::Alu(alu),
                operands,
            } if alu.opcode == Opcode::Resinfo => {
                let alu = alu.clone();
                let operands = operands.clone();
                let holder = register.map_or(Lane::Uint, |place| self.register_holder(place));
                self.resinfo(name, holder, &alu, &operands)
            }
            ExpressionKind::Operation {
                operation: Op::Alu(alu),
                operands,
            } => {
                let alu = alu.clone();
                let operands = operands.clone();
                let mask = alu.dest.clone().unwrap_or(DestAccess { mask: 0b1111 });
                let lanes = mask_lanes(mask.mask);
                let holder = register.map_or(Lane::Uint, |place| self.register_holder(place));
                let (rendered, produced) = self.alu(&alu, &operands, &lanes, holder)?;
                let rendered = converted(rendered, produced, holder);
                let scalar_place =
                    matches!(register, Some(place) if place.file == RegFile::OutputDepth);
                if mask.mask == 0b1111 || scalar_place {
                    self.line(&format!("{name} = {rendered};"));
                } else {
                    self.line(&format!("{name}.{} = {rendered};", suffix(&lanes)));
                }
                Ok(())
            }
            _ => {
                let holder = register.map_or(Lane::Uint, |place| self.register_holder(place));
                let rendered = self.source_value(value, &[0, 1, 2, 3], holder, None)?;
                self.line(&format!("{name} = {rendered};"));
                Ok(())
            }
        }
    }

    fn store(&mut self, reference: &MemRef, operands: &[ExpressionId]) -> Result<(), Unsupported> {
        // Store uses: [value, address?]; the value writes the masked lanes.
        let Some(&value) = operands.first() else {
            return Err(Unsupported("store without a value".into()));
        };
        let holder = if reference.file == RegFile::IndexableTemp {
            Lane::Uint
        } else {
            Lane::Float
        };
        let place = self.memory(reference, operands.get(1).copied())?;
        let lanes = match &reference.components {
            ComponentSelect::Mask(mask) => mask_lanes(*mask),
            _ => vec![0, 1, 2, 3],
        };
        let rendered = self.source_value(value, &lanes, holder, None)?;
        if lanes.len() == 4 {
            self.line(&format!("{place} = {rendered};"));
        } else {
            self.line(&format!("{place}.{} = {rendered};", suffix(&lanes)));
        }
        Ok(())
    }

    fn register_of(&self, variable: cfglib::ir::hlil::VariableId) -> Result<Register, Unsupported> {
        self.function
            .variables()
            .get(variable.index())
            .and_then(|declared| declared.native)
            .ok_or_else(|| Unsupported("write to a synthetic temporary".into()))
    }

    /// The rendered name of any variable: its register, or a synthetic
    /// temporary name for materialized operand values.
    fn variable_name(&self, variable: cfglib::ir::hlil::VariableId) -> String {
        self.register_of(variable)
            .map_or_else(|_| format!("tmp{}", variable.index()), Self::register_name)
    }

    /// The lane type a place's bits are stored in. Temps are bit-honest
    /// `uint4` locals — fxc's constant folder flushes provable denormal
    /// floats, so integer data must never round-trip float storage —
    /// while interface registers and constant buffers stay `float4`.
    fn register_holder(&self, register: Register) -> Lane {
        match register.file {
            RegFile::Temp | RegFile::IndexableTemp | RegFile::ImmediateConstantBuffer => Lane::Uint,
            RegFile::Input => {
                let integral = self
                    .semantics
                    .inputs
                    .get(&register.index)
                    .is_some_and(|row| row.integral);
                if integral { Lane::Uint } else { Lane::Float }
            }
            _ => Lane::Float,
        }
    }

    /// The storage lane type behind one variable's rendered name.
    fn variable_holder(&self, variable: cfglib::ir::hlil::VariableId) -> Lane {
        self.register_of(variable)
            .map_or(Lane::Uint, |register| self.register_holder(register))
    }

    /// Renders a branch-condition test with the requested polarity.
    fn condition(&self, id: ExpressionId, negated: bool) -> Result<String, Unsupported> {
        let expression = self.expression(id)?;
        let ExpressionKind::Operation {
            operation:
                Op::Branch {
                    nonzero,
                    components,
                },
            operands,
        } = expression.kind()
        else {
            return Err(Unsupported("condition is not a branch".into()));
        };
        self.tested(operands, components, *nonzero != negated)
    }

    /// Renders `operand-tested-against-zero` with the given polarity.
    fn tested(
        &self,
        operands: &[ExpressionId],
        components: &ComponentSelect,
        nonzero: bool,
    ) -> Result<String, Unsupported> {
        let Some(&operand) = operands.first() else {
            return Err(Unsupported("test without an operand".into()));
        };
        let lane = select_lane(components);
        // A comparison feeding the test renders as the bare boolean.
        if let ExpressionKind::Operation {
            operation: Op::Alu(alu),
            operands: inner,
        } = self.expression(operand)?.kind()
            && let Some(comparison) = comparison_symbol(alu.opcode)
        {
            let text = self.comparison(alu, inner, comparison, &[lane])?;
            return Ok(if nonzero { text } else { format!("!({text})") });
        }
        let value = self.source_value(operand, &[lane], Lane::Uint, None)?;
        Ok(if nonzero {
            format!("{value} != 0u")
        } else {
            format!("{value} == 0u")
        })
    }

    /// Renders one comparison as a boolean expression over its lanes.
    fn comparison(
        &self,
        alu: &AluOp,
        operands: &[ExpressionId],
        symbol: &str,
        lanes: &[u8],
    ) -> Result<String, Unsupported> {
        let lane_type = operand_lane(alu.opcode);
        let left = self.aligned_source(alu, operands, 0, lanes, lane_type)?;
        let right = self.aligned_source(alu, operands, 1, lanes, lane_type)?;
        Ok(format!("{left} {symbol} {right}"))
    }

    /// One source operand re-swizzled through the destination lanes.
    fn aligned_source(
        &self,
        alu: &AluOp,
        operands: &[ExpressionId],
        position: usize,
        lanes: &[u8],
        lane_type: Lane,
    ) -> Result<String, Unsupported> {
        let access = alu
            .sources
            .get(position)
            .ok_or_else(|| Unsupported("missing source access".into()))?;
        let operand = operands
            .get(position)
            .copied()
            .ok_or_else(|| Unsupported("missing source operand".into()))?;
        let mapped = map_lanes(&access.components, lanes)?;
        let mut text = self.source_value(operand, &mapped, lane_type, Some(access))?;
        if access.abs {
            text = format!("abs({text})");
        }
        if access.negate {
            text = format!("-{text}");
        }
        Ok(text)
    }

    /// Renders one operand value selected down to `lanes`, cast to the
    /// consumer's lane type.
    fn source_value(
        &self,
        id: ExpressionId,
        lanes: &[u8],
        lane_type: Lane,
        access: Option<&SourceAccess>,
    ) -> Result<String, Unsupported> {
        let _ = access;
        let expression = self.expression(id)?;
        match expression.kind() {
            ExpressionKind::Variable(variable) => {
                let name = self.variable_name(*variable);
                let selected = format!("{name}.{}", suffix(lanes));
                Ok(converted(
                    selected,
                    self.variable_holder(*variable),
                    lane_type,
                ))
            }
            ExpressionKind::Constant(immediate) => Ok(literal(immediate, lanes, lane_type)),
            ExpressionKind::Operation {
                operation,
                operands,
            } => match operation {
                Op::Const(immediate) => Ok(literal(immediate, lanes, lane_type)),
                Op::Load(reference) => {
                    let holder = match reference.file {
                        RegFile::IndexableTemp | RegFile::ImmediateConstantBuffer => Lane::Uint,
                        _ => Lane::Float,
                    };
                    // A dynamically indexed icb read is bounds-defined in
                    // DXBC (out of range reads zero) but undefined for an
                    // HLSL const array, where fxc lowers assuming the
                    // index is in range. An explicit guard restores the
                    // exact semantics.
                    if reference.file == RegFile::ImmediateConstantBuffer
                        && reference.relative_to.is_some()
                    {
                        let address = operands.first().copied().ok_or_else(|| {
                            Unsupported("relative reference without address".into())
                        })?;
                        let lane = reference.relative_select.as_ref().map_or(0, select_lane);
                        let index = self.source_value(address, &[lane], Lane::Uint, None)?;
                        let offset = reference.indices.first().copied().unwrap_or(0);
                        let index = if offset == 0 {
                            index
                        } else {
                            format!("({index} + {offset}u)")
                        };
                        let count = self.declarations.immediate_constants.len();
                        let mapped = map_lanes(&reference.components, lanes)?;
                        let selected = format!("icb[{index}].{}", suffix(&mapped));
                        let zeros = splat("0u", mapped.len());
                        let text = format!("{index} < {count}u ? {selected} : {zeros}");
                        return Ok(cast_wrapped(&text, Lane::Uint, lane_type));
                    }
                    let place = self.memory(reference, operands.first().copied())?;
                    let mapped = map_lanes(&reference.components, lanes)?;
                    let selected = format!("{place}.{}", suffix(&mapped));
                    Ok(converted(selected, holder, lane_type))
                }
                Op::Alu(alu) => {
                    // Partial-write producers never inline: the dialect
                    // declares their previous-value operand, so the lift
                    // materializes them (see `previous_value_operand`).
                    if alu.merges {
                        return Err(Unsupported("inlined partial write".into()));
                    }
                    let (text, produced) = self.alu(alu, operands, lanes, lane_type)?;
                    Ok(cast_wrapped(&text, produced, lane_type))
                }
                Op::Copy => {
                    let Some(&inner) = operands.first() else {
                        return Err(Unsupported("copy without a source".into()));
                    };
                    self.source_value(inner, lanes, lane_type, access)
                }
                other => Err(Unsupported(format!("value {}", other.mnemonic()))),
            },
        }
    }

    /// Renders one retained operation's value over the written lanes,
    /// returning the text and the lane type its bits are produced in.
    ///
    /// `hint` steers the type-transparent transports (`mov`, `movc`):
    /// they produce whatever lane the consumer wants, so integer bits
    /// never round-trip float-typed text.
    fn alu(
        &self,
        alu: &AluOp,
        operands: &[ExpressionId],
        lanes: &[u8],
        hint: Lane,
    ) -> Result<(String, Lane), Unsupported> {
        let opcode = alu.opcode;
        // Saturate and the float source modifiers force float meaning
        // even on a transport.
        let modified = alu.saturate || alu.sources.iter().any(|source| source.negate || source.abs);
        let transport = if modified { Lane::Float } else { hint };
        let mut produced = producer_lane(opcode);
        let source = |position: usize| {
            self.aligned_source(alu, operands, position, lanes, operand_lane(opcode))
        };
        let vector_source = |position: usize, count: usize| {
            let full: Vec<u8> = (0..u8::try_from(count).unwrap_or(4)).collect();
            self.aligned_source(alu, operands, position, &full, operand_lane(opcode))
        };
        let mut text = match opcode {
            Opcode::Mov => {
                produced = transport;
                self.aligned_source(alu, operands, 0, lanes, transport)?
            }
            Opcode::Movc => {
                produced = transport;
                let condition = self.select_condition(alu, operands, lanes)?;
                let left = self.aligned_source(alu, operands, 1, lanes, transport)?;
                let right = self.aligned_source(alu, operands, 2, lanes, transport)?;
                format!("({condition}) ? {left} : {right}")
            }
            Opcode::Add | Opcode::Iadd => format!("{} + {}", source(0)?, source(1)?),
            Opcode::Mul => format!("{} * {}", source(0)?, source(1)?),
            Opcode::Div => format!("{} / {}", source(0)?, source(1)?),
            Opcode::Mad | Opcode::IMad | Opcode::UMad => {
                format!("{} * {} + {}", source(0)?, source(1)?, source(2)?)
            }
            Opcode::Min | Opcode::IMin | Opcode::UMin => {
                format!("min({}, {})", source(0)?, source(1)?)
            }
            Opcode::Max | Opcode::IMax | Opcode::UMax => {
                format!("max({}, {})", source(0)?, source(1)?)
            }
            Opcode::Frc => format!("frac({})", source(0)?),
            Opcode::Sqrt => format!("sqrt({})", source(0)?),
            Opcode::Rsq => format!("rsqrt({})", source(0)?),
            Opcode::Exp => format!("exp2({})", source(0)?),
            Opcode::Log => format!("log2({})", source(0)?),
            Opcode::RoundNe => format!("round({})", source(0)?),
            Opcode::RoundNi => format!("floor({})", source(0)?),
            Opcode::RoundPi => format!("ceil({})", source(0)?),
            Opcode::RoundZ => format!("trunc({})", source(0)?),
            Opcode::Dp2 => format!("dot({}, {})", vector_source(0, 2)?, vector_source(1, 2)?),
            Opcode::Dp3 => format!("dot({}, {})", vector_source(0, 3)?, vector_source(1, 3)?),
            Opcode::Dp4 => format!("dot({}, {})", vector_source(0, 4)?, vector_source(1, 4)?),
            Opcode::INeg => format!("-{}", source(0)?),
            Opcode::And => format!("{} & {}", source(0)?, source(1)?),
            Opcode::Or => format!("{} | {}", source(0)?, source(1)?),
            Opcode::Xor => format!("{} ^ {}", source(0)?, source(1)?),
            Opcode::Not => format!("~{}", source(0)?),
            Opcode::Ishl => format!("{} << {}", source(0)?, source(1)?),
            Opcode::Ishr | Opcode::Ushr => format!("{} >> {}", source(0)?, source(1)?),
            Opcode::Itof | Opcode::Utof => format!("float{}({})", width(lanes), source(0)?),
            Opcode::Ftoi => format!("int{}({})", width(lanes), source(0)?),
            Opcode::Ftou => format!("uint{}({})", width(lanes), source(0)?),
            Opcode::Lt
            | Opcode::Ge
            | Opcode::Eq
            | Opcode::Ne
            | Opcode::ILt
            | Opcode::IGe
            | Opcode::IEq
            | Opcode::INe
            | Opcode::ULt
            | Opcode::UGe => self.materialized_comparison(alu, operands, lanes)?,
            Opcode::Sincos => {
                // Operand slot 0 is sin, slot 1 cos; the surviving
                // destination selects the meaning.
                let intrinsic = if alu.dest_position == 1 { "cos" } else { "sin" };
                format!("{intrinsic}({})", source(0)?)
            }
            Opcode::Rcp => format!("rcp({})", source(0)?),
            Opcode::DerivRtx => format!("ddx({})", source(0)?),
            Opcode::DerivRty => format!("ddy({})", source(0)?),
            Opcode::DerivRtxCoarse => format!("ddx_coarse({})", source(0)?),
            Opcode::DerivRtxFine => format!("ddx_fine({})", source(0)?),
            Opcode::DerivRtyCoarse => format!("ddy_coarse({})", source(0)?),
            Opcode::DerivRtyFine => format!("ddy_fine({})", source(0)?),
            Opcode::LdStructured => self.structured_load(alu, operands, lanes)?,
            Opcode::UDiv => {
                // Operand slot 0 is the quotient, slot 1 the remainder;
                // the surviving destination selects the meaning.
                let symbol = if alu.dest_position == 1 { "%" } else { "/" };
                format!("{} {symbol} {}", source(0)?, source(1)?)
            }
            Opcode::Ld
            | Opcode::LdMs
            | Opcode::Gather4
            | Opcode::Sample
            | Opcode::SampleL
            | Opcode::SampleB => {
                produced = self.resource_result_lane(alu);
                self.fetch(alu, operands, lanes)?
            }
            other => {
                return Err(Unsupported(format!("opcode {}", other.name())));
            }
        };
        if alu.saturate {
            text = format!("saturate({text})");
        }
        Ok((text, produced))
    }

    /// Dispatches one resource-fetch opcode to its renderer.
    fn fetch(
        &self,
        alu: &AluOp,
        operands: &[ExpressionId],
        lanes: &[u8],
    ) -> Result<String, Unsupported> {
        match alu.opcode {
            Opcode::Ld => self.texel_load(alu, operands, lanes),
            Opcode::LdMs => self.texel_load_ms(alu, operands, lanes),
            Opcode::Gather4 => self.gather(alu, operands, lanes),
            _ => self.sample(alu, operands, lanes),
        }
    }

    /// The lane type of a texel fetch's result: integer-typed resources
    /// produce their declared integer lanes.
    fn resource_result_lane(&self, alu: &AluOp) -> Lane {
        let Some(resource) = alu.bindings.first() else {
            return Lane::Float;
        };
        self.declarations
            .resources
            .iter()
            .find(|(index, ..)| *index == resource.register.index)
            .map_or(Lane::Float, |(_, _, return_type)| match return_type[0] {
                dxbc::shex::ReturnType::Uint => Lane::Uint,
                dxbc::shex::ReturnType::Sint => Lane::Int,
                _ => Lane::Float,
            })
    }

    /// The boolean condition of one `movc`, unwrapping an inlined
    /// comparison to its bare relational form.
    fn select_condition(
        &self,
        alu: &AluOp,
        operands: &[ExpressionId],
        lanes: &[u8],
    ) -> Result<String, Unsupported> {
        if let Some(&tested) = operands.first()
            && let ExpressionKind::Operation {
                operation: Op::Alu(inner),
                operands: inner_operands,
            } = self.expression(tested)?.kind()
            && let Some(symbol) = comparison_symbol(inner.opcode)
        {
            let mapped = map_lanes(&alu.sources[0].components, lanes)?;
            return self.comparison(inner, inner_operands, symbol, &mapped);
        }
        let condition = self.aligned_source(alu, operands, 0, lanes, Lane::Uint)?;
        Ok(format!("{condition} != 0u"))
    }

    /// A comparison kept as a value: the all-ones mask per true lane.
    fn materialized_comparison(
        &self,
        alu: &AluOp,
        operands: &[ExpressionId],
        lanes: &[u8],
    ) -> Result<String, Unsupported> {
        let symbol =
            comparison_symbol(alu.opcode).ok_or_else(|| Unsupported("comparison symbol".into()))?;
        let comparison = self.comparison(alu, operands, symbol, lanes)?;
        let ones = splat("0xffffffffu", lanes.len());
        let zeros = splat("0u", lanes.len());
        Ok(format!("({comparison}) ? {ones} : {zeros}"))
    }

    fn sample(
        &self,
        alu: &AluOp,
        operands: &[ExpressionId],
        lanes: &[u8],
    ) -> Result<String, Unsupported> {
        let [resource, sampler] = alu.bindings.as_slice() else {
            return Err(Unsupported("sample without resource and sampler".into()));
        };
        let dimension = self
            .declarations
            .resources
            .iter()
            .find(|(index, ..)| *index == resource.register.index)
            .map_or("texture2d", |(_, dimension, _)| dimension);
        let coordinates: &[u8] = match dimension {
            "texture1d" => &[0],
            "texture3d" | "texturecube" | "texture2darray" => &[0, 1, 2],
            "texturecubearray" => &[0, 1, 2, 3],
            _ => &[0, 1],
        };
        let coordinate = self.aligned_source(alu, operands, 0, coordinates, Lane::Float)?;
        let texture = format!("t{}", resource.register.index);
        let sampler = format!("s{}", sampler.register.index);
        let offsets = texel_offsets(alu, coordinates.len().min(3));
        let call = match alu.opcode {
            Opcode::SampleL => {
                let level = self.aligned_source(alu, operands, 1, &[0], Lane::Float)?;
                format!("{texture}.SampleLevel({sampler}, {coordinate}, {level}{offsets})")
            }
            Opcode::SampleB => {
                let bias = self.aligned_source(alu, operands, 1, &[0], Lane::Float)?;
                format!("{texture}.SampleBias({sampler}, {coordinate}, {bias}{offsets})")
            }
            _ => format!("{texture}.Sample({sampler}, {coordinate}{offsets})"),
        };
        select_call_lanes(call, &resource.components, lanes)
    }

    /// Renders one `resinfo` as a `GetDimensions` block assigning the
    /// selected dimensions into the destination lanes.
    fn resinfo(
        &mut self,
        name: &str,
        holder: Lane,
        alu: &AluOp,
        operands: &[ExpressionId],
    ) -> Result<(), Unsupported> {
        let [resource] = alu.bindings.as_slice() else {
            return Err(Unsupported("resinfo without a resource".into()));
        };
        let dimension = self
            .declarations
            .resources
            .iter()
            .find(|(index, ..)| *index == resource.register.index)
            .map_or("texture2d", |(_, dimension, _)| dimension);
        let mip = self.aligned_source(alu, operands, 0, &[0], Lane::Uint)?;
        let texture = format!("t{}", resource.register.index);
        // Result layout: [width, height, depth-or-elements, mip count].
        let (arguments, filled): (&str, [bool; 4]) = match dimension {
            "texture1d" => ("rw, rm", [true, false, false, true]),
            "texture3d" | "texture2darray" | "texturecubearray" => {
                ("rw, rh, rd, rm", [true, true, true, true])
            }
            _ => ("rw, rh, rm", [true, true, false, true]),
        };
        let mask = alu.dest.clone().unwrap_or(DestAccess { mask: 0b1111 });
        let lanes = mask_lanes(mask.mask);
        let mapped = map_lanes(&resource.components, &lanes)?;
        let uint = alu.resinfo_return_type == Some(2);
        let produced = if uint { Lane::Uint } else { Lane::Float };
        let scalars = ["rw", "rh", "rd", "rm"];
        let selected: Vec<String> = mapped
            .iter()
            .map(|&lane| {
                let index = usize::from(lane);
                if filled[index] {
                    let scalar = scalars[index];
                    if uint {
                        scalar.to_owned()
                    } else {
                        format!("float({scalar})")
                    }
                } else if uint {
                    "0u".to_owned()
                } else {
                    "0.0".to_owned()
                }
            })
            .collect();
        let constructor = if uint { "uint" } else { "float" };
        let value = if selected.len() == 1 {
            selected.into_iter().next().expect("one lane renders")
        } else {
            format!("{constructor}{}({})", selected.len(), selected.join(", "))
        };
        let value = converted(value, produced, holder);
        self.line("{");
        self.indent += 1;
        self.line("uint rw, rh, rd, rm;");
        self.line("rd = 0; rw = 0; rh = 0; rm = 0;");
        self.line(&format!("{texture}.GetDimensions({mip}, {arguments});"));
        if mask.mask == 0b1111 {
            self.line(&format!("{name} = {value};"));
        } else {
            self.line(&format!("{name}.{} = {value};", suffix(&lanes)));
        }
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    /// Renders one `ld` texel fetch.
    fn texel_load(
        &self,
        alu: &AluOp,
        operands: &[ExpressionId],
        lanes: &[u8],
    ) -> Result<String, Unsupported> {
        let [resource] = alu.bindings.as_slice() else {
            return Err(Unsupported("ld without a resource".into()));
        };
        let dimension = self
            .declarations
            .resources
            .iter()
            .find(|(index, ..)| *index == resource.register.index)
            .map_or("texture2d", |(_, dimension, _)| dimension);
        // Load takes int coordinates plus the mip level in the last lane.
        let coordinates: &[u8] = match dimension {
            "texture1d" => &[0, 3],
            "texture3d" | "texture2darray" => &[0, 1, 2, 3],
            "buffer" => &[0],
            _ => &[0, 1, 3],
        };
        let coordinate = self.aligned_source(alu, operands, 0, coordinates, Lane::Int)?;
        let offsets = texel_offsets(alu, coordinates.len().saturating_sub(1).max(1));
        let call = format!("t{}.Load({coordinate}{offsets})", resource.register.index);
        select_call_lanes(call, &resource.components, lanes)
    }

    /// Renders one `ld_ms` multisampled texel fetch.
    fn texel_load_ms(
        &self,
        alu: &AluOp,
        operands: &[ExpressionId],
        lanes: &[u8],
    ) -> Result<String, Unsupported> {
        let [resource] = alu.bindings.as_slice() else {
            return Err(Unsupported("ld_ms without a resource".into()));
        };
        let coordinate = self.aligned_source(alu, operands, 0, &[0, 1], Lane::Int)?;
        let sample = self.aligned_source(alu, operands, 1, &[0], Lane::Int)?;
        let call = format!("t{}.Load({coordinate}, {sample})", resource.register.index);
        select_call_lanes(call, &resource.components, lanes)
    }

    /// Renders one `gather4`: the sampler operand's swizzle selects the
    /// gathered channel.
    fn gather(
        &self,
        alu: &AluOp,
        operands: &[ExpressionId],
        lanes: &[u8],
    ) -> Result<String, Unsupported> {
        let [resource, sampler] = alu.bindings.as_slice() else {
            return Err(Unsupported("gather4 without resource and sampler".into()));
        };
        let coordinate = self.aligned_source(alu, operands, 0, &[0, 1], Lane::Float)?;
        let channel = match &sampler.components {
            ComponentSelect::Scalar(lane) => *lane,
            ComponentSelect::Swizzle(swizzle) => swizzle[0],
            _ => 0,
        };
        let intrinsic = match channel {
            1 => "GatherGreen",
            2 => "GatherBlue",
            3 => "GatherAlpha",
            _ => "GatherRed",
        };
        let call = format!(
            "t{}.{intrinsic}(s{}, {coordinate})",
            resource.register.index, sampler.register.index
        );
        select_call_lanes(call, &resource.components, lanes)
    }

    /// Renders one `ld_structured` element read: per result lane, one
    /// dword of the element at the byte offset, selected by the resource
    /// swizzle.
    fn structured_load(
        &self,
        alu: &AluOp,
        operands: &[ExpressionId],
        lanes: &[u8],
    ) -> Result<String, Unsupported> {
        let [resource] = alu.bindings.as_slice() else {
            return Err(Unsupported("ld_structured without a resource".into()));
        };
        let element = self.aligned_source(alu, operands, 0, &[0], Lane::Uint)?;
        // The byte offset must be an immediate to name a struct field.
        let Some(&offset_expression) = operands.get(1) else {
            return Err(Unsupported("ld_structured without an offset".into()));
        };
        let offset_bits = match self.expression(offset_expression)?.kind() {
            ExpressionKind::Constant(immediate)
            | ExpressionKind::Operation {
                operation: Op::Const(immediate),
                ..
            } => immediate.bits.first().copied().unwrap_or(0),
            _ => return Err(Unsupported("computed structured offset".into())),
        };
        if offset_bits % 4 != 0 {
            return Err(Unsupported("unaligned structured offset".into()));
        }
        let base = offset_bits / 4;
        let swizzle = match &resource.components {
            ComponentSelect::Swizzle(swizzle) => *swizzle,
            _ => [0, 1, 2, 3],
        };
        let texture = format!("t{}", resource.register.index);
        let rendered: Vec<String> = lanes
            .iter()
            .map(|&lane| {
                let dword = base + u32::from(swizzle[usize::from(lane)]);
                format!("{texture}[{element}].value[{dword}]")
            })
            .collect();
        Ok(if rendered.len() == 1 {
            rendered.into_iter().next().expect("one lane renders")
        } else {
            format!("uint{}({})", rendered.len(), rendered.join(", "))
        })
    }

    /// Renders one addressable-file place expression.
    fn memory(
        &self,
        reference: &MemRef,
        address: Option<ExpressionId>,
    ) -> Result<String, Unsupported> {
        // The element index position depends on the file: `icb` has no
        // register-selecting index.
        let (base, element_position) = match reference.file {
            RegFile::ConstantBuffer => (
                format!("cb{}", reference.indices.first().copied().unwrap_or(0)),
                1,
            ),
            RegFile::IndexableTemp => (
                format!("x{}", reference.indices.first().copied().unwrap_or(0)),
                1,
            ),
            RegFile::ImmediateConstantBuffer => ("icb".to_owned(), 0),
            _ => return Err(Unsupported("addressable file".into())),
        };
        let Some(&offset) = reference.indices.get(element_position) else {
            return Err(Unsupported("unindexed addressable read".into()));
        };
        let index = if reference.relative_to == Some(element_position) {
            let address =
                address.ok_or_else(|| Unsupported("relative reference without address".into()))?;
            let lane = reference.relative_select.as_ref().map_or(0, select_lane);
            let rendered = self.source_value(address, &[lane], Lane::Int, None)?;
            if offset == 0 {
                rendered
            } else {
                format!("{rendered} + {offset}")
            }
        } else if reference.relative_to.is_some() {
            return Err(Unsupported("relative register-selecting index".into()));
        } else {
            offset.to_string()
        };
        let mut place = String::new();
        write!(place, "{base}[{index}]").expect("string writes are infallible");
        Ok(place)
    }
}

/// The optional immediate texel-offset argument of a sample or load.
fn texel_offsets(alu: &AluOp, dimensions: usize) -> String {
    let Some(offsets) = alu.tex_offsets else {
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

/// Applies the destination-lane selection through a resource swizzle to
/// a vector-valued call.
fn select_call_lanes(
    call: String,
    components: &ComponentSelect,
    lanes: &[u8],
) -> Result<String, Unsupported> {
    let mapped = map_lanes(components, lanes)?;
    if mapped == [0, 1, 2, 3] {
        return Ok(call);
    }
    Ok(format!("{call}.{}", suffix(&mapped)))
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

/// The scalar lane a test/select component names.
fn select_lane(components: &ComponentSelect) -> u8 {
    match components {
        ComponentSelect::Scalar(lane) => *lane,
        ComponentSelect::Swizzle(swizzle) => swizzle[0],
        _ => 0,
    }
}

/// Applies a source swizzle to the destination lanes.
fn map_lanes(components: &ComponentSelect, lanes: &[u8]) -> Result<Vec<u8>, Unsupported> {
    match components {
        ComponentSelect::Swizzle(swizzle) => Ok(lanes
            .iter()
            .map(|&lane| swizzle[usize::from(lane)])
            .collect()),
        ComponentSelect::Scalar(lane) => Ok(vec![*lane; lanes.len()]),
        ComponentSelect::OneComponent | ComponentSelect::ZeroComponent => Ok(vec![0]),
        ComponentSelect::Mask(_) => Err(Unsupported("mask on a source operand".into())),
    }
}

fn width(lanes: &[u8]) -> String {
    if lanes.len() == 1 {
        String::new()
    } else {
        lanes.len().to_string()
    }
}

fn splat(value: &str, count: usize) -> String {
    if count == 1 {
        value.to_owned()
    } else {
        format!("uint{count}({})", vec![value; count].join(", "))
    }
}

/// Reinterprets `text` — bits held in `produced` form — for a consumer
/// wanting `wanted`. Bit casts only; no value conversion.
fn converted(text: String, produced: Lane, wanted: Lane) -> String {
    match (produced, wanted) {
        (from, to) if from == to => text,
        (_, Lane::Float) => format!("asfloat({text})"),
        (_, Lane::Int) => format!("asint({text})"),
        (_, Lane::Uint) => format!("asuint({text})"),
    }
}

/// Adapts a nested producer's value to the consumer's lane type. The
/// result always parenthesizes or wraps: a nested composite expression
/// must never rebind under the consumer's operators (`a * b + c ? x :
/// y` would parse as `(a * b + c) ? x : y`).
fn cast_wrapped(text: &str, produced: Lane, wanted: Lane) -> String {
    converted(format!("({text})"), produced, wanted)
}

/// The lane interpretation an opcode imposes on its arithmetic sources.
fn operand_lane(opcode: Opcode) -> Lane {
    match opcode {
        Opcode::Iadd
        | Opcode::IMad
        | Opcode::IMin
        | Opcode::IMax
        | Opcode::INeg
        | Opcode::Ishl
        | Opcode::Ishr
        | Opcode::ILt
        | Opcode::IGe
        | Opcode::IEq
        | Opcode::INe
        | Opcode::Itof
        | Opcode::Ld => Lane::Int,
        Opcode::And
        | Opcode::Or
        | Opcode::Xor
        | Opcode::Not
        | Opcode::UMad
        | Opcode::UMin
        | Opcode::UMax
        | Opcode::Ushr
        | Opcode::ULt
        | Opcode::UGe
        | Opcode::UDiv
        | Opcode::Utof
        | Opcode::LdStructured => Lane::Uint,
        _ => Lane::Float,
    }
}

/// The lane interpretation of an opcode's result.
fn producer_lane(opcode: Opcode) -> Lane {
    match opcode {
        Opcode::Iadd
        | Opcode::IMad
        | Opcode::IMin
        | Opcode::IMax
        | Opcode::INeg
        | Opcode::Ishl
        | Opcode::Ishr
        | Opcode::Ftoi => Lane::Int,
        Opcode::And
        | Opcode::Or
        | Opcode::Xor
        | Opcode::Not
        | Opcode::UMad
        | Opcode::UMin
        | Opcode::UMax
        | Opcode::Ushr
        | Opcode::Ftou
        | Opcode::UDiv
        | Opcode::Lt
        | Opcode::Ge
        | Opcode::Eq
        | Opcode::Ne
        | Opcode::ILt
        | Opcode::IGe
        | Opcode::IEq
        | Opcode::INe
        | Opcode::ULt
        | Opcode::UGe
        | Opcode::LdStructured => Lane::Uint,
        _ => Lane::Float,
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

/// Formats an immediate's selected lanes in the consumer's lane type.
fn literal(immediate: &Immediate, lanes: &[u8], lane_type: Lane) -> String {
    let lane_bits = |lane: u8| -> u32 {
        let index = usize::from(lane);
        if immediate.bits.len() == 1 {
            immediate.bits[0]
        } else {
            immediate.bits.get(index).copied().unwrap_or(0)
        }
    };
    let rendered: Vec<String> = lanes
        .iter()
        .map(|&lane| scalar_literal(lane_bits(lane), lane_type))
        .collect();
    if rendered.len() == 1 {
        rendered.into_iter().next().expect("one lane renders")
    } else {
        let constructor = match lane_type {
            Lane::Float => "float",
            Lane::Int => "int",
            Lane::Uint => "uint",
        };
        format!("{constructor}{}({})", rendered.len(), rendered.join(", "))
    }
}

/// One 32-bit lane as an exact HLSL literal.
fn scalar_literal(bits: u32, lane_type: Lane) -> String {
    match lane_type {
        Lane::Uint => format!("{bits}u"),
        Lane::Int => {
            #[allow(
                clippy::cast_possible_wrap,
                reason = "the wrap is the reinterpretation"
            )]
            let value = i64::from(bits as i32);
            value.to_string()
        }
        Lane::Float => {
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
    }
}
