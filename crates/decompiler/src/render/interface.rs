//! Interface rendering: headers, signatures, and semantic names.

use std::fmt::Write as _;

use crate::dialect::{RegFile, Register, SystemInput};
use crate::lift::Unsupported;
use cfglib::ir::hlil::ExpressionId;
use cfglib::ir::rtl::ScalarType;

use super::{
    Expr, LANES, PatchRow, Rendered, Renderer, Web, interpolation_keywords, patch_field_name,
    patch_groups, scalar_keyword, semantic_parts,
};

impl Renderer<'_> {
    pub(super) fn header(&mut self) {
        self.line("// Decompiled by d3dasm-decompiler.");
        for (index, size) in &self.declarations.constant_buffers {
            let naming = self.resources.constant_buffers.get(index).cloned();
            let block = naming
                .as_ref()
                .map_or_else(|| format!("cb{index}_buffer"), |n| n.block.clone());
            self.line(&format!("cbuffer {block} : register(b{index})"));
            self.line("{");
            if let Some(fields) = naming.and_then(|n| n.fields) {
                for field in fields {
                    let name = &field.name;
                    let register = field.register;
                    if field.span == 1 {
                        self.line(&format!("    float4 {name} : packoffset(c{register});"));
                    } else {
                        let span = field.span;
                        self.line(&format!(
                            "    float4 {name}[{span}] : packoffset(c{register});"
                        ));
                    }
                }
            } else {
                self.line(&format!("    float4 cb{index}[{size}];"));
            }
            self.line("};");
        }
        for (index, _mode) in &self.declarations.samplers {
            let name = self.sampler_name(*index);
            self.line(&format!("SamplerState {name} : register(s{index});"));
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
            let name = self.texture_name(*index);
            self.line(&format!(
                "{texture_type}<{element}> {name} : register(t{index});"
            ));
        }
        for (index, dimension, return_type) in &self.declarations.uavs {
            let uav_type = match *dimension {
                "buffer" => "RWBuffer",
                "texture1d" => "RWTexture1D",
                "texture3d" => "RWTexture3D",
                "texture2darray" => "RWTexture2DArray",
                _ => "RWTexture2D",
            };
            let element = match return_type[0] {
                dxbc::shex::ReturnType::Uint => "uint4",
                dxbc::shex::ReturnType::Sint => "int4",
                _ => "float4",
            };
            let name = self.uav_name(*index);
            self.line(&format!(
                "{uav_type}<{element}> {name} : register(u{index});"
            ));
        }
        for (index, stride) in &self.declarations.structured {
            let dwords = stride.div_ceil(4);
            let name = self.texture_name(*index);
            self.line(&format!(
                "struct {name}_element {{ uint value[{dwords}]; }};"
            ));
            self.line(&format!(
                "StructuredBuffer<{name}_element> {name} : register(t{index});"
            ));
        }
        if !self.declarations.constant_buffers.is_empty()
            || !self.declarations.samplers.is_empty()
            || !self.declarations.resources.is_empty()
            || !self.declarations.structured.is_empty()
            || !self.declarations.uavs.is_empty()
        {
            self.line("");
        }
    }

    /// The rendered name of one `t#` binding.
    pub(super) fn texture_name(&self, index: u32) -> String {
        self.resources
            .textures
            .get(&index)
            .cloned()
            .unwrap_or_else(|| format!("t{index}"))
    }

    /// Emits the hull-shader signature: interface structs, the stage
    /// attributes, and the control-point entry.
    pub(super) fn hull_signature(&mut self) {
        let tessellation = self.declarations.tessellation.clone().unwrap_or_default();
        self.line("struct HsInput");
        self.line("{");
        for (register, row) in &self.semantics.inputs.clone() {
            let declared = if row.integral { "uint4" } else { "float4" };
            let text = &row.text;
            self.line(&format!("    {declared} v{register} : {text};"));
        }
        self.line("};");
        self.line("struct HsControlPoint");
        self.line("{");
        for (register, _, system_value) in &self.declarations.outputs.clone() {
            let semantic = self.semantic(*register, *system_value, true);
            let name = Self::register_name(*register);
            let declared = if self.output_integral(*register) {
                "uint"
            } else {
                "float4"
            };
            self.line(&format!("    {declared} {name} : {semantic};"));
        }
        self.line("};");
        self.line("struct HsConstants");
        self.line("{");
        for group in patch_groups(&self.semantics.patch.clone()) {
            let keyword = if group.integral { "uint" } else { "float" };
            let field = &group.field;
            let semantic = &group.semantic;
            let declared = if group.width > 1 {
                format!("{keyword}{}", group.width)
            } else {
                keyword.to_owned()
            };
            if group.count > 1 {
                let count = group.count;
                self.line(&format!("    {declared} {field}[{count}] : {semantic};"));
            } else {
                self.line(&format!("    {declared} {field} : {semantic};"));
            }
        }
        self.line("};");
        let domain = if tessellation.domain.is_empty() {
            "quad"
        } else {
            tessellation.domain
        };
        self.line(&format!("[domain(\"{domain}\")]"));
        if let Some(partitioning) = tessellation.partitioning {
            self.line(&format!("[partitioning(\"{partitioning}\")]"));
        }
        if let Some(primitive) = tessellation.output_primitive {
            self.line(&format!("[outputtopology(\"{primitive}\")]"));
        }
        self.line(&format!(
            "[outputcontrolpoints({})]",
            tessellation.output_control_points.max(1)
        ));
        if let Some(factor) = tessellation.max_factor {
            self.line(&format!("[maxtessfactor({factor:.1})]"));
        }
        self.line("[patchconstantfunc(\"constants\")]");
        let count = tessellation.input_control_points.max(1);
        self.line(&format!(
            "HsControlPoint main(InputPatch<HsInput, {count}> patch, \
             uint vOutputControlPointID_raw : SV_OutputControlPointID)"
        ));
        self.input_preambles.push(
            "uint4 vOutputControlPointID = uint4(vOutputControlPointID_raw, 0u, 0u, 0u);"
                .to_owned(),
        );
        self.input_preambles
            .push("HsControlPoint _cp = (HsControlPoint)0;".to_owned());
    }

    /// Emits the domain-shader signature: domain attribute, domain
    /// location, patch constants, and the control-point patch.
    pub(super) fn domain_signature(&mut self) {
        let tessellation = self.declarations.tessellation.clone().unwrap_or_default();
        let domain = if tessellation.domain.is_empty() {
            "quad"
        } else {
            tessellation.domain
        };
        self.line("struct DsControlPoint");
        self.line("{");
        // The control-point fields come from the input signature: the
        // shader may leave them undeclared (unread), but the patch type
        // must still carry the rows.
        for (register, row) in &self.semantics.inputs.clone() {
            let declared = if row.integral { "uint4" } else { "float4" };
            let text = &row.text;
            self.line(&format!("    {declared} v{register} : {text};"));
        }
        self.line("};");
        self.line("struct DsPatch");
        self.line("{");
        for row in &self.semantics.patch.clone() {
            let name = patch_field_name(row);
            let width = row.mask.count_ones().clamp(1, 4);
            let keyword = if row.integral { "uint" } else { "float" };
            let declared = if width == 1 {
                keyword.to_owned()
            } else {
                format!("{keyword}{width}")
            };
            let semantic = &row.text;
            self.line(&format!("    {declared} {name} : {semantic};"));
        }
        self.line("};");
        let location = match domain {
            "tri" => "float3",
            _ => "float2",
        };
        self.line(&format!("[domain(\"{domain}\")]"));
        let count = tessellation.input_control_points.max(1);
        let signature = format!(
            "void main({location} vDomain_raw : SV_DomainLocation, DsPatch constants, \
             const OutputPatch<DsControlPoint, {count}> patch{})",
            self.output_parameters()
        );
        self.line(&signature);
        let filler = if location == "float3" {
            "float4 vDomain = float4(vDomain_raw, 0.0);"
        } else {
            "float4 vDomain = float4(vDomain_raw, 0.0, 0.0);"
        };
        self.input_preambles.push(filler.to_owned());
    }

    /// The `out` parameter list of ordinary output registers.
    pub(super) fn output_parameters(&self) -> String {
        let mut parameters = String::new();
        for (register, _, system_value) in &self.declarations.outputs {
            let semantic = self.semantic(*register, *system_value, true);
            let name = Self::register_name(*register);
            let declared = if register.file == RegFile::OutputDepth {
                "float"
            } else {
                "float4"
            };
            let _ = write!(parameters, ", out precise {declared} {name} : {semantic}");
        }
        parameters
    }

    /// Renders one patch-constant read as struct-field accesses.
    pub(super) fn patch_read(
        &self,
        register: u32,
        web: &Web,
        positions: &[u8],
        wanted: ScalarType,
    ) -> Result<Rendered, Unsupported> {
        let mut parts: Vec<(String, ScalarType)> = Vec::new();
        for &position in positions {
            let lane = web
                .lanes
                .get(usize::from(position))
                .copied()
                .unwrap_or(position);
            let row = self
                .patch_row(register, lane)
                .ok_or_else(|| Unsupported("patch constant outside the signature".into()))?;
            let name = patch_field_name(row);
            let width = row.mask.count_ones().clamp(1, 4);
            let offset = (row.mask & ((1u8 << lane) - 1)).count_ones();
            let holder = if row.integral {
                ScalarType::U32
            } else {
                ScalarType::F32
            };
            let text = if width == 1 {
                format!("constants.{name}")
            } else {
                format!(
                    "constants.{name}.{}",
                    LANES[usize::try_from(offset).unwrap_or(0)]
                )
            };
            parts.push((text, holder));
        }
        if parts.len() == 1 {
            let (text, holder) = parts.remove(0);
            let value = Rendered::atom(text, holder);
            return Ok(Rendered::atom(value.cast(wanted), wanted));
        }
        let rendered: Vec<String> = parts
            .into_iter()
            .map(|(text, holder)| Rendered::atom(text, holder).cast_bare(wanted))
            .collect();
        Ok(Rendered::atom(
            format!(
                "{}{}({})",
                scalar_keyword(wanted),
                rendered.len(),
                rendered.join(", ")
            ),
            wanted,
        ))
    }

    /// Renders one patch-constant store in the hull constants phase.
    pub(super) fn store_patch_constant(
        &mut self,
        register: Register,
        lanes: &[u8],
        dynamic: bool,
        values: &[Expr],
        operands: &[ExpressionId],
    ) -> Result<(), Unsupported> {
        let [lane] = lanes else {
            return Err(Unsupported("vector patch-constant store".into()));
        };
        let row = self
            .patch_row(register.index, *lane)
            .ok_or_else(|| Unsupported("patch constant outside the signature".into()))?
            .clone();
        let wanted = if row.integral {
            ScalarType::U32
        } else {
            ScalarType::F32
        };
        let value = values
            .first()
            .ok_or_else(|| Unsupported("patch store without a value".into()))?;
        let mut cursor = 0usize;
        let rendered = self.value(value, operands, &mut cursor, wanted)?;
        let groups = patch_groups(&self.semantics.patch);
        let (row_name, row_index) = semantic_parts(&row.text);
        let group = groups
            .iter()
            .find(|group| {
                let (group_name, group_index) = semantic_parts(&group.semantic);
                group_name == row_name
                    && row_index >= group_index
                    && row_index < group_index + group.count
            })
            .ok_or_else(|| Unsupported("patch constant outside its groups".into()))?;
        // A vector row selects the written component within its element.
        let component = if group.width > 1 {
            let offset = (row.mask & ((1u8 << lane) - 1)).count_ones();
            format!(".{}", LANES[usize::try_from(offset).unwrap_or(0)])
        } else {
            String::new()
        };
        let place = if dynamic {
            // The computed index is absolute in patch-constant registers.
            let index = values
                .get(1)
                .ok_or_else(|| Unsupported("dynamic store without an index".into()))?;
            let index = self
                .value(index, operands, &mut cursor, ScalarType::U32)?
                .cast(ScalarType::U32);
            let delta = register.index.wrapping_sub(group.first_register);
            let field = &group.field;
            if delta == 0 {
                format!("_patch.{field}[{index}]{component}")
            } else {
                format!("_patch.{field}[{index} + {delta}]{component}")
            }
        } else if group.count > 1 {
            format!(
                "_patch.{}[{}]{component}",
                group.field,
                register.index - group.first_register
            )
        } else {
            let name = &group.field;
            let width = row.mask.count_ones().clamp(1, 4);
            let offset = (row.mask & ((1u8 << lane) - 1)).count_ones();
            if width == 1 {
                format!("_patch.{name}")
            } else {
                format!(
                    "_patch.{name}.{}",
                    LANES[usize::try_from(offset).unwrap_or(0)]
                )
            }
        };
        let text = rendered.cast_bare(wanted);
        self.line(&format!("{place} = {text};"));
        Ok(())
    }

    /// The patch-constant row covering one register lane.
    pub(super) fn patch_row(&self, register: u32, lane: u8) -> Option<&PatchRow> {
        self.semantics
            .patch
            .iter()
            .find(|row| row.register == register && row.mask & (1 << lane) != 0)
    }

    /// Whether one output register's signature row is integer-typed.
    pub(super) fn output_integral(&self, register: Register) -> bool {
        register.file == RegFile::Output
            && self
                .semantics
                .outputs
                .get(&register.index)
                .is_some_and(|row| row.integral)
    }

    /// The rendered name of one `u#` binding.
    pub(super) fn uav_name(&self, index: u32) -> String {
        self.resources
            .uavs
            .get(&index)
            .cloned()
            .unwrap_or_else(|| format!("u{index}"))
    }

    /// The rendered name of one `s#` binding.
    pub(super) fn sampler_name(&self, index: u32) -> String {
        self.resources
            .samplers
            .get(&index)
            .cloned()
            .unwrap_or_else(|| format!("s{index}"))
    }

    /// The register's HLSL spelling.
    pub(super) fn register_name(register: Register) -> String {
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
            RegFile::PatchConstant => format!("vpc{}", register.index),
            RegFile::System(system) => match system {
                SystemInput::ThreadId => "vThreadID",
                SystemInput::ThreadGroupId => "vThreadGroupID",
                SystemInput::ThreadIdInGroup => "vThreadIDInGroup",
                SystemInput::Coverage => "vCoverage",
                SystemInput::PrimitiveId => "vPrim",
                SystemInput::DomainPoint => "vDomain",
                SystemInput::ControlPointId => "vOutputControlPointID",
                SystemInput::ForkInstanceId => "vForkInstanceID",
            }
            .to_owned(),
        }
    }

    /// The semantic string of one declared input or output.
    pub(super) fn semantic(
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
            RegFile::System(SystemInput::ControlPointId) => "SV_OutputControlPointID".to_owned(),
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

    #[expect(clippy::too_many_lines, reason = "one arm per interface row kind")]
    pub(super) fn signature(&mut self) {
        if self.program.shader_type == "ds" {
            return self.domain_signature();
        }
        if self.program.shader_type == "hs" {
            return self.hull_signature();
        }
        let geometry = self.declarations.geometry.clone();
        if let Some(geometry) = &geometry {
            self.line("struct GsInput");
            self.line("{");
            for row in &self.declarations.inputs {
                if row.register.file != RegFile::Input {
                    continue;
                }
                let semantic = self.semantic(row.register, row.system_value, false);
                let name = Self::register_name(row.register);
                self.line(&format!("    float4 {name} : {semantic};"));
            }
            self.line("};");
            self.line("struct GsOutput");
            self.line("{");
            for (register, _, system_value) in &self.declarations.outputs.clone() {
                let semantic = self.semantic(*register, *system_value, true);
                let name = Self::register_name(*register);
                // Integral signature rows (SV_RenderTargetArrayIndex,
                // SV_ViewportArrayIndex) demand integral fields.
                let declared = if self.output_integral(*register) {
                    "uint"
                } else {
                    "float4"
                };
                self.line(&format!("    {declared} {name} : {semantic};"));
            }
            self.line("};");
            let mut parameters = vec![format!(
                "{} GsInput vertices[{}]",
                geometry.primitive, geometry.vertices
            )];
            for row in &self.declarations.inputs {
                if let RegFile::System(_) = row.register.file {
                    let semantic = self.semantic(row.register, row.system_value, false);
                    let name = Self::register_name(row.register);
                    parameters.push(format!("uint {name}_raw : {semantic}"));
                    self.input_preambles
                        .push(format!("uint4 {name} = uint4({name}_raw, 0u, 0u, 0u);"));
                }
            }
            parameters.push("inout TriangleStream<GsOutput> stream0".to_owned());
            self.line(&format!("[maxvertexcount({})]", geometry.max_vertices));
            let mut signature = String::from("void main(");
            signature.push_str(&parameters.join(", "));
            signature.push(')');
            self.line(&signature);
            self.input_preambles
                .push("GsOutput _vertex = (GsOutput)0;".to_owned());
            return;
        }
        let mut parameters = Vec::new();
        for row in &self.declarations.inputs {
            let semantic = self.semantic(row.register, row.system_value, false);
            let interpolation = interpolation_keywords(row.interpolation);
            let name = Self::register_name(row.register);
            // Integer-typed signature rows (SV_VertexID, SV_InstanceID)
            // and system-value registers (thread ids, primitive id) must
            // be integral parameters; a preamble local restores the
            // register's raw-bits view.
            let system_width = match row.register.file {
                RegFile::System(
                    SystemInput::ThreadId
                    | SystemInput::ThreadGroupId
                    | SystemInput::ThreadIdInGroup,
                ) => Some(3usize),
                RegFile::System(
                    SystemInput::PrimitiveId | SystemInput::Coverage | SystemInput::ControlPointId,
                ) => Some(1),
                _ => None,
            };
            let integral = system_width.is_some()
                || (row.register.file == RegFile::Input
                    && self
                        .semantics
                        .inputs
                        .get(&row.register.index)
                        .is_some_and(|row| row.integral));
            if integral {
                let mask = self
                    .semantics
                    .inputs
                    .get(&row.register.index)
                    .map_or(0b0001, |row| row.mask);
                let width = system_width
                    .unwrap_or_else(|| usize::try_from(mask.count_ones()).unwrap_or(4).clamp(1, 4));
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
                format!(
                    "out precise float {} : {semantic}",
                    Self::register_name(*register)
                )
            } else {
                format!(
                    "out precise float4 {} : {semantic}",
                    Self::register_name(*register)
                )
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
}
