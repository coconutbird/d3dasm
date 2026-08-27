//! Statement rendering: assignments, effects, and conditions.

use cfglib::ir::hlil::{ExpressionId, ExpressionKind, StatementId, StatementKind};
use cfglib::ir::rtl::{ResolvedRead, ScalarType, VarExpr};
use dxbc::shex::Opcode;

use crate::dialect::{AluHead, EffectOp, RegFile, RtlOp};
use crate::lift::Unsupported;

use super::*;

impl<'a> Renderer<'a> {
    pub(super) fn statements(&mut self, ids: &[StatementId]) -> Result<(), Unsupported> {
        for &id in ids {
            self.statement(id)?;
        }
        Ok(())
    }

    pub(super) fn statement_kind(
        &self,
        id: StatementId,
    ) -> Result<&'a StatementKind<ShaderDialect>, Unsupported> {
        self.function
            .statement(id)
            .map(cfglib::ir::hlil::Statement::kind)
            .ok_or_else(|| Unsupported("unresolvable statement identity".into()))
    }

    pub(super) fn expression(
        &self,
        id: ExpressionId,
    ) -> Result<&'a cfglib::ir::hlil::Expression<ShaderDialect>, Unsupported> {
        self.function
            .expression(id)
            .ok_or_else(|| Unsupported("unresolvable expression identity".into()))
    }

    pub(super) fn block(&mut self, ids: &[StatementId]) -> Result<(), Unsupported> {
        self.line("{");
        self.indent += 1;
        self.statements(ids)?;
        self.indent -= 1;
        self.line("}");
        Ok(())
    }

    pub(super) fn statement(&mut self, id: StatementId) -> Result<(), Unsupported> {
        match self.statement_kind(id)? {
            StatementKind::Assign { value, .. } => self.assign(*value),
            StatementKind::Expression(expression) => self.effect(*expression),
            StatementKind::If {
                condition,
                then_body,
                else_body,
            } => {
                let condition = self.condition(*condition)?;
                self.line(&format!("if ({condition})"));
                self.block(&then_body.clone())?;
                if !else_body.is_empty() {
                    self.line("else");
                    self.block(&else_body.clone())?;
                }
                Ok(())
            }
            StatementKind::While { condition, body } => {
                let condition = self.condition(*condition)?;
                self.line(&format!("while ({condition})"));
                self.block(&body.clone())
            }
            StatementKind::DoWhile { body, condition } => {
                let body = body.clone();
                self.line("do");
                self.block(&body)?;
                let condition = self.condition(*condition)?;
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
            StatementKind::Return { .. } => {
                match self.hull_mode {
                    HullMode::None => self.line("return;"),
                    HullMode::ControlPoint => self.line("return _cp;"),
                    // The phase bodies concatenate; the single trailing
                    // return is emitted by the orchestrator.
                    HullMode::Constants => {}
                }
                Ok(())
            }
            other => Err(Unsupported(format!("statement {other:?}"))),
        }
    }

    /// Renders one assignment statement.
    pub(super) fn assign(&mut self, value: ExpressionId) -> Result<(), Unsupported> {
        let expression = self.expression(value)?;
        let ExpressionKind::Operation {
            operation,
            operands,
        } = expression.kind()
        else {
            return Err(Unsupported("assignment of a bare value".into()));
        };
        let Op::Assign {
            target,
            positions,
            width,
            value,
            ..
        } = operation
        else {
            return Err(Unsupported("assignment of a non-assign operation".into()));
        };
        let (index, web) = self.mlil_web(*target)?;
        let name = self.web_names[index].clone();
        let scalar = normalize(web.shape.scalar);
        let mut reads = Reads::new(self.function, operands);
        // `resinfo` renders as a statement block, not an expression.
        if let VarExpr::Apply {
            operator: RtlOp::Alu(head),
            operands: inner,
            ..
        } = value
            && head.opcode == Opcode::Resinfo
        {
            let place = Self::place_text(&name, positions, *width);
            return self.resinfo(&place, scalar, head, inner, &mut reads);
        }
        let rendered = self.value(value, &mut reads, scalar)?;
        let place = Self::place_text(&name, positions, *width);
        let text = rendered.cast_bare(scalar);
        self.line(&format!("{place} = {text};"));
        Ok(())
    }

    /// The written-place spelling of one target.
    pub(super) fn place_text(name: &str, positions: &[u8], width: u8) -> String {
        let full: Vec<u8> = (0..width).collect();
        if width <= 1 || positions == full.as_slice() {
            name.to_owned()
        } else {
            format!("{name}.{}", suffix(positions))
        }
    }

    /// An expression standing as a statement: effect operations.
    #[expect(clippy::too_many_lines, reason = "one arm per effect operation")]
    pub(super) fn effect(&mut self, id: ExpressionId) -> Result<(), Unsupported> {
        let expression = self.expression(id)?;
        let ExpressionKind::Operation {
            operation,
            operands,
        } = expression.kind()
        else {
            return Err(Unsupported("value expression as statement".into()));
        };
        let Op::Effect {
            operation: effect,
            operands: values,
            ..
        } = operation
        else {
            return Err(Unsupported("statement of a non-effect operation".into()));
        };
        match effect {
            EffectOp::StoreOutput {
                register,
                lanes,
                dynamic,
            } => {
                if self.hull_mode == HullMode::Constants {
                    return self.store_patch_constant(*register, lanes, *dynamic, values, operands);
                }
                if *dynamic {
                    return Err(Unsupported("dynamic output store outside a hull".into()));
                }
                let value = values
                    .first()
                    .ok_or_else(|| Unsupported("output store without a value".into()))?;
                let mut reads = Reads::new(self.function, operands);
                let integral = (self.declarations.geometry.is_some()
                    || self.hull_mode == HullMode::ControlPoint)
                    && self.output_integral(*register);
                let wanted = if integral {
                    ScalarType::U32
                } else {
                    ScalarType::F32
                };
                let rendered = self.value(value, &mut reads, wanted)?;
                let name = if self.declarations.geometry.is_some() {
                    format!("_vertex.{}", Self::register_name(*register))
                } else if self.hull_mode == HullMode::ControlPoint {
                    format!("_cp.{}", Self::register_name(*register))
                } else {
                    Self::register_name(*register)
                };
                let place = if register.file == RegFile::OutputDepth
                    || integral
                    || lanes.as_slice() == [0, 1, 2, 3]
                {
                    name
                } else {
                    format!("{name}.{}", suffix(lanes))
                };
                let text = rendered.cast_bare(wanted);
                self.line(&format!("{place} = {text};"));
                Ok(())
            }
            EffectOp::StoreMem(reference) => {
                let value = values
                    .first()
                    .ok_or_else(|| Unsupported("store without a value".into()))?;
                let mut reads = Reads::new(self.function, operands);
                let rendered = self.value(value, &mut reads, ScalarType::U32)?;
                let address = values.get(1);
                let place = self.memory(reference, address, &mut reads)?;
                let place = if reference.lanes.len() == 4 {
                    place
                } else {
                    format!("{place}.{}", suffix(&reference.lanes))
                };
                let text = rendered.cast_bare(ScalarType::U32);
                self.line(&format!("{place} = {text};"));
                Ok(())
            }
            EffectOp::StoreUav { register, .. } => {
                let element = self
                    .declarations
                    .uavs
                    .iter()
                    .find(|(index, ..)| *index == register.index)
                    .map_or(ScalarType::F32, |(_, _, kinds)| match kinds[0] {
                        dxbc::shex::ReturnType::Uint => ScalarType::U32,
                        dxbc::shex::ReturnType::Sint => ScalarType::I32,
                        _ => ScalarType::F32,
                    });
                let coordinate = values
                    .first()
                    .ok_or_else(|| Unsupported("uav store without a coordinate".into()))?;
                let value = values
                    .get(1)
                    .ok_or_else(|| Unsupported("uav store without a value".into()))?;
                let mut reads = Reads::new(self.function, operands);
                let coordinate = self
                    .value(coordinate, &mut reads, ScalarType::U32)?
                    .cast_bare(ScalarType::U32);
                let rendered = self.value(value, &mut reads, element)?;
                let name = self.uav_name(register.index);
                let text = rendered.cast_bare(element);
                self.line(&format!("{name}[{coordinate}] = {text};"));
                Ok(())
            }
            EffectOp::Emit => {
                self.line("stream0.Append(_vertex);");
                Ok(())
            }
            EffectOp::Cut => {
                self.line("stream0.RestartStrip();");
                Ok(())
            }
            EffectOp::Discard => {
                let value = values
                    .first()
                    .ok_or_else(|| Unsupported("discard without a condition".into()))?;
                let mut reads = Reads::new(self.function, operands);
                let condition = self.test_text(value, &mut reads)?;
                self.line(&format!("if ({condition}) discard;"));
                Ok(())
            }
            EffectOp::Bare(opcode) => Err(Unsupported(format!("opcode {}", opcode.name()))),
        }
    }

    /// Renders a branch condition.
    pub(super) fn condition(&self, id: ExpressionId) -> Result<String, Unsupported> {
        let expression = self.expression(id)?;
        let ExpressionKind::Operation {
            operation,
            operands,
        } = expression.kind()
        else {
            return Err(Unsupported("condition is not a branch".into()));
        };
        let Op::Branch { condition } = operation else {
            return Err(Unsupported("condition is not a branch".into()));
        };
        let mut reads = Reads::new(self.function, operands);
        self.test_text(condition, &mut reads)
    }

    /// Renders one `Test` application (or a bare tested value) as a
    /// boolean, folding an inlined comparison to its relational form.
    pub(super) fn test_text(
        &self,
        expr: &Expr,
        reads: &mut Reads<'a>,
    ) -> Result<String, Unsupported> {
        if let VarExpr::Apply {
            operator: RtlOp::Test { nonzero },
            operands: inner,
            ..
        } = expr
        {
            let tested = inner
                .first()
                .ok_or_else(|| Unsupported("test without an operand".into()))?;
            // A comparison feeding the test renders as the bare boolean.
            if let Some(comparison) = self.inlined_comparison(tested, reads)? {
                return Ok(if *nonzero {
                    comparison
                } else {
                    format!("!({comparison})")
                });
            }
            let value = self.value(tested, reads, ScalarType::U32)?;
            let text = value.cast(ScalarType::U32);
            return Ok(if *nonzero {
                format!("{text} != 0u")
            } else {
                format!("{text} == 0u")
            });
        }
        let value = self.value(expr, reads, ScalarType::U32)?;
        Ok(format!("{} != 0u", value.cast(ScalarType::U32)))
    }

    /// When a tested value is an inlined comparison producer read in the
    /// producer's own lane order, renders the bare relational text
    /// (consuming the read). A reordered or partial read must go through
    /// the generic value path so the position remap applies.
    pub(super) fn inlined_comparison(
        &self,
        expr: &Expr,
        reads: &mut Reads<'a>,
    ) -> Result<Option<String>, Unsupported> {
        let VarExpr::Read { positions, .. } = expr else {
            // A directly nested comparison application.
            if let VarExpr::Apply {
                operator: RtlOp::Alu(head),
                operands: inner,
                ..
            } = expr
                && comparison_symbol(head.opcode).is_some()
            {
                let text = self.comparison(head, inner, reads)?;
                return Ok(Some(text));
            }
            return Ok(None);
        };
        // Probe on a clone: the read is consumed only when it resolves
        // to a comparison producer read in the producer's own lane order
        // (a remapped read must go through the generic value path).
        let mut probe = reads.clone();
        let resolved = probe
            .resolve(positions)
            .map_err(|error| Unsupported(error.to_string()))?;
        let ResolvedRead::Inlined {
            value:
                VarExpr::Apply {
                    operator: RtlOp::Alu(head),
                    operands: inner,
                    ..
                },
            operands: producer_operands,
            remap: None,
        } = resolved
        else {
            return Ok(None);
        };
        if comparison_symbol(head.opcode).is_none() {
            return Ok(None);
        }
        *reads = probe;
        let mut inner_reads = reads.nested(producer_operands);
        let text = self.comparison(head, inner, &mut inner_reads)?;
        Ok(Some(text))
    }

    /// Renders one comparison application as relational text.
    pub(super) fn comparison(
        &self,
        head: &AluHead,
        inner: &[Expr],
        reads: &mut Reads<'a>,
    ) -> Result<String, Unsupported> {
        let symbol = comparison_symbol(head.opcode)
            .ok_or_else(|| Unsupported("comparison symbol".into()))?;
        let left = self
            .value(
                inner
                    .first()
                    .ok_or_else(|| Unsupported("comparison operand".into()))?,
                reads,
                ScalarType::F32,
            )?
            .wrapped();
        let right = self
            .value(
                inner
                    .get(1)
                    .ok_or_else(|| Unsupported("comparison operand".into()))?,
                reads,
                ScalarType::F32,
            )?
            .wrapped();
        Ok(format!("{left} {symbol} {right}"))
    }
}
