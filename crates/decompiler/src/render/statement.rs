//! Statement rendering: assignments, effects, and conditions.

use cfglib::ir::hlil::{ExpressionId, ExpressionKind, StatementId, StatementKind};
use cfglib::ir::rtl::{ResolvedRead, ScalarType, VarExpr};

use crate::dialect::{AluHead, EffectOp, RegFile, RtlOp, resinfo_application};
use crate::lift::Unsupported;

use super::*;

/// One parsed assignment whose expression can be rendered in a statement or
/// a `for` clause.
struct Assignment<'a> {
    place: String,
    scalar: ScalarType,
    written: cfglib::ir::hlil::VariableId,
    positions: &'a [u8],
    value: &'a Expr,
    operands: &'a [ExpressionId],
}

impl<'a> Renderer<'a> {
    pub(super) fn statements(&mut self, ids: &[StatementId]) -> Result<(), Unsupported> {
        for &id in ids {
            self.web_declarations_before(id)?;
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
            StatementKind::Assign { target, value } => self.assign(*target, *value),
            StatementKind::Expression(expression) => self.effect(*expression),
            StatementKind::If {
                condition,
                then_body,
                else_body,
            } => {
                let condition = self.condition(*condition)?;
                self.line(&format!("if ({condition})"));
                self.block(then_body)?;
                if !else_body.is_empty() {
                    self.line("else");
                    self.block(else_body)?;
                }
                Ok(())
            }
            StatementKind::While { condition, body } => {
                let condition = self.condition(*condition)?;
                self.line(&format!("while ({condition})"));
                self.block(body)
            }
            StatementKind::DoWhile { body, condition } => {
                self.line("do");
                self.block(body)?;
                let condition = self.condition(*condition)?;
                let last = self.out.pop();
                debug_assert_eq!(last, Some('\n'));
                writeln!(self.out, " while ({condition});").expect("string writes are infallible");
                Ok(())
            }
            StatementKind::Loop { body } => {
                self.line("while (true)");
                self.block(body)
            }
            StatementKind::For {
                initializer,
                condition,
                update,
                body,
            } => {
                let initializer = self.for_clause(initializer)?;
                let condition = condition
                    .map(|condition| self.condition(condition))
                    .transpose()?
                    .unwrap_or_default();
                let update = self.for_clause(update)?;
                self.line(&format!("for ({initializer}; {condition}; {update})"));
                self.block(body)
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
    pub(super) fn assign(
        &mut self,
        target: ExpressionId,
        value: ExpressionId,
    ) -> Result<(), Unsupported> {
        if let Some(text) = self.assignment_expression_text(target, value)? {
            self.line(&format!("{text};"));
            return Ok(());
        }
        let assignment = self.assignment(target, value)?;
        let (head, inner) = resinfo_application(assignment.value)
            .ok_or_else(|| Unsupported("non-expression assignment".into()))?;
        let mut reads = Reads::new(self.function, assignment.operands);
        self.resinfo(
            &assignment.place,
            assignment.scalar,
            head,
            inner,
            &mut reads,
        )
    }

    /// Renders any single-expression HLIL assignment, including universal
    /// constant and variable expressions.
    fn assignment_expression_text(
        &self,
        target: ExpressionId,
        value: ExpressionId,
    ) -> Result<Option<String>, Unsupported> {
        let expression = self.expression(value)?;
        match expression.kind() {
            ExpressionKind::Constant(bits) => self
                .constant_assignment_text(target, bits, *expression.value_type())
                .map(Some),
            ExpressionKind::Variable(variable) => self
                .variable_assignment_text(target, *variable, *expression.value_type())
                .map(Some),
            ExpressionKind::Operation { .. } => {
                let assignment = self.assignment(target, value)?;
                self.assignment_text(&assignment)
            }
        }
    }

    /// Renders a generic HLIL constant assignment produced by direct or
    /// materialized MLIL constants.
    fn constant_assignment_text(
        &self,
        target: ExpressionId,
        bits: &[u64],
        shape: ValueShape,
    ) -> Result<String, Unsupported> {
        let ExpressionKind::Variable(written) = self.expression(target)?.kind() else {
            return Err(Unsupported(
                "constant assignment into a non-variable place".into(),
            ));
        };
        let (index, web) = self.hlil_web(*written)?;
        let holder = normalize(web.shape.scalar);
        let produced = if shape.scalar == ScalarType::Bits {
            holder
        } else {
            normalize(shape.scalar)
        };
        let value = Rendered::atom(literal(bits, produced), produced).cast_bare(holder);
        let name = self.web_name(index)?;
        Ok(format!("{name} = {value}"))
    }

    /// Renders a generic HLIL copy assignment.
    fn variable_assignment_text(
        &self,
        target: ExpressionId,
        source: cfglib::ir::hlil::VariableId,
        shape: ValueShape,
    ) -> Result<String, Unsupported> {
        let ExpressionKind::Variable(written) = self.expression(target)?.kind() else {
            return Err(Unsupported(
                "copy assignment into a non-variable place".into(),
            ));
        };
        let (index, web) = self.hlil_web(*written)?;
        let holder = normalize(web.shape.scalar);
        let wanted = if shape.scalar == ScalarType::Bits {
            holder
        } else {
            normalize(shape.scalar)
        };
        let positions: Vec<u8> = (0..shape.lanes).collect();
        let value = self
            .variable_read(source, &positions, wanted)?
            .cast_bare(holder);
        let name = self.web_name(index)?;
        Ok(format!("{name} = {value}"))
    }

    /// Parses one HLIL assignment into its HLSL place and retained value.
    fn assignment(
        &self,
        target: ExpressionId,
        value: ExpressionId,
    ) -> Result<Assignment<'a>, Unsupported> {
        let ExpressionKind::Variable(written) = self.expression(target)?.kind() else {
            return Err(Unsupported("assignment into a non-variable place".into()));
        };
        let written = *written;
        let expression = self.expression(value)?;
        let ExpressionKind::Operation {
            operation,
            operands,
        } = expression.kind()
        else {
            return Err(Unsupported("assignment of a bare value".into()));
        };
        let Op::Assign {
            positions,
            width,
            value,
            ..
        } = operation
        else {
            return Err(Unsupported("assignment of a non-assign operation".into()));
        };
        let (index, web) = self.hlil_web(written)?;
        let name = self.web_name(index)?.to_owned();
        let scalar = normalize(web.shape.scalar);
        let place = Self::place_text(&name, positions, *width);
        Ok(Assignment {
            place,
            scalar,
            written,
            positions,
            value,
            operands,
        })
    }

    /// Renders an assignment as one expression, or returns `None` when the
    /// retained operation requires a statement block.
    fn assignment_text(&self, assignment: &Assignment<'a>) -> Result<Option<String>, Unsupported> {
        if resinfo_application(assignment.value).is_some() {
            return Ok(None);
        }
        if let Some(text) = self.compound_assignment_text(assignment)? {
            return Ok(Some(text));
        }
        let mut reads = Reads::new(self.function, assignment.operands);
        let rendered = self.value(assignment.value, &mut reads, assignment.scalar)?;
        let text = rendered.cast_bare(assignment.scalar);
        Ok(Some(format!("{} = {text}", assignment.place)))
    }

    /// Renders `place op= rhs` when the assignment is structurally
    /// compound: its value is a binary operation with a compound spelling
    /// whose first operand reads back exactly the written place — the
    /// target variable at the written positions, in the interpretation it
    /// is stored under, with no cast on the read or on the produced value.
    /// The decision never inspects rendered text.
    fn compound_assignment_text(
        &self,
        assignment: &Assignment<'a>,
    ) -> Result<Option<String>, Unsupported> {
        let VarExpr::Apply {
            operator: RtlOp::Alu(head),
            operands: inner,
            shape,
        } = assignment.value
        else {
            return Ok(None);
        };
        let Some(compound) = compound_operator(head) else {
            return Ok(None);
        };
        let [place_operand, value_operand] = inner.as_slice() else {
            return Ok(None);
        };
        let VarExpr::Read { positions, scalar } = place_operand else {
            return Ok(None);
        };
        if positions.as_slice() != assignment.positions {
            return Ok(None);
        }
        // The compound spelling stores the produced value back uncast and
        // reads the place in its stored interpretation: a mismatch on
        // either side means the plain spelling's casts are load-bearing.
        let produced = if shape.scalar == ScalarType::Bits {
            assignment.scalar
        } else {
            shape.scalar
        };
        if normalize(produced) != normalize(assignment.scalar) {
            return Ok(None);
        }
        // Binary operands render in the float context (`Renderer::alu`);
        // a `Bits` read takes that context as its interpretation.
        let wanted = if *scalar == ScalarType::Bits {
            ScalarType::F32
        } else {
            *scalar
        };
        let mut reads = Reads::new(self.function, assignment.operands);
        let resolved = reads
            .resolve(positions)
            .map_err(|error| Unsupported(error.to_string()))?;
        let ResolvedRead::Variable(variable) = resolved else {
            return Ok(None);
        };
        if variable != assignment.written {
            return Ok(None);
        }
        let (_, web) = self.hlil_web(variable)?;
        if Self::is_input_web(web) || normalize(self.web_holder(web)) != normalize(wanted) {
            return Ok(None);
        }
        let rhs = self
            .value(value_operand, &mut reads, ScalarType::F32)?
            .wrapped();
        Ok(Some(format!("{} {compound} {rhs}", assignment.place)))
    }

    /// Renders the comma-separated assignment expressions of one `for`
    /// initializer or update clause.
    fn for_clause(&self, statements: &[StatementId]) -> Result<String, Unsupported> {
        statements
            .iter()
            .map(|&statement| {
                let StatementKind::Assign { target, value } = self.statement_kind(statement)?
                else {
                    return Err(Unsupported("non-assignment in for clause".into()));
                };
                self.assignment_expression_text(*target, *value)?
                    .ok_or_else(|| Unsupported("multi-statement operation in for clause".into()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| parts.join(", "))
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
            let polarity = if *nonzero {
                ComparisonPolarity::Direct
            } else {
                ComparisonPolarity::Negated
            };
            if let Some(comparison) = self.inlined_comparison(tested, reads, polarity)? {
                return Ok(comparison);
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
        polarity: ComparisonPolarity,
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
                let text = self.comparison(head, inner, reads, polarity)?;
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
        let text = self.comparison(head, inner, &mut inner_reads, polarity)?;
        Ok(Some(text))
    }

    /// Renders one comparison application as relational text.
    pub(super) fn comparison(
        &self,
        head: &AluHead,
        inner: &[Expr],
        reads: &mut Reads<'a>,
        polarity: ComparisonPolarity,
    ) -> Result<String, Unsupported> {
        let direct = comparison_symbol(head.opcode)
            .ok_or_else(|| Unsupported("comparison symbol".into()))?;
        let inverted = negated_comparison_symbol(head.opcode);
        let symbol = match polarity {
            ComparisonPolarity::Direct => direct,
            ComparisonPolarity::Negated => inverted.unwrap_or(direct),
        };
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
        let comparison = format!("{left} {symbol} {right}");
        if polarity == ComparisonPolarity::Negated && inverted.is_none() {
            Ok(format!("!({comparison})"))
        } else {
            Ok(comparison)
        }
    }
}

/// The compound-assignment spelling of one retained binary operation,
/// preserving its operand order.
fn compound_operator(head: &AluHead) -> Option<&'static str> {
    if head.saturate {
        return None;
    }
    match head.opcode {
        Opcode::Add | Opcode::Iadd => Some("+="),
        Opcode::Mul => Some("*="),
        Opcode::Div => Some("/="),
        Opcode::And => Some("&="),
        Opcode::Or => Some("|="),
        Opcode::Xor => Some("^="),
        Opcode::Ishl => Some("<<="),
        Opcode::Ishr | Opcode::Ushr => Some(">>="),
        _ => None,
    }
}
