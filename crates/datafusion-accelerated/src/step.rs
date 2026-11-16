// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A program's step, in SQL or as a DataFusion plan, and the rewrites
//! `lower` makes to a step's plan.

use std::sync::Arc;

use datafusion::catalog::TableProvider;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, Result, ScalarValue, Spans, TableReference};
use datafusion::dataframe::DataFrame;
use datafusion::datasource::provider_as_source;
use datafusion::logical_expr::expr::Placeholder;
use datafusion::logical_expr::expr_rewriter::NamePreserver;
use datafusion::logical_expr::{
    Expr, LogicalPlan, LogicalPlanBuilder, Subquery, TableScanBuilder, TableSource,
};

use crate::Error;
use crate::lowered_program::Params;

/// One step of a [`Program`](crate::Program): a `SELECT` in SQL, or a
/// DataFusion `LogicalPlan`, from a `DataFrame`, a plan built by hand, or a
/// Substrait plan read by datafusion-substrait (design §5.4).
///
/// A step reads the program's tables, its state tables, inputs and other
/// steps, by name. In SQL, that's a table's name. In a plan, it's a
/// `TableScan` of that name, whose source can hold just a schema, as
/// DataFusion's `LogicalTableSource` does, since the program's tables don't
/// exist until `lower`. A parameter is `$name` in SQL, and a placeholder,
/// `placeholder("$name")`, in a plan.
#[derive(Clone, Debug)]
pub struct Step(pub(crate) StepKind);

#[derive(Clone, Debug)]
pub(crate) enum StepKind {
    Sql(String),
    Plan(LogicalPlan),
}

impl Step {
    /// A step in SQL: one `SELECT`.
    #[must_use]
    pub fn sql(sql: impl Into<String>) -> Self {
        Self(StepKind::Sql(sql.into()))
    }

    /// A step as a DataFusion plan, unoptimized or not.
    #[must_use]
    pub fn plan(plan: LogicalPlan) -> Self {
        Self(StepKind::Plan(plan))
    }
}

impl From<&str> for Step {
    fn from(sql: &str) -> Self {
        Self::sql(sql)
    }
}

impl From<&String> for Step {
    fn from(sql: &String) -> Self {
        Self::sql(sql.as_str())
    }
}

impl From<String> for Step {
    fn from(sql: String) -> Self {
        Self::sql(sql)
    }
}

impl From<LogicalPlan> for Step {
    fn from(plan: LogicalPlan) -> Self {
        Self::plan(plan)
    }
}

impl From<DataFrame> for Step {
    fn from(frame: DataFrame) -> Self {
        Self::plan(frame.into_unoptimized_plan())
    }
}

/// The tables `plan` scans, in it and in its subqueries, by name, with
/// each scan's source.
pub(crate) fn scans(plan: &LogicalPlan) -> Result<Vec<(TableReference, Arc<dyn TableSource>)>> {
    let mut found = Vec::new();
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            found.push((scan.table_name.clone(), Arc::clone(&scan.source)));
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok(found)
}

/// `plan`, with each scan that `table` gives a provider for reading that
/// provider, and every schema above recomputed. Fails with
/// `Error::ScanSchema` on a scan whose fields aren't the provider's, by
/// name and type.
pub(crate) fn bind_scans(
    plan: LogicalPlan,
    step: &str,
    table: impl Fn(&TableReference) -> Option<Arc<dyn TableProvider>>,
) -> Result<LogicalPlan> {
    plan.transform_up_with_subqueries(|node| {
        let LogicalPlan::TableScan(scan) = node else {
            return node.recompute_schema().map(Transformed::yes);
        };
        let Some(provider) = table(&scan.table_name) else {
            return Ok(Transformed::no(LogicalPlan::TableScan(scan)));
        };
        let (expected, found) = (provider.schema(), scan.source.schema());
        if !crate::program::fits(&expected, &found) {
            return Err(Error::ScanSchema {
                step: step.to_string(),
                table: scan.table_name.to_quoted_string(),
                expected,
                found,
            }
            .into());
        }
        let bound = TableScanBuilder::new(scan.table_name, provider_as_source(provider))
            .with_projection(scan.projection)
            .with_filters(scan.filters)
            .with_fetch(scan.fetch)
            .with_statistics_requests(scan.statistics_requests)
            .build()?;
        Ok(Transformed::yes(LogicalPlan::TableScan(bound)))
    })
    .map(|bound| bound.data)
}

/// The names of the parameters `plan` has, without their `$`.
pub(crate) fn parameter_names(plan: &LogicalPlan) -> Result<Vec<String>> {
    let mut names: Vec<String> = plan
        .get_parameter_names()?
        .into_iter()
        .map(|id| id.trim_start_matches('$').to_string())
        .collect();
    names.sort();
    Ok(names)
}

/// Where in `plan` a parameter is where planning needs its value, which a
/// scalar subquery can't give: a `Values` node, whose subqueries DataFusion
/// 55 doesn't run (#48), or a `Limit`'s fetch or skip, which must be
/// literals.
pub(crate) fn planned_position(plan: &LogicalPlan) -> Result<Option<&'static str>> {
    let mut found = None;
    plan.apply_with_subqueries(|node| {
        found = match node {
            LogicalPlan::Values(values) => values
                .values
                .iter()
                .flatten()
                .any(has_placeholder)
                .then_some("VALUES"),
            LogicalPlan::Limit(limit) if limit.fetch.as_deref().is_some_and(has_placeholder) => {
                Some("LIMIT")
            }
            LogicalPlan::Limit(limit) if limit.skip.as_deref().is_some_and(has_placeholder) => {
                Some("OFFSET")
            }
            _ => None,
        };
        Ok(if found.is_some() {
            TreeNodeRecursion::Stop
        } else {
            TreeNodeRecursion::Continue
        })
    })?;
    Ok(found)
}

fn has_placeholder(expr: &Expr) -> bool {
    expr.exists(|expr| Ok(matches!(expr, Expr::Placeholder(_))))
        .unwrap_or(false)
}

/// `plan`, with each parameter replaced by a scalar subquery of the
/// parameters' one-row table, so that each run reads its new value without
/// planning again (supplement S3.2).
pub(crate) fn read_params(plan: LogicalPlan, params: &Params) -> Result<LogicalPlan> {
    let table = params.handle.name().to_string();
    let source = provider_as_source(Arc::new(crate::handle_table::HandleTable::new(
        params.handle.clone(),
    )));
    replace_placeholders(plan, |name| {
        let column = Column::new(Some(TableReference::bare(table.as_str())), name);
        let read = LogicalPlanBuilder::scan(
            TableReference::bare(table.as_str()),
            Arc::clone(&source),
            None,
        )?
        .project(vec![Expr::Column(column)])?
        .build()?;
        Ok(Expr::ScalarSubquery(Subquery {
            subquery: Arc::new(read),
            outer_ref_columns: vec![],
            spans: Spans::new(),
        }))
    })
}

/// `plan`, with each parameter replaced by its initial value, as a
/// literal.
pub(crate) fn with_literals(plan: LogicalPlan, params: &Params) -> Result<LogicalPlan> {
    replace_placeholders(plan, |name| {
        let value = params
            .values
            .iter()
            .find(|(param, _)| param == name)
            .map(|(_, value)| value.clone());
        Ok(Expr::Literal(value.unwrap_or(ScalarValue::Null), None))
    })
}

/// `plan`, with each placeholder replaced by `value` of its name, without
/// its `$`, as DataFusion's `LogicalPlan::with_param_values` replaces them
/// with literals: each expression keeps its name, and every schema is
/// recomputed.
fn replace_placeholders(
    plan: LogicalPlan,
    value: impl Fn(&str) -> Result<Expr>,
) -> Result<LogicalPlan> {
    plan.transform_up_with_subqueries(|node| {
        let names = NamePreserver::new(&node);
        node.map_expressions(|expr| {
            if !has_placeholder(&expr) {
                return Ok(Transformed::no(expr));
            }
            let name = names.save(&expr);
            let replaced = expr.transform_up(|expr| match expr {
                Expr::Placeholder(Placeholder { id, .. }) => {
                    value(id.trim_start_matches('$')).map(Transformed::yes)
                }
                expr => Ok(Transformed::no(expr)),
            })?;
            Ok(replaced.update_data(|expr| name.restore(expr)))
        })?
        .map_data(LogicalPlan::recompute_schema)
    })
    .map(|replaced| replaced.data)
}
