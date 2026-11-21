// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Splitting a `SELECT` into its top-level CTEs and its final query.

use std::ops::ControlFlow;

use datafusion::sql::sqlparser::ast::{Expr, ObjectName, Query, Statement, Visit, Visitor};
use datafusion::sql::sqlparser::dialect::GenericDialect;
use datafusion::sql::sqlparser::parser::Parser;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// One part of a `SELECT`: a top-level CTE, or the final query, whose name
/// is empty. Each part has its SQL, the tables it names directly and the
/// functions it calls.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Part {
    /// The CTE's name, unquoted, and lowered if it wasn't quoted. Empty for
    /// the final query.
    pub name: String,
    /// The part's query, as SQL.
    pub sql: String,
    /// The tables the part names with one identifier, in order, unquoted,
    /// and lowered if they weren't quoted. They include the CTEs it reads.
    pub tables: Vec<String>,
    /// The functions the part calls, in order, lowered.
    pub functions: Vec<String>,
}

#[derive(Default)]
struct References {
    tables: Vec<String>,
    functions: Vec<String>,
    nested_with: bool,
}

impl Visitor for References {
    type Break = ();

    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<()> {
        self.nested_with |= query.with.is_some();
        ControlFlow::Continue(())
    }

    fn pre_visit_relation(&mut self, name: &ObjectName) -> ControlFlow<()> {
        if let [part] = name.0.as_slice()
            && let Some(id) = part.as_ident()
        {
            self.tables.push(if id.quote_style.is_some() {
                id.value.clone()
            } else {
                id.value.to_lowercase()
            });
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &Expr) -> ControlFlow<()> {
        if let Expr::Function(function) = expr {
            self.functions
                .push(function.name.to_string().to_lowercase());
        }
        ControlFlow::Continue(())
    }
}

/// The top-level CTEs of `sql`, one `SELECT`, in order, then its final
/// query.
///
/// Fails on SQL that isn't one `SELECT`, on a recursive `WITH`, on a CTE
/// with column aliases, and, unless `allow_nested`, on a part with a
/// `WITH` of its own.
pub fn shared_ctes(sql: &str, allow_nested: bool) -> Result<Vec<Part>, String> {
    let mut statements =
        Parser::parse_sql(&GenericDialect {}, sql).map_err(|error| error.to_string())?;
    if statements.len() != 1 {
        return Err("shared execution requires one SELECT".to_string());
    }
    let Statement::Query(mut query) = statements.remove(0) else {
        return Err("shared execution requires a SELECT".to_string());
    };
    let mut parts = Vec::new();
    if let Some(with) = query.with.take() {
        if with.recursive {
            return Err("recursive CTEs cannot be staged".to_string());
        }
        for cte in with.cte_tables {
            if !cte.alias.columns.is_empty() {
                return Err("CTE column aliases cannot be staged".to_string());
            }
            let id = cte.alias.name;
            let name = if id.quote_style.is_some() {
                id.value
            } else {
                id.value.to_lowercase()
            };
            parts.push(part(name, &cte.query, allow_nested)?);
        }
    }
    parts.push(part(String::new(), &query, allow_nested)?);
    Ok(parts)
}

fn part(name: String, query: &Query, allow_nested: bool) -> Result<Part, String> {
    let mut references = References::default();
    let _ = query.visit(&mut references);
    if references.nested_with && !allow_nested {
        return Err("nested WITH scopes cannot be staged".to_string());
    }
    Ok(Part {
        name,
        sql: query.to_string(),
        tables: references.tables,
        functions: references.functions,
    })
}

/// A [`Part`] as Python sees it: `(name, sql, tables, functions)`.
type PyPart = (String, String, Vec<String>, Vec<String>);

/// [`shared_ctes`] for Python: each part as `(name, sql, tables,
/// functions)`, and a `ValueError` for SQL it can't split.
#[pyfunction]
#[pyo3(name = "shared_ctes", signature = (sql, allow_nested=false))]
pub(crate) fn py_shared_ctes(sql: &str, allow_nested: bool) -> PyResult<Vec<PyPart>> {
    let parts = shared_ctes(sql, allow_nested).map_err(PyValueError::new_err)?;
    Ok(parts
        .into_iter()
        .map(|part| (part.name, part.sql, part.tables, part.functions))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_select_splits_into_its_ctes_then_its_final_query() {
        let parts = shared_ctes(
            "WITH \"H\" AS (SELECT exp(v) AS v FROM X), g AS (SELECT v FROM \"H\")              SELECT sum(v) FROM g JOIN h ON true",
            false,
        )
        .unwrap();
        let names: Vec<_> = parts.iter().map(|part| part.name.as_str()).collect();
        assert_eq!(names, ["H", "g", ""]);
        assert_eq!(parts[0].tables, ["x"]);
        assert_eq!(parts[0].functions, ["exp"]);
        assert_eq!(parts[1].tables, ["H"]);
        assert_eq!(parts[2].tables, ["g", "h"]);
        assert_eq!(parts[2].functions, ["sum"]);
    }

    #[test]
    fn sql_it_cant_stage_fails() {
        for (sql, allow_nested, error) in [
            ("SELECT 1; SELECT 2", false, "one SELECT"),
            ("INSERT INTO t VALUES (1)", false, "requires a SELECT"),
            (
                "WITH RECURSIVE r AS (SELECT 1) SELECT * FROM r",
                false,
                "recursive",
            ),
            (
                "WITH r(a) AS (SELECT 1) SELECT * FROM r",
                false,
                "column aliases",
            ),
            (
                "SELECT * FROM (WITH r AS (SELECT 1) SELECT * FROM r)",
                false,
                "nested",
            ),
        ] {
            let found = shared_ctes(sql, allow_nested).unwrap_err();
            assert!(found.contains(error), "{sql}: {found}");
        }
        assert!(shared_ctes("SELECT * FROM (WITH r AS (SELECT 1) SELECT * FROM r)", true).is_ok());
    }
}
