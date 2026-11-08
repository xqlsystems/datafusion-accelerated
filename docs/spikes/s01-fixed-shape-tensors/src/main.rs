// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike 1: how far does a fixed-shape tensor column pass through
//! DataFusion's operators without being copied or losing its type?
//!
//! Arrow's canonical *fixed-shape tensor* extension type is a
//! `FixedSizeList` column whose field carries the extension's name and
//! shape as metadata, so that each row's value is a dense array of one
//! shape. Design C (supplement S2.3) would have DataFusion carry tiles of
//! arrays this way. The spike runs one table, with a tensor column `t` of
//! 8 x 8 values per row, through each kind of operator, and records for
//! each output:
//! - whether the query runs at all;
//! - whether `t` keeps its extension metadata;
//! - whether `t`'s values are the input's own buffer (zero-copy) or a copy.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Array, FixedSizeListArray, Float64Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::common::Result;
use datafusion::datasource::MemTable;
use datafusion::prelude::*;

const SHAPE: usize = 64; // 8 x 8
const EXT: &str = "ARROW:extension:name";

fn tensor_field() -> Field {
    let item = Arc::new(Field::new("item", DataType::Float64, false));
    Field::new("t", DataType::FixedSizeList(item, SHAPE as i32), false).with_metadata(HashMap::from([
        (EXT.to_string(), "arrow.fixed_shape_tensor".to_string()),
        ("ARROW:extension:metadata".to_string(), r#"{"shape":[8,8]}"#.to_string()),
    ]))
}

fn x_table() -> (SchemaRef, Vec<Vec<RecordBatch>>) {
    let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int64, false), tensor_field()]));
    let item = Arc::new(Field::new("item", DataType::Float64, false));
    let mut batches = vec![];
    for b in 0..4 {
        let ks: Vec<i64> = (0..250).map(|i| (b * 250 + i) as i64).collect();
        let vs: Vec<f64> = (0..250 * SHAPE).map(|i| (b * 250 * SHAPE + i) as f64).collect();
        let t = FixedSizeListArray::new(item.clone(), SHAPE as i32, Arc::new(Float64Array::from(vs)), None);
        batches.push(RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(ks)), Arc::new(t)]).unwrap());
    }
    (schema, vec![batches])
}

fn y_table() -> (SchemaRef, Vec<Vec<RecordBatch>>) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("w", DataType::Float64, false),
    ]));
    let ks: Vec<i64> = (0..1000).step_by(3).collect();
    let ws: Vec<f64> = ks.iter().map(|k| *k as f64 * 0.1).collect();
    let b = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int64Array::from(ks)), Arc::new(Float64Array::from(ws))]).unwrap();
    (schema, vec![vec![b]])
}

/// Address ranges of the input tensor values, to tell a zero-copy output
/// from a copy.
fn ranges(parts: &[Vec<RecordBatch>]) -> Vec<(usize, usize)> {
    let mut out = vec![];
    for b in parts.iter().flatten() {
        let fsl = b.column(1).as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        let data = fsl.values().to_data();
        let buf = &data.buffers()[0];
        out.push((buf.as_ptr() as usize, buf.as_ptr() as usize + buf.len()));
    }
    out
}

struct Outcome {
    rows: usize,
    tensor_columns: usize,
    keeps_type: Option<bool>,
    zero_copy: Option<String>,
}

fn examine(batches: &[RecordBatch], inputs: &[(usize, usize)]) -> Outcome {
    let rows = batches.iter().map(|b| b.num_rows()).sum();
    let schema = match batches.first() {
        Some(b) => b.schema(),
        None => return Outcome { rows, tensor_columns: 0, keeps_type: None, zero_copy: None },
    };
    let tensor_cols: Vec<usize> = (0..schema.fields().len())
        .filter(|i| matches!(schema.field(*i).data_type(), DataType::FixedSizeList(_, _)))
        .collect();
    if tensor_cols.is_empty() {
        return Outcome { rows, tensor_columns: 0, keeps_type: None, zero_copy: None };
    }
    let keeps = tensor_cols.iter().all(|i| schema.field(*i).metadata().get(EXT).map(|s| s.as_str()) == Some("arrow.fixed_shape_tensor"));
    let (mut same, mut copied) = (0, 0);
    for b in batches {
        for i in &tensor_cols {
            let fsl = b.column(*i).as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            if fsl.is_empty() {
                continue;
            }
            let data = fsl.values().to_data();
            let p = data.buffers()[0].as_ptr() as usize + data.offset() * 8;
            if inputs.iter().any(|(s, e)| p >= *s && p < *e) {
                same += 1;
            } else {
                copied += 1;
            }
        }
    }
    let zc = match (same, copied) {
        (s, 0) if s > 0 => "zero-copy".to_string(),
        (0, c) if c > 0 => "copied".to_string(),
        (s, c) => format!("{s} batches zero-copy, {c} copied"),
    };
    Outcome { rows, tensor_columns: tensor_cols.len(), keeps_type: Some(keeps), zero_copy: Some(zc) }
}

const QUERIES: &[(&str, &str)] = &[
    ("projection", "SELECT k, t FROM x"),
    ("filter", "SELECT k, t FROM x WHERE k % 2 = 0"),
    ("limit", "SELECT k, t FROM x LIMIT 10"),
    ("sort", "SELECT k, t FROM x ORDER BY k DESC"),
    ("hash join, tensor as payload", "SELECT x.k, x.t, y.w FROM x JOIN y ON x.k = y.k"),
    ("union all", "SELECT k, t FROM x UNION ALL SELECT k, t FROM x"),
    ("repartition by key", "SELECT k, t FROM x DISTRIBUTE BY k"),
    ("window", "SELECT k, t, ROW_NUMBER() OVER (ORDER BY k) AS n FROM x"),
    ("group by, first_value", "SELECT k % 10 AS g, first_value(t) AS t FROM x GROUP BY k % 10"),
    ("group by, array_agg", "SELECT k % 10 AS g, array_agg(t) AS ts FROM x GROUP BY k % 10"),
    ("element access", "SELECT k, t[1] AS t1 FROM x"),
    ("elementwise arithmetic", "SELECT k, t * 2.0 AS t2 FROM x"),
    ("sum of tensors", "SELECT SUM(t) FROM x"),
    ("join on tensor equality", "SELECT a.k FROM x a JOIN x b ON a.t = b.t"),
    ("cast to list", "SELECT k, CAST(t AS DOUBLE[]) AS l FROM x"),
];

#[tokio::main]
async fn main() -> Result<()> {
    let (xs, xp) = x_table();
    let inputs = ranges(&xp);
    let (ys, yp) = y_table();
    println!("| target partitions | operator | runs | rows | tensor columns out | keeps its extension type | values |");
    println!("|---|---|---|---|---|---|---|");
    for tp in [1, 4] {
        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(tp));
        ctx.register_table("x", Arc::new(MemTable::try_new(xs.clone(), xp.clone())?))?;
        ctx.register_table("y", Arc::new(MemTable::try_new(ys.clone(), yp.clone())?))?;
        for (name, sql) in QUERIES {
            let r = async { ctx.sql(sql).await?.collect().await }.await;
            match r {
                Ok(batches) => {
                    let o = examine(&batches, &inputs);
                    let keeps = o.keeps_type.map(|k| if k { "yes" } else { "**no**" }.to_string()).unwrap_or("n/a".into());
                    println!("| {tp} | {name} | yes | {} | {} | {keeps} | {} |", o.rows, o.tensor_columns, o.zero_copy.unwrap_or("n/a".into()));
                }
                Err(e) => {
                    let msg = e.to_string().replace('|', "/");
                    let first = msg.lines().next().unwrap_or("").chars().take(110).collect::<String>();
                    println!("| {tp} | {name} | **no**: {first} | | | | |");
                }
            }
        }
    }
    Ok(())
}
