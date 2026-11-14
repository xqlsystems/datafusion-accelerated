// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The workspace builds against DataFusion, and a session runs a query.

use datafusion::arrow::array::{Array, Int64Array};
use datafusion::prelude::SessionContext;

#[tokio::test]
async fn a_session_runs_a_query() -> datafusion::error::Result<()> {
    let ctx = SessionContext::new();
    let batches = ctx.sql("SELECT 1 + 1 AS two").await?.collect().await?;
    let two = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("an Int64 column");
    assert_eq!((two.len(), two.value(0)), (1, 2));
    Ok(())
}
