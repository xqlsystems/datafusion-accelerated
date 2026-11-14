// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Attaching the runtime, and the `accelerated` settings.

use datafusion::error::Result;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_accelerated::{Accelerator, AcceleratorOptions, SessionStateBuilderExt};

fn accelerated(options: AcceleratorOptions) -> SessionContext {
    SessionContext::new_with_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_accelerator(options)
            .build(),
    )
}

async fn set(ctx: &SessionContext, sql: &str) -> Result<()> {
    ctx.sql(sql).await?.collect().await?;
    Ok(())
}

fn strict(ctx: &SessionContext) -> bool {
    Accelerator::settings(&ctx.state()).unwrap().strict
}

#[test]
fn a_session_built_with_the_runtime_has_it() {
    let ctx = accelerated(AcceleratorOptions::new());
    assert!(Accelerator::of(&ctx.state()).is_ok());
    assert!(Accelerator::of_task(&ctx.task_ctx()).is_ok());
    assert!(!strict(&ctx));
}

#[test]
fn a_plain_session_says_how_to_attach_it() {
    let ctx = SessionContext::new();
    let message = Accelerator::of(&ctx.state()).unwrap_err().to_string();
    assert!(message.contains("with_accelerator()"), "{message}");
    assert!(Accelerator::of_task(&ctx.task_ctx()).is_err());
    assert!(Accelerator::settings(&ctx.state()).is_err());
}

#[test]
fn with_config_after_with_accelerator_drops_it() {
    let state = SessionStateBuilder::new()
        .with_accelerator(AcceleratorOptions::new())
        .with_config(SessionConfig::new())
        .build();
    assert!(Accelerator::of(&state).is_err());
}

#[test]
fn with_accelerator_keeps_the_builders_config() {
    let state = SessionStateBuilder::new()
        .with_config(SessionConfig::new().with_target_partitions(3))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    assert_eq!(state.config().target_partitions(), 3);
    assert!(Accelerator::of(&state).is_ok());
}

#[test]
fn the_later_call_wins() {
    let state = SessionStateBuilder::new()
        .with_accelerator(AcceleratorOptions::new().strict(true))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    assert!(!Accelerator::settings(&state).unwrap().strict);
}

#[tokio::test]
async fn options_give_the_initial_settings_and_set_changes_them() -> Result<()> {
    let ctx = accelerated(AcceleratorOptions::new().strict(true));
    assert!(strict(&ctx));
    set(&ctx, "SET accelerated.strict = false").await?;
    assert!(!strict(&ctx));
    set(&ctx, "SET accelerated.strict = true").await?;
    assert!(strict(&ctx));
    // The runtime's state survives a SET.
    assert!(Accelerator::of(&ctx.state()).is_ok());
    Ok(())
}

#[tokio::test]
async fn an_unknown_setting_is_an_error() {
    let ctx = accelerated(AcceleratorOptions::new());
    let err = set(&ctx, "SET accelerated.nonsense = 1").await.unwrap_err();
    assert!(err.to_string().contains("nonsense"), "{err}");
}

#[tokio::test]
async fn a_plain_session_has_no_accelerated_settings() {
    let ctx = SessionContext::new();
    assert!(set(&ctx, "SET accelerated.strict = true").await.is_err());
}
