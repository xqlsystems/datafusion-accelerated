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
    Accelerator::settings(&ctx.state())
        .expect("the session was built with the runtime")
        .strict
}

#[test]
fn a_session_built_with_the_runtime_has_it() {
    let ctx = accelerated(AcceleratorOptions::new());
    assert!(
        Accelerator::of(&ctx.state()).is_ok(),
        "the session's state should hold the runtime"
    );
    assert!(
        Accelerator::of_task(&ctx.task_ctx()).is_ok(),
        "a task context should carry the session's runtime"
    );
    assert!(!strict(&ctx), "strict should default to false");
}

#[test]
fn a_plain_session_says_how_to_attach_it() {
    let ctx = SessionContext::new();
    let message = Accelerator::of(&ctx.state()).unwrap_err().to_string();
    assert!(
        message.contains("with_accelerator()"),
        "the error should say how to attach: {message}"
    );
    assert!(
        Accelerator::of_task(&ctx.task_ctx()).is_err(),
        "a plain task context has no runtime"
    );
    assert!(
        Accelerator::settings(&ctx.state()).is_err(),
        "a plain session has no settings"
    );
}

#[test]
fn with_config_after_with_accelerator_drops_it() {
    let state = SessionStateBuilder::new()
        .with_accelerator(AcceleratorOptions::new())
        .with_config(SessionConfig::new())
        .build();
    assert!(
        Accelerator::of(&state).is_err(),
        "with_config() replaces the config, and the runtime with it"
    );
}

#[test]
fn with_accelerator_keeps_the_builders_config() {
    let state = SessionStateBuilder::new()
        .with_config(SessionConfig::new().with_target_partitions(3))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    assert_eq!(
        state.config().target_partitions(),
        3,
        "the builder's own settings should survive"
    );
    assert!(
        Accelerator::of(&state).is_ok(),
        "the runtime should be added to the builder's config"
    );
}

#[test]
fn the_later_call_wins() {
    let state = SessionStateBuilder::new()
        .with_accelerator(AcceleratorOptions::new().strict(true))
        .with_accelerator(AcceleratorOptions::new())
        .build();
    let settings = Accelerator::settings(&state).unwrap();
    assert!(
        !settings.strict,
        "the second call's options should replace the first's"
    );
}

#[tokio::test]
async fn options_give_the_initial_settings_and_set_changes_them() -> Result<()> {
    let ctx = accelerated(AcceleratorOptions::new().strict(true));
    assert!(strict(&ctx), "the options should set the initial value");
    set(&ctx, "SET accelerated.strict = false").await?;
    assert!(!strict(&ctx), "SET should override the options");
    set(&ctx, "SET accelerated.strict = true").await?;
    assert!(strict(&ctx), "SET should apply again");
    assert!(
        Accelerator::of(&ctx.state()).is_ok(),
        "the runtime's state should survive a SET"
    );
    Ok(())
}

#[tokio::test]
async fn an_unknown_setting_is_an_error() {
    let ctx = accelerated(AcceleratorOptions::new());
    let err = set(&ctx, "SET accelerated.nonsense = 1").await.unwrap_err();
    assert!(
        err.to_string().contains("nonsense"),
        "the error should name the setting: {err}"
    );
}

#[tokio::test]
async fn a_plain_session_has_no_accelerated_settings() {
    let ctx = SessionContext::new();
    assert!(
        set(&ctx, "SET accelerated.strict = true").await.is_err(),
        "without the runtime, the namespace shouldn't exist"
    );
}
