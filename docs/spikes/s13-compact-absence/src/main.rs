// SPDX-FileCopyrightText: 2026 Alexander Sky Merose <al@merose.com> & datafusion-accelerated Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike 13: where do compact arrays stop paying for their absent cells, and
//! does a blocked layout keep them unstored?
//!
//! One 3x3 convolution with C input and C output channels, the core of a
//! U-Net layer such as Samudra's, runs on each depth level of a real ocean
//! mask, in three layouts:
//! - rows: long-format tables in DataFusion, one row per present value, and
//!   the convolution as a join and a grouped sum in SQL;
//! - compact: one dense array per level, absent cells stored as masked
//!   zeros, every cell computed;
//! - blocked: dense B x B blocks, only the blocks with an ocean cell stored,
//!   each computed from a tile gathered with a one-cell halo.
//!
//! Every layout's output is checked against a direct sum over present
//! neighbours, to the float bound of design §6.3.

use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{Array, Float64Array, Int32Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};

const NZ: usize = 33;
const NY: usize = 180;
const NX: usize = 360;
const CELLS: usize = NY * NX;
const REPS: usize = 3;
const BLOCKS: [usize; 3] = [8, 16, 32];

// ---------------------------------------------------------------- data

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A value in [-1, 1) from a key.
fn value(key: u64) -> f64 {
    (splitmix(key) >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

fn input(z: usize, y: usize, x: usize, c: usize) -> f64 {
    value((((z * NY + y) * NX + x) * 64 + c) as u64)
}

/// Weights w[dy][dx][ci][co], flattened, with dy and dx in 0..3 for -1..=1.
fn weights(ch: usize) -> Vec<f64> {
    (0..9 * ch * ch).map(|i| value(0xDEAD_0000 + i as u64) * 0.3).collect()
}

fn read_mask() -> Vec<Vec<u8>> {
    let bytes = std::fs::read("mask.bin").expect("run export_mask.py first, to write mask.bin");
    assert_eq!(bytes.len(), NZ * CELLS);
    bytes.chunks(CELLS).map(|c| c.to_vec()).collect()
}

fn read_depths() -> Vec<String> {
    std::fs::read_to_string("depths.txt").unwrap().lines().map(str::to_owned).collect()
}

/// The same number of ocean cells as `real`, placed uniformly at random.
fn scattered(real: &[u8], seed: u64) -> Vec<u8> {
    let n = real.iter().filter(|&&m| m == 1).count();
    let mut idx: Vec<usize> = (0..CELLS).collect();
    for i in 0..n {
        let j = i + (splitmix(seed ^ i as u64) as usize) % (CELLS - i);
        idx.swap(i, j);
    }
    let mut out = vec![0u8; CELLS];
    for &i in &idx[..n] {
        out[i] = 1;
    }
    out
}

// ---------------------------------------------------------------- reference

/// out[cell][co] and the sum of |terms| for each, over present neighbours.
fn reference(z: usize, mask: &[u8], w: &[f64], ch: usize) -> (Vec<f64>, Vec<f64>) {
    let mut out = vec![0.0; CELLS * ch];
    let mut mag = vec![0.0; CELLS * ch];
    for y in 0..NY {
        for x in 0..NX {
            if mask[y * NX + x] == 0 {
                continue;
            }
            for (k, (dy, dx)) in offsets().enumerate() {
                let (ny, nx) = (y as isize + dy, x as isize + dx);
                if !inside(ny, nx) || mask[ny as usize * NX + nx as usize] == 0 {
                    continue;
                }
                for ci in 0..ch {
                    let a = input(z, ny as usize, nx as usize, ci);
                    for co in 0..ch {
                        let t = a * w[(k * ch + ci) * ch + co];
                        out[(y * NX + x) * ch + co] += t;
                        mag[(y * NX + x) * ch + co] += t.abs();
                    }
                }
            }
        }
    }
    (out, mag)
}

fn offsets() -> impl Iterator<Item = (isize, isize)> {
    (-1..=1).flat_map(|dy| (-1..=1).map(move |dx| (dy, dx)))
}

fn inside(y: isize, x: isize) -> bool {
    (0..NY as isize).contains(&y) && (0..NX as isize).contains(&x)
}

/// Whether `got` matches the reference at every present cell, within
/// (n - 1) u sum|terms| with n = 9 C terms, and no value is missing.
fn check(got: &[f64], have: &[bool], refs: &(Vec<f64>, Vec<f64>), mask: &[u8], ch: usize) -> bool {
    let n = (9 * ch) as f64;
    (0..CELLS).filter(|&i| mask[i] == 1).all(|i| {
        (0..ch).all(|co| {
            let j = i * ch + co;
            have[j] && (got[j] - refs.0[j]).abs() <= (n - 1.0) * f64::EPSILON / 2.0 * refs.1[j] + 1e-300
        })
    })
}

// ---------------------------------------------------------------- rows

async fn rows_context(mask: &[u8], z: usize, w: &[f64], ch: usize) -> SessionContext {
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_limit(4 << 30, 1.0)
        .build_arc()
        .unwrap();
    let config = SessionConfig::new().with_target_partitions(1);
    let ctx = SessionContext::new_with_config_rt(config, runtime);

    let (mut ys, mut xs, mut cs, mut vs) = (vec![], vec![], vec![], vec![]);
    let (mut cy, mut cx) = (vec![], vec![]);
    for y in 0..NY {
        for x in 0..NX {
            if mask[y * NX + x] == 1 {
                cy.push(y as i32);
                cx.push(x as i32);
                for c in 0..ch {
                    ys.push(y as i32);
                    xs.push(x as i32);
                    cs.push(c as i32);
                    vs.push(input(z, y, x, c));
                }
            }
        }
    }
    let int = |n: &str| Field::new(n, DataType::Int32, false);
    let x_schema = Arc::new(Schema::new(vec![int("y"), int("x"), int("c"), Field::new("v", DataType::Float64, false)]));
    let x_batch = RecordBatch::try_new(
        x_schema.clone(),
        vec![
            Arc::new(Int32Array::from(ys)),
            Arc::new(Int32Array::from(xs)),
            Arc::new(Int32Array::from(cs)),
            Arc::new(Float64Array::from(vs)),
        ],
    )
    .unwrap();
    ctx.register_table("x", Arc::new(MemTable::try_new(x_schema, vec![vec![x_batch]]).unwrap()))
        .unwrap();

    let c_schema = Arc::new(Schema::new(vec![int("y"), int("x")]));
    let c_batch = RecordBatch::try_new(
        c_schema.clone(),
        vec![Arc::new(Int32Array::from(cy)), Arc::new(Int32Array::from(cx))],
    )
    .unwrap();
    ctx.register_table("cells", Arc::new(MemTable::try_new(c_schema, vec![vec![c_batch]]).unwrap()))
        .unwrap();

    let (mut dys, mut dxs, mut cis, mut cos, mut wv) = (vec![], vec![], vec![], vec![], vec![]);
    for (k, (dy, dx)) in offsets().enumerate() {
        for ci in 0..ch {
            for co in 0..ch {
                dys.push(dy as i32);
                dxs.push(dx as i32);
                cis.push(ci as i32);
                cos.push(co as i32);
                wv.push(w[(k * ch + ci) * ch + co]);
            }
        }
    }
    let w_schema = Arc::new(Schema::new(vec![
        int("dy"),
        int("dx"),
        int("ci"),
        int("co"),
        Field::new("v", DataType::Float64, false),
    ]));
    let w_batch = RecordBatch::try_new(
        w_schema.clone(),
        vec![
            Arc::new(Int32Array::from(dys)),
            Arc::new(Int32Array::from(dxs)),
            Arc::new(Int32Array::from(cis)),
            Arc::new(Int32Array::from(cos)),
            Arc::new(Float64Array::from(wv)),
        ],
    )
    .unwrap();
    ctx.register_table("w", Arc::new(MemTable::try_new(w_schema, vec![vec![w_batch]]).unwrap()))
        .unwrap();
    ctx
}

/// An input at (y, x) reaches the output at (y - dy, x - dx), so
/// out(y, x) = sum over dy, dx of in(y + dy, x + dx) w(dy, dx).
const CONV_SQL: &str = "
WITH p AS (
  SELECT x.y - w.dy AS y, x.x - w.dx AS x, w.co AS c, SUM(x.v * w.v) AS v
  FROM x JOIN w ON x.c = w.ci
  GROUP BY x.y - w.dy, x.x - w.dx, w.co)
SELECT p.y, p.x, p.c, p.v FROM p JOIN cells ON p.y = cells.y AND p.x = cells.x";

async fn rows_run(ctx: &SessionContext, ch: usize) -> (Vec<f64>, Vec<bool>) {
    let batches = ctx.sql(CONV_SQL).await.unwrap().collect().await.unwrap();
    let mut out = vec![0.0; CELLS * ch];
    let mut have = vec![false; CELLS * ch];
    for b in &batches {
        let col = |i: usize| b.column(i).as_any().downcast_ref::<Int32Array>().unwrap().clone();
        let (y, x, c) = (col(0), col(1), col(2));
        let v = b.column(3).as_any().downcast_ref::<Float64Array>().unwrap();
        for r in 0..b.num_rows() {
            let j = (y.value(r) as usize * NX + x.value(r) as usize) * ch + c.value(r) as usize;
            assert!(!have[j], "duplicate output row");
            out[j] = v.value(r);
            have[j] = true;
        }
    }
    (out, have)
}

// ---------------------------------------------------------------- compact

struct Compact {
    values: Vec<f64>, // NY x NX x C, absent cells stored as 0
    mask: Vec<u8>,
}

fn compact(mask: &[u8], z: usize, ch: usize) -> Compact {
    let mut values = vec![0.0; CELLS * ch];
    for i in 0..CELLS {
        if mask[i] == 1 {
            for c in 0..ch {
                values[i * ch + c] = input(z, i / NX, i % NX, c);
            }
        }
    }
    Compact { values, mask: mask.to_vec() }
}

/// Every cell is computed, absent ones included, as a dense kernel does.
fn compact_run(a: &Compact, w: &[f64], ch: usize) -> Vec<f64> {
    let mut out = vec![0.0; CELLS * ch];
    for y in 0..NY {
        for x in 0..NX {
            let o = (y * NX + x) * ch;
            for (k, (dy, dx)) in offsets().enumerate() {
                let (ny, nx) = (y as isize + dy, x as isize + dx);
                if !inside(ny, nx) {
                    continue;
                }
                let s = (ny as usize * NX + nx as usize) * ch;
                for ci in 0..ch {
                    let v = a.values[s + ci];
                    let wk = &w[(k * ch + ci) * ch..(k * ch + ci + 1) * ch];
                    for co in 0..ch {
                        out[o + co] += v * wk[co];
                    }
                }
            }
        }
    }
    // The mask is applied to the output, as a masked value's validity.
    for i in 0..CELLS {
        if a.mask[i] == 0 {
            out[i * ch..(i + 1) * ch].fill(0.0);
        }
    }
    out
}

fn compact_bytes(ch: usize) -> usize {
    CELLS * ch * 8 + CELLS.div_ceil(8)
}

// ---------------------------------------------------------------- blocked

struct Blocked {
    b: usize,
    nby: usize,
    nbx: usize,
    index: Vec<i32>, // block (by, bx) -> slot, or -1 when absent
    values: Vec<f64>, // slot x B x B x C
    origin: Vec<(usize, usize)>,
}

fn blocked(mask: &[u8], z: usize, ch: usize, b: usize) -> Blocked {
    let (nby, nbx) = (NY.div_ceil(b), NX.div_ceil(b));
    let mut index = vec![-1i32; nby * nbx];
    let mut origin = vec![];
    for by in 0..nby {
        for bx in 0..nbx {
            let any = (by * b..((by + 1) * b).min(NY))
                .any(|y| (bx * b..((bx + 1) * b).min(NX)).any(|x| mask[y * NX + x] == 1));
            if any {
                index[by * nbx + bx] = origin.len() as i32;
                origin.push((by * b, bx * b));
            }
        }
    }
    let mut values = vec![0.0; origin.len() * b * b * ch];
    for (s, &(y0, x0)) in origin.iter().enumerate() {
        for ly in 0..b {
            for lx in 0..b {
                let (y, x) = (y0 + ly, x0 + lx);
                if y < NY && x < NX && mask[y * NX + x] == 1 {
                    for c in 0..ch {
                        values[((s * b + ly) * b + lx) * ch + c] = input(z, y, x, c);
                    }
                }
            }
        }
    }
    Blocked { b, nby, nbx, index, values, origin }
}

impl Blocked {
    fn get(&self, y: isize, x: isize, c: usize, ch: usize) -> f64 {
        if !inside(y, x) {
            return 0.0;
        }
        let (y, x) = (y as usize, x as usize);
        let slot = self.index[(y / self.b) * self.nbx + x / self.b];
        if slot < 0 {
            return 0.0;
        }
        self.values[((slot as usize * self.b + y % self.b) * self.b + x % self.b) * ch + c]
    }

    fn bytes(&self, ch: usize) -> usize {
        let cells = self.origin.len() * self.b * self.b;
        cells * ch * 8 + cells.div_ceil(8) + self.nby * self.nbx * 4
    }

    fn cells(&self) -> usize {
        self.origin.len() * self.b * self.b
    }
}

/// Each stored block gathers a (B + 2)^2 tile with its halo, as an exchange
/// of neighbouring blocks would, then computes every cell of the block.
fn blocked_run(a: &Blocked, w: &[f64], ch: usize) -> (Vec<f64>, Vec<bool>) {
    let (b, t) = (a.b, a.b + 2);
    let mut out = vec![0.0; CELLS * ch];
    let mut have = vec![false; CELLS * ch];
    let mut tile = vec![0.0; t * t * ch];
    let mut acc = vec![0.0; b * b * ch];
    for &(y0, x0) in &a.origin {
        for ty in 0..t {
            for tx in 0..t {
                let (y, x) = (y0 as isize + ty as isize - 1, x0 as isize + tx as isize - 1);
                for c in 0..ch {
                    tile[(ty * t + tx) * ch + c] = a.get(y, x, c, ch);
                }
            }
        }
        acc.fill(0.0);
        for ly in 0..b {
            for lx in 0..b {
                let o = (ly * b + lx) * ch;
                for (k, (dy, dx)) in offsets().enumerate() {
                    let s = (((ly as isize + 1 + dy) as usize) * t + (lx as isize + 1 + dx) as usize) * ch;
                    for ci in 0..ch {
                        let v = tile[s + ci];
                        let wk = &w[(k * ch + ci) * ch..(k * ch + ci + 1) * ch];
                        for co in 0..ch {
                            acc[o + co] += v * wk[co];
                        }
                    }
                }
            }
        }
        for ly in 0..b {
            for lx in 0..b {
                let (y, x) = (y0 + ly, x0 + lx);
                if y < NY && x < NX {
                    for c in 0..ch {
                        out[(y * NX + x) * ch + c] = acc[(ly * b + lx) * ch + c];
                        have[(y * NX + x) * ch + c] = true;
                    }
                }
            }
        }
    }
    (out, have)
}

// ---------------------------------------------------------------- driver

fn best<T>(mut f: impl FnMut() -> T) -> (f64, T) {
    let mut t = f64::INFINITY;
    let mut last = None;
    for _ in 0..REPS {
        let s = Instant::now();
        let r = f();
        t = t.min(s.elapsed().as_secs_f64() * 1e3);
        last = Some(r);
    }
    (t, last.unwrap())
}

struct Row {
    depth: String,
    mask: &'static str,
    density: f64,
    ocean: usize,
    rows_ms: f64,
    compact_ms: f64,
    blocked_ms: [f64; 3],
    blocked_cells: [usize; 3],
    blocked_bytes: [usize; 3],
    ok: bool,
}

#[tokio::main]
async fn main() {
    let ch: usize = std::env::var("CHANNELS").ok().map(|s| s.parse().unwrap()).unwrap_or(4);
    let only: Option<Vec<usize>> = std::env::var("LEVELS")
        .ok()
        .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect());
    let masks = read_mask();
    let depths = read_depths();
    let w = weights(ch);
    let mut rows = vec![];

    for z in 0..NZ {
        if only.as_ref().is_some_and(|o| !o.contains(&z)) {
            continue;
        }
        for (name, mask) in [("real", masks[z].clone()), ("scattered", scattered(&masks[z], 0x13 + z as u64))] {
            let ocean = mask.iter().filter(|&&m| m == 1).count();
            if ocean == 0 {
                continue;
            }
            let refs = reference(z, &mask, &w, ch);

            let ctx = rows_context(&mask, z, &w, ch).await;
            let mut rows_ms = f64::INFINITY;
            let mut rows_out = None;
            for _ in 0..REPS {
                let s = Instant::now();
                let r = rows_run(&ctx, ch).await;
                rows_ms = rows_ms.min(s.elapsed().as_secs_f64() * 1e3);
                rows_out = Some(r);
            }
            let (ro, rh) = rows_out.unwrap();
            let mut ok = check(&ro, &rh, &refs, &mask, ch);

            let a = compact(&mask, z, ch);
            let (compact_ms, co) = best(|| compact_run(&a, &w, ch));
            ok &= check(&co, &vec![true; CELLS * ch], &refs, &mask, ch);

            let mut blocked_ms = [0.0; 3];
            let mut blocked_cells = [0; 3];
            let mut blocked_bytes = [0; 3];
            for (i, &b) in BLOCKS.iter().enumerate() {
                let a = blocked(&mask, z, ch, b);
                let (ms, (bo, bh)) = best(|| blocked_run(&a, &w, ch));
                ok &= check(&bo, &bh, &refs, &mask, ch);
                blocked_ms[i] = ms;
                blocked_cells[i] = a.cells();
                blocked_bytes[i] = a.bytes(ch);
            }
            eprintln!(
                "z={z:2} {name:9} density {:.3} rows {rows_ms:8.1} ms compact {compact_ms:6.1} ms blocked {:?} ok={ok}",
                ocean as f64 / CELLS as f64,
                blocked_ms.map(|m| (m * 10.0).round() / 10.0)
            );
            rows.push(Row {
                depth: depths[z].clone(),
                mask: name,
                density: ocean as f64 / CELLS as f64,
                ocean,
                rows_ms,
                compact_ms,
                blocked_ms,
                blocked_cells,
                blocked_bytes,
                ok,
            });
        }
    }
    print!("{}", report(&rows, ch));
}

fn report(rows: &[Row], ch: usize) -> String {
    let mut s = String::new();
    let rows_bytes = |ocean: usize| ocean * ch * (3 * 4 + 8);
    s += &format!(
        "<!-- Written by `cargo run --release` with CHANNELS={ch}: {} ocean levels, best of {REPS} runs each. -->\n\n",
        rows.iter().filter(|r| r.mask == "real").count()
    );
    s += "## E1. Cells stored, per ocean cell\n\n";
    s += "A layout's stored cells divided by the level's ocean cells. Rows store exactly the ocean cells (1.00).\n\n";
    s += "| depth (m) | mask | density | compact | blocked 8 | blocked 16 | blocked 32 |\n|---|---|---|---|---|---|---|\n";
    for r in rows {
        let o = r.ocean as f64;
        s += &format!(
            "| {} | {} | {:.3} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
            r.depth,
            r.mask,
            r.density,
            CELLS as f64 / o,
            r.blocked_cells[0] as f64 / o,
            r.blocked_cells[1] as f64 / o,
            r.blocked_cells[2] as f64 / o
        );
    }
    s += &format!("\n## E2. Time per convolution, ms, and bytes stored, with C = {ch}\n\n");
    s += "Rows run through DataFusion on one partition; the arrays run on one core. Bytes are the input's: \
           rows hold y, x and c as Int32 and v as Float64; arrays hold Float64 values and a bit per cell.\n\n";
    s += "| depth (m) | mask | density | rows ms | compact ms | blocked 8 ms | blocked 16 ms | blocked 32 ms | rows MB | compact MB | blocked 8 MB | blocked 32 MB | matches |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
    let mb = |b: usize| b as f64 / 1e6;
    for r in rows {
        s += &format!(
            "| {} | {} | {:.3} | {:.1} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {} |\n",
            r.depth,
            r.mask,
            r.density,
            r.rows_ms,
            r.compact_ms,
            r.blocked_ms[0],
            r.blocked_ms[1],
            r.blocked_ms[2],
            mb(rows_bytes(r.ocean)),
            mb(compact_bytes(ch)),
            mb(r.blocked_bytes[0]),
            mb(r.blocked_bytes[2]),
            if r.ok { "yes" } else { "**no**" }
        );
    }
    s += "\n## E3. Totals over every level\n\n";
    s += "| mask | ocean cells | rows ms | compact ms | blocked 8 ms | blocked 16 ms | blocked 32 ms | rows MB | compact MB | blocked 8 MB | blocked 16 MB | blocked 32 MB |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n";
    for m in ["real", "scattered"] {
        let sel: Vec<&Row> = rows.iter().filter(|r| r.mask == m).collect();
        let sum = |f: &dyn Fn(&Row) -> f64| sel.iter().map(|r| f(r)).sum::<f64>();
        s += &format!(
            "| {m} | {} | {:.0} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} | {:.1} |\n",
            sel.iter().map(|r| r.ocean).sum::<usize>(),
            sum(&|r| r.rows_ms),
            sum(&|r| r.compact_ms),
            sum(&|r| r.blocked_ms[0]),
            sum(&|r| r.blocked_ms[1]),
            sum(&|r| r.blocked_ms[2]),
            sum(&|r| mb(rows_bytes(r.ocean))),
            sum(&|_| mb(compact_bytes(ch))),
            sum(&|r| mb(r.blocked_bytes[0])),
            sum(&|r| mb(r.blocked_bytes[1])),
            sum(&|r| mb(r.blocked_bytes[2])),
        );
    }
    s
}
