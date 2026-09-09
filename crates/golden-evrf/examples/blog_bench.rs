//! Ad-hoc benchmark for the alinush.github.io Chunky-vs-Golden blog post.
//!
//! Measures, for the exact (t, n) grid used in that post's "Full benchmarks"
//! table, the same three columns reported there for every scheme:
//! transcript size (one dealer's serialized `DealerMessage`), Deal time
//! (`create_dealing` for one dealer), and Verify time (`verify_dealing` for
//! that one dealer's message).
//!
//! `t` here is the blog's polynomial degree (a `t`-out-of-`n` scheme needs
//! `t+1` shares), so `DkgConfig::threshold` (the *coefficient count*) is
//! `t + 1`.
//!
//! ```bash
//! cargo run --profile optimized --example blog_bench \
//!     --features golden-evrf/bls12-381-jubjub,golden-evrf/parallel -- <t> <n>
//! ```
//!
//! Takes one `(t, n)` pair as CLI args and exits, rather than looping over
//! the whole grid in one process: `create_dealing`/`verify_dealing` route
//! through `BatchedEvrfPublicParams::shared`, which caches each distinct
//! `(threshold, receiver_count)` shape's Bulletproofs generators in a
//! process-wide static for the life of the process. Looping over every grid
//! row in one process accumulates every prior shape's generators and OOMs
//! well before reaching `n=1024`; a fresh process per row lets the OS
//! reclaim that shape's generators on exit.

#![allow(non_snake_case)]
#![allow(missing_docs)]
#![allow(clippy::unwrap_used)]

#[path = "../benches/bls_support_dir/mod.rs"]
mod support;

use std::time::{Duration, Instant};

use golden_bls_jubjub::golden_group::JubjubGoldenGroup;
use golden_core::wire::to_wire_bytes;
use golden_core::{
    create_dealing, verify_dealing, DealerMessageNonce, EvrfProofBackend, EvrfStatement,
    EvrfWitness, GoldenGroup, GoldenScalar, Result, Share,
};
use golden_evrf::paper::bls_jubjub::{BatchedEvrfPublicParams, BlsJubjubBackend};
use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};
use rand_core::CryptoRngCore;
use support::{build_config, identity_secret, idx, BENCH_SEED};

fn median(mut xs: Vec<Duration>) -> Duration {
    xs.sort();
    xs[xs.len() / 2]
}

/// Stand-in backend for shapes too large to actually prove/verify in this
/// environment's memory budget (e.g. n=1024's ~8.4M-multiplier circuit OOMs
/// an 11 GiB sandbox). Delegates `derive_pad` to the real backend (so
/// encrypted shares, and hence decrypt-share timing, stay real) but replaces
/// the expensive Bulletproofs proof with a zero-filled placeholder of the
/// *exact* real wire length from `batched_proof_wire_len`, which is a closed
/// -form function of the circuit shape and does not require building a
/// proof. This makes the resulting `DealerMessage`'s serialized size exactly
/// correct while skipping the memory-hungry prover entirely; it must never
/// be used to measure Deal/Verify time, since `prove_batch` here does none
/// of the real work.
struct SizeOnlyBackend;

impl EvrfProofBackend<JubjubGoldenGroup> for SizeOnlyBackend {
    const PROOF_ID: &'static [u8] = BlsJubjubBackend::PROOF_ID;

    fn derive_pad(
        msg_i: DealerMessageNonce,
        beta: &<JubjubGoldenGroup as GoldenGroup>::Scalar,
        identity_secret: &<JubjubGoldenGroup as GoldenGroup>::Scalar,
        peer_public_key: &<JubjubGoldenGroup as GoldenGroup>::Element,
        receiver_public_key: &<JubjubGoldenGroup as GoldenGroup>::Element,
    ) -> Result<<JubjubGoldenGroup as GoldenGroup>::Scalar> {
        BlsJubjubBackend::derive_pad(
            msg_i,
            beta,
            identity_secret,
            peer_public_key,
            receiver_public_key,
        )
    }

    fn prove_batch(
        statements: &[EvrfStatement<JubjubGoldenGroup>],
        _witnesses: &[EvrfWitness<JubjubGoldenGroup>],
        _rng: &mut impl CryptoRngCore,
    ) -> Result<Vec<u8>> {
        let threshold = statements[0].threshold;
        let len = BatchedEvrfPublicParams::batched_proof_wire_len(threshold, statements.len())?;
        Ok(vec![0u8; len])
    }

    fn verify_batch(
        _statements: &[EvrfStatement<JubjubGoldenGroup>],
        _proof: &[u8],
    ) -> Result<()> {
        unreachable!("SizeOnlyBackend never verifies; it only exists to size a transcript")
    }
}

fn reps_for(n: usize) -> usize {
    if n <= 64 {
        5
    } else if n <= 256 {
        3
    } else {
        1
    }
}

fn decrypt_time(
    config: &golden_core::DkgConfig<JubjubGoldenGroup>,
    dealer: golden_core::ParticipantIndex,
    receiver: golden_core::ParticipantIndex,
    receiver_secret: &<JubjubGoldenGroup as GoldenGroup>::Scalar,
    message: &golden_core::DealerMessage<JubjubGoldenGroup>,
    reps: usize,
) -> Duration {
    let dealer_public_key = config.registry.public_key(dealer).unwrap();
    let receiver_public_key = config.registry.public_key(receiver).unwrap();
    let encrypted_share = message.encrypted_shares.get(&receiver).unwrap().clone();
    let mut decrypt_times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        let pad = BlsJubjubBackend::derive_pad(
            message.msg_i,
            &config.beta,
            receiver_secret,
            dealer_public_key,
            receiver_public_key,
        )
        .unwrap();
        let share = Share {
            participant: receiver,
            value: encrypted_share.encrypted_share.sub(&pad),
        };
        assert!(message.commitment.verify_share(&share).unwrap());
        decrypt_times.push(start.elapsed());
    }
    median(decrypt_times)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let t: usize = args.get(1).expect("usage: blog_bench <t> <n> [size-only]").parse().unwrap();
    let n: usize = args.get(2).expect("usage: blog_bench <t> <n> [size-only]").parse().unwrap();
    let size_only = args.get(3).map(String::as_str) == Some("size-only");

    let config = build_config(n, t + 1);
    let dealer = idx(1);
    let secret = identity_secret(dealer);
    let receiver = idx(2);
    let receiver_secret = identity_secret(receiver);

    if size_only {
        // Shape too large to actually prove/verify in this environment's
        // memory budget. `SizeOnlyBackend` still runs every other real step
        // of `create_dealing` (Feldman commit, share/pad derivation,
        // encrypted shares, transcript root), so the resulting message's
        // wire size and decrypt-share time are exact; Deal/Verify are
        // unmeasured (printed as "NA").
        let mut rng = ChaCha20Rng::from_seed(BENCH_SEED);
        let dealing = create_dealing::<JubjubGoldenGroup, SizeOnlyBackend>(
            dealer, &secret, &config, &mut rng,
        )
        .unwrap();
        let transcript_bytes = to_wire_bytes(&dealing.message).len();
        let decrypt_ms =
            decrypt_time(&config, dealer, receiver, &receiver_secret, &dealing.message, 5)
                .as_secs_f64()
                * 1000.0;
        println!("{t},{n},{transcript_bytes},NA,NA,{decrypt_ms:.3},size-only");
        return;
    }

    let reps = reps_for(n);

    let mut deal_times = Vec::with_capacity(reps);
    let mut transcript_bytes = 0usize;
    let mut last_message = None;
    for _ in 0..reps {
        let mut rng = ChaCha20Rng::from_seed(BENCH_SEED);
        let start = Instant::now();
        let dealing = create_dealing::<JubjubGoldenGroup, BlsJubjubBackend>(
            dealer, &secret, &config, &mut rng,
        )
        .unwrap();
        deal_times.push(start.elapsed());
        transcript_bytes = to_wire_bytes(&dealing.message).len();
        last_message = Some(dealing.message);
    }
    let message = last_message.unwrap();

    let mut verify_times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        verify_dealing::<JubjubGoldenGroup, BlsJubjubBackend>(&message, &config).unwrap();
        verify_times.push(start.elapsed());
    }

    let decrypt_ms =
        decrypt_time(&config, dealer, receiver, &receiver_secret, &message, reps).as_secs_f64()
            * 1000.0;
    let deal_ms = median(deal_times).as_secs_f64() * 1000.0;
    let verify_ms = median(verify_times).as_secs_f64() * 1000.0;
    println!("{t},{n},{transcript_bytes},{deal_ms:.3},{verify_ms:.3},{decrypt_ms:.3},{reps}");
}
