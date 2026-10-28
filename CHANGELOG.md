# Changelog

## Unreleased

- English (`language: "english"`): the `english_2026-04` bundle at the same pinned
  revision, SHA-256 checked, with its own Alba embedding (`voices/alba-casual.en.emb.npy`,
  encoded with the English encoder). German stays the default.
- Sentence rules per language: English abbreviations (Mr., Dr., e.g., i.e.) and no
  German ordinal-date rule. English keeps 3 frames after end-of-speech (5 for <= 4 words);
  with 1, endings such as "afternoon?" were cut.
- `requireBos()` checks BOS for every bundle that declares it; `requireGermanBos` stays as
  a deprecated alias.
- Tools, model tests and the bench take `--language` / `POCKET_TTS_LANGUAGE`; English
  endings corpus `test/models/endings.en.json` and bench phrases `bench/prompts.en.json`.

## 0.1.0-alpha.0 (2026-10-20)

First extraction of the browser runtime into its own package.

- Dynamic KV caches, patched in memory from the stock graphs (matching PCM in tested German/WASM cases).
- Precision profiles `fp32`, `int8-core`, `int8-wide`, with a pinned, SHA-256-checked asset manifest.
- Revision-keyed Cache Storage, finite retries for 429/5xx/network errors, a single download per asset.
- Stock voice as a precision-independent encoder embedding (Alba, CC BY 4.0).
- Optional local cloning with a lazily loaded encoder.
- `speak()` with queued (1 s), immediate, adaptive and complete playback.
- Sentence endings, ported from the Codex runtime snapshot (codex-pocket-runtime-2026-10-20):
  one text chunk per sentence; the end-of-speech tail is computed per sentence (3 frames
  for <= 4 words, else 1, unless the model metadata says otherwise); a sentence scanner
  that keeps decimals, abbreviations, initials, ordinal dates and closing quotes intact;
  per-sentence diagnostics (`metrics.textChunks`, `metrics.complete`).
- Typographic punctuation outside the tokenizer vocabulary („“ ’ – …) is normalized: on
  real models it lost a quoted sentence and turned "geht’s" into babble.
- A chunk that hits the 500-frame cap now still delivers its generated frames, reported
  as `frame-cap`, not complete.
- Worker/client build handshake: a stale cached worker.js is refused.

Alpha packaging: npm name pocket-tts-fast; includes the attributed
CC-BY-4.0 Alba embedding for consumer self-hosting, automatic prepack build,
and explicit public alpha publish configuration.
