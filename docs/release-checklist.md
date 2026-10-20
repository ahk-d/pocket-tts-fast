# Release checklist

Status as of 2026-10-20. Nothing has been published, pushed, uploaded or sent.

## Done in this repository

- [x] One framework-independent ESM package with checked declarations (`npm run test:types`).
- [x] Pinned, SHA-256-verified asset manifest; revision-keyed cache; retries; integrity failures never fall back.
- [x] In-memory dynamic-cache patch plus the offline `tools/make_dynamic_cache.py`; graph-level unit tests.
- [x] Real-model PCM identity, fixed vs dynamic (stock voice, short and multi-sentence) and against the original runtime's hashes.
- [x] Stock-voice-first demo (`demo/`), with the encoder loaded only for cloning.
- [x] Packed-tarball installs into vanilla, Vite and React-Vite examples (`npm run test:consumers`), with the React consumer exercised in a browser.
- [x] EOS / sentence-pause repair ported from the Codex snapshot; hard real-model ending test (14 items) plus a Whisper transcript check; browser check of raw vs scheduled endings in immediate / queued / complete modes, both cache modes, stop→retry.
- [x] Source inventory with hashes (`docs/source-inventory.json`) and notices (`THIRD_PARTY_NOTICES.md`).

## Open before an alpha

- [ ] Human listening check of the ending corpus (`test/models/endings.de.json`), especially the utterance-initial "Ich" that Whisper-small missed in 2 items.
- [ ] Decide on number normalization ("3.15 Uhr").
- [ ] Decide the copyright line in `LICENSE` (currently "Flowent contributors").
- [ ] Confirm ownership of the npm scope `@flowent` (`@flowent/pocket-tts` returned 404, so the name is free, but scope ownership is unverified) and of the GitHub and Hugging Face accounts.
- [ ] Upload `voices/alba-casual.emb.npy` (with a model card: source, license, preparation, hash) to a public, CORS-enabled, commit-pinned location, then set it as the default `voiceBaseUrl`.
- [ ] Controlled benchmarks per `bench/protocol.md` on an idle machine and at least one other device and browser; browser runner; the `pocket-tts-js@0.1.0` comparison row.
- [ ] Threaded run of the demo in Chrome, Firefox and Safari with isolation on (the Claude browser pane can only run 1 thread).
- [ ] Vue, Svelte and Next.js consumers, if they will be listed as supported.
- [ ] Blind listening check per precision; transcript check on the full corpus.

## Later, only after explicit approval

- [ ] Push the repository, deploy the demo with COOP/COEP, publish the `alpha` dist-tag, install from npm in a clean consumer.
- [ ] Focused PR to the graph exporter (symbolic cache axis) and/or `vlapky/pocket-tts-js` (runtime allocation), with the identity evidence.
- [ ] Minimal README listing PR to `kyutai-labs/pocket-tts` ("In-browser implementations").
