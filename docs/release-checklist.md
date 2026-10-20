# Release checklist

Status as of 2026-10-20: initial source is public on GitHub main. The npm alpha is being prepared; no npm publication has been verified yet.

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
- [x] Retain the existing Flowent contributors and upstream vlapky copyright lines; voice assets remain separately CC-BY-4.0.
- [x] Verify npm account ali-dulaimi and select unscoped pocket-tts-fast (no organization required); GitHub repository ahk-d/pocket-tts-fast is public on main. No new Hugging Face account/upload is needed for this alpha.
- [x] Include the 512128-byte manifest-matching Alba embedding in npm voices/ with CC-BY-4.0 attribution, preparation recipe and hash; document consumer self-hosting under /voices/ with voiceBaseUrl set to its parent. Large models remain remote.
- [ ] Controlled benchmarks per `bench/protocol.md` on an idle machine and at least one other device and browser; browser runner; the `pocket-tts-js@0.1.0` comparison row.
- [ ] Threaded run of the demo in Chrome, Firefox and Safari with isolation on (the Claude browser pane can only run 1 thread).
- [ ] Vue, Svelte and Next.js consumers, if they will be listed as supported.
- [ ] Blind listening check per precision; transcript check on the full corpus.

## Later, only after explicit approval

- [ ] Push the repository, deploy the demo with COOP/COEP, publish the `alpha` dist-tag, install from npm in a clean consumer.
- [ ] Focused PR to the graph exporter (symbolic cache axis) and/or `vlapky/pocket-tts-js` (runtime allocation), with the identity evidence.
- [ ] Minimal README listing PR to `kyutai-labs/pocket-tts` ("In-browser implementations").

## Alpha release preparation

The user selected the unscoped package pocket-tts-fast. Account login was verified as ali-dulaimi; no npm organization is required. Public alpha publishConfig and prepack build are configured. Packed-consumer checks cover the stock voice hash/size and package allowlist. Controlled benchmarks, broader browsers/devices and subjective listening remain outstanding; the alpha README labels these boundaries.

Fresh checks for the prepared alpha: 85 unit tests passed; public types passed; packed installs/builds passed for vanilla, Vite and React/Vite. Real-model checks: fixed/dynamic PCM identity passed on three inputs, and all fourteen ending-corpus cases passed (every generated frame delivered, EOS reached, numeric tail thresholds satisfied). Optional original-reference hash comparison was skipped because POCKET_TTS_REFERENCE_F32 was unset. Human naturalness and transcript completeness are not inferred from these checks. npm login succeeded; the unscoped name was selected and publication is pending. No npm publication has occurred.
