// Text normalization and sentence chunking (pure helpers, no model needed).
import test from "node:test";
import assert from "node:assert/strict";
import { prepareTextPrompt, normalizePunctuation, splitIntoChunks, splitTextIntoSentences, framesAfterEosForWordCount, utf8Encode, utf8Decode } from "../dist/index.js";

// Whitespace-word "tokenizer": one id per word, for predictable chunk limits.
function wordTokenizer() {
  const vocab = [];
  return {
    encodeIds: (t) => t.split(/\s+/).filter(Boolean).map((w) => (vocab.includes(w) ? vocab.indexOf(w) : vocab.push(w) - 1)),
    decodeIds: (ids) => ids.map((i) => vocab[i]).join(" "),
  };
}

test("prompt normalization: whitespace, capitalization, final period, semicolons", () => {
  assert.deepEqual(prepareTextPrompt("  guten\n\nmorgen  "), { text: "Guten morgen.", framesAfterEos: 3 });
  assert.equal(prepareTextPrompt("Ja; nein", { remove_semicolons: true }).text, "Ja, nein.");
  assert.equal(prepareTextPrompt("Wie spät ist es?").text, "Wie spät ist es?", "existing punctuation kept");
  assert.equal(prepareTextPrompt("über alles").text, "Über alles.", "umlaut capitalized");
  assert.equal(prepareTextPrompt("eins zwei drei vier fünf").framesAfterEos, 1);
  assert.equal(prepareTextPrompt("a b", { model_recommended_frames_after_eos: 2 }).framesAfterEos, 2);
  assert.deepEqual(prepareTextPrompt("   "), { text: "", framesAfterEos: 1 });
});

test("every complete sentence becomes its own chunk (no packing)", () => {
  const tok = wordTokenizer();
  const { chunks } = splitIntoChunks("Eins zwei drei. Vier fünf. Sechs sieben acht neun.", tok, { max_token_per_chunk: 50 });
  assert.deepEqual(chunks, ["Eins zwei drei.", "Vier fünf.", "Sechs sieben acht neun."]);
});

// Ported from codex-pocket-runtime-2026-10-20/runtime/test/chunk-diagnostics.test.js.
test("complete sentences in the reported slow phrase become separate model chunks", () => {
  const { chunks } = splitIntoChunks("hello! ich heisse anna und ich komme aus berlin. wie heisst du .", wordTokenizer());
  assert.deepEqual(chunks, ["Hello!", "ich heisse anna und ich komme aus berlin.", "wie heisst du ."]);
});

test("decimal points and common abbreviations stay inside their sentence", () => {
  const { chunks } = splitIntoChunks("Der Zug fährt um 3.14 Uhr. Dr. Müller wartet. Danach geht es weiter.", wordTokenizer());
  assert.deepEqual(chunks, ["Der Zug fährt um 3.14 Uhr.", "Dr. Müller wartet.", "Danach geht es weiter."]);
});

test("sentence boundary after closing quotes keeps punctuation and all words", () => {
  // Scanner level (quotes kept): the closing quote stays with its sentence.
  assert.deepEqual(splitTextIntoSentences("„Hallo!“, sagte sie. „Guten Morgen!“, antwortete er."), ["„Hallo!“, sagte sie.", "„Guten Morgen!“, antwortete er."]);
  assert.deepEqual(splitTextIntoSentences("„Hallo!“ Wie geht es?"), ["„Hallo!“", "Wie geht es?"]);
  // Prompt level: quotation marks are removed before chunking, all words kept.
  assert.deepEqual(splitIntoChunks("„Hallo!“, sagte sie. „Guten Morgen!“, antwortete er.", wordTokenizer()).chunks, ["Hallo!, sagte sie.", "Guten Morgen!, antwortete er."]);
  assert.deepEqual(splitIntoChunks("„Hallo!“ Wie geht es?", wordTokenizer()).chunks, ["Hallo!", "Wie geht es?"]);
});

test("typographic punctuation outside the tokenizer vocabulary is normalized", () => {
  assert.equal(normalizePunctuation("„Hallo“ «Welt» ‚so‘ \"ja\""), "Hallo Welt so ja");
  assert.equal(normalizePunctuation("Wie geht’s? Mir geht‘s gut."), "Wie geht's? Mir geht's gut.");
  assert.equal(normalizePunctuation("morgen – vielleicht—später"), "morgen - vielleicht - später");
  assert.equal(normalizePunctuation("Na ja… mal sehen."), "Na ja... mal sehen.");
  assert.equal(normalizePunctuation("Über 3,5 % für 10 €."), "Über 3,5 % für 10 €.", "letters, numbers and symbols untouched");
});

test("ordinal date stays intact while the following sentence is separate", () => {
  assert.deepEqual(splitIntoChunks("Ich komme am 3. Oktober. Wie geht es?", wordTokenizer()).chunks, ["Ich komme am 3. Oktober.", "Wie geht es?"]);
});

test("spaced abbreviations and initials do not end a sentence", () => {
  assert.deepEqual(splitTextIntoSentences("Wir kaufen z. B. Brot. Ca. zehn Leute kommen."), ["Wir kaufen z. B. Brot.", "Ca. zehn Leute kommen."]);
  // Consecutive initials stay together; an initial right before a surname still
  // ends a sentence (ambiguous with "Plan B. Meier kommt."). No words are lost.
  assert.deepEqual(splitTextIntoSentences("A. B. Meier kommt mit."), ["A. B.", "Meier kommt mit."]);
  assert.deepEqual(splitTextIntoSentences("Wirklich?! Ja..."), ["Wirklich?!", "Ja..."]);
});

test("long sentences still obey the 50-token limit without losing words", () => {
  const words = Array.from({ length: 63 }, (_, i) => `wort${i + 1}`);
  const { chunks } = splitIntoChunks(`${words.join(" ")}.`, wordTokenizer(), { max_token_per_chunk: 50 });
  assert.deepEqual(chunks.map((c) => c.split(/\s+/).length), [50, 13]);
  assert.equal(chunks.join(" "), `Wort1 ${words.slice(1).join(" ")}.`);
});

test("EOS tail: 3 frames for <= 4 words, 1 otherwise, model recommendation wins", () => {
  assert.equal(framesAfterEosForWordCount(1), 3);
  assert.equal(framesAfterEosForWordCount(4), 3);
  assert.equal(framesAfterEosForWordCount(5), 1);
  assert.equal(framesAfterEosForWordCount(2, { model_recommended_frames_after_eos: 2 }), 2);
  assert.equal(framesAfterEosForWordCount(9, { model_recommended_frames_after_eos: 0 }), 0);
});

test("a sentence longer than the limit is split by tokens", () => {
  const tok = wordTokenizer();
  const { chunks } = splitIntoChunks("a b c d e f g", tok, { max_token_per_chunk: 3 });
  assert.deepEqual(chunks, ["A b c", "d e f", "g."]);
});

test("question and exclamation marks end sentences; trailing text without punctuation is kept", () => {
  const tok = wordTokenizer();
  const { chunks } = splitIntoChunks("Kommst du? Ja! Dann los", tok);
  assert.deepEqual(chunks, ["Kommst du?", "Ja!", "Dann los."]);
});

test("UTF-8 helpers match TextEncoder/TextDecoder (React Native has neither everywhere)", () => {
  for (const s of ["", "abc", "Grüße aus Köln – ß ✓", "emoji 😀 and CJK 漢字", "▁Hallo"]) {
    assert.deepEqual(utf8Encode(s), new TextEncoder().encode(s));
    assert.equal(utf8Decode(new TextEncoder().encode(s)), s);
  }
  const bad = Uint8Array.from([0x61, 0xc3, 0x28, 0xe2, 0x82, 0x62]);
  assert.equal(utf8Decode(bad), new TextDecoder().decode(bad));
});
