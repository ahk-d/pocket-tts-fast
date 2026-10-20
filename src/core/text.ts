// Text normalization and chunking, ported from vlapky/pocket-tts-js (MIT),
// with the sentence scanner and per-sentence EOS tail from the Codex runtime
// snapshot (codex-pocket-runtime-2026-10-20).

export interface TextOptions {
  remove_semicolons?: boolean;
  model_recommended_frames_after_eos?: number | null;
  pad_with_spaces_for_short_inputs?: boolean;
  max_token_per_chunk?: number;
}

export interface TokenCounter {
  encodeIds(text: string): number[];
  decodeIds(ids: readonly number[]): string;
}

export interface PreparedText {
  text: string;
  framesAfterEos: number;
}

/**
 * Replace typographic punctuation the tokenizer doesn't know. „ “ ’ – … etc.
 * are not in the vocabulary and become 3 raw-byte tokens each, which derailed
 * generation on real models (a quoted sentence was lost, "geht’s" turned into
 * babble). Quotation marks carry no sound and are removed; apostrophes, dashes
 * and ellipses become their ASCII forms, which are in the vocabulary.
 */
export function normalizePunctuation(text: string): string {
  return text
    .replace(/(\p{L})‘(?=\p{L})/gu, "$1'") // "geht‘s": ‘ inside a word is an apostrophe
    .replace(/[„“”«»‹›‚‘"]/g, "") // quotation marks (‘ elsewhere closes a ‚single quote‘)
    .replace(/[’ʼ′]/g, "'")
    .replace(/\s*[–—]\s*/g, " - ")
    .replace(/…/g, "...");
}

export function prepareTextPrompt(text: string, options: TextOptions = {}): PreparedText {
  let prompt = normalizePunctuation(text).trim();
  if (!prompt) return { text: "", framesAfterEos: 1 };
  prompt = prompt.replace(/\r/g, " ").replace(/\n/g, " ").replace(/\s+/g, " ");
  if (options.remove_semicolons) prompt = prompt.replace(/;/g, ",");

  const wordCount = countWords(prompt);
  const framesAfterEos = framesAfterEosForWordCount(wordCount, options);
  if (prompt && !/[A-ZÀ-Þ]/.test(prompt[0])) prompt = prompt[0].toUpperCase() + prompt.slice(1);
  if (prompt && /[0-9A-Za-zÀ-ÿ]/.test(prompt[prompt.length - 1])) prompt += ".";
  if (options.pad_with_spaces_for_short_inputs && wordCount < 5) prompt = "        " + prompt;
  return { text: prompt, framesAfterEos };
}

export function countWords(text: string): number {
  return text.split(/\s+/).filter(Boolean).length;
}

/**
 * Frames to keep generating after the model signals end-of-speech. The model's
 * recommendation wins; otherwise a local heuristic: 3 for short (<= 4 word)
 * chunks, whose final sound is otherwise cut, and 1 for longer ones. Apply it
 * per text chunk, never to the whole input.
 */
export function framesAfterEosForWordCount(wordCount: number, options: TextOptions = {}): number {
  if (options.model_recommended_frames_after_eos != null) return Number(options.model_recommended_frames_after_eos);
  return wordCount <= 4 ? 3 : 1;
}

const NON_TERMINAL_ABBREVIATIONS = new Set(["dr", "prof", "bzw", "ca", "usw", "nr", "z.b", "u.a", "d.h"]);
const GERMAN_MONTHS = new Set(["januar", "februar", "märz", "april", "mai", "juni", "juli", "august", "september", "oktober", "november", "dezember"]);

/**
 * Split at sentence-ending punctuation, keeping decimal numbers ("3.14"),
 * common German abbreviations ("Dr.", "z. B."), initials, ordinal dates
 * ("3. Oktober") and closing quotes with their sentence.
 */
export function splitTextIntoSentences(text: string): string[] {
  const sentences: string[] = [];
  let start = 0;
  for (let i = 0; i < text.length; i++) {
    if (!/[.!?]/.test(text[i])) continue;
    if (text[i] === ".") {
      // A decimal point or abbreviation is part of the current sentence.
      if (/\d/.test(text[i - 1] || "") && /\d/.test(text[i + 1] || "")) continue;
      const nextWord = text.slice(i + 1).match(/^\s+(\p{L}+)/u)?.[1].toLowerCase();
      if (/\d/.test(text[i - 1] || "") && nextWord && GERMAN_MONTHS.has(nextWord)) continue;
      const before = text.slice(start, i).trimEnd();
      const lastWord = before.match(/\p{L}+$/u)?.[0].toLowerCase();
      const abbreviation = before.match(/(?:z\.\s*b|u\.\s*a|d\.\s*h)$/iu)?.[0].replace(/\s+/g, "").toLowerCase();
      if ((lastWord && NON_TERMINAL_ABBREVIATIONS.has(lastWord)) || (abbreviation && NON_TERMINAL_ABBREVIATIONS.has(abbreviation))) continue;
      // Preserve spaced initialisms such as "z. B." and personal initials.
      if (lastWord?.length === 1 && /^\s+\p{Lu}[.\s]/u.test(text.slice(i + 1))) continue;
    }
    let end = i + 1;
    while (/[.!?]/.test(text[end] || "")) end++;
    while (/[“”’"'»）)\]}]/.test(text[end] || "")) end++;
    if (end < text.length && !/\s/.test(text[end])) continue;
    const sentence = text.slice(start, end).trim();
    if (sentence) sentences.push(sentence);
    start = end;
    i = end - 1;
  }
  const rest = text.slice(start).trim();
  if (rest) sentences.push(rest);
  return sentences;
}

function splitTokenIdsIntoChunks(tokenizer: TokenCounter, tokenIds: number[], maxTokens: number): string[] {
  const chunks: string[] = [];
  for (let i = 0; i < tokenIds.length; i += maxTokens) {
    const chunkText = tokenizer.decodeIds(tokenIds.slice(i, i + maxTokens)).trim();
    if (chunkText) chunks.push(chunkText);
  }
  return chunks;
}

/**
 * Normalize `text` and make every complete sentence its own text chunk; a
 * sentence over `max_token_per_chunk` tokens is split by tokens. Each chunk is
 * generated from the clean voice-conditioned state with its own EOS tail, and a
 * short pause is inserted between chunks. (Packing several sentences into one
 * chunk let end-of-speech cut the final sound of the last sentence.)
 */
export function splitIntoChunks(
  text: string,
  tokenizer: TokenCounter,
  options: TextOptions = {},
): { chunks: string[]; framesAfterEos: number } {
  const maxTokens = Number(options.max_token_per_chunk || 50);
  const prepared = prepareTextPrompt(text, options);
  if (!prepared.text) return { chunks: [], framesAfterEos: prepared.framesAfterEos };
  const sentences = splitTextIntoSentences(prepared.text);
  if (!sentences.length) return { chunks: [prepared.text], framesAfterEos: prepared.framesAfterEos };

  const chunks: string[] = [];
  for (const sentence of sentences) {
    const ids = tokenizer.encodeIds(sentence);
    if (ids.length > maxTokens) {
      for (const piece of splitTokenIdsIntoChunks(tokenizer, ids, maxTokens)) chunks.push(piece.trim());
    } else {
      chunks.push(sentence);
    }
  }
  return { chunks, framesAfterEos: prepared.framesAfterEos };
}
