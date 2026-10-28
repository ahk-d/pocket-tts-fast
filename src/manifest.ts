// Pinned, checksummed model assets. The runtime downloads exactly these files
// and rejects any byte that does not match; nothing is fetched at npm install.

export type Precision = "fp32" | "int8-core" | "int8-wide";

export type AssetRole = "metadata" | "tokenizer" | "bos" | "conditioner" | "flow-main" | "flow-head" | "decoder" | "encoder" | "voice";

export interface Asset {
  file: string;
  sha256: string;
  bytes: number;
  role: AssetRole;
  precision: "fp32" | "int8" | "none";
  /** Only needed for some uses (the encoder only for cloning). */
  optional: boolean;
}

export interface StockVoice {
  id: string;
  label: string;
  /** Precision-independent mimi_encoder embedding (.npy, float32 [1, frames, 1024]). */
  file: string;
  sha256: string;
  bytes: number;
  source: { url: string; sha256: string; license: string; attribution: string };
}

export interface LanguageManifest {
  id: string;
  language: string;
  repo: string;
  revision: string;
  /** Default base for graphs: `${modelBaseUrl}/${file}`. */
  modelBaseUrl: string;
  sampleRate: number;
  assets: Asset[];
  precisions: Record<Precision, string[]>;
  voices: StockVoice[];
}

const REVISION = "58a6d00cf13d239b6748cb0769f35c580a8f606c";

export const GERMAN: LanguageManifest = {
  id: `german@${REVISION.slice(0, 7)}`,
  language: "german",
  repo: "KevinAHM/pocket-tts-onnx",
  revision: REVISION,
  modelBaseUrl: `https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/${REVISION}/onnx/german`,
  sampleRate: 24000,
  assets: [
    { file: "bundle.json", sha256: "7f032e4df0db3a02e24871dfa496e2b7ffa095dd98640bfcf12c04b8671ac60e", bytes: 24362, role: "metadata", precision: "none", optional: false },
    { file: "tokenizer.model", sha256: "389079b9c67cd17ff1c565d53bc695df390b4ca8abf2300947e659585edd9d2a", bytes: 59837, role: "tokenizer", precision: "none", optional: false },
    { file: "bos_before_voice.npy", sha256: "8c866aec4bcd5f0e6ca788f4b24eacba799b02262905e9f5d6f967409872f8df", bytes: 4224, role: "bos", precision: "none", optional: false },
    { file: "text_conditioner.onnx", sha256: "a004c74047bc1c4208f6ca150c6337fb541ab03e154493f456d43baa0fd469f1", bytes: 16388344, role: "conditioner", precision: "fp32", optional: false },
    { file: "mimi_encoder.onnx", sha256: "71a00b2c6270af5f99389609bcd3d4682bf692fccb8f0ff9b21c3d2e473a204b", bytes: 39768446, role: "encoder", precision: "fp32", optional: true },
    { file: "flow_lm_main.onnx", sha256: "368c05f4b2752a9624ae380fffa065ba46db261ee8fac6019994c6a78c307f6c", bytes: 302742149, role: "flow-main", precision: "fp32", optional: false },
    { file: "flow_lm_main_int8.onnx", sha256: "3d5e9cd303ecd26fc75b91e2766bce6de3d274d397d3ee9a392f6299366681ef", bytes: 76341079, role: "flow-main", precision: "int8", optional: false },
    { file: "flow_lm_flow.onnx", sha256: "3b6dceddb6ea594f25d871d5466112b1c7a18bad0ef43b967a7a857b518143a1", bytes: 39097095, role: "flow-head", precision: "fp32", optional: false },
    { file: "flow_lm_flow_int8.onnx", sha256: "2b0eb8381c49672ac1050c50221db02033b5dc6c346ab45c67bd05344257f06c", bytes: 9962530, role: "flow-head", precision: "int8", optional: false },
    { file: "mimi_decoder.onnx", sha256: "e068e4b2a30e721a48171f15f09bdc082ad9f531434e9e2bcee2b2cd4e3e77f6", bytes: 41471926, role: "decoder", precision: "fp32", optional: false },
    { file: "mimi_decoder_int8.onnx", sha256: "3fe868810d66fa0c26bb3ae0d5ee65cae83028c0fe83398bcf3d96478f68083b", bytes: 22684077, role: "decoder", precision: "int8", optional: false },
  ],
  precisions: {
    fp32: [],
    "int8-core": ["flow_lm_main"],
    "int8-wide": ["flow_lm_main", "flow_lm_flow", "mimi_decoder"],
  },
  voices: [
    {
      id: "alba",
      label: "Alba (casual)",
      file: "voices/alba-casual.emb.npy",
      // Output of tools/prepare-stock-voice.mjs (see docs/release-assets.json).
      sha256: "e4549924b9054a02c33907431ebce573011837f68084785499c43df9945741e0",
      bytes: 512128,
      source: {
        url: "https://huggingface.co/kyutai/tts-voices/resolve/323332d33f997de8394f24a193e1a76df720e01a/alba-mackenna/casual.wav",
        sha256: "46264e83cb99115c3d210260e029117566d9c64f20266d10daa78107759ede3e",
        license: "CC-BY-4.0",
        attribution: "Voice: Alba MacKenna, via kyutai/tts-voices (CC BY 4.0)",
      },
    },
  ],
};

// Same revision, folder onnx/english_2026-04. The graphs share German's layout and
// state shapes, but every file (encoder included) has its own weights, so the
// stock voice embedding is encoded separately.
export const ENGLISH: LanguageManifest = {
  id: `english@${REVISION.slice(0, 7)}`,
  language: "english",
  repo: "KevinAHM/pocket-tts-onnx",
  revision: REVISION,
  modelBaseUrl: `https://huggingface.co/KevinAHM/pocket-tts-onnx/resolve/${REVISION}/onnx/english_2026-04`,
  sampleRate: 24000,
  assets: [
    { file: "bundle.json", sha256: "bab643150f437f37df080a710520ff39ed9ebd9a339f8ebdc739f7eddfc28b3f", bytes: 24381, role: "metadata", precision: "none", optional: false },
    { file: "tokenizer.model", sha256: "d461765ae179566678c93091c5fa6f2984c31bbe990bf1aa62d92c64d91bc3f6", bytes: 59339, role: "tokenizer", precision: "none", optional: false },
    { file: "bos_before_voice.npy", sha256: "f46edf4f7007b7ba4ea58831f49d003e59e167b4641c44bb3addfe9231a780b1", bytes: 4224, role: "bos", precision: "none", optional: false },
    { file: "text_conditioner.onnx", sha256: "4ecee995fb69f85c7a7493d11f7b5ee15d9950facc7ab3f5c9c49ef1e03847bb", bytes: 16388344, role: "conditioner", precision: "fp32", optional: false },
    { file: "mimi_encoder.onnx", sha256: "853e2ca623b8782d94c3745ec6133bfdff7ce33d9b11128bd29ea03f28d76e3d", bytes: 39768446, role: "encoder", precision: "fp32", optional: true },
    { file: "flow_lm_main.onnx", sha256: "6d18315e2c33ca3e3aa4a4e3dca22f56d007fd823127e24948b37695bf54190f", bytes: 302742149, role: "flow-main", precision: "fp32", optional: false },
    { file: "flow_lm_main_int8.onnx", sha256: "f9bd8106b79a0192c1c43399ab938fb24900a95c1c599870d75a884e99000116", bytes: 76341079, role: "flow-main", precision: "int8", optional: false },
    { file: "flow_lm_flow.onnx", sha256: "085d239f68897e28fb06e95c743738ad8b8c092ee6dc55f5491313e81ff08062", bytes: 39097095, role: "flow-head", precision: "fp32", optional: false },
    { file: "flow_lm_flow_int8.onnx", sha256: "3dd781ee5abee9e195320bf0106bebd6372a852b3b36352524ee78b40554635d", bytes: 9962530, role: "flow-head", precision: "int8", optional: false },
    { file: "mimi_decoder.onnx", sha256: "86f038caa02a96a0ff9c25526a0ff43a4906c418197ed72d3e30f720ac7ce802", bytes: 41471926, role: "decoder", precision: "fp32", optional: false },
    { file: "mimi_decoder_int8.onnx", sha256: "3630450a3297a101792a6ac66619ebc70ab916b265e6220c2afaef8b1673f925", bytes: 22684077, role: "decoder", precision: "int8", optional: false },
  ],
  precisions: GERMAN.precisions,
  voices: [
    {
      ...GERMAN.voices[0],
      file: "voices/alba-casual.en.emb.npy",
      // Output of tools/prepare-stock-voice.mjs --language english.
      sha256: "5094e4009bc23a6738a024e73b41bb459711d767c8ee45f1fdd39ca5613d147b",
      bytes: 512128,
    },
  ],
};

export const MANIFESTS: Record<string, LanguageManifest> = { german: GERMAN, english: ENGLISH };

export function assetFor(manifest: LanguageManifest, file: string): Asset | undefined {
  return manifest.assets.find((a) => a.file === file);
}

/** Files a precision needs; the encoder only with `cloning`. */
export function filesFor(manifest: LanguageManifest, precision: Precision, { cloning = false } = {}): Asset[] {
  const int8 = new Set(manifest.precisions[precision]);
  const stemOf = (file: string) => file.replace(/(_int8)?\.onnx$/, "");
  return manifest.assets.filter((a) => {
    if (a.role === "encoder") return cloning;
    if (!a.file.endsWith(".onnx")) return true;
    return (a.precision === "int8") === int8.has(stemOf(a.file));
  });
}

/** Total bytes a fresh visitor downloads for a precision (graphs and metadata, excluding the onnxruntime-web runtime). */
export function downloadBytes(manifest: LanguageManifest, precision: Precision, opts: { cloning?: boolean } = {}): number {
  return filesFor(manifest, precision, opts).reduce((n, a) => n + a.bytes, 0);
}
