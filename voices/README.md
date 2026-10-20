# Alba stock voice embedding — CC BY 4.0

Voice: Alba MacKenna, via Kyutai tts-voices. Source recording casual.wav:
https://huggingface.co/kyutai/tts-voices/resolve/323332d33f997de8394f24a193e1a76df720e01a/alba-mackenna/casual.wav

License: Creative Commons Attribution 4.0 International:
https://creativecommons.org/licenses/by/4.0/

This is a derived Mimi encoder embedding, not the source recording. Prepared from the first ten seconds using Pocket TTS's encoder; preparation recipe:
https://github.com/ahk-d/pocket-tts-fast/blob/main/tools/prepare-stock-voice.mjs

Embedding SHA-256: e4549924b9054a02c33907431ebce573011837f68084785499c43df9945741e0
Size: 512128 bytes. Source recording SHA-256: 46264e83cb99115c3d210260e029117566d9c64f20266d10daa78107759ede3e.

Retain attribution and license information when redistributing this file. It is not covered by this package's MIT code license. No endorsement is implied.

Copy this folder into your application's static/public directory. When served as /voices/alba-casual.emb.npy, configure new PocketTTS({ voiceBaseUrl: "/" }). The large ONNX models remain downloaded separately at runtime.
