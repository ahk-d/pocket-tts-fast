"""Transcribe generated WAVs with Whisper and check that every word, and
especially the last one, was spoken. An intelligibility check, not a
naturalness score.

    pip install faster-whisper
    POCKET_TTS_WAV_DIR=out npm run test:models      # writes out/<id>.wav
    python3 tools/check_transcripts.py out test/models/endings.de.json [--model small] [--language de|en]

Exit code 1 if a last word is missing or word recall < --min-recall.
Numbers ("3.15") are compared loosely: Pocket TTS does not normalize digits and
ASR writes them back in varied forms, so digit tokens are skipped.
"""
import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path


def words(text, lang="de"):
    text = unicodedata.normalize("NFC", text.lower())
    text = re.sub(r"[„“”«»‚‘\"]", "", text).replace("’", "'")
    if lang == "de":
        text = re.sub(r"(\w)'s\b", r"\1 es", text)  # "geht's" is transcribed as "geht es"
    return [w for w in re.findall(r"[\w']+", text) if not w.isdigit()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav_dir", type=Path)
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--model", default="small")
    ap.add_argument("--language", default="de")
    ap.add_argument("--min-recall", type=float, default=0.8)
    args = ap.parse_args()
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device="cpu", compute_type="int8")
    failures = 0
    for item in json.loads(args.corpus.read_text()):
        wav = args.wav_dir / f"{item['id']}.wav"
        if not wav.exists():
            print(f"skip   {item['id']} (no wav)")
            continue
        segments, _ = model.transcribe(str(wav), language=args.language, beam_size=5, word_timestamps=True)
        heard_text = " ".join(s.text.strip() for s in segments)
        expected, heard = words(item["text"], args.language), words(heard_text, args.language)
        pool = list(heard)
        hits = 0
        for w in expected:
            if w in pool:
                pool.remove(w)
                hits += 1
        recall = hits / max(1, len(expected))
        last_ok = bool(heard) and heard[-1] == expected[-1]
        ok = last_ok and recall >= args.min_recall
        failures += not ok
        print(f"{'ok    ' if ok else 'FAIL  '} {item['id']:26} recall {recall:.2f} last {'ok' if last_ok else 'MISSING'} | {heard_text}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
