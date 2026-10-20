// UTF-8 helpers that do not depend on TextEncoder/TextDecoder, which are not
// available on every React Native JavaScript engine.

export function utf8Encode(text: string): Uint8Array {
  const out: number[] = [];
  for (const ch of text) {
    let cp = ch.codePointAt(0)!;
    if (cp < 0x80) out.push(cp);
    else if (cp < 0x800) out.push(0xc0 | (cp >> 6), 0x80 | (cp & 63));
    else if (cp < 0x10000) {
      if (cp >= 0xd800 && cp <= 0xdfff) cp = 0xfffd; // lone surrogate
      out.push(0xe0 | (cp >> 12), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
    } else out.push(0xf0 | (cp >> 18), 0x80 | ((cp >> 12) & 63), 0x80 | ((cp >> 6) & 63), 0x80 | (cp & 63));
  }
  return Uint8Array.from(out);
}

// Lenient decoder: invalid sequences become U+FFFD (matches TextDecoder fatal:false
// for the inputs the tokenizer produces).
export function utf8Decode(bytes: Uint8Array): string {
  let out = "";
  let i = 0;
  while (i < bytes.length) {
    const b = bytes[i];
    let cp = 0xfffd;
    let need = 0;
    if (b < 0x80) cp = b;
    else if (b >= 0xc2 && b < 0xe0) { cp = b & 31; need = 1; }
    else if (b >= 0xe0 && b < 0xf0) { cp = b & 15; need = 2; }
    else if (b >= 0xf0 && b < 0xf5) { cp = b & 7; need = 3; }
    i++;
    let ok = true;
    for (let k = 0; k < need; k++) {
      const c = bytes[i];
      if (c === undefined || (c & 0xc0) !== 0x80) { ok = false; break; }
      cp = (cp << 6) | (c & 63);
      i++;
    }
    if (!ok || (need === 2 && (cp < 0x800 || (cp >= 0xd800 && cp <= 0xdfff))) || (need === 3 && (cp < 0x10000 || cp > 0x10ffff))) {
      cp = 0xfffd;
    }
    out += String.fromCodePoint(cp);
  }
  return out;
}

export function asciiDecode(bytes: Uint8Array): string {
  let out = "";
  for (let i = 0; i < bytes.length; i++) out += String.fromCharCode(bytes[i]);
  return out;
}
