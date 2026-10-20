// .npy parsing (bos_before_voice and voice embeddings) and writing.
// Parser ported from vlapky/pocket-tts-js (MIT).

import { asciiDecode } from "./utf8.js";

export interface Float32Tensor {
  data: Float32Array;
  shape: number[];
}

function bytesOf(input: ArrayBuffer | Uint8Array): Uint8Array {
  return input instanceof Uint8Array ? input : new Uint8Array(input);
}

// Copy so typed arrays are aligned regardless of the source offset.
function copy(bytes: Uint8Array, start: number, end: number): ArrayBuffer {
  return bytes.slice(start, end).buffer;
}

export function parseNpyFloat32(input: ArrayBuffer | Uint8Array): Float32Tensor {
  const bytes = bytesOf(input);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const expected = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59];
  for (let i = 0; i < expected.length; i++) {
    if (bytes[i] !== expected[i]) throw new Error("Invalid NPY file");
  }
  const major = view.getUint8(6);
  const headerLen = major === 1 ? view.getUint16(8, true) : view.getUint32(8, true);
  const headerOffset = major === 1 ? 10 : 12;
  const header = asciiDecode(bytes.subarray(headerOffset, headerOffset + headerLen));
  if (!/'descr':\s*'<f4'/.test(header)) throw new Error("NPY file is not little-endian float32");
  if (/'fortran_order':\s*True/.test(header)) throw new Error("Fortran-ordered NPY is not supported");
  const shapeMatch = header.match(/\(\s*([0-9,\s]+)\)/);
  if (!shapeMatch) throw new Error("Could not parse NPY shape");
  const shape = shapeMatch[1]
    .split(",")
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => Number.parseInt(p, 10));
  const dataOffset = headerOffset + headerLen;
  return { data: new Float32Array(copy(bytes, dataOffset, bytes.length)), shape };
}

/** Serialize a float32 tensor as .npy (v1.0, little-endian, C order). */
export function encodeNpyFloat32(data: Float32Array, shape: readonly number[]): Uint8Array {
  const dims = shape.length === 1 ? `${shape[0]},` : shape.join(", ");
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${dims}), }`;
  const unpadded = 10 + header.length + 1;
  header += " ".repeat((64 - (unpadded % 64)) % 64) + "\n";
  const out = new Uint8Array(10 + header.length + data.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  new DataView(out.buffer).setUint16(8, header.length, true);
  for (let i = 0; i < header.length; i++) out[10 + i] = header.charCodeAt(i);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), 10 + header.length);
  return out;
}
