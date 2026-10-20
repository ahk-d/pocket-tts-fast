"""Write *_dyncache.onnx copies whose KV-cache inputs/outputs have a dynamic length.

The exported graphs declare every attention cache as 1000 slots, so onnxruntime-web
copies the whole buffer (49 MB for FlowLM, 16 MB for the Mimi decoder) from JS into
WASM and back out on every call. The graphs themselves never read past the filled
prefix:

* FlowLM self-attention writes with ScatterND at `step` and reads Slice[0:step+len],
  so the cache only needs `step + len` slots. Only the declared dim changes.
* The decoder's encoder_transformer caches are Identity passthroughs, so they can be
  zero-length. The decoder_transformer ring caches (Mod on capacity) stay at 1000.

Weights and nodes are untouched; only the declared shapes of those graph inputs and
outputs become symbolic and stale intermediate shape annotations are dropped so
onnxruntime re-infers them.

The runtime does the same rewrite in memory at load time (src/core/onnx-patch.ts),
so this script is only needed to prebuild graphs for hosting or inspection.
Verify with `npm run test:models` or `node bench/bench.mjs --dynamic prebuilt`
(PCM hashes must match the fixed-cache run).

    pip install onnx
    python3 tools/make_dynamic_cache.py [--models models/german]
"""
import argparse
import collections
import json
from pathlib import Path

import onnx

MODELS = Path(__file__).resolve().parent.parent / 'models' / 'german'
FLOW = ['flow_lm_main.onnx', 'flow_lm_main_int8.onnx']
DECODER = ['mimi_decoder.onnx', 'mimi_decoder_int8.onnx']


def set_dim(value, index, name):
    dim = value.type.tensor_type.shape.dim[index]
    assert dim.dim_value == 1000, (value.name, dim)
    dim.ClearField('dim_value')
    dim.dim_param = name


def patch(filename, select):
    source = MODELS / filename
    model = onnx.load(source)
    graph = model.graph
    consumers = collections.defaultdict(list)
    for node in graph.node:
        for name in node.input:
            consumers[name].append(node.op_type)
    outputs = {o.name: o for o in graph.output}
    changed = []
    for value in graph.input:
        hit = select(value, consumers[value.name])
        if hit is None:
            continue
        index, dim_name, output_name = hit
        set_dim(value, index, dim_name)
        set_dim(outputs[output_name], index, dim_name)
        changed.append(value.name)
    del graph.value_info[:]
    target = source.with_name(source.stem + '_dyncache.onnx')
    onnx.save(model, target)
    print(f'{target.name}: {len(changed)} cache inputs made dynamic')
    return changed


def main():
    global MODELS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--models', type=Path, default=MODELS, help='folder with bundle.json and the graphs')
    MODELS = parser.parse_args().models.resolve()
    bundle = json.loads((MODELS / 'bundle.json').read_text())
    flow = {e['input_name']: e['output_name'] for e in bundle['flow_lm_state_manifest'] if e['key'] == 'cache'}
    mimi = {e['input_name']: e['output_name'] for e in bundle['mimi_state_manifest'] if e['key'] == 'cache'}

    def flow_select(value, ops):
        return (2, 'cache_len', flow[value.name]) if value.name in flow else None

    def decoder_select(value, ops):
        # Only caches that nothing reads except an Identity to the matching output.
        return (3, 'unused_len', mimi[value.name]) if value.name in mimi and ops == ['Identity'] else None

    for filename in FLOW:
        if (MODELS / filename).exists():
            assert len(patch(filename, flow_select)) == len(flow)
    for filename in DECODER:
        if (MODELS / filename).exists():
            assert len(patch(filename, decoder_select)) == 2


if __name__ == '__main__':
    main()
