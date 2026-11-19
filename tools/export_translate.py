"""
Готовит models/translate/ - перевод английского на русский для режима «Перевод».

Модель - Helsinki-NLP/opus-mt-tc-big-en-zle (Apache-2.0, Marian transformer-big):
заметно точнее базовой opus-mt-en-ru («функция, которая вызывает себя», а не
«называет»), в int8 по каналам - 370 МБ и 80-160 мс на фразу.

Шаги (один раз, у разработчика):
  1. экспорт в ONNX из окружения с torch (optimum-onnx):
       optimum-cli export onnx --model Helsinki-NLP/opus-mt-tc-big-en-zle \\
           --task text2text-generation-with-past build/onnx/tc-big-en-zle
  2. этот скрипт из build/venv-tools (onnx, onnxruntime, sentencepiece):
       build/venv-tools/Scripts/python tools/export_translate.py

Скрипт квантует кодировщик и объединённый декодер в int8 (по каналам: так
перевод не отличается от fp32) и сохраняет словарь в translate.json, чтобы
программе не нужны были sentencepiece и transformers.
"""
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'build' / 'onnx' / 'tc-big-en-zle'
OUT = ROOT / 'models' / 'translate'
TARGET = '>>rus<<'


def main():
    import sentencepiece as spm
    from onnxruntime.quantization import QuantType, quantize_dynamic

    OUT.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(SRC / 'encoder_model.onnx'), str(OUT / 'encoder.int8.onnx'),
                     weight_type=QuantType.QInt8, per_channel=True)
    quantize_dynamic(str(SRC / 'decoder_model_merged.onnx'), str(OUT / 'decoder.int8.onnx'),
                     weight_type=QuantType.QInt8, per_channel=True, extra_options={'EnableSubgraph': True})

    cfg = json.loads((SRC / 'config.json').read_text(encoding='utf-8'))
    vocab = json.loads((SRC / 'vocab.json').read_text(encoding='utf-8'))
    sp = spm.SentencePieceProcessor(model_file=str(SRC / 'source.spm'))
    unk = vocab['<unk>']
    source = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if sp.is_control(i) or sp.is_unknown(i):
            continue
        source.append([piece, sp.get_score(i), vocab.get(piece, unk)])
    target = [''] * (max(vocab.values()) + 1)
    for piece, i in vocab.items():
        target[i] = piece
    meta = {
        'pad': cfg['pad_token_id'], 'eos': cfg['eos_token_id'], 'start': cfg['decoder_start_token_id'],
        'unk': unk, 'target': vocab[TARGET],
        'heads': cfg['decoder_attention_heads'], 'head_dim': cfg['d_model'] // cfg['decoder_attention_heads'],
        'source': source, 'target_vocab': target,
    }
    (OUT / 'translate.json').write_text(json.dumps(meta, ensure_ascii=False), encoding='utf-8')
    print('готово:', ', '.join(f'{p.name} {p.stat().st_size // 1_000_000} МБ' for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
