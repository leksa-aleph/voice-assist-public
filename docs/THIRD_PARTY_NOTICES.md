# Лицензии и исходники компонентов Voice Assist

Исходный код Voice Assist доступен по лицензии MIT. Windows-сборка
Voice Assist с PyQt6 распространяется на условиях GNU GPL версии 3.
Модели и отдельные компоненты сохраняют указанные ниже лицензии. Тексты
лицензий из Python-пакетов, Node.js и моделей установлены в _internal/licenses,
а лицензии пакетов Discord лежат рядом с пакетами в
_internal/discord_bridge/node_modules.

## Программа

| Компонент | Условия и исходники |
|---|---|
| Voice Assist | [Репозиторий](https://github.com/leksa-aleph/voice-assist-public) и [релизы с исходниками](https://github.com/leksa-aleph/voice-assist-public/releases). Инструкция по сборке находится в docs/DEVELOPMENT.md в исходниках. |
| PyQt6 | © Riverbank Computing Limited. [GPL-3.0-only и исходники PyQt6 6.11.0](https://pypi.org/project/PyQt6/6.11.0/#files). [Исходники PyQt6-sip 13.13.0](https://pypi.org/project/PyQt6-sip/13.13.0/#files). |
| Qt | [PyQt6-Qt6 6.11.2, LGPL-3.0](https://pypi.org/project/PyQt6-Qt6/6.11.2/). [Исходники Qt 6.11.2](https://download.qt.io/archive/qt/6.11/6.11.2/single/). |
| PyAV и FFmpeg | [PyAV 19.0.1](https://github.com/PyAV-Org/PyAV/tree/v19.0.1), BSD-3-Clause. Windows wheel включает FFmpeg, который сообщает LGPL 3.0 or later. [Конфигурация поставки PyAV](https://github.com/PyAV-Org/PyAV/blob/v19.0.1/scripts/ffmpeg-latest.json) и [сборка FFmpeg для PyAV](https://github.com/PyAV-Org/pyav-ffmpeg/tree/9.0.2-1). |
| ONNX Runtime | [ONNX Runtime 1.24.4](https://github.com/microsoft/onnxruntime/tree/v1.24.4), MIT, с Windows-провайдером [DirectML](https://github.com/microsoft/DirectML). |
| Python, Node.js | [Исходники CPython](https://www.python.org/downloads/source/) и [официальные выпуски Node.js](https://nodejs.org/en/download/archive). Версии Python и Node.js для конкретной сборки указываются рядом с установщиками. |
| PyInstaller | [Лицензия и исключение для загрузчика](https://pyinstaller.org/en/stable/license.html). [Исходники PyInstaller 6.22.3](https://github.com/pyinstaller/pyinstaller/tree/v6.22.3). |
| Остальные библиотеки | Версии Python-зависимостей указаны в requirements-windows.lock.txt, версии и источники пакетов Discord — в discord_bridge/package-lock.json. |

## Модели

| Модель | Источник и преобразования |
|---|---|
| Silero VAD v6 | [Silero Team](https://github.com/snakers4/silero-vad), MIT. Поставляется модель ONNX. |
| GigaAM v3 RNNT | [Salute Developers](https://github.com/salute-developers/GigaAM) и [ONNX-экспорт istupakov](https://huggingface.co/istupakov/gigaam-v3-onnx), MIT. Voice Assist разделяет графы кодировщика для CPU и DirectML и хранит общие веса в int8. Расчёт признаков основан на [onnx-asr](https://github.com/istupakov/onnx-asr), MIT. |
| WeSpeaker ResNet34-LM | [WeSpeaker и pyannote](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Для Voice Assist модель экспортирована в ONNX. |
| Parakeet TDT 0.6B v3 | [NVIDIA](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) и [ONNX-экспорт istupakov](https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). В Voice Assist кодировщик разделён на графы для CPU и DirectML с общими весами int8. Только в полной сборке. |
| opus-mt-tc-big-en-zle | [Helsinki-NLP](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-zle), [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0). Для Voice Assist модель экспортирована в ONNX и квантована в int8. Только в полной сборке. |
