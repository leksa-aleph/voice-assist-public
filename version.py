"""
Версия программы - одна на всё: окно, трей, журнал.

Установщик (installer/voice-assist.iss) держит те же два значения у себя -
tests.test_core сверяет, что они не разъехались.
"""

VERSION = '1.0.0'
STAGE = 'beta'      # '' - обычный выпуск

LABEL = f'{VERSION} {STAGE}'.strip()            # «1.0.0 beta» - в интерфейсе
TAG = f'{VERSION}-{STAGE}' if STAGE else VERSION  # «1.0.0-beta» - установщик, релиз, журнал
