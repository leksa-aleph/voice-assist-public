import sys
from pathlib import Path

IS_FROZEN = getattr(sys, "frozen", False)


def resource_path(rel: str) -> Path:
    """Файлы только для чтения (модели, ресурсы). В сборке лежат внутри бандла."""
    if IS_FROZEN:
        base = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    else:
        base = Path(__file__).resolve().parent
    return base / rel

def user_data_path(name: str, legacy: str | None = None) -> Path:
    """Файл для записи рядом с config.json. Один раз копируется со старого пути."""
    import shutil
    from config import config_dir
    target = config_dir() / name
    if legacy and not target.exists():
        old = Path(__file__).resolve().parent / legacy
        if old.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, target)
    return target