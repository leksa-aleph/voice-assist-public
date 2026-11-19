# ============================================================
# Voice Assist - подготовка запуска из исходников на Windows
# Запускать через install.bat (двойной клик) или вручную:
#   PowerShell -ExecutionPolicy Bypass -File .\setup.ps1
# ============================================================

$ErrorActionPreference = 'Stop'

# Цветной вывод
function Write-Step($msg)  { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-OK($msg)    { Write-Host "[OK] $msg"      -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "[!]  $msg"      -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "[X]  $msg"      -ForegroundColor Red }

# Кодировка вывода - чтобы кириллица не превращалась в кракозябры
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

# ------------------------------------------------------------
# 0. Проверка ExecutionPolicy текущего процесса
# ------------------------------------------------------------
$policy = Get-ExecutionPolicy -Scope Process
if ($policy -notin @('Bypass','Unrestricted','RemoteSigned')) {
    Write-Err "ExecutionPolicy = $policy. Скрипт нужно запускать через install.bat"
    Write-Host ""
    Write-Host "Или вручную из PowerShell в этой папке:" -ForegroundColor Yellow
    Write-Host "  PowerShell -ExecutionPolicy Bypass -File .\setup.ps1" -ForegroundColor Cyan
    exit 1
}

# Все пути - относительно расположения скрипта
$Root       = $PSScriptRoot
$VenvDir    = Join-Path $Root 'venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'
$VenvPip    = Join-Path $VenvDir 'Scripts\pip.exe'
$Reqs       = Join-Path $Root 'requirements-windows.lock.txt'

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Voice Assist - подготовка окружения" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Папка проекта: $Root"

# ------------------------------------------------------------
# 1. Проверка / установка Python 3.12
# ------------------------------------------------------------
Write-Step "Проверка Python"

function Find-SystemPython {
    # Ищем подходящий python в PATH. Версия 3.12 - целевая,
    # 3.11/3.13 тоже примем но предупредим.
    $candidates = @('py -3.12', 'python3.12', 'python')
    foreach ($cmd in $candidates) {
        try {
            $parts = $cmd.Split(' ')
            $exe   = $parts[0]
            $pyArgs = @($parts | Select-Object -Skip 1)
            $ver = & $exe @pyArgs --version 2>&1
            if ($LASTEXITCODE -eq 0 -and $ver -match 'Python (\d+)\.(\d+)') {
                $major = [int]$Matches[1]; $minor = [int]$Matches[2]
                if ($major -eq 3 -and $minor -ge 10) {
                    return @{ Cmd = $exe; Args = $pyArgs; Version = "$major.$minor" }
                }
            }
        } catch {}
    }
    return $null
}

$py = Find-SystemPython

if ($null -eq $py) {
    Write-Warn "Python 3.10+ не найден. Пробую установить через winget..."

    # Проверка наличия winget
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($null -eq $winget) {
        Write-Err "winget не найден. Установи Python вручную:"
        Write-Host "  https://www.python.org/downloads/release/python-3128/" -ForegroundColor Cyan
        Write-Host "  При установке поставь галку 'Add python.exe to PATH'"  -ForegroundColor Cyan
        Write-Host "  Затем запусти install.bat ещё раз"                       -ForegroundColor Cyan
        exit 1
    }

    & winget install --id Python.Python.3.12 --source winget --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-Err "winget не смог установить Python. Поставь вручную с python.org"
        exit 1
    }

    # PATH обновляется только в новых процессах - перечитаем его руками
    $env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ';' +
                [System.Environment]::GetEnvironmentVariable('Path','User')

    $py = Find-SystemPython
    if ($null -eq $py) {
        Write-Err "Python все ещё не находится. Закрой это окно, открой новое и запусти install.bat ещё раз."
        exit 1
    }
}

Write-OK "Найден Python $($py.Version): $($py.Cmd) $($py.Args -join ' ')"

if ($py.Version -ne '3.12') {
    Write-Warn "Рекомендуемая версия - Python 3.12. У тебя $($py.Version). Скорее всего сработает, но если упадет - поставь 3.12."
}

# ------------------------------------------------------------
# 2. Создание venv
# ------------------------------------------------------------
Write-Step "Виртуальное окружение"

if (Test-Path $VenvPython) {
    Write-OK "venv уже существует: $VenvDir"
} else {
    Write-Host "Создаю venv в $VenvDir..."
    & $py.Cmd @($py.Args + @('-m','venv', $VenvDir))
    if (-not (Test-Path $VenvPython)) {
        Write-Err "Не удалось создать venv"
        exit 1
    }
    Write-OK "venv создан"
}

# ------------------------------------------------------------
# 3. Обновление pip/wheel/setuptools внутри venv
# ------------------------------------------------------------
Write-Step "Обновление pip/wheel/setuptools"
& $VenvPython -m pip install --upgrade pip wheel setuptools
if ($LASTEXITCODE -ne 0) { Write-Err "Не удалось обновить pip"; exit 1 }
Write-OK "pip обновлен"

# ------------------------------------------------------------
# 4. Зависимости
# ------------------------------------------------------------
Write-Step "Зависимости из requirements-windows.lock.txt"

if (-not (Test-Path $Reqs)) {
    Write-Err "Не найден requirements-windows.lock.txt рядом со скриптом"
    exit 1
}

& $VenvPip install -r $Reqs
if ($LASTEXITCODE -ne 0) {
    Write-Err "Установка зависимостей завершилась с ошибкой"
    exit 1
}
Write-OK "Все зависимости установлены"

# ------------------------------------------------------------
# 5. Проверка: модули и модели
# ------------------------------------------------------------
Write-Step "Проверка установки"
$checkCode = @'
import sys

try:
    import onnxruntime, numpy, sounddevice, av
    from PyQt6 import QtCore
    from asr_onnx import CPU_MODEL, WEIGHTS, model_dir
    print(f"onnxruntime   : {onnxruntime.__version__}")
    print(f"numpy         : {numpy.__version__}")
    print("sounddevice   : ok")
    print(f"PyQt6         : {QtCore.PYQT_VERSION_STR}")
except Exception as e:
    print(f"FAIL: {e}")
    sys.exit(1)
need = [CPU_MODEL, WEIGHTS, "rnnt_head.npz", "fbanks.npz",
        "silero_vad_v6.onnx", "wespeaker_resnet34.onnx", "v3_e2e_rnnt_vocab.txt"]
missing = [n for n in need if not (model_dir() / n).is_file()]
if missing:
    print("models/onnx: нет " + ", ".join(missing))
    sys.exit(2)
print("models/onnx   : ok")
'@
try {
    Push-Location $Root
    $checkCode | & $VenvPython -
    $checkExit = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($checkExit -eq 0) {
    Write-OK "Всё на месте"
} elseif ($checkExit -eq 2) {
    Write-Warn "Скопируйте _internal\models из установленной программы рядом с app.py. Подробнее в docs\DEVELOPMENT.md."
    exit 2
} else {
    Write-Err "Что-то не импортируется. Лог выше."
    exit 1
}

# ------------------------------------------------------------
# 6. Список аудио-устройств для проверки
# ------------------------------------------------------------
Write-Step "Доступные аудио-устройства"

@'
import sounddevice as sd
print('--- Host APIs ---')
for i, api in enumerate(sd.query_hostapis()):
    print(f'  [{i}] {api["name"]}')
print()
print('--- Input devices ---')
for i, d in enumerate(sd.query_devices()):
    if d.get('max_input_channels', 0) > 0:
        api = sd.query_hostapis(d['hostapi'])['name']
        print(f'  [{i:3}] ({api}) {d["name"]}')
'@ | & $VenvPython -

# ------------------------------------------------------------
# Финал
# ------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Установка завершена" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host "Запуск приложения: двойной клик по run.bat"
Write-Host "Лог приложения:    %APPDATA%\voice-transcriber\voice-transcriber.log"
Write-Host ""
