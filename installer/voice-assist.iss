; voice-assist installer
;   ISCC.exe installer\voice-assist.iss                 - всё
;   ISCC.exe /DVariant=ru installer\voice-assist.iss    - только русская речь
; Одна сборка (dist) на все варианты: установщик берёт из неё нужные модели.
; Внутри варианта модели - компоненты, лишнее можно снять при установке.

#define AppName "voice-assist"
; версия и стадия - те же, что в version.py (тест сверяет); Stage "" - обычный выпуск
#define AppVersion "1.0.0"
#define Stage "beta"
#define AppPublisher "leksa-aleph"
#define AppURL "https://github.com/leksa-aleph/voice-assist-public"
#define AppExe "voice-assist.exe"
#define SrcDir "..\dist\voice-assist"
#define M "\_internal\models"

#ifndef Variant
  #define Variant "full"
#endif
#if Variant == "full"
  #define WithEnglish
  #define WithGpu
  #define Suffix ""
#elif Variant == "ru"
  #define WithGpu
  #define Suffix "-ru"
#else
  #error Variant: full или ru
#endif
#if Stage != ""
  #define FullVersion AppVersion + "-" + Stage
  #define VerName AppVersion + " " + Stage
  #define StageNote "%n%nЭто бета-версия: основное работает, но возможны ошибки."
#else
  #define FullVersion AppVersion
  #define VerName AppVersion
  #define StageNote ""
#endif

[Setup]
AppId={{8F3C1A42-6D71-4B90-9E2A-7C51D8A4E103}
AppName={#AppName}
AppVersion={#FullVersion}
AppVerName={#AppName} {#VerName}
VersionInfoVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=..\LICENSE
OutputDir=.\output
OutputBaseFilename=voice-assist-{#FullVersion}{#Suffix}-setup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExe}
SetupIconFile=icon.ico
MinVersion=10.0.22000

[Languages]
Name: "russian"; MessagesFile: "compiler:Languages\Russian.isl"

[Types]
#ifdef WithEnglish
Name: "full"; Description: "Всё: русская и английская речь, перевод"
Name: "compact"; Description: "Только русская речь"
  #define AllTypes "full compact custom"
#else
Name: "full"; Description: "Русская речь"
  #define AllTypes "full custom"
#endif
Name: "custom"; Description: "Выбрать"; Flags: iscustom

[Components]
Name: "main"; Description: "Субтитры русской речи"; Types: {#AllTypes}; Flags: fixed
#ifdef WithEnglish
Name: "english"; Description: "Английская речь и перевод на русский (+1 ГБ)"; Types: full
#endif
#ifdef WithGpu
Name: "gpu"; Description: "Распознавание на видеокарте (несколько МБ: веса общие с процессором)"; Types: {#AllTypes}
#endif

[Files]
Source: "{#SrcDir}\*"; Excludes: "{#M}\parakeet,{#M}\translate,{#M}\onnx\v3_e2e_rnnt_encoder.gpu.*"; DestDir: "{app}"; Components: main; Flags: ignoreversion recursesubdirs createallsubdirs
#ifdef WithGpu
Source: "{#SrcDir}{#M}\onnx\v3_e2e_rnnt_encoder.gpu.*"; DestDir: "{app}{#M}\onnx"; Components: gpu; Flags: ignoreversion
#endif
#ifdef WithEnglish
Source: "{#SrcDir}{#M}\parakeet\*"; Excludes: "encoder-model.gpu.*"; DestDir: "{app}{#M}\parakeet"; Components: english; Flags: ignoreversion
Source: "{#SrcDir}{#M}\translate\*"; DestDir: "{app}{#M}\translate"; Components: english; Flags: ignoreversion
  #ifdef WithGpu
Source: "{#SrcDir}{#M}\parakeet\encoder-model.gpu.*"; DestDir: "{app}{#M}\parakeet"; Components: english and gpu; Flags: ignoreversion
  #endif
#endif
Source: "..\README.md"; DestDir: "{app}"; DestName: "README.txt"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "..\docs\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; DestName: "THIRD_PARTY_NOTICES.txt"; Flags: ignoreversion
Source: "..\docs\licenses\*.txt"; DestDir: "{app}\_internal\licenses\models"; Flags: ignoreversion

[InstallDelete]
; модели прошлых версий: теперь у процессора и видеокарты общие веса (*.npz)
Type: files; Name: "{app}{#M}\onnx\v3_e2e_rnnt_encoder.fp16.onnx"
Type: files; Name: "{app}{#M}\onnx\v3_e2e_rnnt_encoder.int8.onnx"
Type: files; Name: "{app}{#M}\onnx\v3_e2e_rnnt_encoder.gpu.npz"
Type: files; Name: "{app}{#M}\parakeet\encoder-model.int8.onnx"
Type: files; Name: "{app}{#M}\parakeet\encoder-model.gpu.npz"
; сняли компонент при обновлении - его модели больше не нужны
#ifdef WithEnglish
Type: filesandordirs; Name: "{app}{#M}\parakeet"; Components: not english
Type: filesandordirs; Name: "{app}{#M}\translate"; Components: not english
#else
Type: filesandordirs; Name: "{app}{#M}\parakeet"
Type: filesandordirs; Name: "{app}{#M}\translate"
#endif
#ifdef WithGpu
Type: files; Name: "{app}{#M}\onnx\v3_e2e_rnnt_encoder.gpu.*"; Components: not gpu
Type: files; Name: "{app}{#M}\parakeet\encoder-model.gpu.*"; Components: not gpu
#endif

[Icons]
Name: "{group}\Voice Assist"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\Voice Assist"; Filename: "{app}\{#AppExe}"

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}\_internal\__pycache__"

[Messages]
russian.WelcomeLabel2=Программа покажет субтитры для микрофона, звука выбранного приложения и Discord.%n%nМодели уже внутри. Видеокарта не обязательна.{#StageNote}
