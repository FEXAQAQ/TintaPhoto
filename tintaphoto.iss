; Inno Setup script for TintaPhoto.
;
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" tintaphoto.iss
;
; Run the PyInstaller build first (see tintaphoto.spec): this packs up what
; is in dist\TintaPhoto and leaves the installer in dist as well.
;
; It installs for one user and asks for no administrator rights, so the
; program lands in the user's own folder and nothing system wide is
; touched. Qt stays a set of separate files that can be replaced, which is
; what using it under the LGPL v3 asks for (PLAN.md 13.2), and the licence
; files are installed beside the program (PLAN.md 13.5).
;
; Uninstalling removes the program and leaves the user's own settings,
; presets and parameter files alone.

#define AppName "TintaPhoto"
#define AppVersion "0.41.0"
#define AppExe "TintaPhoto.exe"
#define BuildDir "dist\TintaPhoto"

[Setup]
; Never change AppId: it is what tells Windows that a new version is this
; same program rather than a second copy of it.
AppId={{F5B0BC55-C809-46C0-A4C5-7AEFD7F17F08}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppName}
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
SetupIconFile=resources\app.ico

InfoAfterFile=THIRD_PARTY_NOTICES.md
OutputDir=dist
OutputBaseFilename={#AppName}-{#AppVersion}-setup
UninstallDisplayName={#AppName} {#AppVersion}
UninstallDisplayIcon={app}\{#AppExe}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
; No administrator rights: the program goes in the user's own folder, so
; installing it is not a decision about the whole machine.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
; Inno Setup ships no Traditional Chinese; its translation is an unofficial
; download, so the installer speaks English while the program itself is in
; whichever language its settings say.
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The whole of the PyInstaller folder, licences included.
Source: "{#BuildDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
