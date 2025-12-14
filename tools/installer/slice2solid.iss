; Inno Setup script for slice2solid
; Build: iscc.exe tools\installer\slice2solid.iss

#define AppName "slice2solid"
#define AppPublisher "slice2solid"
#define AppURL "https://github.com/kastigator/slice2solid"
#define AppExeName "slice2solid.exe"

; Change this if you want a different default version string.
#define AppVersion "0.1.0"

[Setup]
AppId={{9A48DAB2-95C7-4B7D-9B0A-5B4F9E1D5A3F}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={pf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
OutputBaseFilename={#AppName}-setup
Compression=lzma
SolidCompression=yes
ArchitecturesInstallIn64BitMode=x64
PrivilegesRequired=admin
UninstallDisplayIcon={app}\{#AppExeName}
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
; This expects a PyInstaller --onedir build in dist_exe\slice2solid\
Source: "..\\..\\dist_exe\\slice2solid\\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\\{#AppName}"; Filename: "{app}\\{#AppExeName}"
Name: "{commondesktop}\\{#AppName}"; Filename: "{app}\\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\\{#AppExeName}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

