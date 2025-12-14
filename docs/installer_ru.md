# Установщик slice2solid (Windows, Program Files)

## Что получится “на выходе”

1) **Одна папка приложения** (результат PyInstaller):
- `dist_exe\\slice2solid\\slice2solid.exe` и рядом нужные DLL/пакеты.

2) **Один установщик** (Inno Setup):
- один файл `slice2solid-setup.exe`
- при запуске он:
  - копирует файлы в `C:\\Program Files\\slice2solid\\`
  - создаёт пункт в меню Пуск
  - (опционально) создаёт ярлык на рабочем столе
  - добавляет удаление через “Apps & Features / Программы и компоненты”

## Как собрать

1) Собрать папку приложения:
- `powershell -ExecutionPolicy Bypass -File .\\build_installer.ps1`

2) Если установлен Inno Setup (есть `iscc.exe`):
- `build_installer.ps1` автоматически соберёт установщик.
Иначе установите Inno Setup и выполните:
- `iscc.exe tools\\installer\\slice2solid.iss`

