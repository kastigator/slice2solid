# Сборка установщика slice2solid (Windows)

## Что получится "на выходе"

1) **Папка приложения для инсталлятора (PyInstaller `--onedir`)**:
- `dist_exe\\slice2solid\\slice2solid.exe` и рядом нужные DLL/пакеты.

2) **Установщик (Inno Setup)**:
- `tools\\installer\\Output\\slice2solid-setup-vX.Y.Z.exe`
- при запуске он:
  - копирует файлы в `C:\\Program Files\\slice2solid\\`
  - создаёт пункт в меню Пуск
  - (опционально) создаёт ярлык на рабочем столе
  - добавляет удаление через “Apps & Features / Программы и компоненты”

## Как собрать

0) Подготовка:
- нужен Python (проверка: `python -V`), рекомендуется `python -m venv .venv`
- зависимости ставятся автоматически внутри `build_installer.ps1` (через `requirements.txt`)

1) Собрать **установщик** (и папку приложения для него):
- `powershell -ExecutionPolicy Bypass -File .\\build_installer.ps1`

Примечание: в GUI есть вкладка "Просмотр" (3D-превью сетки). Для максимально корректного отображения используется VTK (`vtk` + `pyvista` + `pyvistaqt`). Как лёгкий fallback возможен `pyqtgraph` + `PyOpenGL`.

`build_installer.ps1` собирает установщик автоматически, если Inno Setup найден (есть `iscc.exe` в `PATH` или в стандартных путях установки).
Если Inno Setup не установлен, установите его и соберите вручную:
- `iscc.exe tools\\installer\\slice2solid.iss`

Примечание: иконка установщика/приложения (`tools\\installer\\slice2solid.ico`) генерируется автоматически в `build_installer.ps1`. Если вы запускаете `iscc.exe` вручную, предварительно выполните:
- `python tools\\generate_windows_icon.py`

Если сборка падает с `WinError 32` / “file is being used by another process” — обычно это значит, что запущен `slice2solid.exe` (или его держит Проводник/антивирус). Закройте приложение и повторите сборку.

## Про версии и «почему запускается старая сборка»

- Версия приложения хранится в `src\\slice2solid\\__init__.py` (`__version__`).
- `build_installer.ps1` передаёт эту версию в Inno Setup как `AppVersion`, чтобы инсталлятор не «отставал» от кода.
- Инсталлятор пакует **`--onedir`** сборку из `dist_exe\\slice2solid\\...` (см. `tools\\installer\\slice2solid.iss`).
- При проверке не путайте, откуда запускаете программу: меню Пуск/`Program Files` (установленная версия) vs `dist_exe\\...` (локальная сборка). Если сомневаетесь — удалите старую установку через “Apps & Features” и установите заново.

## Публикация релиза (GitHub)

В репозитории настроен GitHub Actions workflow, который собирает установщик и прикрепляет его к GitHub Release при пуше тега `vX.Y.Z`.

1) Обновите `__version__` в `src\\slice2solid\\__init__.py`
2) Создайте тег и отправьте его в GitHub:
- `git tag vX.Y.Z`
- `git push origin vX.Y.Z`
3) Дождитесь завершения workflow “Release (Windows installer)” и скачайте установщик из `Releases`.
