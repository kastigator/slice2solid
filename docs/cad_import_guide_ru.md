# Импорт результата slice2solid в CAD/mesh-инструменты (универсально)

Цель: из сетки `*_s2s_preview_structure.stl` получить максимально корректный объект для:
- импорта в CAD как **объект на основе сетки** (без конвертации в B-Rep),
- или (если инструмент поддерживает) конвертации сетки/объёма в **твердое тело (B-Rep)** и экспорт `STEP`.

## Что даёт slice2solid

В папке результата (Output folder) обычно есть:
- `*_s2s_preview_structure.stl` — сетка восстановленной структуры (имя включает параметры, например `..._vox0p25_ds4_sig1_it20_...`).
- `metadata.json` — параметры запуска, матрица, статистика, `voxel_size_mm` (и доп. сведения о mesh, если включён экспорт геометрии).
- (Опционально) `*_s2s_preview_structure_mesh.ply`, `voxel_points.csv`, `cad_import_notes.txt` — “CAD bundle”:
  - PLY как альтернативный формат импорта mesh,
  - point cloud из занятых вокселей,
  - заметки с подсказками по стартовым параметрам (spacing/resolution).
- (Опционально) `*_healed.stl` и `*_healed_report.json` — результат `Mesh Healer` (если включён).

## Типовой пайплайн (mesh -> repair -> STEP) в стороннем инструменте

1) Импортируйте mesh:
- файл: `*_s2s_preview_structure.stl` (или `_mesh.ply`)
- единицы: `mm`

2) При необходимости выполните операции “ремонта”:
- Close holes / Repair / Remove self-intersections (если есть),
- Orient normals / Fix winding,
- удаление мелкого мусора (islands), если вокруг есть артефакты.

3) Если инструмент поддерживает конвертацию:
- Convert mesh/implicit/volume -> solid (B-Rep)

4) Экспортируйте CAD-формат:
- `STEP` (или Parasolid/другой формат, который подходит вашему CAD)

## Альтернатива: point cloud -> implicit/volume -> solid

Для решёток/заполнений иногда лучше работать не с сеткой, а с point cloud:
1) Импортируйте `voxel_points.csv` как список точек (формат: каждая строка `x,y,z`, без заголовка).
2) Постройте implicit/volume по точкам (название инструмента зависит от ПО).
3) Выполните solidify/repair и экспортируйте `STEP`.

## Как выбрать spacing/resolution

Ориентир от slice2solid:
- В `metadata.json` есть `voxel.voxel_size_mm`.
- Если включён downsample для meshing (в имени есть `dsN`), эффективный шаг сетки ≈ `voxel_size_mm * N`.
- Стартовые значения:
  - по точкам/вокселям: примерно `0.5 * voxel_size_mm`
  - по mesh: примерно `0.5 * voxel_size_mm * N` (если `dsN > 1`)
  - если тяжело по памяти/времени: увеличивайте spacing
  - если теряются тонкие элементы: уменьшайте spacing

Если включён “CAD bundle”, в `cad_import_notes.txt` уже записан рекомендуемый старт для spacing/resolution.
