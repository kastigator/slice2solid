# Импорт результата slice2solid в nTop (гладкий STEP)

Цель: из сетки `preview_structure.stl` получить в nTop гладкое твердое тело и экспортировать `STEP`.

## Что даёт slice2solid

В папке результата (Output folder) обычно есть:
- `preview_structure.stl` — сетка восстановленной структуры.
- `metadata.json` — параметры запуска, матрица, статистика, `voxel_size_mm`.
- (Опционально) `preview_structure.ply`, `ntop_points.csv` и `ntop_recipe.txt` — “bundle” для nTop (альтернативный импорт + point cloud + подсказки).

## Типовой пайплайн в nTop (для решёток/заполнений)

1) `Utilities` → `Import Mesh`
- File: `preview_structure.stl` (или `preview_structure.ply`)
- Units: `mm`

2) Поиск (Ctrl+F) и добавить блок: `Implicit Body from Mesh`
- Вход: импортированный Mesh
- Важно: выставить разумную “resolution/spacing” (детальнее ниже)

3) (По ситуации) операции на implicit-теле
- `Smooth` — если видны “ступеньки”/шум
- `Close` / `Repair` — если есть микро-разрывы/дырки
- Удаление мелких компонент (“islands”), если вокруг есть “мусор”

4) Конвертация implicit → CAD/Solid body
- В nTop это отдельный блок конвертации (название зависит от версии)

5) Экспорт
- `Export` → `STEP`

## Альтернатива: point cloud → implicit (иногда лучше для решёток)

Если импорт mesh даёт артефакты или слишком тяжёлый:

1) Импортируйте `ntop_points.csv` как список точек (Point List/Point Cloud).
   Формат файла: каждая строка — `x, y, z` (без заголовка).
2) Создайте implicit из point cloud (название блока зависит от версии nTop).
3) Дальше: Convert to CAD/Solid → Export STEP.

## Как выбрать resolution/spacing

Ориентир от slice2solid:
- В `metadata.json` есть `voxel.voxel_size_mm`.
- Стартовое значение для nTop: примерно `0.5 * voxel_size_mm`.
  - Если тяжело по памяти/времени: увеличивайте spacing.
  - Если теряются тонкие элементы: уменьшайте spacing.

Если включён “nTop bundle”, в `ntop_recipe.txt` уже записан рекомендуемый старт.
