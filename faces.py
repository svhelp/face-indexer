import argparse
import json
from core import extract_faces, cluster_local, save_previews, match_global, apply_global_matches, resolve_matches_cli, annotate_frames

# ─────────────────────────────────────────────
# ТОЧКА ВХОДА
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection and clustering pipeline.")

    parser.add_argument("video_path", help="Путь к видеофайлу")
    parser.add_argument("-i", "--interval", type=float, default=10.0,
                        help="Интервал выборки кадров в секундах (по умолчанию: 10.0)")

    args = parser.parse_args()

    # Шаг 0: Извлечение кадров и распознавание лиц
    frames_dir = extract_faces(args.video_path, args.interval)
    if not frames_dir:
        exit(1)

    # Шаг 1: Локальная кластеризация
    local_path = cluster_local(frames_dir)
    if not local_path:
        exit(1)

    # Шаг 1.5: Сохранение превью групп
    save_previews(frames_dir)

    # Проверяем, были ли уже проставлены глобальные лейблы
    with open(local_path, "r", encoding="utf-8") as f:
        _local_data = json.load(f)

    if "global_labels" in _local_data:
        print(f"\nГлобальные лейблы уже проставлены. Пропускаем шаги 2 и 3.")
        exit()
        
    # Шаг 2: Сопоставление с глобальными кластерами
    result, local_data, global_clusters = match_global(frames_dir)

    resolved_data = resolve_matches_cli(result, local_data, global_clusters)

    apply_global_matches(frames_dir, resolved_data, local_data, global_clusters)

    # Шаг 3: Отрисовка
    annotate_frames(frames_dir, local_data, global_clusters)