from faces_utils import _min_distance_to_cluster, _find_medoid
import cv2
import os
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis

# ─────────────────────────────────────────────
# КОНСТАНТЫ
# ─────────────────────────────────────────────

# Шаг 1: Локальная кластеризация (DBSCAN внутри одного видео)
EPSILON_FACE_CLUSTERS = 1.05   # Порог DBSCAN — выше = охотнее объединяет лица

# Шаг 2: Сравнение с глобальными кластерами (два порога)
# Сравнение идёт по минимальному расстоянию до любого вектора в глобальном кластере.
GLOBAL_MATCH_STRICT = 0.70     # < этого: точное совпадение, вектор НЕ добавляем (уже представлен)
GLOBAL_MATCH_ADD    = 1.05     # < этого: совпадение, но вектор ДОБАВЛЯЕМ (новый ракурс/освещение)
                               # >= этого: новый человек

MAX_ENCODINGS_PER_CLUSTER = 25 # Максимум векторов на глобальный кластер (против разрастания JSON)

def extract_frames(video_path, interval_seconds=5.0):
    """
    Извлекает кадры из видео с заданным интервалом.

    :param video_path: Путь к исходному видеофайлу.
    :param interval_seconds: Интервал между кадрами в секундах.
    :return: Путь к директории с сохраненными кадрами или None в случае ошибки.
    """
    if not os.path.exists(video_path):
        print(f"Ошибка: Видеофайл '{video_path}' не найден.")
        return None

    # Получаем название видео без расширения
    video_name = Path(video_path).stem
    output_dir = os.path.join("temp", video_name)

    # Проверяем, существует ли папка и есть ли в ней файлы кадров
    if os.path.exists(output_dir):
        existing_frames = [f for f in os.listdir(output_dir) if f.startswith("frame_") and f.endswith(".jpg")]
        if existing_frames:
            print(f"Папка '{output_dir}' уже содержит {len(existing_frames)} кадров. Пропускаем извлечение.")
            return output_dir

    # Создаем папку для сохранения кадров, если её нет
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка: Не удалось открыть видео '{video_path}'.")
        return None

    # Получаем FPS (количество кадров в секунду) и общее количество кадров
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        print("Ошибка: Не удалось определить FPS видео.")
        cap.release()
        return None

    frame_interval = max(1, int(fps * interval_seconds))
    frame_idx = 0
    extracted_count = 0

    print(f"Начинаем извлечение. Сохранение в '{output_dir}'...")

    while frame_idx < total_frames:
        # Быстрый переход к нужному кадру
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break

        output_path = os.path.join(output_dir, f"frame_{extracted_count:04d}.jpg")
        cv2.imwrite(output_path, frame)
        extracted_count += 1
        
        # Переходим к следующему кадру для извлечения
        frame_idx += frame_interval

    cap.release()
    print(f"Извлечение завершено. Сохранено {extracted_count} кадров.")
    return output_dir
    
# ─────────────────────────────────────────────
# ШАГ 0.5: Детектирование и сохранение признаков
# ─────────────────────────────────────────────

def recognize_faces(frames_dir):
    """
    Применяет InsightFace ко всем кадрам, сохраняет локации и encoding'и в faces_data.json.
    """
    output_json = os.path.join(frames_dir, "faces_data.json")
    if os.path.exists(output_json):
        print(f"\nФайл '{output_json}' уже существует. Пропускаем этап детектирования лиц.")
        return output_json

    print(f"\nНачинаем поиск лиц в '{frames_dir}'...")

    # Инициализация InsightFace (buffalo_l — самая точная модель)
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Инициализирована модель InsightFace (будет использован GPU, если доступен)")

    results = {}
    
    valid_exts = {".jpg", ".jpeg", ".png"}

    for filename in sorted(os.listdir(frames_dir)):
        if Path(filename).suffix.lower() in valid_exts and os.path.isfile(os.path.join(frames_dir, filename)):
            file_path = os.path.join(frames_dir, filename)
            image = cv2.imread(file_path)
            faces = app.get(image)

            face_list = []
            for face in faces:
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                face_list.append({
                    "location": [y1, x2, y2, x1],   # top, right, bottom, left
                    "encoding": face.normed_embedding.tolist()
                })

            results[filename] = face_list

            if face_list:
                print(f"[{filename}] Найдено лиц: {len(face_list)}")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Распознавание завершено. Данные сохранены в '{output_json}'.")
    return output_json

# ─────────────────────────────────────────────
# ШАГ 1: Локальная кластеризация внутри видео
# ─────────────────────────────────────────────

def cluster_local(frames_dir):
    """
    Шаг 1: DBSCAN-кластеризация лиц внутри одного видео.
    Оптимизация: сохраняем только лейблы и ОДИН медоид на человека (экономия места в 100 раз).
    """
    json_path = os.path.join(frames_dir, "faces_data.json")
    if not os.path.exists(json_path):
        print(f"Ошибка: '{json_path}' не найден. Запустите recognize_faces() сначала.")
        return None

    local_clusters_path = os.path.join(frames_dir, "local_clusters.json")

    with open(json_path, "r", encoding="utf-8") as f:
        faces_data = json.load(f)

    print(f"\n[Шаг 1] Локальная кластеризация (Medoid-based) в '{frames_dir}'...")

    encodings = []
    face_meta = [] # Здесь храним только filename и location (без тяжелых векторов)

    for filename in sorted(faces_data.keys()):
        for face_data in faces_data[filename]:
            if isinstance(face_data, dict) and "encoding" in face_data:
                encodings.append(np.array(face_data["encoding"]))
                face_meta.append({
                    "filename": filename,
                    "location": face_data["location"]
                })

    if not encodings:
        print("Не удалось получить признаки ни одного лица.")
        return None

    clt = DBSCAN(eps=EPSILON_FACE_CLUSTERS, metric="euclidean", min_samples=2, n_jobs=-1)
    clt.fit(encodings)
    labels = clt.labels_.tolist()

    unique_labels = set(labels)
    num_faces = len([l for l in unique_labels if l != -1])
    num_noise = labels.count(-1)
    print(f"  Найдено локальных людей: {num_faces}, шум: {num_noise}")

    # Находим медоид для каждого кластера
    medoids = {}
    for label in unique_labels:
        if label == -1:
            continue
        # Собираем все векторы этого кластера
        cluster_encs = [encodings[i] for i, l in enumerate(labels) if l == label]
        medoids[str(label)] = _find_medoid(cluster_encs)

    result = {
        "labels": labels,
        "face_meta": face_meta,
        "medoids": medoids
    }

    with open(local_clusters_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"  Оптимизированные локальные кластеры сохранены в '{local_clusters_path}'.")
    return local_clusters_path


# ─────────────────────────────────────────────
# ШАГ 2: Сравнение с глобальными кластерами
# ─────────────────────────────────────────────

def match_global(frames_dir):
    """
    Шаг 2: Сопоставление локальных медоидов с глобальными кластерами.
    """
    local_clusters_path = os.path.join(frames_dir, "local_clusters.json")
    if not os.path.exists(local_clusters_path):
        return None

    with open(local_clusters_path, "r", encoding="utf-8") as f:
        local_data = json.load(f)

    # Загружаем/инициализируем глобальные кластеры
    global_clusters_path = os.path.join("temp", "clusters_data.json")
    if os.path.exists(global_clusters_path):
        with open(global_clusters_path, "r", encoding="utf-8") as f:
            global_clusters = json.load(f)
    else:
        global_clusters = []

    print(f"\n[Шаг 2] Глобальное сопоставление (по медоидам)...")

    medoids = local_data["medoids"] # local_label_str -> vector
    local_to_global = {-1: -1}

    for label_str, medoid_enc in medoids.items():
        local_label = int(label_str)
        
        best_match_idx = -1
        best_distance = float("inf")

        for gi, gc in enumerate(global_clusters):
            dist = _min_distance_to_cluster(medoid_enc, gc["encodings"])
            if dist < best_distance:
                best_distance = dist
                best_match_idx = gi

        if best_distance < GLOBAL_MATCH_ADD:
            gc = global_clusters[best_match_idx]
            global_label_id = int(gc["label"].split("_")[1])
            local_to_global[local_label] = global_label_id

            if best_distance < GLOBAL_MATCH_STRICT:
                print(f"  {local_label} -> {gc['label']} (точное совпадение, d={best_distance:.3f})")
            else:
                # Добавляем медоид в глобальный список
                if len(gc["encodings"]) < MAX_ENCODINGS_PER_CLUSTER:
                    gc["encodings"].append(medoid_enc)
                    print(f"  {local_label} -> {gc['label']} (добавлен новый ракурс, d={best_distance:.3f})")
        else:
            # Новый человек
            max_id = max((int(gc["label"].split("_")[1]) for gc in global_clusters), default=-1)
            new_id = max_id + 1
            global_label_str = f"person_{new_id}"
            local_to_global[local_label] = new_id

            global_clusters.append({
                "label": global_label_str,
                "name": global_label_str, # По умолчанию имя совпадает с лейблом
                "encodings": [medoid_enc]
            })
            print(f"  {local_label} -> новый {global_label_str} (d={best_distance:.3f})")

    # Сохраняем обновлённые глобальные кластеры
    with open(global_clusters_path, "w", encoding="utf-8") as f:
        json.dump(global_clusters, f, indent=4, ensure_ascii=False)

    # Обновляем local_clusters.json (добавляем global_labels)
    local_data["global_labels"] = [local_to_global.get(lbl, -1) for lbl in local_data["labels"]]
    
    with open(local_clusters_path, "w", encoding="utf-8") as f:
        json.dump(local_data, f, indent=4, ensure_ascii=False)

    print(f"  Глобальное сопоставление завершено.")
    return local_clusters_path

# ─────────────────────────────────────────────
# ШАГ 3: Отрисовка аннотаций на кадрах
# ─────────────────────────────────────────────

def get_color(label):
    if label == -1:
        return (128, 128, 128)
    import colorsys
    hue = (label * 137.508) % 360 / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def annotate_frames(frames_dir):
    """
    Шаг 3: Рисует рамки с именами на кадрах на основе данных из local_clusters.json.
    Результаты сохраняются в <frames_dir>/annotated/.
    """
    local_clusters_path = os.path.join(frames_dir, "local_clusters.json")
    if not os.path.exists(local_clusters_path):
        print(f"Ошибка: '{local_clusters_path}' не найден. Сначала выполните cluster_local() и match_global().")
        return

    with open(local_clusters_path, "r", encoding="utf-8") as f:
        local_data = json.load(f)

    face_meta = local_data["face_meta"]
    global_labels = local_data.get("global_labels", local_data.get("labels", []))

    # Группируем по кадрам
    from collections import defaultdict
    annotations = defaultdict(list)
    for i, meta in enumerate(face_meta):
        global_label = global_labels[i] if i < len(global_labels) else -1
        annotations[meta["filename"]].append({
            "location": meta["location"],
            "label": global_label
        })

    annotated_dir = os.path.join(frames_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    print(f"\n[Шаг 3] Отрисовка рамок → '{annotated_dir}'...")

    for filename, faces in annotations.items():
        file_path = os.path.join(frames_dir, filename)
        if not os.path.exists(file_path):
            continue

        image = cv2.imread(file_path)
        if image is None:
            continue

        for face in faces:
            top, right, bottom, left = face["location"]
            label = face["label"]
            color = get_color(label)
            text = f"Person {label}" if label != -1 else "Noise"

            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.putText(image, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imwrite(os.path.join(annotated_dir, filename), image)

    print("  Отрисовка завершена.")
