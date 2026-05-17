from utils import _min_distance_to_cluster, _find_medoid
import cv2
import os
import json
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN
from insightface.app import FaceAnalysis

if not hasattr(np, "int"):
    np.int = int

# ─────────────────────────────────────────────
# КОНСТАНТЫ
# ─────────────────────────────────────────────

FACE_DET_THRESHOLD = 0.8

# Шаг 1: Локальная кластеризация (DBSCAN внутри одного видео)
EPSILON_FACE_CLUSTERS = 1.05   # Порог DBSCAN — выше = охотнее объединяет лица

# Веса для многокритериальной оценки качества лиц в превью (Шаг 1.5)
WEIGHT_DET_SCORE = 0.6      # Вес уверенности InsightFace (достоверность обнаружения)
WEIGHT_RESOLUTION = 0.3     # Вес разрешения лица (размер лица на кадре)
WEIGHT_SHARPNESS = 0.1      # Вес четкости лица (резкость по Лапласиану)

# Шаг 2: Сравнение с глобальными кластерами (два порога)
# Сравнение идёт по минимальному расстоянию до любого вектора в глобальном кластере.
GLOBAL_MATCH_STRICT = 0.70     # < этого: точное совпадение, вектор НЕ добавляем (уже представлен)
GLOBAL_MATCH_ADD    = 1.05     # < этого: совпадение, но вектор ДОБАВЛЯЕМ (новый ракурс/освещение)
                               # >= этого: новый человек

MAX_ENCODINGS_PER_CLUSTER = 25 # Максимум векторов на глобальный кластер (против разрастания JSON)

def extract_faces(video_path, interval_seconds=5.0):
    """
    Объединенный шаг: извлекает кадры из видео с заданным интервалом,
    сразу же распознает лица с помощью InsightFace и сохраняет кадр на диск
    только в том случае, если на нем обнаружено хотя бы одно лицо.
    Данные о лицах сохраняются в faces_data.json.
    """
    if not os.path.exists(video_path):
        print(f"Ошибка: Видеофайл '{video_path}' не найден.")
        return None

    # Получаем название видео без расширения
    video_name = Path(video_path).stem
    output_dir = os.path.join("temp", video_name)
    output_json = os.path.join(output_dir, "faces_data.json")

    # Проверяем, выполнено ли уже распознавание
    if os.path.exists(output_json):
        print(f"Файл '{output_json}' уже существует. Пропускаем этап извлечения и распознавания.")
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
    saved_count = 0
    scanned_count = 0

    print(f"\nНачинаем совмещенный процесс извлечения и распознавания...")
    print(f"Сохранение кадров с лицами в '{output_dir}'...")

    # Инициализация InsightFace (buffalo_l — самая точная модель)
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Инициализирована модель InsightFace (будет использован GPU, если доступен)")

    results = {}

    while frame_idx < total_frames:
        # Быстрый переход к нужному кадру
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            break

        scanned_count += 1

        # Распознавание лиц прямо на кадре в памяти
        faces = app.get(frame)

        face_list = []
        for face in faces:
            if face.det_score < FACE_DET_THRESHOLD:
                continue
            
            x1, y1, x2, y2 = [int(v) for v in face.bbox]
            
            # Вычисляем резкость (дисперсию Лапласиана) вырезанного лица
            h, w = frame.shape[:2]
            t_c = max(0, y1)
            b_c = min(h, y2)
            l_c = max(0, x1)
            r_c = min(w, x2)
            face_img = frame[t_c:b_c, l_c:r_c]
            
            if face_img.size > 0:
                gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                blur_value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            else:
                blur_value = 0.0

            face_list.append({
                "location": [y1, x2, y2, x1],   # top, right, bottom, left
                "encoding": face.normed_embedding.tolist(),
                "blur_value": blur_value,
                "det_score": float(face.det_score) if hasattr(face, "det_score") else 0.0
            })

        # Если найдено хотя бы одно лицо, сохраняем кадр на диск и заносим в результаты
        if face_list:
            filename = f"frame_{saved_count:04d}.jpg"
            output_path = os.path.join(output_dir, filename)
            # === ВРЕМЕННАЯ ОТЛАДОЧНАЯ АННОТАЦИЯ (УДАЛИТЬ ПОСЛЕ ОТЛАДКИ) ===
            debug_frame = frame.copy()
            for face, face_meta in zip(faces, face_list):
                x1, y1, x2, y2 = [int(v) for v in face.bbox]
                h, w = debug_frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                # Рисуем зеленую рамку вокруг лица
                cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Получаем все характеристики InsightFace
                info = []
                if hasattr(face, "det_score") and face.det_score is not None:
                    info.append(f"Score: {face.det_score:.2f}")
                if hasattr(face, "gender") and face.gender is not None:
                    gender_str = "M" if face.gender == 1 else "F"
                    info.append(f"Sex: {gender_str}")
                if hasattr(face, "age") and face.age is not None:
                    info.append(f"Age: {int(face.age)}")
                
                # Добавляем рассчитанную нами дисперсию Лапласиана (резкость)
                blur_val = face_meta.get("blur_value", 0.0)
                info.append(f"Sharp: {blur_val:.1f}")
                
                text_str = ", ".join(info)
                cv2.putText(debug_frame, text_str, (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            
            cv2.imwrite(output_path, debug_frame)
            # ==============================================================
            
            results[filename] = face_list
            print(f"[{filename} (кадр {frame_idx})] Сохранено! Найдено лиц: {len(face_list)}")
            saved_count += 1
        
        # Переходим к следующему кадру для извлечения
        frame_idx += frame_interval

    cap.release()

    # Сохраняем faces_data.json
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"Процесс завершен. Всего просканировано кадров: {scanned_count}, сохранено {saved_count} кадров с лицами.")
    print(f"Данные о распознанных лицах сохранены в '{output_json}'.")
    return output_dir


# ─────────────────────────────────────────────
# ШАГ 1: Локальная кластеризация внутри видео
# ─────────────────────────────────────────────

def cluster_local(frames_dir):
    """
    Шаг 1: DBSCAN-кластеризация лиц внутри одного видео.
    Оптимизация: сохраняем только лейблы и ОДИН медоид на человека (экономия места в 100 раз).
    Иерархическая структура: группируем результаты по кадрам.
    """
    local_clusters_path = os.path.join(frames_dir, "local_clusters.json")

    if os.path.exists(local_clusters_path):
        print(f"Файл '{local_clusters_path}' уже существует. Пропускаем этап кластеризации.")
        return local_clusters_path

    json_path = os.path.join(frames_dir, "faces_data.json")
    if not os.path.exists(json_path):
        print(f"Ошибка: '{json_path}' не найден. Сначала выполните extract_faces().")
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        faces_data = json.load(f)

    print(f"\n[Шаг 1] Локальная кластеризация (Medoid-based) в '{frames_dir}'...")

    encodings = []
    flat_detections = [] # Временный плоский список для связи индексов с DBSCAN

    for filename in sorted(faces_data.keys()):
        for face_data in faces_data[filename]:
            if isinstance(face_data, dict) and "encoding" in face_data:
                encodings.append(np.array(face_data["encoding"]))
                flat_detections.append({
                    "filename": filename,
                    "location": face_data["location"],
                    "blur_value": face_data.get("blur_value", 0.0),
                    "det_score": face_data.get("det_score", 0.0)
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

    # Группируем результаты по кадрам
    frames_dict = {}
    for i, det in enumerate(flat_detections):
        filename = det["filename"]
        if filename not in frames_dict:
            frames_dict[filename] = []
        
        frames_dict[filename].append({
            "location": det["location"],
            "local_label": labels[i],
            "global_label": -1,  # Инициализируем до Шага 2
            "blur_value": det.get("blur_value", 0.0),
            "det_score": det.get("det_score", 0.0)
        })

    result = {
        "frames": frames_dict,
        "medoids": medoids
    }

    with open(local_clusters_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"  Оптимизированные локальные кластеры сохранены в '{local_clusters_path}'.")
    return local_clusters_path


# ─────────────────────────────────────────────
# ШАГ 1.5: Сохранение превью групп
# ─────────────────────────────────────────────

def save_previews(frames_dir):
    """
    Шаг 1.5: Сохранение превью групп.
    Для каждого локального кластера вычисляет финальный скор для каждого кадра по трем параметрам:
    уверенности детекции, разрешению лица и четкости по Лапласиану (без жестких порогов).
    Сохраняет ТОП-3 превью (кадры и cropped версии) в папку <frames_dir>/previews/.
    """
    local_clusters_path = os.path.join(frames_dir, "local_clusters.json")
    if not os.path.exists(local_clusters_path):
        print(f"Ошибка: '{local_clusters_path}' не найден.")
        return

    with open(local_clusters_path, "r", encoding="utf-8") as f:
        local_data = json.load(f)

    frames_dict = local_data.get("frames", {})

    previews_dir = os.path.join(frames_dir, "previews")
    os.makedirs(previews_dir, exist_ok=True)

    print(f"\n[Шаг 1.5] Сохранение превью групп (ТОП-3) в '{previews_dir}'...")

    # Группируем детекции по локальным лейблам
    from collections import defaultdict
    label_detections = defaultdict(list)

    for filename, faces in frames_dict.items():
        for face in faces:
            lbl = face.get("local_label", -1)
            if lbl == -1:
                continue
            label_detections[lbl].append({
                "filename": filename,
                "location": face["location"],
                "blur_value": face.get("blur_value", 0.0),
                "det_score": face.get("det_score", 0.0)
            })

    for lbl, detections in label_detections.items():
        # Подготавливаем данные: вычисляем линейные размеры, площадь и финальный скор
        processed_dets = []
        for d in detections:
            t, r, b, l = d["location"]
            h_face = b - t
            w_face = r - l
            area = h_face * w_face
            face_size = min(h_face, w_face)
            processed_dets.append({
                "filename": d["filename"],
                "location": d["location"],
                "blur_value": d["blur_value"],
                "det_score": d["det_score"],
                "area": area,
                "width": w_face,
                "height": h_face,
                "face_size": face_size
            })

        # Вычисляем финальный скор с абсолютными и ограниченными параметрами
        for d in processed_dets:
            # Абсолютная уверенность детектора (0.0 - 1.0)
            norm_det = d["det_score"]
            # Линейный размер с оптимальным порогом в 120 пикселей (все лица крупнее 120x120 получают 1.0)
            norm_res = min(d["face_size"], 120.0) / 120.0
            # Резкость с порогом насыщения в 100.0 (все лица резче 100.0 получают 1.0)
            norm_sharp = min(d["blur_value"], 100.0) / 100.0

            d["final_score"] = (
                WEIGHT_DET_SCORE * norm_det +
                WEIGHT_RESOLUTION * norm_res +
                WEIGHT_SHARPNESS * norm_sharp
            )

        # Сортируем детекции по финальному скору по убыванию
        processed_dets.sort(key=lambda x: x["final_score"], reverse=True)

        # Сохраняем топ-3
        top_k = min(3, len(processed_dets))
        print(f"  Группа {lbl}: найдено кадров {len(processed_dets)}, экспортируем ТОП-{top_k}:")
        
        for rank in range(top_k):
            best_det = processed_dets[rank]
            frame_path = os.path.join(frames_dir, best_det["filename"])
            image = cv2.imread(frame_path)
            
            if image is not None:
                rank_num = rank + 1
                full_path = os.path.join(previews_dir, f"group_{lbl}_rank{rank_num}.jpg")
                cv2.imwrite(full_path, image)

                # Вырезаем лицо
                t, r, b, l = best_det["location"]
                h, w = image.shape[:2]
                t_c = max(0, t)
                b_c = min(h, b)
                l_c = max(0, l)
                r_c = min(w, r)
                cropped_face = image[t_c:b_c, l_c:r_c]

                if cropped_face.size > 0:
                    cropped_path = os.path.join(previews_dir, f"group_{lbl}_rank{rank_num}_cropped.jpg")
                    cv2.imwrite(cropped_path, cropped_face)
                    
                print(f"    [Rank {rank_num}] {best_det['filename']} (Скор: {best_det['final_score']:.3f}, Резкость: {best_det['blur_value']:.1f}, Score: {best_det['det_score']:.2f}, Разрешение: {best_det['width']}x{best_det['height']}px)")

    print("  Шаг 1.5 завершен.")


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

    # Обновляем local_clusters.json (заполняем global_label во всех кавлях)
    for filename, faces in local_data["frames"].items():
        for face in faces:
            face["global_label"] = local_to_global.get(face["local_label"], -1)
    
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

    frames_dict = local_data.get("frames", {})

    annotated_dir = os.path.join(frames_dir, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    print(f"\n[Шаг 3] Отрисовка рамок → '{annotated_dir}'...")

    for filename, faces in frames_dict.items():
        file_path = os.path.join(frames_dir, filename)
        if not os.path.exists(file_path):
            continue

        image = cv2.imread(file_path)
        if image is None:
            continue

        for face in faces:
            top, right, bottom, left = face["location"]
            label = face.get("global_label", face.get("local_label", -1))
            color = get_color(label)
            text = f"Person {label}" if label != -1 else "Noise"

            cv2.rectangle(image, (left, top), (right, bottom), color, 2)
            cv2.putText(image, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imwrite(os.path.join(annotated_dir, filename), image)

    print("  Отрисовка завершена.")