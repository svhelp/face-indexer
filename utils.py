import numpy as np
from pathlib import Path

def _find_medoid(encodings_list):
    """
    Находит медоид — реальный вектор из списка, наиболее близкий к центру группы.
    """
    if not encodings_list:
        return None
    if len(encodings_list) <= 2:
        # Важно: конвертируем в список, иначе json.dump упадет
        return encodings_list[0].tolist() if isinstance(encodings_list[0], np.ndarray) else encodings_list[0]
    
    # Если кластер слишком большой, берем выборку для скорости
    if len(encodings_list) > 500:
        indices = np.random.choice(len(encodings_list), 500, replace=False)
        sample = np.array([encodings_list[i] for i in indices])
    else:
        sample = np.array(encodings_list)
        
    from sklearn.metrics import pairwise_distances
    dist_matrix = pairwise_distances(sample, metric="euclidean")
    dist_sums = dist_matrix.sum(axis=1)
    best_idx = np.argmin(dist_sums)
    
    return sample[best_idx].tolist()


def _min_distance_to_cluster(query_enc, cluster_encs):
    """
    Вычисляет минимальное евклидово расстояние от query_enc до любого вектора в глобальном кластере.
    """
    arr = np.array(cluster_encs)  # (N, D)
    dists = np.linalg.norm(arr - query_enc, axis=1)
    return float(np.min(dists))