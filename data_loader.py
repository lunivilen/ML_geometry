import torch
from torch.utils.data import Dataset
import numpy as np


class CubeDataset(Dataset):
    def __init__(self, cube_vertices: np.ndarray[3, 8], n: int, seed: int, data_type: str):
        """
        Args:
            cube_vertices: координаты 8 точек, задающих куб
            n: количество точек внутри/на границе куба и вне его
            seed: для воспроизводимости генерации точек
            data_type: "train", "valid", "test" для разбиения на соответствующие наборы
        """
        self.data_type = data_type
        self.n = n
        self.seed = seed

        np.random.seed(seed)

        # Определяем границы куба
        min_coords = np.min(cube_vertices, axis=0)
        max_coords = np.max(cube_vertices, axis=0)

        # Генерируем точки внутри/на границе куба
        points_inside = np.random.uniform(low=min_coords, high=max_coords, size=(n, 3))
        labels_inside = np.ones((n, 1))  # Класс 1

        # Генерируем точки за пределами куба
        points_outside = self._generate_outside_points(min_coords, max_coords, n)
        labels_outside = np.zeros((n, 1))  # Класс 0

        # Объединяем данные
        data = np.vstack((points_inside, points_outside))
        labels = np.vstack((labels_inside, labels_outside))

        # Перемешиваем данные
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]

        # Разделяем данные на train, valid, test
        total_size = len(data)
        valid_test_size = int(0.15 * total_size)
        train_size = total_size - 2 * valid_test_size

        if data_type == "train":
            self.data = data[:train_size]
            self.labels = labels[:train_size]
        elif data_type == "valid":
            self.data = data[train_size:train_size + valid_test_size]
            self.labels = labels[train_size:train_size + valid_test_size]
        else:  # test
            self.data = data[train_size + valid_test_size:]
            self.labels = labels[train_size + valid_test_size:]

    def _generate_outside_points(self, min_coords, max_coords, n):
        """Генерирует точки за пределами куба"""
        points = []
        while len(points) < n:
            # Генерация случайных точек в большем диапазоне
            candidate = np.random.uniform(low=min_coords - 1, high=max_coords + 1, size=(1, 3))
            if not self._is_inside_cube(candidate[0], min_coords, max_coords):
                points.append(candidate[0])
        return np.array(points)

    def _is_inside_cube(self, point, min_coords, max_coords):
        """Проверяет, находится ли точка внутри или на границе куба"""
        return np.all(point >= min_coords) and np.all(point <= max_coords)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long).squeeze()
