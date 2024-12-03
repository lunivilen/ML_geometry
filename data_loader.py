import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.utils import resample, shuffle


class TetraDataset(Dataset):
    def __init__(self, cube_vertices: np.ndarray[3, 4], n: int, seed: int, data_type: str):
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

        data = [[cube_vertices[0], cube_vertices[1], cube_vertices[2], cube_vertices[3] ],
                [cube_vertices[1], cube_vertices[2], cube_vertices[3], cube_vertices[0] ],
                [cube_vertices[0], cube_vertices[1], cube_vertices[3], cube_vertices[2] ],
                [cube_vertices[0], cube_vertices[2], cube_vertices[3],  cube_vertices[1]],]

        plats = []

        for i in data:
            A, B, C, marker = i
            x1, y1, z1 = A
            x2, y2, z2 = B
            x3, y3, z3 = C
            x_marker, y_marker, z_marker = marker
            plats.append([x1, y1, z1, x2, y2, z2, x3, y3, z3, x_marker, y_marker, z_marker])

        # Размер куба
        size = 10  # Куб с координатами от 0 до size
        # Генерация точек
        x = np.random.uniform(-2, size, n)
        y = np.random.uniform(-2, size, n)
        z = np.random.uniform(-2, size, n)

        # Объединяем точки в один массив
        data = np.column_stack((x, y, z))
        labels = []

        for point in data:
            det = []
            x, y, z = point

            flag = True
            for plat in plats:
                x1, y1, z1, x2, y2, z2, x3, y3, z3, x_marker, y_marker, z_marker = plat
                ar = np.array([[x - x1, y - y1, z - z1],
                               [x2 - x1, y2 - y1, z2 - z1],
                               [x3 - x1, y3 - y1, z3 - z1]])
                d = np.linalg.det(ar)

                ar_marker = np.array([[x_marker - x1, y_marker - y1, z_marker - z1],
                               [x2 - x1, y2 - y1, z2 - z1],
                               [x3 - x1, y3 - y1, z3 - z1]])
                d_marker = np.linalg.det(ar_marker)

                if d_marker * d <0:
                    flag = False


            if flag:
                labels.append(1)
            else:
                labels.append(0)


        # Разделяем данные на классы
        data_class_0 = [point for point, label in zip(data, labels) if label == 0]
        data_class_1 = [point for point, label in zip(data, labels) if label == 1]

        # Применяем undersampling для класса 0
        data_class_0_undersampled = resample(data_class_0,
                                             replace=False,  # без замены
                                             n_samples=len(data_class_1),  # выбираем столько же, сколько в классе 1
                                             random_state=self.seed)

        # Сливаем класс 1 и undersampled класс 0
        data_balanced = np.vstack([data_class_1, data_class_0_undersampled])
        labels_balanced = [1] * len(data_class_1) + [0] * len(data_class_0_undersampled)
        # Разделение на train, valid и test
        data_balanced, labels_balanced = shuffle(data_balanced, labels_balanced, random_state=42)

        # Разделение на train, valid и test
        total_size = len(data_balanced)
        valid_test_size = int(0.15 * total_size)
        train_size = total_size - 2 * valid_test_size
        if data_type == "train":
            self.data = data_balanced[:train_size]
            self.labels = labels_balanced[:train_size]
        elif data_type == "valid":
            self.data = data_balanced[train_size:train_size + valid_test_size]
            self.labels = labels_balanced[train_size:train_size + valid_test_size]
        else:  # test
            self.data = data_balanced[train_size + valid_test_size:]
            self.labels = labels_balanced[train_size + valid_test_size:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx],
                                                                               dtype=torch.long).squeeze()
