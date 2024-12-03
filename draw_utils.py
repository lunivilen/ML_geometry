import plotly.graph_objects as go
import numpy as np
import torch


def calculate_cube_edges(vertices):
    # Пары индексов вершин, задающих рёбра
    edges = [
        [0, 1], [1, 5], [5, 4], [4, 0], [3, 2], [2, 6],
        [7, 3], [0, 3], [1, 2], [5, 6], [4, 7], [4, 6],
        [5, 7], [0, 2], [0, 5], [1, 3], [3, 6], [6, 7],
        [2, 4], [2, 7], [3, 5], [1, 4], [0, 6], [1, 7]
    ]

    edge_x, edge_y, edge_z = [], [], []
    for edge in edges:
        # Добавляем координаты вершин рёбер
        edge_x.extend([vertices[edge[0]][0], vertices[edge[1]][0], None])
        edge_y.extend([vertices[edge[0]][1], vertices[edge[1]][1], None])
        edge_z.extend([vertices[edge[0]][2], vertices[edge[1]][2], None])

    return edge_x, edge_y, edge_z


def visualize_classifier_plotly(model, valid_loader, cube_vertices, device):
    model.to(device)
    model.eval()

    # Списки для точек, предсказаний и истинных меток
    points = []
    true_labels = []
    predicted_labels = []

    # Прогнозирование на данных valid_loader
    with torch.no_grad():
        for inputs, labels in valid_loader:  # labels — истинные метки
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Прогноз модели
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            # Сохранение данных
            points.append(inputs.cpu().numpy())
            true_labels.append(labels.cpu().numpy())
            predicted_labels.append(preds.cpu().numpy())

    # Преобразуем списки в numpy массивы
    points = np.vstack(points)
    true_labels = np.hstack(true_labels)
    predicted_labels = np.hstack(predicted_labels)

    print(points)

    # Определяем правильно и неправильно классифицированные точки
    correct_points = points[true_labels == predicted_labels]
    incorrect_points = points[true_labels != predicted_labels]

    # Разделение правильно классифицированных точек по классам
    inside_correct = correct_points[predicted_labels[true_labels == predicted_labels] == 1]
    outside_correct = correct_points[predicted_labels[true_labels == predicted_labels] == 0]

    # Вычисление координат рёбер куба
    # edge_x, edge_y, edge_z = calculate_cube_edges(cube_vertices)

    # Создаем фигуру Plotly
    fig = go.Figure()

    # Добавляем правильно классифицированные точки
    fig.add_trace(go.Scatter3d(
        x=inside_correct[:, 0], y=inside_correct[:, 1], z=inside_correct[:, 2],
        mode='markers',
        marker=dict(size=5, color='cyan'),
        name='Верно: класс 1'
    ))
    fig.add_trace(go.Scatter3d(
        x=outside_correct[:, 0], y=outside_correct[:, 1], z=outside_correct[:, 2],
        mode='markers',
        marker=dict(size=5, color='orange'),
        name='Верно: класс 0'
    ))

    # Добавляем неправильно классифицированные точки
    fig.add_trace(go.Scatter3d(
        x=incorrect_points[:, 0], y=incorrect_points[:, 1], z=incorrect_points[:, 2],
        mode='markers',
        marker=dict(size=5, color='purple'),
        name='Неправильно классифицировано'
    ))

    # # Добавляем рёбра куба
    # fig.add_trace(go.Scatter3d(
    #     x=edge_x, y=edge_y, z=edge_z,
    #     mode='lines',
    #     line=dict(color='red', width=5),  # Толстые красные рёбра
    #     name='Рёбра куба'
    # ))

    # Настройка визуализации
    fig.update_layout(
        title="Визуализация классификатора в 3D",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        )
    )

    fig.show()
