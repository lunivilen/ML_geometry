import torch


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs, device):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        # Обучение на тренировочных данных
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        # Средняя ошибка на обучении
        train_loss = running_train_loss / len(train_loader)

        # Оценка на тестовых данных
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)

        # Вывод статистики за эпоху
        print(f"Epoch {epoch + 1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Accuracy: {test_accuracy:.4f}")


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            # Подсчет правильных предсказаний
            predicted_classes = torch.argmax(outputs, dim=1)
            correct += (predicted_classes == labels).sum().item()
            total += labels.size(0)

    # Вычисление средней потери и точности
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy
