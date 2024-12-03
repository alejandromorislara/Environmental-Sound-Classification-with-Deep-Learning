import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchsummary import summary

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold

import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam


from utils import prepare_datasets


import torch
import torch.nn as nn

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.25):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Passing input through LSTM
        lstm_out, (h_n, c_n) = self.lstm(
            x
        )  # lstm_out is the output tensor, h_n and c_n are hidden states
        last_hidden_state = h_n[-1]  # Taking the last hidden state from the LSTM output

        # Pass through fully connected layers
        x = self.fc(last_hidden_state)
        x = self.dropout(x)

        return x


class ConvLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_rate=0.25,
        conv_filters=64,
        kernel_size=3,
    ):
        super(ConvLSTMClassifier, self).__init__()

        # Capa de convolución
        self.conv1 = nn.Conv1d(
            input_dim, conv_filters, kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(
            conv_filters, conv_filters, kernel_size, padding=kernel_size // 2
        )

        # LSTM
        self.lstm = nn.LSTM(conv_filters, hidden_dim, batch_first=True)

        # Capa totalmente conectada (fully connected)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Capa de dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Aplicamos las convoluciones
        x = x.permute(
            0, 2, 1
        )  # Cambiamos las dimensiones para que coincidan con Conv1d (batch_size, channels, sequence_length)
        x = self.conv1(x)  # Primera capa de convolución
        x = torch.relu(x)
        x = self.conv2(x)  # Segunda capa de convolución
        x = torch.relu(x)

        x = x.permute(
            0, 2, 1
        )  # Revertimos las dimensiones a (batch_size, sequence_length, channels) para el LSTM

        # Pasamos la salida por el LSTM
        lstm_out, (h_n, c_n) = self.lstm(
            x
        )  # lstm_out es el tensor de salida, h_n y c_n son los estados ocultos

        # Tomamos el último estado oculto de la secuencia
        last_hidden_state = h_n[-1]

        # Pasamos a través de la capa fully connected
        x = self.fc(last_hidden_state)

        # Aplicamos dropout
        x = self.dropout(x)

        return x


class ComplexLSTMModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim1,
        hidden_dim2,
        hidden_dim3,
        output_dim,
        dropout_rate=0.25,
    ):
        super(ComplexLSTMModel, self).__init__()

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_dim, hidden_dim1, batch_first=True, bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            hidden_dim1 * 2, hidden_dim2, batch_first=True, bidirectional=False
        )
        self.lstm3 = nn.LSTM(hidden_dim2, hidden_dim3, batch_first=True)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim3, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        # Batch normalization
        self.batchnorm = nn.BatchNorm1d(hidden_dim3)

    def forward(self, x):
        # Forward pass through LSTM layers
        x, _ = self.lstm1(x)  # Extracting only the output from the tuple
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        # Using the last output of the sequence for classification
        x = self.batchnorm(x[:, -1, :])  # Batch normalization on the last output
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


import torch
import torch.nn as nn


class AudioLSTM(nn.Module):
    def __init__(
        self, input_dim, hidden_dim=256, output_dim=10, n_layers=2, dropout_rate=0.3
    ):
        super(AudioLSTM, self).__init__()

        # Definir la LSTM
        self.lstm1 = nn.LSTM(
            input_dim, hidden_dim, n_layers, batch_first=True, dropout=dropout_rate
        )

        # Capas densas
        self.fc1 = nn.Linear(
            hidden_dim, hidden_dim // 2
        )  # Reducimos las dimensiones a la mitad
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)  # Capa de salida

        # Definir Dropout y ReLU
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x, hidden=None):
        # Pasamos la entrada a través de la LSTM
        lstm_out, hidden = self.lstm1(x, hidden)

        # Aplicamos el Dropout
        out = self.dropout(lstm_out)

        # Usamos solo la última salida de la secuencia
        out = out[:, -1, :]  # [batch_size, hidden_dim]

        # Pasamos por las capas densas
        out = self.fc1(out)
        out = self.relu(out)  # ReLU en la primera capa densa
        out = self.fc2(out)  # Capa de salida

        return out

    def init_hidden(self, batch_size):
        # Inicializar el estado oculto
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
        )
        return hidden


class ComplexGRUModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, output_dim, dropout_rate
    ):
        super(ComplexGRUModel, self).__init__()
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True)
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)
        self.gru3 = nn.GRU(hidden_dim2, hidden_dim3, batch_first=True)

        # BatchNorm aplicada a la salida completa de la última capa GRU
        self.batchnorm = nn.BatchNorm1d(hidden_dim3)

        self.fc1 = nn.Linear(hidden_dim3, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x, _ = self.gru1(x)
        x, _ = self.gru2(x)
        x, _ = self.gru3(x)

        # BatchNorm en la secuencia completa y selecciona el último paso
        x = self.batchnorm(x.transpose(1, 2)).transpose(
            1, 2
        )  # (batch, features, sequence)
        x = x[:, -1, :]  # Usa el último paso después de la BatchNorm

        # Cambiamos a tanh para la activación
        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)
        return x


class ImprovedGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate):
        super(ImprovedGRUModel, self).__init__()

        # Primera capa GRU
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(
            dropout_rate
        )  # Regularización después de la primera GRU

        # Segunda capa GRU
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(
            dropout_rate
        )  # Regularización después de la segunda GRU

        # Global pooling para capturar toda la secuencia
        self.global_max_pooling = nn.AdaptiveMaxPool1d(
            1
        )  # Alternativa: AdaptiveAvgPool1d
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)

        self.activation = nn.Tanh()
        # Capa densa final
        self.fc1 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Primera capa GRU
        x, _ = self.gru1(x)  # (batch, seq_len, hidden_dim1)
        x = self.bn1(x.transpose(1, 2)).transpose(
            1, 2
        )  # BatchNorm en la dimensión de características
        x = self.dropout1(x)

        # Segunda capa GRU
        x, _ = self.gru2(x)  # (batch, seq_len, hidden_dim2)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout2(x)

        # Global pooling para reducir la dimensión temporalSSSS
        # Pooling
        x = x.transpose(1, 2)  # (batch, hidden_dim2, seq_len)

        x_max = self.global_max_pooling(x).squeeze(2)
        x_avg = self.global_avg_pooling(x).squeeze(2)
        x = x_max

        # Capa completamente conectada
        x = self.activation(x)
        x = self.fc1(x)  # Salida sin activación (para usar con CrossEntropyLoss)
        return x


class BasicRNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate):
        super(BasicRNNModel, self).__init__()

        # Primera capa RNN
        self.rnn1 = nn.RNN(input_dim, hidden_dim1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(
            dropout_rate
        )  # Regularización después de la primera RNN

        # Segunda capa RNN
        self.rnn2 = nn.RNN(hidden_dim1, hidden_dim2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(
            dropout_rate
        )  # Regularización después de la segunda RNN

        # Capa densa final
        self.fc1 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Primera capa RNN
        x, _ = self.rnn1(x)  # (batch, seq_len, hidden_dim1)
        x = self.bn1(x.transpose(1, 2)).transpose(
            1, 2
        )  # BatchNorm en la dimensión de características
        x = self.dropout1(x)

        # Segunda capa RNN
        x, _ = self.rnn2(x)  # (batch, seq_len, hidden_dim2)
        x = self.bn2(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout2(x)

        # Tomar la última salida de la secuencia
        x = x[:, -1, :]  # Selecciona el último estado oculto de la secuencia

        # Capa completamente conectada
        x = self.fc1(x)  # Salida sin activación (para usar con CrossEntropyLoss)
        return x


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(GRU, self).__init__()

        # Primera capa GRU
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True)

        # Segunda capa GRU
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)

        self.activation = nn.Tanh()

        # Capa densa final
        self.fc1 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Primera capa GRU
        x, _ = self.gru1(x)

        # Segunda capa GRU
        x, _ = self.gru2(x)

        x = x[:, -1, :]  # Usar solo la última salida de la secuencia

        # Aplicar activación
        x = self.activation(x)

        # Capa completamente conectada
        x = self.fc1(x)

        return x


import torch
import torch.nn as nn


class CRNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CRNN, self).__init__()
        self.conv1 = nn.Conv1d(
            2, 32, kernel_size=3, stride=1, padding=1
        )  # Adaptado para canales de entrada
        self.bn1 = nn.BatchNorm1d(32)
        self.elu1 = nn.ELU()
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.dropout1 = nn.Dropout(p=0.1)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.elu2 = nn.ELU()
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(p=0.1)

        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.elu3 = nn.ELU()
        self.pool3 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout3 = nn.Dropout(p=0.1)

        self.lstm = nn.LSTM(
            input_size=64, hidden_size=64, num_layers=2, batch_first=True
        )

        self.fc_dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: [batch_size, channels, features]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)

        # Ajustar para LSTM
        x = x.permute(0, 2, 1)  # [batch_size, sequence_length, features]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Tomar la última salida de la LSTM

        x = self.fc_dropout(x)
        x = self.fc(x)

        return x


class AttGRUModel(nn.Module):
    def __init__(
        self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout_rate, num_heads=4
    ):
        super(AttGRUModel, self).__init__()

        # Primera capa GRU
        self.gru1 = nn.GRU(input_dim, hidden_dim1, batch_first=True)
        self.bn1 = nn.BatchNorm1d(hidden_dim1)
        self.dropout1 = nn.Dropout(
            dropout_rate
        )  # Regularización después de la primera GRU

        # Segunda capa GRU
        self.gru2 = nn.GRU(hidden_dim1, hidden_dim2, batch_first=True)
        self.bn2 = nn.BatchNorm1d(hidden_dim2)
        self.dropout2 = nn.Dropout(
            dropout_rate
        )  # Regularización después de la segunda GRU

        # Capa de atención multi-cabeza
        self.attention = nn.MultiheadAttention(
            hidden_dim2, num_heads=num_heads, batch_first=True
        )

        # Global pooling para capturar toda la secuencia
        self.global_max_pooling = nn.AdaptiveMaxPool1d(
            1
        )  # Puedes alternar entre MaxPool o AvgPool

        # Capa de activación
        self.activation = nn.Tanh()

        # Capa densa final
        self.fc1 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # Primera capa GRU
        x, _ = self.gru1(x)  # (batch_size, seq_len, hidden_dim1)
        x = self.bn1(x.transpose(1, 2)).transpose(
            1, 2
        )  # BatchNorm en la dimensión de características
        x = self.dropout1(x)

        # Segunda capa GRU
        x, _ = self.gru2(x)  # (batch_size, seq_len, hidden_dim2)
        x = self.bn2(x.transpose(1, 2)).transpose(
            1, 2
        )  # BatchNorm después de la segunda GRU
        x = self.dropout2(x)

        # Capa de atención multi-cabeza
        # La entrada de la atención debe ser (seq_len, batch_size, hidden_dim2)
        x_attn, _ = self.attention(x, x, x)  # Aplicamos la atención sobre sí misma

        # Global pooling para reducir la dimensión temporal
        x = x_attn.transpose(1, 2)  # (batch_size, hidden_dim2, seq_len)
        x_max = self.global_max_pooling(x).squeeze(
            2
        )  # Aplicamos Max Pooling y eliminamos la dimensión extra

        # Capa completamente conectada
        x = self.activation(x_max)
        x = self.fc1(x)  # Salida sin activación (para usar con CrossEntropyLoss)

        return x


def train_model(
    model,
    train_loader,
    test_loader,
    criterion,
    optimizer,
    epochs=100,
    device="cpu",
    verbose=1,
):
    """
    Train and validate the model with metrics per class.

    Args:
        model: PyTorch model to be trained.
        train_loader: DataLoader for training data.
        test_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for updating model weights.
        epochs: Number of epochs for training.
        device: Device to perform computation on ('cpu' or 'cuda').

    Returns:
        Dictionary containing training and validation losses, accuracies, F1-scores, recalls,
        and per-class metrics.
    """
    # Ensure model is on the specified device
    model.to(device)

    # Initialize metrics storage
    metrics = {
        "train_losses": [],
        "valid_losses": [],
        "train_accuracies": [],
        "train_f1_scores": [],
        "train_recalls": [],
        "valid_accuracies": [],
        "valid_f1_scores": [],
        "valid_recalls": [],
        "train_class_metrics": [],
        "valid_class_metrics": [],
    }

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_labels = []
        all_preds = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

        # Calculate training metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_preds)
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
        epoch_recall = recall_score(all_labels, all_preds, average="weighted")

        # Per-class metrics
        train_class_metrics = classification_report(
            all_labels, all_preds, output_dict=True, zero_division=0
        )

        metrics["train_losses"].append(epoch_loss)
        metrics["train_accuracies"].append(epoch_accuracy)
        metrics["train_f1_scores"].append(epoch_f1)
        metrics["train_recalls"].append(epoch_recall)
        metrics["train_class_metrics"].append(train_class_metrics)

        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_labels = []
        valid_preds = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                valid_labels.extend(labels.cpu().tolist())
                valid_preds.extend(preds.cpu().tolist())

        # Calculate validation metrics
        valid_loss /= len(test_loader)
        valid_accuracy = accuracy_score(valid_labels, valid_preds)
        valid_f1 = f1_score(valid_labels, valid_preds, average="weighted")
        valid_recall = recall_score(valid_labels, valid_preds, average="weighted")

        # Per-class metrics
        valid_class_metrics = classification_report(
            valid_labels, valid_preds, output_dict=True, zero_division=0
        )

        metrics["valid_losses"].append(valid_loss)
        metrics["valid_accuracies"].append(valid_accuracy)
        metrics["valid_f1_scores"].append(valid_f1)
        metrics["valid_recalls"].append(valid_recall)
        metrics["valid_class_metrics"].append(valid_class_metrics)

        # Log the metrics for the current epoch
        if verbose == 1:
            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_accuracy:.4f} - Train F1: {epoch_f1:.4f} - Train Recall: {epoch_recall:.4f} - "
                f"Valid Loss: {valid_loss:.4f} - Valid Acc: {valid_accuracy:.4f} - Valid F1: {valid_f1:.4f} - Valid Recall: {valid_recall:.4f}"
            )

        if verbose > 2:
            print("\nPer-class training metrics:")
            for cls, metrics in train_class_metrics.items():
                if cls.isdigit():  # Only show metrics for actual classes
                    print(
                        f"  Class {cls}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}"
                    )

            print("\nPer-class validation metrics:")
            for cls, metrics in valid_class_metrics.items():
                if cls.isdigit():  # Only show metrics for actual classes
                    print(
                        f"  Class {cls}: Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1-score']:.4f}"
                    )

    return [metrics, valid_labels, valid_preds]


def data_loader(
    X_train_padded,
    X_test_padded,
    y_tensor_train,
    y_tensor_test,
    batch_size=64,
    device="cpu",
    shuffle=False,
):
    """
    Prepara DataLoaders para entrenamiento y validación.
    """
    # Crear tensores y verificar dimensiones
    train_data = TensorDataset(
        torch.tensor(X_train_padded, dtype=torch.float32).to(device),
        torch.tensor(y_tensor_train, dtype=torch.long).to(device),
    )
    test_data = TensorDataset(
        torch.tensor(X_test_padded, dtype=torch.float32).to(device),
        torch.tensor(y_tensor_test, dtype=torch.long).to(device),
    )

    # Crear DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def preprocess_data(X_train, X_test, y_train, y_test, verbose=False):
    sequence_lengths = [len(seq) for seq in X_train]
    max_timesteps = max(sequence_lengths)

    X_train_padded = pad_sequences(
        X_train, maxlen=max_timesteps, padding="post", dtype="float32"
    )
    X_test_padded = pad_sequences(
        X_test, maxlen=max_timesteps, padding="post", dtype="float32"
    )
    if verbose:
        print("X_train shape before padding:", X_train.shape)
    if verbose:
        print("X_train shape after padding:", X_train_padded.shape)

    num_classes = len(np.unique(y_train))
    label_encoder = LabelEncoder()

    y_train_numeric = label_encoder.fit_transform(y_train)
    y_test_numeric = label_encoder.transform(y_test)

    if verbose:
        print("Classes:", label_encoder.classes_)
    # y_train_one_hot = to_categorical(y_train_numeric, num_classes=num_classes)
    # y_test_one_hot = to_categorical(y_test_numeric, num_classes=num_classes)

    return (
        X_train_padded,
        X_test_padded,
        y_train_numeric,
        y_test_numeric,
    )  # y_train_one_hot, y_test_one_hot)


def aux_cross_validation(lap, total_folds=10):

    test_folds = [lap]
    train_folds = [x for x in range(1, total_folds + 1) if x != lap]
    return train_folds, test_folds


def create_criteria(args, model):

    optimizer = optim.Adam(
        model.parameters(),
        lr=args["learing_rate"],
        weight_decay=args.get("weight_decay", 0),
    )

    criterion = nn.CrossEntropyLoss()

    return optimizer, criterion


def cross_validate(
    model_name,
    config,
    features_list,
    labels_list,
    folds_list,
    epochs=20,
    device="cpu",
    verbose=1,
    summary=False,
):
    """
    Función que recibe una función de entrenamiento y la ejecuta sobre un conjunto de datos para validación cruzada.
    """
    results = []
    train_class = []
    valid_class = []
    true_labels, pred_labels = [], []

    for i in range(1, 11):

        train_folds, test_folds = aux_cross_validation(lap=i)
        X_train, X_test, y_train, y_test = prepare_datasets(
            features_list, labels_list, folds_list, train_folds, test_folds
        )
        X_train_padded, X_test_padded, y_train_one_hot, y_test_one_hot = (
            preprocess_data(X_train, X_test, y_train, y_test)
        )

        if model_name == "ImprovedGRUModel":
            model = ImprovedGRUModel(
                input_dim=X_train_padded.shape[2],
                hidden_dim1=config["hidden_dim1"],
                hidden_dim2=config["hidden_dim2"],
                output_dim=10,
                dropout_rate=config["dropout_rate"],
            ).to(device)

        elif model_name == "AttentionGRU":
            model = AttGRUModel(
                input_dim=X_train_padded.shape[2],
                hidden_dim1=config["hidden_dim1"],
                hidden_dim2=config["hidden_dim2"],
                output_dim=10,
                dropout_rate=config["dropout_rate"],
            ).to(device)
        elif model_name == "GRU":
            model = GRU(
                input_dim=X_train_padded.shape[2],
                hidden_dim1=config["hidden_dim1"],
                hidden_dim2=config["hidden_dim2"],
                output_dim=10,
                dropout_rate=config["dropout_rate"],
            ).to(device)

        elif model_name == "ComplexGRU":
            model = ComplexGRUModel(
                input_dim=X_train_padded.shape[2],
                hidden_dim1=config["hidden_dim1"],
                hidden_dim2=config["hidden_dim2"],
                hidden_dim3=config["hidden_dim3"],
                output_dim=10,
                dropout_rate=config["dropout_rate"],
            ).to(device)

        elif model_name == "ImprovedGRU":
            model = ImprovedGRUModel(
                input_dim=X_train_padded.shape[2],
                hidden_dim1=config["hidden_dim1"],
                hidden_dim2=config["hidden_dim2"],
                output_dim=10,
                dropout_rate=config["dropout_rate"],
            ).to(device)

        elif model_name == "LSTM":
            model = LSTMClassifier(
                input_dim=X_train_padded.shape[2],
                hidden_dim=config["hidden_dim1"],
                output_dim=10,
                dropout_rate=config["dropout_rate"],
            ).to(device)

        elif model_name == "CLSTM":
            model = ConvLSTMClassifier(
                input_dim=X_train_padded.shape[2],
                hidden_dim=config["hidden_dim1"],
                output_dim=10,
                dropout_rate=config["dropout_rate"],
            ).to(device)

        elif model_name == "BasicRNN":
            model = BasicRNNModel(
                input_dim=X_train_padded.shape[2],
                hidden_dim1=config["hidden_dim1"],
                hidden_dim2=config["hidden_dim2"],
                output_dim=10,
                dropout_rate=config["dropout_rate"],
            ).to(device)

        elif model_name == "CRNN":
            model = CRNN().to(device)

        if i == 1 and summary:
            summary(model, (X_train_padded.shape[1], X_train_padded.shape[2]))

        optimizer, criterion = create_criteria(config, model)

        # Preparar DataLoaders
        train_loader, test_loader = data_loader(
            X_train_padded,
            X_test_padded,
            y_train_one_hot,
            y_test_one_hot,
            device=device,
        )

        lista = train_model(
            model,
            train_loader,
            test_loader,
            criterion,
            optimizer,
            epochs,
            device,
            verbose=verbose,
        )
        metrics = lista[0]
        results.append(metrics)
        true_labels.append(lista[1])
        pred_labels.append(lista[2])

    avg_metrics = {
        "train_losses": [],
        "valid_losses": [],
        "train_accuracies": [],
        "train_f1_scores": [],
        "train_recalls": [],
        "valid_accuracies": [],
        "valid_f1_scores": [],
        "valid_recalls": [],
    }

    # Promediar las métricas a través de todos los pliegues
    for fold in results:
        for key, value in fold.items():
            if key not in ["train_class_metrics", "valid_class_metrics"]:
                avg_metrics[key].append(value)
            else:
                (
                    train_class.append(value)
                    if key == "train_class_metrics"
                    else valid_class.append(value)
                )

    # Calcular la media de cada métrica
    avg_results = {}
    for key, values in avg_metrics.items():
        avg_results[key] = np.mean(values)

    print(avg_results)

    return avg_results, train_class, valid_class, results, true_labels, pred_labels


# 10-fold cross validation splits


def prepare_splits(x, y, metadata, n_splits=10):
    group_kfold = GroupKFold(n_splits=n_splits)
    groups = metadata["fold"]

    for train_idx, test_idx in group_kfold.split(x, y, groups):
        val_idx = train_idx[: len(train_idx) // 10]
        train_idx = train_idx[len(train_idx) // 10 :]

        x_train, x_val, x_test = x[train_idx], x[val_idx], x[test_idx]
        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        yield x_train[..., np.newaxis], x_val[..., np.newaxis], x_test[
            ..., np.newaxis
        ], y_train, y_val, y_test


def new_cnn_model(input_shape=(128, 128, 1), num_classes=10):
    model = Sequential()

    # 1st conv + MaxPooling + Dropout
    model.add(
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape)
    )
    model.add(BatchNormalization())  # added because of overfitting
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Conv + MaxPooling + Dropout
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # 3rd Conv + MaxPooling + Dropout
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    # Flattening
    model.add(Flatten())

    # Fully connected layers
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    # Output layer
    model.add(Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Adaptive gradient optimizer
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if __name__ == "__main__":

    print(aux_cross_validation(lap=2))
