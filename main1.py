import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Fonctions d'activation
def relu(x):
    """ReLU activation: max(0, x)"""
    assert isinstance(x, np.ndarray), "Input to ReLU must be a numpy array"
    result = np.maximum(0, x)
    assert np.all(result >= 0), "ReLU output must be non-negative"
    return result

def relu_derivative(x):
    """Derivative of ReLU: 1 if x > 0, else 0"""
    assert isinstance(x, np.ndarray), "Input to ReLU derivative must be a numpy array"
    result = (x > 0).astype(float)
    assert np.all((result == 0) | (result == 1)), "ReLU derivative must be 0 or 1"
    return result

def softmax(x):
    """Softmax activation: exp(x) / sum(exp(x))"""
    assert isinstance(x, np.ndarray), "Input to softmax must be a numpy array"
    # Stabilité numérique: soustraire le max pour éviter les overflow
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    result = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    assert np.all((result >= 0) & (result <= 1)), "Softmax output must be in [0, 1]"
    assert np.allclose(np.sum(result, axis=1), 1, atol=1e-6), "Softmax output must sum to 1 per sample"
    return result

# Classe MultiClassNeuralNetwork
class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01):
        """Initialize the neural network with given layer sizes and learning rate."""
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2, "layer_sizes must be a list with at least 2 elements"
        assert all(isinstance(size, int) and size > 0 for size in layer_sizes), "All layer sizes must be positive integers"
        assert isinstance(learning_rate, (int, float)) and learning_rate > 0, "Learning rate must be a positive number"
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialisation des poids et biais
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def forward(self, X):
        """Forward propagation: Z^[l] = A^[l-1]W^[l] + b^[l], A^[l] = g(Z^[l])"""
        assert isinstance(X, np.ndarray), "Input X must be a numpy array"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        
        self.activations = [X]  # A0 = X
        self.z_values = []      # Stockage des Z
        
        # Couches cachées (ReLU)
        a = X
        for i in range(len(self.weights) - 1):
            z = a @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            a = relu(z)
            self.activations.append(a)
        
        # Couche de sortie (Softmax)
        z = a @ self.weights[-1] + self.biases[-1]
        self.z_values.append(z)
        output = softmax(z)
        self.activations.append(output)
        
        return output
    
    def compute_loss(self, y_true, y_pred):
        """Categorical Cross-Entropy: J = -1/m * sum(y_true * log(y_pred))"""
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to loss must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        
        m = y_true.shape[0]
        # Clip pour éviter log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred)) / m
        assert not np.isnan(loss), "Loss computation resulted in NaN"
        return loss
    
    def compute_accuracy(self, y_true, y_pred):
        """Compute accuracy: proportion of correct predictions"""
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to accuracy must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        
        predictions = np.argmax(y_pred, axis=1)
        true_labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(predictions == true_labels)
        assert 0 <= accuracy <= 1, "Accuracy must be between 0 and 1"
        return accuracy
    
    def backward(self, X, y, outputs):
        """Backpropagation: compute gradients and update parameters"""
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray) and isinstance(outputs, np.ndarray), "Inputs to backward must be numpy arrays"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y.shape == outputs.shape, "y and outputs must have the same shape"
        
        m = X.shape[0]
        self.d_weights = [None] * len(self.weights)
        self.d_biases = [None] * len(self.biases)
        
        # Gradient pour la couche de sortie (softmax + cross-entropy)
        dZ = outputs - y
        self.d_weights[-1] = self.activations[-2].T @ dZ / m
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Rétropropagation à travers les couches cachées
        for i in range(len(self.weights)-2, -1, -1):
            dA = dZ @ self.weights[i+1].T
            dZ = dA * relu_derivative(self.z_values[i])
            self.d_weights[i] = self.activations[i].T @ dZ / m
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Mise à jour des paramètres
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.d_weights[i]
            self.biases[i] -= self.learning_rate * self.d_biases[i]
    
    def train(self, X, y, X_val, y_val, epochs, batch_size):
        """Train the neural network using mini-batch SGD, with validation"""
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray), "X and y must be numpy arrays"
        assert isinstance(X_val, np.ndarray) and isinstance(y_val, np.ndarray), "X_val and y_val must be numpy arrays"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y.shape[1] == self.layer_sizes[-1], f"Output dimension ({y.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
        assert X_val.shape[1] == self.layer_sizes[0], f"Validation input dimension ({X_val.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y_val.shape[1] == self.layer_sizes[-1], f"Validation output dimension ({y_val.shape[1]}) must match output layer size ({self.layer_sizes[-1]})"
        assert isinstance(epochs, int) and epochs > 0, "Epochs must be a positive integer"
        assert isinstance(batch_size, int) and batch_size > 0, "Batch size must be a positive integer"
        
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(epochs):
            # Mélanger les données
            indices = np.random.permutation(X.shape[0])
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            # Parcours par mini-batches
            for i in range(0, X.shape[0], batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Propagation avant
                outputs = self.forward(X_batch)
                batch_loss = self.compute_loss(y_batch, outputs)
                epoch_loss += batch_loss * X_batch.shape[0]
                
                # Rétropropagation
                self.backward(X_batch, y_batch, outputs)
            
            # Calcul des métriques
            train_loss = epoch_loss / X.shape[0]
            train_pred = self.forward(X)
            train_accuracy = self.compute_accuracy(y, train_pred)
            
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_accuracy = self.compute_accuracy(y_val, val_pred)
            
            # Stockage des résultats
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            # Affichage périodique
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")
        
        return train_losses, val_losses, train_accuracies, val_accuracies
    
    def predict(self, X):
        """Predict class labels"""
        assert isinstance(X, np.ndarray), "Input X must be a numpy array"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        
        outputs = self.forward(X)
        predictions = np.argmax(outputs, axis=1)
        return predictions

# Chargement et prétraitement des données
def load_and_preprocess_data(data_dir):
    # Charger les labels
    try:
        labels_df = pd.read_csv(os.path.join(data_dir, 'amhcd-data-64/labels-map.csv'))
        assert 'image_path' in labels_df.columns and 'label' in labels_df.columns, "CSV must contain 'image_path' and 'label' columns"
    except FileNotFoundError:
        print("labels-map.csv not found. Building DataFrame from folders...")
        image_paths = []
        labels = []
        for label_dir in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label_dir)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    image_paths.append(os.path.join(label_path, img_name))
                    labels.append(label_dir)
        labels_df = pd.DataFrame({'image_path': image_paths, 'label': labels})
    
    # Vérification des données
    assert not labels_df.empty, "No data loaded. Check dataset files."
    print(f"Loaded {len(labels_df)} samples with {labels_df['label'].nunique()} unique classes.")
    
    # Encoder les étiquettes
    label_encoder = LabelEncoder()
    labels_df['label_encoded'] = label_encoder.fit_transform(labels_df['label'])
    num_classes = len(label_encoder.classes_)
    
    # Fonction de chargement d'image
    def load_and_preprocess_image(image_path, target_size=(32, 32)):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0  # Normalisation
        return img.flatten()  # Aplatir en vecteur
    
    # Charger toutes les images
    X = np.array([load_and_preprocess_image(os.path.join(data_dir, path)) for path in labels_df['image_path']])
    y = labels_df['label_encoded'].values
    
    # Vérifier les dimensions
    assert X.shape[0] == y.shape[0], "Mismatch between number of images and labels"
    assert X.shape[1] == 32 * 32, f"Expected flattened image size of {32*32}, got {X.shape[1]}"
    
    # Division des données
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
    
    # Encodage one-hot
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    y_train_one_hot = one_hot_encoder.fit_transform(y_train.reshape(-1, 1))
    y_val_one_hot = one_hot_encoder.transform(y_val.reshape(-1, 1))
    y_test_one_hot = one_hot_encoder.transform(y_test.reshape(-1, 1))
    
    return X_train, X_val, X_test, y_train_one_hot, y_val_one_hot, y_test_one_hot, num_classes, label_encoder

# Point d'entrée principal
if __name__ == "__main__":
    data_dir = os.path.join(os.getcwd(), 'amhcd-data-64/tifinagh-images/')
    
    # Charger les données
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, label_encoder = load_and_preprocess_data(data_dir)
    
    print(f"Train: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Créer et entraîner le modèle
    layer_sizes = [X_train.shape[1], 64, 32, num_classes]
    nn = MultiClassNeuralNetwork(layer_sizes, learning_rate=0.01)
    train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
        X_train, y_train, X_val, y_val, epochs=100, batch_size=32
    )
    
    # Évaluation finale
    y_pred = nn.predict(X_test)
    print("\nClassification Report (Test set):")
    print(classification_report(
        np.argmax(y_test, axis=1), 
        y_pred, 
        target_names=label_encoder.classes_
    ))
    
    # Matrice de confusion
    cm = confusion_matrix(np.argmax(y_test, axis=1), y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix (Test set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Courbes d'apprentissage
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Loss Curve')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Accuracy Curve')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('loss_accuracy_plot.png')
    plt.close()
