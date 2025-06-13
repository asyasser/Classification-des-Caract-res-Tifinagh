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
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Classe MultiClassNeuralNetwork avec régularisation L2
class MultiClassNeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, l2_lambda=0.01):
        """Initialize the neural network with given layer sizes and learning rate."""
        assert isinstance(layer_sizes, list) and len(layer_sizes) >= 2, "layer_sizes must be a list with at least 2 elements"
        assert all(isinstance(size, int) and size > 0 for size in layer_sizes), "All layer sizes must be positive integers"
        assert isinstance(learning_rate, (int, float)) and learning_rate > 0, "Learning rate must be a positive number"
        assert isinstance(l2_lambda, (int, float)) and l2_lambda >= 0, "L2 regularization coefficient must be non-negative"
        
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda  # Coefficient de régularisation L2
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
        """Categorical Cross-Entropy avec régularisation L2"""
        assert isinstance(y_true, np.ndarray) and isinstance(y_pred, np.ndarray), "Inputs to loss must be numpy arrays"
        assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
        
        m = y_true.shape[0]
        # Clip pour éviter log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        cross_entropy_loss = -np.sum(y_true * np.log(y_pred)) / m
        
        # Calcul du terme de régularisation L2
        l2_penalty = 0
        for w in self.weights:
            l2_penalty += np.sum(w ** 2)
        l2_penalty = (self.l2_lambda / (2 * m)) * l2_penalty
        
        total_loss = cross_entropy_loss + l2_penalty
        assert not np.isnan(total_loss), "Loss computation resulted in NaN"
        return total_loss
    
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
        """Backpropagation avec régularisation L2"""
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray) and isinstance(outputs, np.ndarray), "Inputs to backward must be numpy arrays"
        assert X.shape[1] == self.layer_sizes[0], f"Input dimension ({X.shape[1]}) must match input layer size ({self.layer_sizes[0]})"
        assert y.shape == outputs.shape, "y and outputs must have the same shape"
        
        m = X.shape[0]
        self.d_weights = [None] * len(self.weights)
        self.d_biases = [None] * len(self.biases)
        
        # Gradient pour la couche de sortie (softmax + cross-entropy)
        dZ = outputs - y
        self.d_weights[-1] = (self.activations[-2].T @ dZ) / m
        self.d_biases[-1] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # Ajout du terme de régularisation L2 aux poids
        self.d_weights[-1] += (self.l2_lambda / m) * self.weights[-1]
        
        # Rétropropagation à travers les couches cachées
        for i in range(len(self.weights)-2, -1, -1):
            dA = dZ @ self.weights[i+1].T
            dZ = dA * relu_derivative(self.z_values[i])
            self.d_weights[i] = (self.activations[i].T @ dZ) / m
            self.d_biases[i] = np.sum(dZ, axis=0, keepdims=True) / m
            
            # Ajout du terme de régularisation L2 aux poids
            self.d_weights[i] += (self.l2_lambda / m) * self.weights[i]
        
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
    # Chemin pour le CSV
    csv_path = os.path.join(data_dir, 'labels-map.csv')
    
    try:
        # Charger le CSV sans header et avec les bons noms de colonnes
        labels_df = pd.read_csv(csv_path, header=None, names=['image_path', 'label'])
        print("labels-map.csv found. Loading data from CSV...")
        
        # Correction des chemins
        labels_df['image_path'] = labels_df['image_path'].str.replace(
            './images-data-64/tifinagh-images/', 
            'tifinagh-images/', 
            regex=False
        )
        labels_df['image_path'] = labels_df['image_path'].str.replace('./', '', regex=False)
        labels_df['image_path'] = labels_df['image_path'].apply(
            lambda x: os.path.join(data_dir, x)
        )
    except FileNotFoundError:
        print("labels-map.csv not found. Building DataFrame from folders...")
        base_dir = os.path.join(data_dir, 'tifinagh-images')
        image_paths = []
        labels = []
        for label_dir in os.listdir(base_dir):
            label_path = os.path.join(base_dir, label_dir)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    if img_name.lower().endswith(('.jpeg', '.jpg', '.png')):
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
        if not os.path.exists(image_path):
            print(f"Warning: File not found - {image_path}")
            return None
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Failed to load image - {image_path}")
            return None
        
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        return img.flatten()
    
    # Charger toutes les images
    images = []
    valid_indices = []
    
    for idx, path in enumerate(labels_df['image_path']):
        img = load_and_preprocess_image(path)
        if img is not None:
            images.append(img)
            valid_indices.append(idx)
    
    # Filtrer les données valides
    valid_df = labels_df.iloc[valid_indices]
    X = np.array(images)
    y = valid_df['label_encoded'].values
    
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
    # Chemin pour le dataset
    data_dir = os.path.join(os.getcwd(), 'amhcd-data-64')
    
    # Charger les données
    X_train, X_val, X_test, y_train, y_val, y_test, num_classes, label_encoder = load_and_preprocess_data(data_dir)
    
    print(f"Train: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples, Test: {X_test.shape[0]} samples")
    
    # Créer et entraîner le modèle avec régularisation L2
    layer_sizes = [X_train.shape[1], 128, 64, num_classes]
    nn = MultiClassNeuralNetwork(
        layer_sizes, 
        learning_rate=0.01,
        l2_lambda=0.001  # Coefficient de régularisation L2
    )
    train_losses, val_losses, train_accuracies, val_accuracies = nn.train(
        X_train, y_train, X_val, y_val, epochs=100, batch_size=64
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
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix (Test set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
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
    plt.show()

