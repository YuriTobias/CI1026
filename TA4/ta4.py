# CI1026 - TA04: Aplicando Algoritmos de Classificação utilizando o MNIST. Trabalho desenvolvido por:
# * Izalorran Oliveira Santos Bonaldi (GRR20210582)
# * Yuri Junqueira Tobias (GRR20211767)
# Docente: Eduardo Todt
# Data de entrega: 09/06/2025

# Bibliotecas
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter

# Dataset MNIST
from tensorflow.keras.datasets import mnist

# Bibliotecas de aprendizado de máquina
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC

# Biblioteca de decomposição de dados
from sklearn.decomposition import PCA

# Bibliotecas para visualização e avaliação de modelos
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def run_linear_classifiers(X_train, y_train, X_test, y_test, use_pca=False):
    print("\nExecutando classificadores lineares...\n")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if use_pca:
        print(f"\nReduzindo dimensionalidade com PCA...")
        print(f"→ Dimensão original: {X_train.shape[1]} componentes")
        pca = PCA(n_components=0.95, random_state=42)
        X_train_scaled = pca.fit_transform(X_train_scaled)
        X_test_scaled = pca.transform(X_test_scaled)
        print(f"→ Dimensão reduzida: {X_train_scaled.shape[1]} componentes")

    results = {}

    # Logistic Regression - Multinomial
    print("\n--- Regressão Logística (Multinomial + lbfgs) ---")
    model_name = "LogReg_Multinomial_LBFGS"
    start = time.time()
    clf = LogisticRegression(solver='lbfgs', max_iter=200, C=1.0, random_state=42, n_jobs=-1)
    clf.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    start_pred = time.time()
    y_pred = clf.predict(X_test_scaled)
    pred_time = time.time() - start_pred
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} - Acurácia: {acc:.4f}, Tempo Treino: {train_time:.2f}s, Tempo Pred./Amostra: {(pred_time/len(X_test))*1e6:.2f}µs")
    print(classification_report(y_test, y_pred))
    results[model_name] = acc

    # Logistic Regression - OvR
    print("\n--- Regressão Logística (OvR + liblinear) ---")
    model_name = "LogReg_OvR_Liblinear"
    start = time.time()
    clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=200, C=1.0, random_state=42))
    clf.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    start_pred = time.time()
    y_pred = clf.predict(X_test_scaled)
    pred_time = time.time() - start_pred
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} - Acurácia: {acc:.4f}, Tempo Treino: {train_time:.2f}s, Tempo Pred./Amostra: {(pred_time/len(X_test))*1e6:.2f}µs")
    print(classification_report(y_test, y_pred))
    results[model_name] = acc

    # SVC com kernel linear
    print("\n--- SVC com kernel linear (pode ser mais demorado) ---")
    model_name = "SVC_Linear_Kernel"
    start = time.time()
    clf = SVC(kernel='linear', C=1.0, random_state=42, max_iter=10000)
    clf.fit(X_train_scaled, y_train)
    train_time = time.time() - start
    start_pred = time.time()
    y_pred = clf.predict(X_test_scaled)
    pred_time = time.time() - start_pred
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} - Acurácia: {acc:.4f}, Tempo Treino: {train_time:.2f}s, Tempo Pred./Amostra: {(pred_time/len(X_test))*1e6:.2f}µs")
    print(classification_report(y_test, y_pred))
    results[model_name] = acc

def run_knn(X_train, y_train, X_test, y_test, use_pca=False):
    print("\nExecutando kNN com GridSearchCV...")
    
    if use_pca:
        print(f"\nReduzindo dimensionalidade com PCA...")
        print(f"→ Dimensão original: {X_train.shape[1]} componentes")
        pca = PCA(n_components=0.95, random_state=42)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        print(f"→ Dimensão reduzida: {X_train.shape[1]} componentes")

    distance_metrics = ['euclidean', 'manhattan', 'cosine']

    for metric in distance_metrics:
        print(f"\n--- Métrica de distância: {metric} ---")
        
        param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
        grid = GridSearchCV(KNeighborsClassifier(metric=metric), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        
        start = time.time()
        grid.fit(X_train, y_train)
        end = time.time()
        
        print(f"Melhor valor de k: {grid.best_params_['n_neighbors']}")
        print(f"Acurácia média na validação cruzada: {grid.best_score_:.4f}")
        print(f"Tempo de treinamento: {end - start:.2f} segundos")
        
        # Extração de resultados da validação cruzada
        results = pd.DataFrame(grid.cv_results_)
        
        # Gráfico 1: Acurácia média por valor de k
        plt.figure(figsize=(8, 5))
        sns.lineplot(x='param_n_neighbors', y='mean_test_score', data=results, marker='o')
        plt.title(f'Acurácia média por valor de k (métrica: {metric})')
        plt.xlabel('Número de vizinhos (k)')
        plt.ylabel('Acurácia média')
        plt.grid(True)
        plt.show()

        best_knn = grid.best_estimator_
        y_pred = best_knn.predict(X_test)
        
        print("\nAvaliação no conjunto de teste:")
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Matriz de Confusão - kNN ({metric})")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.show()

        print("\n" + "-"*60 + "\n")

def load_data(subset_size_train=5000, subset_size_test=1000, random_state=42):
    # Carregar o dataset MNIST
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    # Redimensionar e normalizar os dados
    X_train_flat = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
    X_test_flat = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0

    # Visualizar algumas imagens de exemplo
    plt.figure(figsize=(10,4))
    plt.title('Visualizando algumas imagens do dataset MNIST:\n')
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(X_train[i], cmap='gray')
        plt.title(f"Label: {y_train[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Visualizar a distribuição das classes
    class_counts = Counter(y_train)
    sorted_class_counts = {k: class_counts[k] for k in sorted(class_counts)}

    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(sorted_class_counts.keys()), y=list(sorted_class_counts.values()), hue=(sorted_class_counts.keys()), palette='viridis', legend=False)
    plt.title('Distribuição das Classes no Conjunto de Treinamento MNIST')
    plt.xlabel('Dígito')
    plt.ylabel('Contagem')
    plt.show()

    # Dividir os dados em conjuntos de treinamento, validação e teste
    # Amostragem estratificada para treino e validação
    train_indices = np.arange(X_train_flat.shape[0])
    X_train_idx, _ = train_test_split(
        train_indices, train_size=subset_size_train, stratify=y_train, random_state=random_state
    )
    X_train = X_train_flat[X_train_idx]
    y_train = y_train[X_train_idx]

    # Amostragem estratificada para teste
    test_indices = np.arange(X_test_flat.shape[0])
    X_test_idx, _ = train_test_split(
        test_indices, train_size=subset_size_test, stratify=y_test, random_state=random_state
    )
    X_test = X_test_flat[X_test_idx]
    y_test = y_test[X_test_idx]

    print(f"Conjunto de Treinamento+Validação: {X_train.shape}, Conjunto de Teste: {X_test.shape}")

    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = load_data()

    # Executar kNN
    run_knn(X_train, y_train, X_test, y_test, use_pca=False)

    # Executar kNN com PCA
    run_knn(X_train, y_train, X_test, y_test, use_pca=True)
    
    # Executar classificadores lineares
    run_linear_classifiers(X_train, y_train, X_test, y_test)

    # Executar classificadores lineares com PCA
    run_linear_classifiers(X_train, y_train, X_test, y_test, use_pca=True)