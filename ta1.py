import os
from matplotlib import pyplot as plt
from matplotlib.image import imread
import cv2
import numpy as np
from sklearn.cluster import KMeans
from pprint import pprint

def carrega_imagens(diretorio):
    imagens = []
    caminhos = []
    for arquivo in os.listdir(diretorio):
        if arquivo.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            caminho_completo = os.path.join(diretorio, arquivo)
            img = cv2.imread(caminho_completo, cv2.IMREAD_GRAYSCALE)
            imagens.append(img)
            caminhos.append(caminho_completo)
    return imagens, caminhos

def imprime_imagens(imagens, caminhos, imagens_por_linha=5):
    if not imagens:
        print("Nenhuma imagem encontrada para exibir.")
        return

    total_imagens = len(imagens)
    linhas = (total_imagens + imagens_por_linha - 1) // imagens_por_linha
    fig, axs = plt.subplots(linhas, imagens_por_linha, figsize=(15, 3 * linhas))

    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]

    for idx, (img, caminho) in enumerate(zip(imagens, caminhos)):
        axs[idx].imshow(img, cmap='gray')
        axs[idx].axis('off')
        axs[idx].set_title(caminho, fontsize=12)

    for idx in range(len(imagens), len(axs)):
        axs[idx].axis('off')

    plt.tight_layout()
    plt.show()

def gera_filtros_de_gabor(thetas, ksize=31, sigma=4.0, lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F):
    kernels = []
    for theta in thetas:
        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype)
        kernels.append(kernel)
    return kernels

def gera_filtro_circular(thetas, ksize=31, sigma=4.0, lambd=10.0, gamma=0.5, psi=0, ktype=cv2.CV_32F):
    kernels = gera_filtros_de_gabor(thetas, ksize, sigma, lambd, gamma, psi, ktype)
    combined = np.sum(kernels, axis=0)
    combined /= np.max(np.abs(combined))
    return combined

def imprime_filtros_de_gabor(kernels, titles, imagens_por_linha=5):
    total = len(kernels)
    linhas = (total + imagens_por_linha - 1) // imagens_por_linha
    plt.figure(figsize=(3 * imagens_por_linha, 3 * linhas))

    for idx, kernel in enumerate(kernels):
        plt.subplot(linhas, imagens_por_linha, idx + 1)
        plt.imshow(kernel, cmap='gray')
        plt.title(titles[idx], fontsize=8)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def aplica_filtros(imagens, kernels):
    resultados = []

    for img in imagens:
        img_resultados = []

        escalas = [img]

        blur = cv2.GaussianBlur(img, (5, 5), 0)
        half = cv2.resize(blur, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        escalas.append(half)

        blur2 = cv2.GaussianBlur(half, (5, 5), 0)
        quarter = cv2.resize(blur2, (half.shape[1] // 2, half.shape[0] // 2), interpolation=cv2.INTER_AREA)
        escalas.append(quarter)

        for escala_idx, escala_img in enumerate(escalas):
            for kernel_idx, kernel in enumerate(kernels):
                filtrada = cv2.filter2D(escala_img, -1, kernel)
                img_resultados.append((escala_idx, kernel_idx, filtrada))

        resultados.append(img_resultados)
    return resultados

def imprime_imagens_filtradas(imagens_filtradas, nomes_imagens, nomes_kernels, imagens_por_linha=5):
    for img_idx, resultados_img in enumerate(imagens_filtradas):
        total = len(resultados_img)
        linhas = (total + imagens_por_linha - 1) // imagens_por_linha
        fig, axs = plt.subplots(linhas, imagens_por_linha, figsize=(15, 3 * linhas))

        axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]

        for idx, (escala_idx, kernel_idx, filtrada) in enumerate(resultados_img):
            axs[idx].imshow(filtrada, cmap='gray')
            axs[idx].axis('off')
            axs[idx].set_title(f"{nomes_imagens[img_idx]}\nEscala {escala_idx+1} - {nomes_kernels[kernel_idx]}", fontsize=6)

        for idx in range(len(resultados_img), len(axs)):
            axs[idx].axis('off')

        plt.tight_layout()
        plt.show()

def extrai_features_janelas(gabor_images, window_size=4):
    n, m = gabor_images[0].shape
    n_windows = n // window_size
    m_windows = m // window_size

    features = np.zeros((n_windows, m_windows, 10), dtype=np.float32)

    for i in range(n_windows):
        for j in range(m_windows):
            vetor = []
            x_start = i * window_size
            x_end = x_start + window_size
            y_start = j * window_size
            y_end = y_start + window_size

            for img in gabor_images:
                janela = img[x_start:x_end, y_start:y_end]
                mean_abs = np.mean(np.abs(janela))
                mean = np.mean(janela)
                mean_dev = np.mean(np.abs(janela - mean))
                vetor.extend([mean_abs, mean_dev])

            features[i, j] = vetor
    return features

def segmenta_com_kmeans(features, n_clusters=3):
    n, m, d = features.shape
    X = features.reshape(-1, d)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)

    labels_img = labels.reshape(n, m)
    return labels_img

def imprime_segmentacao(segmentacao):
    plt.figure(figsize=(8, 8))
    plt.imshow(segmentacao, cmap='nipy_spectral')
    plt.title("Segmentação KMeans")
    plt.axis('off')
    plt.show()

def extrai_features_para_kmeans(imagens_filtradas, kernel_size=(3, 3)):
    kh, kw = kernel_size
    all_features = []
    formas = []

    for resultados_img in imagens_filtradas:

        filtrados_por_kernel = dict()
        for escala_idx, kernel_idx, filtrada in resultados_img:
            if escala_idx != 0:
                continue
            filtrados_por_kernel[kernel_idx] = filtrada

        h, w = next(iter(filtrados_por_kernel.values())).shape

        features_img = []

        for i in range(0, h - kh + 1, kh):
            for j in range(0, w - kw + 1, kw):
                features_patch = []

                for kernel_idx in sorted(filtrados_por_kernel.keys()):
                    patch = filtrados_por_kernel[kernel_idx][i:i+kh, j:j+kw]
                    features_patch.append(np.mean(patch))
                    features_patch.append(np.std(patch))

                features_img.append(features_patch)

        n_patches_vertical = (h - kh) // kh + 1
        n_patches_horizontal = (w - kw) // kw + 1
        formas.append((n_patches_vertical, n_patches_horizontal))
        all_features.append(np.array(features_img))

    return all_features, formas

def segmenta_imagem(features, formas, n_clusters=4):
    segmentacoes = []
    for img_features, (h, w) in zip(features, formas):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(img_features)
        segmentacao = labels.reshape(h, w)
        segmentacoes.append(segmentacao)
    return segmentacoes

def imprime_segmentacoes(segmentacoes, caminhos, imagens_por_linha=5):
    total = len(segmentacoes)
    linhas = (total + imagens_por_linha - 1) // imagens_por_linha
    fig, axs = plt.subplots(linhas, imagens_por_linha, figsize=(15, 3 * linhas))

    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]

    for idx, (segmentacao, caminho) in enumerate(zip(segmentacoes, caminhos)):
        axs[idx].imshow(segmentacao, cmap='nipy_spectral')
        axs[idx].axis('off')
        axs[idx].set_title(f"Segmentação\n{caminho}", fontsize=8)

    for idx in range(len(segmentacoes), len(axs)):
        axs[idx].axis('off')

    plt.tight_layout()
    plt.show()

def main():
    diretorio = 'imagens'
    imagens, caminhos = carrega_imagens(diretorio)

    imprime_imagens(imagens, caminhos)

    base_thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    circular_thetas = [i * np.pi/8 for i in range(8)]
    ksize = 11
    sigma = 4
    lambd = 5
    gamma = 0.5
    psi = 0
    ktype = cv2.CV_32F

    gabor_kernels = gera_filtros_de_gabor(base_thetas, ksize, sigma, lambd, gamma, psi, ktype)
    circular_kernel = gera_filtro_circular(circular_thetas, ksize, sigma, lambd, gamma, psi, ktype)
    kernels = gabor_kernels + [circular_kernel]
    titles = ['0°', '45°', '90°', '135°', 'Circular']
    imprime_filtros_de_gabor(kernels, titles)
    imagens_filtradas = aplica_filtros(imagens, kernels)
    imprime_imagens_filtradas(imagens_filtradas, caminhos, titles)
    features, formas = extrai_features_para_kmeans(imagens_filtradas)
    segmentacoes = segmenta_imagem(features, formas, n_clusters=5)
    imprime_segmentacoes(segmentacoes, caminhos)


if __name__ == "__main__":
    main()
