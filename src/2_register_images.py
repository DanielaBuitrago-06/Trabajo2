# ============================================
# REGISTRO DE IMÁGENES DEL COMEDOR (PIPELINE)
# ============================================
"""
Módulo de registro y fusión de imágenes para crear panoramas.

Este módulo implementa un pipeline completo para registrar múltiples imágenes
del mismo escenario (tomadas desde diferentes posiciones) y fusionarlas en una
vista panorámica coherente. El proceso incluye:

1. Proyección cilíndrica: Corrige distorsiones causadas por rotaciones de cámara
2. Detección y matching: Identifica correspondencias entre imágenes usando SIFT/ORB
3. Estimación de homografía: Calcula la transformación geométrica con RANSAC
4. Warping y composición: Fusiona las imágenes en un panorama único

El módulo está diseñado específicamente para registrar imágenes del comedor,
pero puede adaptarse a otros escenarios planos o aproximadamente planos.

Autor: Trabajo de Visión por Computador II
Fecha: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

plt.rcParams['figure.figsize'] = (12,8)

# Configurar carpetas de salida
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
figures_dir = os.path.join(project_root, 'results', 'figures')
os.makedirs(figures_dir, exist_ok=True)

measurements_dir = os.path.join(project_root, 'results', 'measurements')
os.makedirs(measurements_dir, exist_ok=True)

# Clase para capturar y guardar todos los prints
class TeeOutput:
    """
    Clase para duplicar la salida estándar tanto en consola como en archivo.
    
    Esta clase implementa un sistema de logging que captura todos los prints
    y los guarda simultáneamente en un archivo de texto, facilitando el
    análisis posterior y la reproducibilidad de resultados.
    
    Attributes:
        file: Archivo donde se guarda la salida
        stdout: Referencia al stdout original del sistema
    
    Example:
        >>> tee = TeeOutput('output.txt')
        >>> sys.stdout = tee
        >>> print("Esto se guardará en archivo y consola")
        >>> tee.close()
    """
    
    def __init__(self, file_path):
        """
        Inicializa el objeto TeeOutput.
        
        Args:
            file_path (str): Ruta del archivo donde se guardará la salida
        """
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.write(f"=== EJECUCIÓN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def write(self, text):
        """
        Escribe texto tanto en consola como en archivo.
        
        Args:
            text (str): Texto a escribir
        """
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        """Fuerza la escritura de buffers tanto en consola como en archivo."""
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
        """Cierra el archivo y restaura el stdout original."""
        self.file.close()
        sys.stdout = self.stdout

# Configurar archivo de salida (se sobrescribirá en cada ejecución)
log_file = os.path.join(measurements_dir, '2_register_images_results.txt')
tee = TeeOutput(log_file)
sys.stdout = tee

# -----------------------------
# Detector genérico (SIFT -> ORB fallback)
# -----------------------------
def create_detector(name='SIFT'):
    """
    Crea un detector de características con fallback automático.
    
    Intenta crear el detector solicitado y si no está disponible,
    usa ORB como alternativa. Esto asegura portabilidad del código.
    
    Args:
        name (str, optional): Nombre del detector deseado ('SIFT', 'ORB', 'AKAZE').
                              Por defecto 'SIFT'.
    
    Returns:
        tuple: Contiene:
            - detector: Objeto detector de OpenCV
            - method (str): Nombre del método usado ('SIFT', 'ORB' o 'AKAZE')
    
    Note:
        - SIFT es más robusto pero propietario (puede no estar disponible)
        - ORB es libre y eficiente, buena alternativa
        - AKAZE es robusto y libre, también disponible
    
    Example:
        >>> detector, method = create_detector('SIFT')
        >>> print(f"Usando detector: {method}")
    """
    name = name.upper()
    if name == 'SIFT':
        try:
            return cv2.SIFT_create(), 'SIFT'
        except:
            print("SIFT no disponible. Usando ORB.")
            return cv2.ORB_create(5000), 'ORB'
    elif name == 'AKAZE':
        return cv2.AKAZE_create(), 'AKAZE'
    else:
        return cv2.ORB_create(5000), 'ORB'

# -----------------------------
# Proyección cilíndrica (para evitar distorsión en panoramas)
# -----------------------------
def cylindrical_warp(img, f=900):
    """
    Aplica proyección cilíndrica a una imagen para reducir distorsiones.
    
    La proyección cilíndrica es esencial cuando se registran imágenes tomadas
    con rotaciones de cámara. Proyecta la imagen sobre un cilindro, lo que
    reduce las distorsiones angulares y mejora la calidad del panorama final.
    
    El proceso transforma coordenadas de píxeles (x, y) a coordenadas
    cilíndricas, luego proyecta de vuelta a coordenadas de imagen.
    
    Args:
        img (numpy.ndarray): Imagen de entrada en formato BGR
        f (float, optional): Longitud focal estimada en píxeles. Por defecto 900.
                            Este valor debe aproximarse a la longitud focal real
                            de la cámara para mejores resultados.
    
    Returns:
        numpy.ndarray: Imagen proyectada cilíndricamente en formato BGR
    
    Note:
        - La longitud focal f es un parámetro crítico que afecta la cantidad
          de distorsión corregida
        - Valores típicos: 600-1200 píxeles para cámaras comunes
        - La proyección asume que las imágenes fueron tomadas rotando la
          cámara alrededor de su centro óptico
    
    Reference:
        Basado en el modelo de proyección cilíndrica estándar donde:
        - x_cyl = f * sin(x/f)
        - y_cyl = y
        - z_cyl = f * cos(x/f)
    
    Example:
        >>> img = cv2.imread('image.jpg')
        >>> img_warped = cylindrical_warp(img, f=900)
    """
    h, w = img.shape[:2]
    # Matriz de calibración simplificada (asumiendo sin distorsión)
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0, 1]])
    
    # Crear grid de coordenadas de píxeles
    y_i, x_i = np.indices((h, w))
    # Convertir a coordenadas centradas en la imagen
    X = np.stack([x_i - w/2, y_i - h/2, np.ones_like(x_i)], axis=-1)
    X = X.reshape((-1,3))
    
    # Proyectar a coordenadas cilíndricas
    # x_cyl = sin(x/f), y_cyl = y/f, z_cyl = cos(x/f)
    Xc = np.stack([np.sin(X[:,0]/f), X[:,1]/f, np.cos(X[:,0]/f)], axis=-1)
    
    # Proyectar de vuelta a coordenadas de imagen
    x_p = (Xc[:,0]*f / Xc[:,2]) + w/2
    y_p = (Xc[:,1]*f / Xc[:,2]) + h/2
    
    # Crear mapas de remapeo
    mapx = x_p.reshape(h,w).astype(np.float32)
    mapy = y_p.reshape(h,w).astype(np.float32)
    
    # Aplicar remapeo con interpolación lineal
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# -----------------------------
# Detección y matching robusto
# -----------------------------
def detect_and_match(img1, img2, detector_name='SIFT', ratio=0.75):
    """
    Detecta características y encuentra correspondencias entre dos imágenes.
    
    Esta función implementa el proceso completo de detección y matching:
    1. Detecta keypoints y calcula descriptores en ambas imágenes
    2. Empareja descriptores usando el matcher apropiado
    3. Aplica ratio test de Lowe para filtrar matches ambiguos
    4. Retorna puntos correspondientes listos para estimación de homografía
    
    Args:
        img1 (numpy.ndarray): Primera imagen (imagen de referencia)
        img2 (numpy.ndarray): Segunda imagen (imagen a registrar)
        detector_name (str, optional): Nombre del detector ('SIFT', 'ORB', 'AKAZE').
                                       Por defecto 'SIFT'.
        ratio (float, optional): Umbral para ratio test de Lowe. Por defecto 0.75.
                                 Un match se acepta si: distance1 < ratio * distance2
    
    Returns:
        tuple: Contiene cinco elementos:
            - kp1 (list): Keypoints detectados en img1
            - kp2 (list): Keypoints detectados en img2
            - pts1 (numpy.ndarray): Coordenadas de puntos en img1 (Nx2)
            - pts2 (numpy.ndarray): Coordenadas de puntos en img2 (Nx2)
            - good (list): Lista de matches válidos después del ratio test
    
    Note:
        - Si no se detectan descriptores, retorna listas vacías
        - Ratio test: mantiene matches donde el mejor match es significativamente
          mejor que el segundo mejor (reduce falsos positivos)
        - FLANN se usa para SIFT (descriptores float), BFMatcher para ORB/AKAZE
    
    Example:
        >>> img1 = cv2.imread('img1.jpg')
        >>> img2 = cv2.imread('img2.jpg')
        >>> kp1, kp2, pts1, pts2, matches = detect_and_match(img1, img2)
        >>> print(f"Matches encontrados: {len(matches)}")
    """
    detector, method = create_detector(detector_name)
    
    # Detectar keypoints y calcular descriptores
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    
    # Validar que se detectaron descriptores
    if des1 is None or des2 is None:
        return [], [], [], [], []
    
    # Configurar matcher según el método
    if method == 'SIFT':
        # FLANN es más eficiente para descriptores float (SIFT)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # Asegurar tipo float32 para FLANN
        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
    else:
        # Brute Force matcher para descriptores binarios (ORB, AKAZE)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    # Matching con k=2 para ratio test
    knn = matcher.knnMatch(des1, des2, k=2)
    
    # Ratio test de Lowe: filtrar matches ambiguos
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    
    # Extraer coordenadas de puntos correspondientes
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    
    return kp1, kp2, pts1, pts2, good

# -----------------------------
# Estimar homografía robusta (RANSAC)
# -----------------------------
def estimate_homography(pts1, pts2):
    """
    Estima la homografía entre dos conjuntos de puntos correspondientes.
    
    Utiliza RANSAC (Random Sample Consensus) para estimar la homografía de
    forma robusta, filtrando automáticamente los outliers (correspondencias
    erróneas) que podrían degradar la estimación.
    
    Args:
        pts1 (numpy.ndarray): Puntos en la primera imagen (Nx2)
        pts2 (numpy.ndarray): Puntos correspondientes en la segunda imagen (Nx2)
    
    Returns:
        tuple: Contiene dos elementos:
            - H (numpy.ndarray o None): Matriz de homografía 3x3 que transforma
                                        pts2 a pts1. None si no hay suficientes puntos.
            - mask (numpy.ndarray o None): Máscara booleana indicando qué puntos
                                           son inliers (True) y cuáles outliers (False).
                                           None si no se pudo estimar.
    
    Note:
        - Se requieren mínimo 4 puntos correspondientes para estimar homografía
        - RANSAC con umbral de 5.0 píxeles: puntos con error < 5px son inliers
        - La homografía transforma: pts1 = H @ pts2 (en coordenadas homogéneas)
        - La máscara indica la calidad del matching (más inliers = mejor)
    
    Example:
        >>> pts1 = np.array([[100, 200], [150, 250], ...])
        >>> pts2 = np.array([[110, 210], [160, 260], ...])
        >>> H, mask = estimate_homography(pts1, pts2)
        >>> num_inliers = np.sum(mask) if mask is not None else 0
    """
    # Validar número mínimo de puntos
    if len(pts1) < 4:
        return None, None
    
    # Estimar homografía con RANSAC
    # Transforma pts2 -> pts1
    # Umbral de 5.0 píxeles: puntos con error reproyección < 5px son inliers
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    
    return H, mask

# -----------------------------
# Warping y composición al marco de referencia (imagen 0)
# -----------------------------
def warp_and_merge(ref, img, H):
    """
    Registra una imagen respecto a una imagen de referencia y las fusiona.
    
    Esta función realiza el registro completo: transforma la imagen usando
    la homografía, calcula el tamaño necesario del panorama para contener
    ambas imágenes, y fusiona la imagen transformada con la referencia.
    
    El proceso:
    1. Calcula las esquinas de ambas imágenes después de la transformación
    2. Determina el tamaño del canvas necesario para contener todo
    3. Aplica una traslación para que todas las coordenadas sean positivas
    4. Aplica warping y fusiona las imágenes
    
    Args:
        ref (numpy.ndarray): Imagen de referencia (la que se mantiene fija)
        img (numpy.ndarray): Imagen a registrar y fusionar
        H (numpy.ndarray): Matriz de homografía 3x3 que transforma img a ref
    
    Returns:
        numpy.ndarray: Panorama fusionado que contiene ambas imágenes
    
    Note:
        - La imagen de referencia se coloca directamente sin transformar
        - La imagen transformada se fusiona encima (puede sobrescribir áreas)
        - El panorama se dimensiona automáticamente para contener ambas imágenes
        - Esta es una fusión simple (no usa blending avanzado)
    
    Example:
        >>> ref = cv2.imread('img1.jpg')
        >>> img = cv2.imread('img2.jpg')
        >>> H = estimate_homography(pts1, pts2)[0]
        >>> panorama = warp_and_merge(ref, img, H)
    """
    h1, w1 = ref.shape[:2]
    h2, w2 = img.shape[:2]
    
    # Calcular esquinas de la imagen transformada
    corners = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    
    # Combinar esquinas de ambas imágenes para determinar tamaño del panorama
    all_corners = np.concatenate(
        (np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2), warped_corners),
        axis=0)
    
    # Calcular dimensiones del panorama y traslación necesaria
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    trans = [-xmin, -ymin]
    
    # Matriz de traslación para mover todo al primer cuadrante
    Ht = np.array([[1,0,trans[0]],[0,1,trans[1]],[0,0,1]])
    
    # Aplicar transformación completa (homografía + traslación)
    panorama = cv2.warpPerspective(img, Ht.dot(H), (xmax-xmin, ymax-ymin))
    
    # Fusionar: colocar imagen de referencia en su posición
    panorama[trans[1]:trans[1]+h1, trans[0]:trans[0]+w1] = ref
    
    return panorama

# -----------------------------
# Multi-band blending
# -----------------------------
def multiband_blend(img1, img2, mask, levels=5):
    """
    Aplica blending multibanda para fusionar dos imágenes suavemente.
    
    El blending multibanda usa pirámides laplacianas para fusionar imágenes
    de forma que se eviten discontinuidades visibles en las zonas de transición.
    Este método es superior a fusiones simples cuando hay variaciones de
    iluminación o color entre las imágenes.
    
    Args:
        img1 (numpy.ndarray): Primera imagen a fusionar
        img2 (numpy.ndarray): Segunda imagen a fusionar
        mask (numpy.ndarray): Máscara que define la región de transición
                             (valores entre 0 y 1, donde 1 = img1, 0 = img2)
        levels (int, optional): Número de niveles en la pirámide laplaciana.
                                Por defecto 5.
    
    Returns:
        numpy.ndarray: Imagen fusionada con blending multibanda
    
    Note:
        - Esta función no se usa actualmente en el pipeline principal
        - Requiere que las imágenes estén alineadas previamente
        - La máscara debe tener el mismo tamaño que las imágenes
    
    Reference:
        Basado en Burt & Adelson (1983): "The Laplacian Pyramid as a Compact
        Image Code"
    
    Example:
        >>> blended = multiband_blend(img1, img2, mask, levels=5)
    """
    def pyramid(img, down=True):
        imgs = [img.astype(np.float32)]
        for _ in range(levels):
            img = cv2.pyrDown(img) if down else cv2.pyrUp(img)
            imgs.append(img.astype(np.float32))
        return imgs
    gp1, gp2, gpm = pyramid(img1), pyramid(img2), pyramid(mask)
    lp1, lp2 = [], []
    for i in range(levels):
        up1 = cv2.pyrUp(gp1[i+1])
        up2 = cv2.pyrUp(gp2[i+1])
        up1 = cv2.resize(up1, (gp1[i].shape[1], gp1[i].shape[0]))
        up2 = cv2.resize(up2, (gp2[i].shape[1], gp2[i].shape[0]))
        lp1.append(gp1[i] - up1)
        lp2.append(gp2[i] - up2)
    LS = []
    for l1, l2, gm in zip(lp1, lp2, gpm[:-1]):
        gm = gm[...,np.newaxis]
        LS.append(l1 * gm + l2 * (1 - gm))
    res = gp1[-1]*gpm[-1][...,np.newaxis] + gp2[-1]*(1-gpm[-1][...,np.newaxis])
    for i in range(levels-1, -1, -1):
        res = cv2.pyrUp(res)
        res = cv2.resize(res, (LS[i].shape[1], LS[i].shape[0]))
        res += LS[i]
    return np.clip(res, 0, 255).astype(np.uint8)

# -----------------------------
# Pipeline: registro respecto a la primera imagen
# -----------------------------
def stitch_to_reference(img_files, detector_name='SIFT', focal_length=900):
    """
    Pipeline principal para registrar múltiples imágenes en un panorama.
    
    Esta función implementa el pipeline completo de registro:
    1. Carga todas las imágenes
    2. Aplica proyección cilíndrica a cada imagen
    3. Registra cada imagen sucesivamente respecto a la primera (referencia)
    4. Fusiona todas las imágenes en un panorama único
    
    El registro es incremental: cada imagen se registra respecto a la imagen
    de referencia (primera imagen), no respecto a la imagen anterior. Esto
    evita acumulación de errores.
    
    Args:
        img_files (list): Lista de rutas a las imágenes a registrar
        detector_name (str, optional): Nombre del detector a usar ('SIFT', 'ORB', 'AKAZE').
                                       Por defecto 'SIFT'.
        focal_length (float, optional): Longitud focal estimada en píxeles para
                                        proyección cilíndrica. Por defecto 900.
    
    Returns:
        numpy.ndarray: Panorama fusionado que contiene todas las imágenes registradas
    
    Raises:
        FileNotFoundError: Si alguna imagen no se puede cargar
    
    Note:
        - La primera imagen actúa como referencia (no se transforma)
        - Si una homografía falla, se omite esa imagen pero continúa con las demás
        - El panorama se expande automáticamente para contener todas las imágenes
    
    Example:
        >>> files = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        >>> panorama = stitch_to_reference(files, detector_name='SIFT', focal_length=900)
        >>> cv2.imwrite('panorama.jpg', panorama)
    """
    # Cargar todas las imágenes
    imgs = [cv2.imread(f) for f in img_files]
    for i,im in enumerate(imgs):
        if im is None:
            raise FileNotFoundError(f"No se encontró {img_files[i]}")
    
    # Aplicar proyección cilíndrica a todas las imágenes
    # Esto corrige distorsiones causadas por rotaciones de cámara
    imgs = [cylindrical_warp(im, f=focal_length) for im in imgs]
    
    # La primera imagen es la referencia (no se transforma)
    ref = imgs[0]
    panorama = ref.copy()
    
    # Registrar cada imagen sucesivamente respecto a la referencia
    for i in range(1, len(imgs)):
        # Detectar y emparejar características
        kp1,kp2,pts1,pts2,good = detect_and_match(ref, imgs[i], detector_name)
        
        # Estimar homografía
        H, mask = estimate_homography(pts1, pts2)
        
        if H is None:
            print(f"Homografía {i} falló.")
            continue
        
        # Mostrar estadísticas de matching
        print(f"Imagen {i}: {len(good)} matches, {int(mask.sum()) if mask is not None else 0} inliers")
        
        # Fusionar imagen en el panorama
        panorama = warp_and_merge(ref, imgs[i], H)
    
    return panorama

# -----------------------------
# Visualización
# -----------------------------
def show(img, title=""):
    """
    Muestra una imagen usando matplotlib.
    
    Convierte la imagen de BGR (formato OpenCV) a RGB (formato matplotlib)
    y la muestra con un título opcional.
    
    Args:
        img (numpy.ndarray): Imagen en formato BGR
        title (str, optional): Título para la imagen. Por defecto "".
    """
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# -----------------------------
# EJECUCIÓN (Colab)
# -----------------------------
# 1) Sube tus imágenes: img1.jpeg, img2.jpeg, img3.jpg
# 2) Ejecuta:
# from google.colab import files
# files.upload()
# 3) Cambia los nombres si es necesario

if __name__ == "__main__":
    files = ['data/original/img1.jpg', 'data/original/img2.jpg', 'data/original/img3.jpg']
    print("Iniciando registro de imágenes...")
    print(f"Imágenes a procesar: {len(files)}")
    for i, f in enumerate(files):
        print(f"  - {f}")
    
    panorama = stitch_to_reference(files, detector_name='SIFT', focal_length=900)
    
    # Guardar panorama
    panorama_path = os.path.join(figures_dir, 'panorama.jpg')
    cv2.imwrite(panorama_path, panorama)
    print(f"Dimensiones del panorama: {panorama.shape[1]}x{panorama.shape[0]} píxeles")
    
    show(panorama, "Panorama del comedor")
    
    # Cerrar el archivo de salida
    tee.close()
