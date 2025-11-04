# ============================================
# PARTE 1: VALIDACIÓN CON IMÁGENES SINTÉTICAS
# ============================================
"""
Módulo de validación con imágenes sintéticas para el registro de imágenes.

Este módulo implementa un pipeline de validación que genera imágenes sintéticas
con transformaciones geométricas conocidas (rotación, escala, traslación) y
evalúa la precisión del algoritmo de estimación de homografía mediante
comparación entre las transformaciones verdaderas y las estimadas.

El módulo realiza:
1. Generación de imágenes sintéticas base con formas geométricas
2. Aplicación de transformaciones conocidas (rotación, escala, traslación)
3. Detección de puntos de interés y estimación de homografía
4. Comparación cuantitativa mediante métricas (RMSE, error angular, error de escala)
5. Experimentación con diferentes combinaciones de parámetros

Autor: Trabajo de Visión por Computador II
Fecha: 2025
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
import os
import sys
from datetime import datetime

plt.rcParams['figure.figsize'] = (10,6)

# Configurar carpetas de salida
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
output_dir = os.path.join(project_root, 'data', 'synthetic')
os.makedirs(output_dir, exist_ok=True)

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
log_file = os.path.join(measurements_dir, '1_validate_img_synthetic_results.txt')
tee = TeeOutput(log_file)
sys.stdout = tee

# ----------------------------
# 1. Crear imagen sintética base
# ----------------------------
def create_synthetic_image(size=(400, 400)):
    """
    Crea una imagen sintética con formas geométricas para validación.
    
    Genera una imagen de prueba con múltiples elementos geométricos (rectángulo,
    círculo, línea y texto) que proporcionan características distintivas para
    la detección de puntos de interés y el matching entre imágenes.
    
    Args:
        size (tuple, optional): Tamaño de la imagen (alto, ancho) en píxeles.
                                Por defecto (400, 400).
    
    Returns:
        numpy.ndarray: Imagen sintética en formato BGR (uint8) con fondo blanco
                       y elementos geométricos en color.
    
    Example:
        >>> img = create_synthetic_image(size=(400, 400))
        >>> print(img.shape)  # (400, 400, 3)
    """
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (350, 350), (0, 0, 0), 4)
    cv2.circle(img, (200, 200), 60, (0, 0, 255), -1)
    cv2.line(img, (50, 200), (350, 200), (255, 0, 0), 3)
    cv2.putText(img, "TEST", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,0), 2)
    return img

base_img = create_synthetic_image()
cv2.imwrite(os.path.join(output_dir, 'base_img.jpg'), base_img)
plt.imshow(cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)); plt.title("Imagen Base"); plt.axis('off')
plt.show()

# ----------------------------
# 2. Aplicar transformaciones conocidas
# ----------------------------
def apply_known_transform(img, angle=15, scale=1.1, tx=30, ty=20):
    """
    Aplica una transformación afín conocida a una imagen.
    
    Realiza una transformación geométrica compuesta de rotación, escalado y
    traslación. Esta función es crucial para la validación porque conocemos
    la transformación verdadera aplicada, permitiendo comparar con la
    estimada por el algoritmo.
    
    Args:
        img (numpy.ndarray): Imagen de entrada en formato BGR
        angle (float, optional): Ángulo de rotación en grados. Por defecto 15.
        scale (float, optional): Factor de escala (1.0 = sin escalado).
                                 Por defecto 1.1.
        tx (float, optional): Traslación en dirección X en píxeles.
                              Por defecto 30.
        ty (float, optional): Traslación en dirección Y en píxeles.
                              Por defecto 20.
    
    Returns:
        tuple: Contiene dos elementos:
            - transformed (numpy.ndarray): Imagen transformada
            - H_true (numpy.ndarray): Matriz de homografía 3x3 verdadera
                                      que representa la transformación aplicada
    
    Note:
        La transformación se aplica alrededor del centro de la imagen.
        La homografía se convierte desde matriz afín 2x3 a formato 3x3.
    
    Example:
        >>> img = create_synthetic_image()
        >>> transformed, H_true = apply_known_transform(img, angle=20, scale=1.2)
        >>> print(H_true.shape)  # (3, 3)
    """
    h, w = img.shape[:2]
    # Obtener matriz de rotación y escalado alrededor del centro
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    # Agregar traslación
    M_rot[:,2] += [tx, ty]
    # Aplicar transformación afín
    transformed = cv2.warpAffine(img, M_rot, (w, h))
    # Convertir matriz afín 2x3 a homografía 3x3 (agregando fila [0,0,1])
    H_true = np.vstack([M_rot, [0,0,1]])
    return transformed, H_true

angle, scale, tx, ty = 20, 1.2, 40, -15
trans_img, H_true = apply_known_transform(base_img, angle, scale, tx, ty)
cv2.imwrite(os.path.join(output_dir, 'trans_img.jpg'), trans_img)
plt.imshow(cv2.cvtColor(trans_img, cv2.COLOR_BGR2RGB)); plt.title("Imagen Transformada"); plt.axis('off')
plt.show()

print("Homografía Verdadera (H_true):")
print(H_true)

# ----------------------------
# 3. Detección de puntos y estimación de homografía
# ----------------------------
def get_detector():
    """
    Obtiene un detector de características con fallback automático.
    
    Intenta usar SIFT (más robusto) y si no está disponible, usa ORB como
    alternativa. Esto asegura que el código funcione en diferentes entornos.
    
    Returns:
        tuple: Contiene:
            - detector: Objeto detector de OpenCV (SIFT o ORB)
            - method (str): Nombre del método usado ('SIFT' o 'ORB')
    
    Note:
        SIFT es propietario y puede no estar disponible en algunas versiones
        de OpenCV. ORB es una alternativa libre y eficiente.
    """
    try:
        return cv2.SIFT_create(), 'SIFT'
    except:
        return cv2.ORB_create(1000), 'ORB'

def estimate_homography(img1, img2):
    """
    Estima la homografía entre dos imágenes mediante detección de características.
    
    Este es el núcleo del algoritmo de registro. Detecta puntos de interés
    en ambas imágenes, los empareja mediante descriptores y estima la matriz
    de homografía que relaciona las coordenadas de una imagen con la otra.
    
    El proceso incluye:
    1. Detección de keypoints y cálculo de descriptores
    2. Matching de descriptores usando ratio test de Lowe
    3. Filtrado de outliers con RANSAC
    4. Estimación de homografía a partir de correspondencias inliers
    
    Args:
        img1 (numpy.ndarray): Primera imagen (imagen base)
        img2 (numpy.ndarray): Segunda imagen (imagen transformada)
    
    Returns:
        tuple: Contiene cinco elementos:
            - H_est (numpy.ndarray): Matriz de homografía estimada 3x3
            - pts1 (numpy.ndarray): Puntos correspondientes en img1 (Nx2)
            - pts2 (numpy.ndarray): Puntos correspondientes en img2 (Nx2)
            - good (list): Lista de matches válidos después del ratio test
            - method (str): Método de detección usado ('SIFT' o 'ORB')
    
    Note:
        - Ratio test: mantiene matches donde la distancia del mejor match
          es menor al 75% de la distancia del segundo mejor match
        - RANSAC: filtra outliers con umbral de 5.0 píxeles
        - La homografía transforma puntos de img2 a img1: pts1 = H @ pts2
    
    Example:
        >>> img1 = create_synthetic_image()
        >>> img2, H_true = apply_known_transform(img1, angle=20)
        >>> H_est, pts1, pts2, matches, method = estimate_homography(img1, img2)
        >>> print(f"Matches encontrados: {len(matches)}")
    """
    detector, method = get_detector()
    # Detectar keypoints y calcular descriptores
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    
    # Configurar matcher según el método usado
    if method == 'SIFT':
        # FLANN es más eficiente para SIFT (descriptores float32)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        # Asegurar que los descriptores sean float32 para FLANN
        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
    else:
        # Brute Force matcher para ORB (descriptores binarios)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Matching con k=2 para ratio test
    matches = matcher.knnMatch(des1, des2, k=2)
    
    # Ratio test de Lowe: mantener matches donde la distancia del mejor
    # es significativamente menor que la del segundo mejor
    good = [m for m,n in matches if m.distance < 0.75*n.distance]
    
    # Extraer coordenadas de los puntos correspondientes
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    
    # Estimar homografía con RANSAC para filtrar outliers
    # Transforma pts2 -> pts1
    H_est, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    
    return H_est, pts1, pts2, good, method

H_est, pts1, pts2, good, method = estimate_homography(base_img, trans_img)
print(f"Método: {method}")
print("Homografía estimada (H_est):")
print(H_est)

# ----------------------------
# 4. Comparar transformaciones: métricas
# ----------------------------
def compare_homographies(H_true, H_est):
    """
    Compara dos homografías y calcula métricas de error.
    
    Evalúa la precisión de la homografía estimada comparándola con la
    homografía verdadera mediante tres métricas:
    1. RMSE: Error cuadrático medio en la transformación de puntos de esquina
    2. Error angular: Diferencia en la rotación estimada
    3. Error de escala: Diferencia porcentual en el factor de escala
    
    Args:
        H_true (numpy.ndarray): Matriz de homografía verdadera 3x3
        H_est (numpy.ndarray): Matriz de homografía estimada 3x3
    
    Returns:
        tuple: Contiene tres métricas de error:
            - rmse (float): Error cuadrático medio en píxeles, calculado
                            transformando las esquinas de la imagen
            - ang_error (float): Error angular en grados entre las rotaciones
            - scale_error (float): Error de escala en porcentaje
    
    Note:
        - Las homografías se normalizan dividiendo por el elemento [2,2]
        - El RMSE se calcula sobre 4 puntos de esquina de una imagen 400x400
        - El error angular usa la traza de la matriz de rotación normalizada
        - El error de escala compara la norma de la primera columna
    
    Example:
        >>> H_true = np.eye(3)
        >>> H_est = apply_known_transform(img, angle=10)[1]
        >>> rmse, ang_err, scale_err = compare_homographies(H_true, H_est)
        >>> print(f"RMSE: {rmse:.2f} px, Error angular: {ang_err:.2f}°")
    """
    # Normalizar homografías (dividir por elemento [2,2])
    H_true /= H_true[2,2]
    H_est /= H_est[2,2]
    
    # 4 puntos de referencia (esquinas de imagen 400x400)
    pts = np.float32([[0,0],[399,0],[399,399],[0,399]]).reshape(-1,1,2)
    
    # Transformar puntos con ambas homografías
    pts_true = cv2.perspectiveTransform(pts, H_true)
    pts_est = cv2.perspectiveTransform(pts, H_est)
    
    # Calcular RMSE: raíz del error cuadrático medio
    rmse = np.sqrt(np.mean(np.sum((pts_true - pts_est)**2, axis=(1,2))))
    
    # Error de rotación: comparar matrices de rotación extraídas
    # Normalizar submatriz de rotación (primeras 2 filas y columnas)
    R_true = H_true[:2,:2] / np.linalg.norm(H_true[0,:2])
    R_est = H_est[:2,:2] / np.linalg.norm(H_est[0,:2])
    # Calcular ángulo mediante traza: tr(R^T @ R') / 2 = cos(theta)
    cos_theta = np.clip(np.trace(R_true.T @ R_est)/2, -1, 1)
    ang_error = np.degrees(np.arccos(cos_theta))
    
    # Error de escala: comparar norma de la primera columna (vector de escala)
    s_true = np.linalg.norm(H_true[0:2,0])
    s_est = np.linalg.norm(H_est[0:2,0])
    scale_error = abs(s_est - s_true)/s_true * 100
    
    return rmse, ang_error, scale_error

rmse, ang_err, scale_err = compare_homographies(H_true, H_est)
print(f"RMSE de puntos: {rmse:.4f} px")
print(f"Error angular: {ang_err:.4f}°")
print(f"Error de escala: {scale_err:.3f}%")

# ----------------------------
# 5. Visualizar coincidencias
# ----------------------------
# Esta sección crea una visualización de los matches encontrados entre
# las dos imágenes para validar visualmente la calidad del emparejamiento.
match_vis = cv2.drawMatches(base_img, cv2.SIFT_create().detect(base_img, None),
                            trans_img, cv2.SIFT_create().detect(trans_img, None),
                            good[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(os.path.join(output_dir, 'matches.jpg'), match_vis)
plt.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.title("Emparejamientos detectados"); plt.show()

# ----------------------------
# 6. Experimento: evaluar parámetros
# ----------------------------
# Esta sección realiza un barrido sistemático de diferentes combinaciones
# de parámetros de transformación (ángulo y escala) para evaluar cómo
# afectan la precisión del algoritmo. Esto ayuda a identificar los límites
# del método y optimizar los parámetros para mejor rendimiento.
angles = [5, 15, 30, 45]
scales = [1.0, 1.1, 1.3]
results = []

for a in angles:
    for s in scales:
        img_t, Ht = apply_known_transform(base_img, angle=a, scale=s, tx=30, ty=20)
        cv2.imwrite(os.path.join(output_dir, f'transformed_a{a}_s{s:.1f}.jpg'), img_t)
        Hest,_,_,_,_ = estimate_homography(base_img, img_t)
        rmse, ang_err, scale_err = compare_homographies(Ht, Hest)
        results.append((a,s,rmse,ang_err,scale_err))

print("\nVariación de parámetros:")
print("Angulo | Escala | RMSE(px) | Error Rot(°) | Error Escala(%)")
for r in results:
    print(f"{r[0]:>6.1f} | {r[1]:>6.2f} | {r[2]:>9.3f} | {r[3]:>12.3f} | {r[4]:>14.3f}")

# Cerrar el archivo de salida
tee.close()
