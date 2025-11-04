# ============================================
# REGISTRO DE IMÁGENES DEL COMEDOR (PIPELINE)
# ============================================

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
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        self.write(f"=== EJECUCIÓN: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
    
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()
    
    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def close(self):
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
    h, w = img.shape[:2]
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0, 1]])
    y_i, x_i = np.indices((h, w))
    X = np.stack([x_i - w/2, y_i - h/2, np.ones_like(x_i)], axis=-1)
    X = X.reshape((-1,3))
    # coordenadas proyectadas
    Xc = np.stack([np.sin(X[:,0]/f), X[:,1]/f, np.cos(X[:,0]/f)], axis=-1)
    x_p = (Xc[:,0]*f / Xc[:,2]) + w/2
    y_p = (Xc[:,1]*f / Xc[:,2]) + h/2
    mapx = x_p.reshape(h,w).astype(np.float32)
    mapy = y_p.reshape(h,w).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# -----------------------------
# Detección y matching robusto
# -----------------------------
def detect_and_match(img1, img2, detector_name='SIFT', ratio=0.75):
    detector, method = create_detector(detector_name)
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return [], [], [], [], []
    if method == 'SIFT':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            good.append(m)
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return kp1, kp2, pts1, pts2, good

# -----------------------------
# Estimar homografía robusta (RANSAC)
# -----------------------------
def estimate_homography(pts1, pts2):
    if len(pts1) < 4:
        return None, None
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    return H, mask

# -----------------------------
# Warping y composición al marco de referencia (imagen 0)
# -----------------------------
def warp_and_merge(ref, img, H):
    h1, w1 = ref.shape[:2]
    h2, w2 = img.shape[:2]
    corners = np.float32([[0,0],[w2,0],[w2,h2],[0,h2]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, H)
    all_corners = np.concatenate(
        (np.float32([[0,0],[w1,0],[w1,h1],[0,h1]]).reshape(-1,1,2), warped_corners),
        axis=0)
    [xmin, ymin] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
    trans = [-xmin, -ymin]
    Ht = np.array([[1,0,trans[0]],[0,1,trans[1]],[0,0,1]])
    panorama = cv2.warpPerspective(img, Ht.dot(H), (xmax-xmin, ymax-ymin))
    panorama[trans[1]:trans[1]+h1, trans[0]:trans[0]+w1] = ref
    return panorama

# -----------------------------
# Multi-band blending
# -----------------------------
def multiband_blend(img1, img2, mask, levels=5):
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
    imgs = [cv2.imread(f) for f in img_files]
    for i,im in enumerate(imgs):
        if im is None:
            raise FileNotFoundError(f"No se encontró {img_files[i]}")
    # proyección cilíndrica
    imgs = [cylindrical_warp(im, f=focal_length) for im in imgs]
    ref = imgs[0]
    panorama = ref.copy()
    for i in range(1, len(imgs)):
        kp1,kp2,pts1,pts2,good = detect_and_match(ref, imgs[i], detector_name)
        H, mask = estimate_homography(pts1, pts2)
        if H is None:
            print(f"Homografía {i} falló.")
            continue
        print(f"Imagen {i}: {len(good)} matches, {int(mask.sum()) if mask is not None else 0} inliers")
        panorama = warp_and_merge(ref, imgs[i], H)
    return panorama

# -----------------------------
# Visualización
# -----------------------------
def show(img, title=""):
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
