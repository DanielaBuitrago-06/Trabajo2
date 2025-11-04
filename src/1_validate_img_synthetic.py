# ============================================
# PARTE 1: VALIDACIÓN CON IMÁGENES SINTÉTICAS
# ============================================
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
log_file = os.path.join(measurements_dir, '1_validate_img_synthetic_results.txt')
tee = TeeOutput(log_file)
sys.stdout = tee

# ----------------------------
# 1. Crear imagen sintética base
# ----------------------------
def create_synthetic_image(size=(400, 400)):
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
    h, w = img.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M_rot[:,2] += [tx, ty]
    transformed = cv2.warpAffine(img, M_rot, (w, h))
    # Convertir a homografía 3x3
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
    try:
        return cv2.SIFT_create(), 'SIFT'
    except:
        return cv2.ORB_create(1000), 'ORB'

def estimate_homography(img1, img2):
    detector, method = get_detector()
    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)
    if method == 'SIFT':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        if des1.dtype != np.float32:
            des1 = des1.astype(np.float32)
            des2 = des2.astype(np.float32)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = [m for m,n in matches if m.distance < 0.75*n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
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
    H_true /= H_true[2,2]
    H_est /= H_est[2,2]
    # 4 puntos de referencia
    pts = np.float32([[0,0],[399,0],[399,399],[0,399]]).reshape(-1,1,2)
    pts_true = cv2.perspectiveTransform(pts, H_true)
    pts_est = cv2.perspectiveTransform(pts, H_est)
    rmse = np.sqrt(np.mean(np.sum((pts_true - pts_est)**2, axis=(1,2))))
    # error de rotación (grados)
    R_true = H_true[:2,:2] / np.linalg.norm(H_true[0,:2])
    R_est = H_est[:2,:2] / np.linalg.norm(H_est[0,:2])
    cos_theta = np.clip(np.trace(R_true.T @ R_est)/2, -1, 1)
    ang_error = np.degrees(np.arccos(cos_theta))
    # error de escala (comparar norma columnas)
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
match_vis = cv2.drawMatches(base_img, cv2.SIFT_create().detect(base_img, None),
                            trans_img, cv2.SIFT_create().detect(trans_img, None),
                            good[:30], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite(os.path.join(output_dir, 'matches.jpg'), match_vis)
plt.imshow(cv2.cvtColor(match_vis, cv2.COLOR_BGR2RGB)); plt.axis('off')
plt.title("Emparejamientos detectados"); plt.show()

# ----------------------------
# 6. Experimento: evaluar parámetros
# ----------------------------
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
