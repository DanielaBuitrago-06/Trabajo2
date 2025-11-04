"""
Tests para 1_validate_img_synthetic.py
"""
import sys
import os
import pytest
import numpy as np
import cv2

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Importar funciones del módulo (necesitamos extraer las funciones)
# Como las funciones están en el script principal, las importaremos directamente
# o las definiremos aquí para testing

def create_synthetic_image(size=(400, 400)):
    """Crear imagen sintética base"""
    img = np.ones((size[0], size[1], 3), dtype=np.uint8) * 255
    cv2.rectangle(img, (50, 50), (350, 350), (0, 0, 0), 4)
    cv2.circle(img, (200, 200), 60, (0, 0, 255), -1)
    cv2.line(img, (50, 200), (350, 200), (255, 0, 0), 3)
    cv2.putText(img, "TEST", (120, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,128,0), 2)
    return img

def apply_known_transform(img, angle=15, scale=1.1, tx=30, ty=20):
    """Aplicar transformación conocida"""
    h, w = img.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w/2, h/2), angle, scale)
    M_rot[:,2] += [tx, ty]
    transformed = cv2.warpAffine(img, M_rot, (w, h))
    H_true = np.vstack([M_rot, [0,0,1]])
    return transformed, H_true

def get_detector():
    """Obtener detector (SIFT o ORB)"""
    try:
        return cv2.SIFT_create(), 'SIFT'
    except:
        return cv2.ORB_create(1000), 'ORB'

def estimate_homography(img1, img2):
    """Estimar homografía entre dos imágenes"""
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

def compare_homographies(H_true, H_est):
    """Comparar homografías verdaderas y estimadas"""
    H_true /= H_true[2,2]
    H_est /= H_est[2,2]
    pts = np.float32([[0,0],[399,0],[399,399],[0,399]]).reshape(-1,1,2)
    pts_true = cv2.perspectiveTransform(pts, H_true)
    pts_est = cv2.perspectiveTransform(pts, H_est)
    rmse = np.sqrt(np.mean(np.sum((pts_true - pts_est)**2, axis=(1,2))))
    R_true = H_true[:2,:2] / np.linalg.norm(H_true[0,:2])
    R_est = H_est[:2,:2] / np.linalg.norm(H_est[0,:2])
    cos_theta = np.clip(np.trace(R_true.T @ R_est)/2, -1, 1)
    ang_error = np.degrees(np.arccos(cos_theta))
    s_true = np.linalg.norm(H_true[0:2,0])
    s_est = np.linalg.norm(H_est[0:2,0])
    scale_error = abs(s_est - s_true)/s_true * 100
    return rmse, ang_error, scale_error


class TestSyntheticImage:
    """Tests para creación de imágenes sintéticas"""
    
    def test_create_synthetic_image_default_size(self):
        """Test que create_synthetic_image crea una imagen del tamaño correcto"""
        img = create_synthetic_image()
        assert img.shape == (400, 400, 3)
        assert img.dtype == np.uint8
    
    def test_create_synthetic_image_custom_size(self):
        """Test que create_synthetic_image acepta tamaños personalizados"""
        img = create_synthetic_image(size=(200, 300))
        assert img.shape == (200, 300, 3)
    
    def test_create_synthetic_image_not_empty(self):
        """Test que la imagen sintética no está vacía"""
        img = create_synthetic_image()
        assert np.any(img != 0)
        assert np.any(img == 255)  # Fondo blanco


class TestKnownTransform:
    """Tests para transformaciones conocidas"""
    
    def test_apply_known_transform_returns_correct_shape(self):
        """Test que apply_known_transform mantiene el tamaño de la imagen"""
        base_img = create_synthetic_image()
        trans_img, H_true = apply_known_transform(base_img)
        assert trans_img.shape == base_img.shape
        assert H_true.shape == (3, 3)
    
    def test_apply_known_transform_homography_structure(self):
        """Test que la homografía tiene la estructura correcta"""
        base_img = create_synthetic_image()
        _, H_true = apply_known_transform(base_img, angle=20, scale=1.2, tx=40, ty=-15)
        assert H_true[2, 2] == 1.0  # Último elemento debe ser 1
        assert H_true.shape == (3, 3)
    
    def test_apply_known_transform_different_parameters(self):
        """Test que diferentes parámetros producen diferentes transformaciones"""
        base_img = create_synthetic_image()
        trans1, H1 = apply_known_transform(base_img, angle=10, scale=1.0)
        trans2, H2 = apply_known_transform(base_img, angle=30, scale=1.5)
        # Las imágenes transformadas deben ser diferentes
        assert not np.array_equal(trans1, trans2)
        # Las homografías deben ser diferentes
        assert not np.allclose(H1, H2)


class TestDetector:
    """Tests para detectores de características"""
    
    def test_get_detector_returns_detector(self):
        """Test que get_detector retorna un detector válido"""
        detector, method = get_detector()
        assert detector is not None
        assert method in ['SIFT', 'ORB']
    
    def test_detector_can_detect_keypoints(self):
        """Test que el detector puede detectar keypoints"""
        detector, _ = get_detector()
        img = create_synthetic_image()
        kp, des = detector.detectAndCompute(img, None)
        assert len(kp) > 0
        assert des is not None


class TestHomographyEstimation:
    """Tests para estimación de homografía"""
    
    def test_estimate_homography_returns_valid_result(self):
        """Test que estimate_homography retorna resultados válidos"""
        base_img = create_synthetic_image()
        trans_img, _ = apply_known_transform(base_img, angle=15, scale=1.1)
        H_est, pts1, pts2, good, method = estimate_homography(base_img, trans_img)
        assert H_est is not None
        assert H_est.shape == (3, 3)
        assert len(good) > 0
        assert method in ['SIFT', 'ORB']
    
    def test_estimate_homography_matches_found(self):
        """Test que se encuentran matches entre imágenes relacionadas"""
        base_img = create_synthetic_image()
        trans_img, _ = apply_known_transform(base_img, angle=10, scale=1.05)
        H_est, pts1, pts2, good, method = estimate_homography(base_img, trans_img)
        assert len(pts1) > 0
        assert len(pts2) > 0
        assert len(pts1) == len(pts2)


class TestHomographyComparison:
    """Tests para comparación de homografías"""
    
    def test_compare_homographies_returns_metrics(self):
        """Test que compare_homographies retorna métricas válidas"""
        base_img = create_synthetic_image()
        trans_img, H_true = apply_known_transform(base_img, angle=20, scale=1.2)
        H_est, _, _, _, _ = estimate_homography(base_img, trans_img)
        rmse, ang_err, scale_err = compare_homographies(H_true, H_est)
        assert rmse >= 0
        assert ang_err >= 0
        assert scale_err >= 0
        assert not np.isnan(rmse)
        assert not np.isnan(ang_err)
        assert not np.isnan(scale_err)
    
    def test_compare_homographies_identical_returns_low_error(self):
        """Test que homografías idénticas tienen error bajo"""
        H = np.eye(3)
        rmse, ang_err, scale_err = compare_homographies(H, H)
        assert rmse < 1.0  # Debe ser muy bajo
        assert ang_err < 1.0  # Debe ser muy bajo
        assert scale_err < 1.0  # Debe ser muy bajo


class TestIntegration:
    """Tests de integración"""
    
    def test_full_pipeline_small_transformation(self):
        """Test del pipeline completo con transformación pequeña"""
        base_img = create_synthetic_image()
        trans_img, H_true = apply_known_transform(base_img, angle=5, scale=1.0, tx=10, ty=10)
        H_est, _, _, good, _ = estimate_homography(base_img, trans_img)
        assert H_est is not None
        assert len(good) > 0
        rmse, ang_err, scale_err = compare_homographies(H_true, H_est)
        # Para transformaciones pequeñas, el error debe ser razonable
        assert rmse < 50  # RMSE razonable en píxeles
        assert ang_err < 10  # Error angular razonable en grados

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

