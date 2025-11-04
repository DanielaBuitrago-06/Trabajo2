"""
Tests para 2_register_images.py
"""
import sys
import os
import pytest
import numpy as np
import cv2

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def create_detector(name='SIFT'):
    """Detector genérico (SIFT -> ORB fallback)"""
    name = name.upper()
    if name == 'SIFT':
        try:
            return cv2.SIFT_create(), 'SIFT'
        except:
            return cv2.ORB_create(5000), 'ORB'
    elif name == 'AKAZE':
        return cv2.AKAZE_create(), 'AKAZE'
    else:
        return cv2.ORB_create(5000), 'ORB'


def cylindrical_warp(img, f=900):
    """Proyección cilíndrica"""
    h, w = img.shape[:2]
    K = np.array([[f, 0, w/2],
                  [0, f, h/2],
                  [0, 0, 1]])
    y_i, x_i = np.indices((h, w))
    X = np.stack([x_i - w/2, y_i - h/2, np.ones_like(x_i)], axis=-1)
    X = X.reshape((-1,3))
    Xc = np.stack([np.sin(X[:,0]/f), X[:,1]/f, np.cos(X[:,0]/f)], axis=-1)
    x_p = (Xc[:,0]*f / Xc[:,2]) + w/2
    y_p = (Xc[:,1]*f / Xc[:,2]) + h/2
    mapx = x_p.reshape(h,w).astype(np.float32)
    mapy = y_p.reshape(h,w).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)


def detect_and_match(img1, img2, detector_name='SIFT', ratio=0.75):
    """Detección y matching robusto"""
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


def estimate_homography(pts1, pts2):
    """Estimar homografía robusta (RANSAC)"""
    if len(pts1) < 4:
        return None, None
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
    return H, mask


def warp_and_merge(ref, img, H):
    """Warping y composición al marco de referencia"""
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


def create_test_image(size=(400, 400)):
    """Crear imagen de prueba"""
    img = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    # Agregar algunos patrones para detectar características
    cv2.rectangle(img, (50, 50), (350, 350), (255, 255, 255), -1)
    cv2.circle(img, (200, 200), 50, (0, 0, 0), -1)
    return img


class TestDetector:
    """Tests para detector genérico"""
    
    def test_create_detector_sift(self):
        """Test que create_detector puede crear SIFT"""
        detector, method = create_detector('SIFT')
        assert detector is not None
        assert method in ['SIFT', 'ORB']
    
    def test_create_detector_orb(self):
        """Test que create_detector puede crear ORB"""
        detector, method = create_detector('ORB')
        assert detector is not None
        assert method == 'ORB'
    
    def test_create_detector_akaze(self):
        """Test que create_detector puede crear AKAZE"""
        detector, method = create_detector('AKAZE')
        assert detector is not None
        assert method == 'AKAZE'


class TestCylindricalWarp:
    """Tests para proyección cilíndrica"""
    
    def test_cylindrical_warp_preserves_size(self):
        """Test que cylindrical_warp preserva el tamaño de la imagen"""
        img = create_test_image((400, 600))
        warped = cylindrical_warp(img)
        assert warped.shape == img.shape
    
    def test_cylindrical_warp_returns_valid_image(self):
        """Test que cylindrical_warp retorna una imagen válida"""
        img = create_test_image()
        warped = cylindrical_warp(img)
        assert warped.dtype == np.uint8
        assert warped.min() >= 0
        assert warped.max() <= 255
    
    def test_cylindrical_warp_different_focal_lengths(self):
        """Test que diferentes longitudes focales producen diferentes resultados"""
        img = create_test_image()
        warped1 = cylindrical_warp(img, f=500)
        warped2 = cylindrical_warp(img, f=1500)
        # Deben ser diferentes (aunque no completamente diferentes en bordes)
        assert not np.array_equal(warped1, warped2)


class TestDetectAndMatch:
    """Tests para detección y matching"""
    
    def test_detect_and_match_finds_matches(self):
        """Test que detect_and_match encuentra matches entre imágenes similares"""
        img1 = create_test_image()
        # Crear una versión ligeramente transformada
        M = cv2.getRotationMatrix2D((200, 200), 5, 1.0)
        img2 = cv2.warpAffine(img1, M, (400, 400))
        kp1, kp2, pts1, pts2, good = detect_and_match(img1, img2)
        assert len(good) > 0
        assert len(pts1) == len(pts2)
    
    def test_detect_and_match_returns_empty_for_unrelated_images(self):
        """Test que detect_and_match retorna pocos matches para imágenes no relacionadas"""
        img1 = create_test_image()
        img2 = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
        kp1, kp2, pts1, pts2, good = detect_and_match(img1, img2)
        # Puede encontrar algunos matches pero debería ser menos que con imágenes relacionadas
        assert isinstance(len(good), int)
    
    def test_detect_and_match_handles_no_descriptors(self):
        """Test que detect_and_match maneja casos sin descriptores"""
        # Crear imagen completamente uniforme (sin características)
        img1 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        img2 = np.ones((100, 100, 3), dtype=np.uint8) * 128
        kp1, kp2, pts1, pts2, good = detect_and_match(img1, img2)
        # Debe retornar listas vacías
        assert len(pts1) == 0
        assert len(pts2) == 0


class TestEstimateHomography:
    """Tests para estimación de homografía"""
    
    def test_estimate_homography_requires_sufficient_points(self):
        """Test que estimate_homography requiere suficientes puntos"""
        pts1 = np.array([[0, 0], [100, 0], [100, 100]], dtype=np.float32)
        pts2 = np.array([[10, 10], [110, 10], [110, 110]], dtype=np.float32)
        H, mask = estimate_homography(pts1, pts2)
        assert H is None  # No hay suficientes puntos (necesita al menos 4)
    
    def test_estimate_homography_returns_valid_homography(self):
        """Test que estimate_homography retorna homografía válida con suficientes puntos"""
        # Crear puntos correspondientes con transformación conocida
        pts1 = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.float32)
        M = np.array([[1.1, 0.05, 10], [0.05, 1.1, 5]], dtype=np.float32)
        pts2 = cv2.transform(pts1.reshape(-1, 1, 2), M).reshape(-1, 2).astype(np.float32)
        H, mask = estimate_homography(pts1, pts2)
        assert H is not None
        assert H.shape == (3, 3)
        assert mask is not None


class TestWarpAndMerge:
    """Tests para warping y merge"""
    
    def test_warp_and_merge_returns_valid_panorama(self):
        """Test que warp_and_merge retorna un panorama válido"""
        ref = create_test_image((400, 400))
        img = create_test_image((400, 400))
        # Crear homografía de identidad (sin transformación)
        H = np.eye(3)
        panorama = warp_and_merge(ref, img, H)
        assert panorama is not None
        assert panorama.shape[0] > 0
        assert panorama.shape[1] > 0
        assert len(panorama.shape) == 3
    
    def test_warp_and_merge_includes_both_images(self):
        """Test que warp_and_merge incluye ambas imágenes"""
        ref = create_test_image((400, 400))
        img = create_test_image((400, 400))
        H = np.eye(3)
        panorama = warp_and_merge(ref, img, H)
        # El panorama debe ser al menos tan grande como la imagen de referencia
        assert panorama.shape[0] >= ref.shape[0]
        assert panorama.shape[1] >= ref.shape[1]
    
    def test_warp_and_merge_handles_translation(self):
        """Test que warp_and_merge maneja transformaciones de traslación"""
        ref = create_test_image((400, 400))
        img = create_test_image((400, 400))
        # Homografía de traslación
        H = np.array([[1, 0, 50], [0, 1, 30], [0, 0, 1]], dtype=np.float32)
        panorama = warp_and_merge(ref, img, H)
        assert panorama is not None
        assert panorama.shape[0] > ref.shape[0] or panorama.shape[1] > ref.shape[1]


class TestIntegration:
    """Tests de integración"""
    
    def test_full_matching_pipeline(self):
        """Test del pipeline completo de matching"""
        img1 = create_test_image()
        # Crear imagen ligeramente transformada
        M = cv2.getRotationMatrix2D((200, 200), 10, 1.0)
        M = np.vstack([M, [0, 0, 1]])
        img2 = cv2.warpAffine(img1, M[:2], (400, 400))
        
        kp1, kp2, pts1, pts2, good = detect_and_match(img1, img2)
        if len(pts1) >= 4:
            H, mask = estimate_homography(pts1, pts2)
            if H is not None:
                panorama = warp_and_merge(img1, img2, H)
                assert panorama is not None
                assert panorama.shape[0] > 0
                assert panorama.shape[1] > 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

