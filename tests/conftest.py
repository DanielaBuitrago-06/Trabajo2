"""
Configuración de pytest para tests
"""
import pytest
import numpy as np
import cv2
import os
import sys

# Agregar src al path para imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_image():
    """Crear una imagen de muestra para tests"""
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (350, 350), (255, 255, 255), -1)
    cv2.circle(img, (200, 200), 50, (0, 0, 0), -1)
    return img


@pytest.fixture
def sample_panorama():
    """Crear un panorama de muestra para tests"""
    panorama = np.random.randint(0, 255, (800, 1600, 3), dtype=np.uint8)
    cv2.rectangle(panorama, (200, 200), (600, 600), (255, 255, 255), -1)
    return panorama


@pytest.fixture
def test_output_dir(tmp_path):
    """Directorio temporal para outputs de tests"""
    output_dir = tmp_path / "test_outputs"
    output_dir.mkdir()
    return str(output_dir)

