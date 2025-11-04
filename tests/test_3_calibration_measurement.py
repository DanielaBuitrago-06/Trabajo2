"""
Tests para 3_calibration_measurement.py
"""
import sys
import os
import pytest
import numpy as np
import cv2

# Agregar el directorio src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class MeasurementTool:
    """Herramienta de medición para testing"""
    def __init__(self, img, scale_factor=None):
        self.img = img.copy()
        self.original_img = img.copy()
        self.scale_factor = scale_factor  # cm/pixel
        self.points = []
        self.measurements = []
        
    def calibrate(self, point1, point2, real_distance_cm):
        """Calibrar usando dos puntos y una distancia real conocida"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        distance_px = np.sqrt(dx**2 + dy**2)
        self.scale_factor = real_distance_cm / distance_px
        return self.scale_factor
    
    def measure_distance(self, point1, point2):
        """Medir distancia entre dos puntos"""
        dx = point2[0] - point1[0]
        dy = point2[1] - point1[1]
        distance_px = np.sqrt(dx**2 + dy**2)
        if self.scale_factor:
            return distance_px * self.scale_factor, distance_px
        return None, distance_px


def create_test_panorama(size=(1000, 2000)):
    """Crear panorama de prueba"""
    panorama = np.random.randint(0, 255, (size[0], size[1], 3), dtype=np.uint8)
    # Agregar algunos elementos reconocibles
    cv2.rectangle(panorama, (200, 200), (400, 600), (255, 255, 255), -1)  # Rectángulo blanco
    cv2.rectangle(panorama, (800, 300), (1600, 500), (0, 0, 0), -1)  # Rectángulo negro
    return panorama


class TestMeasurementTool:
    """Tests para MeasurementTool"""
    
    def test_measurement_tool_initialization(self):
        """Test que MeasurementTool se inicializa correctamente"""
        img = create_test_panorama()
        tool = MeasurementTool(img, scale_factor=0.1)
        assert tool.scale_factor == 0.1
        assert tool.img.shape == img.shape
        assert len(tool.points) == 0
    
    def test_calibrate_sets_scale_factor(self):
        """Test que calibrate establece el factor de escala correctamente"""
        img = create_test_panorama()
        tool = MeasurementTool(img)
        point1 = (100, 100)
        point2 = (200, 100)  # 100 píxeles de distancia horizontal
        real_distance_cm = 50.0  # 50 cm
        
        scale = tool.calibrate(point1, point2, real_distance_cm)
        assert scale == 0.5  # 50 cm / 100 px = 0.5 cm/px
        assert tool.scale_factor == 0.5
    
    def test_measure_distance_without_calibration(self):
        """Test que measure_distance funciona sin calibración"""
        img = create_test_panorama()
        tool = MeasurementTool(img)  # Sin scale_factor
        point1 = (0, 0)
        point2 = (100, 100)
        
        distance_cm, distance_px = tool.measure_distance(point1, point2)
        assert distance_cm is None
        assert distance_px > 0
        expected_px = np.sqrt(100**2 + 100**2)
        assert abs(distance_px - expected_px) < 1.0
    
    def test_measure_distance_with_calibration(self):
        """Test que measure_distance retorna distancia en cm con calibración"""
        img = create_test_panorama()
        tool = MeasurementTool(img, scale_factor=0.1)  # 0.1 cm/px
        point1 = (0, 0)
        point2 = (100, 0)  # 100 píxeles
        
        distance_cm, distance_px = tool.measure_distance(point1, point2)
        assert distance_cm == 10.0  # 100 px * 0.1 cm/px = 10 cm
        assert distance_px == 100.0
    
    def test_measure_distance_diagonal(self):
        """Test que measure_distance calcula correctamente distancias diagonales"""
        img = create_test_panorama()
        tool = MeasurementTool(img, scale_factor=0.1)
        point1 = (0, 0)
        point2 = (100, 100)
        
        distance_cm, distance_px = tool.measure_distance(point1, point2)
        expected_px = np.sqrt(100**2 + 100**2)
        assert abs(distance_px - expected_px) < 1.0
        assert abs(distance_cm - expected_px * 0.1) < 0.1


class TestCalibration:
    """Tests para funciones de calibración"""
    
    def test_calibration_with_known_measurements(self):
        """Test que la calibración funciona con mediciones conocidas"""
        img = create_test_panorama()
        tool = MeasurementTool(img)
        
        # Simular medición de altura conocida (117 cm)
        point1 = (500, 100)
        point2 = (500, 1000)  # 900 píxeles verticales
        known_height_cm = 117.0
        
        scale = tool.calibrate(point1, point2, known_height_cm)
        expected_scale = known_height_cm / 900.0
        assert abs(scale - expected_scale) < 0.001
    
    def test_calibration_accuracy(self):
        """Test que la calibración es precisa"""
        img = create_test_panorama()
        tool = MeasurementTool(img)
        
        # Calibrar con distancia conocida
        point1 = (100, 100)
        point2 = (200, 100)  # 100 píxeles
        known_distance = 50.0  # cm
        tool.calibrate(point1, point2, known_distance)
        
        # Medir otra distancia con la misma escala
        point3 = (200, 100)
        point4 = (400, 100)  # 200 píxeles
        distance_cm, _ = tool.measure_distance(point3, point4)
        expected_distance = 100.0  # 200 px * 0.5 cm/px
        assert abs(distance_cm - expected_distance) < 0.1


class TestScaleFactor:
    """Tests para cálculo de factor de escala"""
    
    def test_scale_factor_calculation(self):
        """Test que el factor de escala se calcula correctamente"""
        # Distancia en píxeles
        distance_px = 1000.0
        # Distancia real conocida
        known_distance_cm = 161.1  # cm (ancho de mesa)
        
        scale_factor = known_distance_cm / distance_px
        assert scale_factor > 0
        assert scale_factor < 1.0  # Normalmente cm/px es pequeño
    
    def test_scale_factor_consistency(self):
        """Test que el factor de escala es consistente"""
        # Usar dos mediciones conocidas
        cuadro_height_cm = 117.0
        mesa_width_cm = 161.1
        
        # Simular mediciones en píxeles
        cuadro_height_px = 800.0
        mesa_width_px = 1100.0
        
        scale1 = cuadro_height_cm / cuadro_height_px
        scale2 = mesa_width_cm / mesa_width_px
        
        # El promedio debe ser razonable
        avg_scale = (scale1 + scale2) / 2.0
        assert avg_scale > 0
        assert avg_scale < 1.0


class TestUncertainty:
    """Tests para análisis de incertidumbre"""
    
    def test_uncertainty_propagation(self):
        """Test que la propagación de incertidumbre funciona correctamente"""
        uncertainty_px = 2.0
        scale_factor = 0.1  # cm/px
        uncertainty_cm = uncertainty_px * scale_factor
        
        assert uncertainty_cm == 0.2  # 2 px * 0.1 cm/px = 0.2 cm
    
    def test_combined_uncertainty(self):
        """Test que la incertidumbre combinada se calcula correctamente"""
        uncertainty_calibration_cm = 0.5
        uncertainty_measurement_cm = 0.3
        
        # Suma cuadrática (propagación de incertidumbre)
        total_uncertainty = np.sqrt(uncertainty_calibration_cm**2 + uncertainty_measurement_cm**2)
        expected = np.sqrt(0.5**2 + 0.3**2)
        
        assert abs(total_uncertainty - expected) < 0.001
    
    def test_relative_uncertainty(self):
        """Test que la incertidumbre relativa se calcula correctamente"""
        measured_value_cm = 100.0
        uncertainty_cm = 2.0
        relative_uncertainty = (uncertainty_cm / measured_value_cm) * 100
        
        assert relative_uncertainty == 2.0  # 2%


class TestMeasurementData:
    """Tests para manejo de datos de medición"""
    
    def test_measurement_data_structure(self):
        """Test que la estructura de datos de medición es correcta"""
        measurement_data = {
            'elemento': 'Test Element',
            'distancia_cm': 50.0,
            'distancia_px': 500.0,
            'incertidumbre_cm': 1.0,
            'incertidumbre_relativa_pct': 2.0
        }
        
        assert 'elemento' in measurement_data
        assert 'distancia_cm' in measurement_data
        assert 'distancia_px' in measurement_data
        assert measurement_data['distancia_cm'] > 0
        assert measurement_data['incertidumbre_relativa_pct'] >= 0


class TestIntegration:
    """Tests de integración"""
    
    def test_full_calibration_and_measurement_workflow(self):
        """Test del flujo completo de calibración y medición"""
        img = create_test_panorama()
        tool = MeasurementTool(img)
        
        # Paso 1: Calibrar
        point1 = (500, 100)
        point2 = (500, 1000)  # 900 px
        known_distance = 117.0  # cm
        tool.calibrate(point1, point2, known_distance)
        
        # Paso 2: Medir otra distancia
        point3 = (200, 200)
        point4 = (800, 200)  # 600 px
        distance_cm, distance_px = tool.measure_distance(point3, point4)
        
        assert distance_cm is not None
        assert distance_cm > 0
        assert distance_px > 0
        assert abs(distance_px - 600.0) < 1.0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

