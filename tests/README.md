# Tests del Proyecto

Este directorio contiene los tests unitarios para los scripts principales del proyecto.

## Estructura de Tests

- `test_1_validate_img_synthetic.py`: Tests para validación con imágenes sintéticas
- `test_2_register_images.py`: Tests para registro de imágenes
- `test_3_calibration_measurement.py`: Tests para calibración y medición
- `conftest.py`: Configuración y fixtures compartidas de pytest

## Instalación

Asegúrate de tener pytest instalado:

```bash
pip install pytest
```

O instala todas las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecutar Tests

### Ejecutar todos los tests:

```bash
pytest tests/
```

### Ejecutar un archivo de test específico:

```bash
pytest tests/test_1_validate_img_synthetic.py
pytest tests/test_2_register_images.py
pytest tests/test_3_calibration_measurement.py
```

### Ejecutar con más detalles:

```bash
pytest tests/ -v
```

### Ejecutar con cobertura:

```bash
pytest tests/ --cov=src --cov-report=html
```

## Estructura de los Tests

### test_1_validate_img_synthetic.py

Tests para:
- Creación de imágenes sintéticas
- Aplicación de transformaciones conocidas
- Detección de características (SIFT/ORB)
- Estimación de homografía
- Comparación de homografías

### test_2_register_images.py

Tests para:
- Creación de detectores (SIFT, ORB, AKAZE)
- Proyección cilíndrica
- Detección y matching de características
- Estimación de homografía robusta
- Warping y composición de panoramas

### test_3_calibration_measurement.py

Tests para:
- Herramienta de medición (MeasurementTool)
- Calibración usando dimensiones conocidas
- Cálculo de factor de escala
- Análisis de incertidumbre
- Propagación de incertidumbre

## Notas

- Los tests están diseñados para ejecutarse sin dependencias externas (imágenes reales)
- Se utilizan imágenes sintéticas generadas en los tests
- Algunos tests pueden requerir que OpenCV tenga SIFT disponible (depende de la versión)

