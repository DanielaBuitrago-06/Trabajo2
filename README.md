# Proyecto de Registro de Imágenes y Calibración Métrica

Este proyecto implementa un sistema completo para el registro (stitching) de múltiples imágenes y la calibración métrica del resultado, permitiendo realizar mediciones precisas de objetos en el panorama generado.

## 📋 Contenido

1. [Descripción](#descripción)
2. [Requisitos](#requisitos)
3. [Instalación](#instalación)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Uso](#uso)
6. [Parte 1: Validación con Imágenes Sintéticas](#parte-1-validación-con-imágenes-sintéticas)
7. [Parte 2: Registro de Imágenes](#parte-2-registro-de-imágenes)
8. [Parte 3: Calibración y Medición](#parte-3-calibración-y-medición)
9. [Tests](#tests)
10. [Notebooks](#notebooks)
11. [Resultados](#resultados)

## 📖 Descripción

Este proyecto está dividido en tres partes principales:

1. **Validación con Imágenes Sintéticas**: Valida el proceso de registro utilizando imágenes sintéticas con transformaciones conocidas, permitiendo evaluar la precisión del algoritmo.

2. **Registro de Imágenes**: Implementa un pipeline completo para registrar múltiples imágenes del comedor usando proyección cilíndrica y detección de características (SIFT/ORB).

3. **Calibración y Medición**: Establece una escala métrica usando dimensiones conocidas y permite medir distancias en el panorama calibrado.

## 🔧 Requisitos

- Python 3.10 o superior
- pip (gestor de paquetes de Python)
- Imágenes del comedor en formato JPG (opcional, para la Parte 2)

## 📦 Instalación

### 1. Clonar o descargar el proyecto

```bash
cd github
```

### 2. Crear entorno virtual (recomendado)

```bash
python3 -m venv venv
#Mac/Linux
source venv/bin/activate  
#Windows
venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Verificar instalación

```bash
python -c "import cv2, numpy, matplotlib, scipy; print('Instalación correcta')"
```

## 📁 Estructura del Proyecto

```
trabajo2/
├── data/
│   ├── original/          # Imágenes originales del comedor
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── img3.jpg
│   └── synthetic/          # Imágenes sintéticas generadas
│       ├── base_img.jpg
│       ├── trans_img.jpg
│       ├── matches.jpg
│       └── transformed_*.jpg
├── notebooks/             # Jupyter notebooks interactivos
│   ├── 1_validate_img_synthetic.ipynb
│   ├── 2_register_images.ipynb
│   └── 3_calibration_measurement.ipynb
├── results/
│   ├── figures/           # Imágenes de salida
│   │   ├── panorama.jpg
│   │   └── panorama_calibrated.jpg
│   └── measurements/      # Datos de medición y logs
│       ├── 1_validate_img_synthetic_results.txt
│       ├── 2_register_images_results.txt
│       ├── 3_calibration_measurement_results.txt
│       └── measurements_data.json
├── src/                   # Scripts principales
│   ├── 1_validate_img_synthetic.py
│   ├── 2_register_images.py
│   └── 3_calibration_measurement.py
├── tests/                 # Tests unitarios
│   ├── test_1_validate_img_synthetic.py
│   ├── test_2_register_images.py
│   └── test_3_calibration_measurement.py
├── venv/                  # Entorno virtual (no incluido en git)
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Este archivo
```

## 🚀 Uso

### Ejecución Secuencial (Recomendado)

Los scripts están diseñados para ejecutarse en orden:

```bash
# 1. Validación con imágenes sintéticas
python src/1_validate_img_synthetic.py

# 2. Registro de imágenes (requiere imágenes en data/original/)
python src/2_register_images.py

# 3. Calibración y medición (requiere panorama de la parte 2)
python src/3_calibration_measurement.py
```

### Ejecución Individual

Cada script puede ejecutarse de forma independiente si se cumplen los requisitos previos.

## 📊 Parte 1: Validación con Imágenes Sintéticas

### Descripción

Esta parte valida el proceso de registro utilizando imágenes sintéticas con transformaciones conocidas. Esto permite evaluar la precisión del algoritmo sin depender de imágenes reales.

### Ejecución

```bash
python src/1_validate_img_synthetic.py
```

### Qué Genera

#### Imágenes en `data/synthetic/`:

- **`base_img.jpg`**: Imagen sintética base con formas geométricas (rectángulo, círculo, línea, texto)
- **`trans_img.jpg`**: Imagen transformada con rotación, escala y traslación conocidas
- **`matches.jpg`**: Visualización de los emparejamientos de características detectados
- **`transformed_a{angulo}_s{escala}.jpg`**: 12 imágenes transformadas con diferentes combinaciones de ángulos (5°, 15°, 30°, 45°) y escalas (1.0, 1.1, 1.3)

#### Datos en `results/measurements/`:

- **`1_validate_img_synthetic_results.txt`**: Contiene:
  - Homografía verdadera (H_true) aplicada
  - Método de detección usado (SIFT u ORB)
  - Homografía estimada (H_est)
  - Métricas de comparación:
    - RMSE de puntos (error cuadrático medio)
    - Error angular (grados)
    - Error de escala (porcentaje)
  - Tabla completa de resultados para diferentes combinaciones de parámetros

### Ejemplo de Salida

```
=== EJECUCIÓN: 2024-01-15 10:30:45 ===

Homografía Verdadera (H_true):
[[ 1.127  0.342  40.000]
 [ -0.342  1.127 -15.000]
 [ 0.000  0.000   1.000]]

Método: SIFT
Homografía estimada (H_est):
[[ 1.125  0.340  39.850]
 [ -0.341  1.125 -14.950]
 [ 0.000  0.000   1.000]]

RMSE de puntos: 2.3456 px
Error angular: 0.1234°
Error de escala: 0.5678%

Variación de parámetros:
Angulo | Escala | RMSE(px) | Error Rot(°) | Error Escala(%)
  5.0 |   1.00 |    1.234 |        0.056 |          0.123
 15.0 |   1.00 |    2.345 |        0.234 |          0.456
...
```

## 🖼️ Parte 2: Registro de Imágenes

### Descripción

Implementa el registro (stitching) de múltiples imágenes del comedor usando:
- **Proyección cilíndrica**: Reduce distorsiones en panoramas
- **Detección de características**: SIFT (o ORB como fallback)
- **Matching robusto**: Ratio test de Lowe
- **Estimación de homografía**: RANSAC para robustez
- **Composición**: Warping y merge de imágenes

### Requisitos Previos

Las imágenes del comedor deben estar en `data/original/`:
- `img1.jpg`
- `img2.jpg`
- `img3.jpg`

### Ejecución

```bash
python src/2_register_images.py
```

### Qué Genera

#### Imágenes en `results/figures/`:

- **`panorama.jpg`**: Panorama completo del comedor generado por el registro de las 3 imágenes

#### Datos en `results/measurements/`:

- **`2_register_images_results.txt`**: Contiene:
  - Lista de imágenes procesadas
  - Número de matches encontrados para cada par de imágenes
  - Número de inliers después de RANSAC
  - Dimensiones finales del panorama
  - Mensajes de error si alguna homografía falla

### Ejemplo de Salida

```
=== EJECUCIÓN: 2024-01-15 10:35:20 ===

Iniciando registro de imágenes...
Imágenes a procesar: 3
  - data/original/img1.jpg
  - data/original/img2.jpg
  - data/original/img3.jpg

Imagen 1: 245 matches, 198 inliers
Imagen 2: 189 matches, 156 inliers

Dimensiones del panorama: 3456x1200 píxeles
```

### Configuración

Puedes modificar los parámetros en el script:
- `detector_name`: 'SIFT' (recomendado) o 'ORB'
- `focal_length`: Longitud focal para proyección cilíndrica (default: 900)

## 📏 Parte 3: Calibración y Medición

### Descripción

Establece una escala métrica usando dimensiones conocidas y permite medir distancias en el panorama calibrado.

**Dimensiones conocidas utilizadas:**
- Cuadro de la Virgen de Guadalupe: **117 cm** (altura)
- Mesa: **161.1 cm** (ancho)

### Requisitos Previos

Requiere que el panorama esté generado en `results/figures/panorama.jpg` (ejecutar Parte 2 primero).

### Ejecución

```bash
python src/3_calibration_measurement.py
```

### Qué Genera

#### Imágenes en `results/figures/`:

- **`panorama_calibrated.jpg`**: Panorama con barra de escala visual (50 cm) en la esquina inferior derecha

#### Datos en `results/measurements/`:

- **`3_calibration_measurement_results.txt`**: Contiene:
  - Información de calibración:
    - Distancias en píxeles de las dimensiones conocidas
    - Factor de escala calculado (cm/píxel)
    - Escala promedio si se usan múltiples mediciones
  - Mediciones realizadas:
    - Ancho del cuadro
    - Largo de la mesa
    - Ventana (ancho)
    - Silla (ancho)
    - Planta (altura)
  - Análisis de incertidumbre:
    - Incertidumbre en calibración
    - Incertidumbre en cada medición
    - Incertidumbre relativa (porcentaje)
  - Resumen final con todas las dimensiones calculadas

- **`measurements_data.json`**: Datos estructurados en JSON con:
  - Factor de escala
  - Incertidumbre de calibración
  - Mediciones conocidas
  - Todas las mediciones realizadas con incertidumbres
  - Timestamp de la ejecución

### Ejemplo de Salida

```
=== EJECUCIÓN: 2024-01-15 10:40:15 ===

CALIBRACIÓN Y MEDICIÓN EN PANORAMA
============================================================

Panorama cargado: 3456x1200 píxeles

Dimensiones conocidas para calibración:
  - Cuadro de la Virgen de Guadalupe (altura): 117.0 cm
  - Mesa (ancho): 161.1 cm

Usando escala estimada (no interactivo): 0.1345 cm/px
NOTA: Para mayor precisión, cambia USE_INTERACTIVE a True

CÁLCULO DE DIMENSIONES ADICIONALES
============================================================

Mediciones realizadas:
------------------------------------------------------------
Ancho del cuadro              :   92.45 cm (0.924 m) [687.0 px]
Largo de la mesa              :  154.23 cm (1.542 m) [1145.0 px]
Ventana (ancho)               :  185.08 cm (1.851 m) [1375.0 px]
Silla (ancho)                 :   49.35 cm (0.494 m) [367.0 px]
Planta (altura)               :  108.36 cm (1.084 m) [805.0 px]

DIMENSIONES DEL CUADRO Y MESA
============================================================

Cuadro de la Virgen de Guadalupe:
  - Altura (conocida): 117.0 cm
  - Ancho (medido):    92.45 cm

Mesa:
  - Ancho (conocido): 161.1 cm
  - Largo (medido):  154.23 cm

ANÁLISIS DE INCERTIDUMBRE
============================================================

Incertidumbre en calibración:
  - Incertidumbre en píxeles: ±2.0 px
  - Incertidumbre en centímetros: ±0.27 cm
  - Incertidumbre relativa: ±0.23%

Incertidumbre en mediciones:
------------------------------------------------------------
Elemento                      Valor (cm) Incert. (cm) Incert. (%)
------------------------------------------------------------
Ancho del cuadro                 92.45 ±        0.30 ±       0.32%
Largo de la mesa               154.23 ±        0.30 ±       0.19%
Ventana (ancho)                185.08 ±        0.30 ±       0.16%
Silla (ancho)                   49.35 ±        0.30 ±       0.61%
Planta (altura)                108.36 ±        0.30 ±       0.28%

RESUMEN FINAL
============================================================

Escala calibrada: 0.1345 cm/píxel

Dimensiones calculadas:
  - Cuadro (ancho): 92.45 ± 0.30 cm
  - Mesa (largo): 154.23 ± 0.30 cm

Incertidumbre promedio en mediciones: 0.31%
```

### Modo Interactivo

Para mayor precisión, puedes usar el modo interactivo editando el script:

```python
USE_INTERACTIVE = True  # Cambiar a True
USE_INTERACTIVE_MEASUREMENTS = True  # Cambiar a True
```

Esto abrirá ventanas interactivas donde puedes hacer clic en dos puntos para:
1. Calibrar usando las dimensiones conocidas
2. Medir cada elemento adicional

## 🧪 Tests

### Ejecutar Tests

```bash
# Ejecutar todos los tests
pytest tests/

# Ejecutar tests específicos
pytest tests/test_1_validate_img_synthetic.py -v
pytest tests/test_2_register_images.py -v
pytest tests/test_3_calibration_measurement.py -v

# Con más detalles
pytest tests/ -v

# Con cobertura
pytest tests/ --cov=src --cov-report=html
```

### Cobertura de Tests

Los tests cubren:
- Creación y transformación de imágenes sintéticas
- Detección de características (SIFT/ORB)
- Estimación de homografía
- Proyección cilíndrica
- Matching de características
- Calibración y medición
- Análisis de incertidumbre

## 📓 Notebooks

Los notebooks de Jupyter proporcionan una versión interactiva de cada script, ideal para experimentación y análisis.

### Ejecutar Notebooks

```bash
# Desde el directorio del proyecto
jupyter notebook notebooks/

# O abrir directamente
jupyter notebook notebooks/1_validate_img_synthetic.ipynb
```

### Ventajas de los Notebooks

- Ejecución celda por celda
- Visualización interactiva de resultados
- Fácil modificación de parámetros
- Análisis paso a paso

## 📈 Resultados

### Estructura de Resultados

Todos los resultados se guardan automáticamente en:

- **`results/figures/`**: Imágenes generadas
- **`results/measurements/`**: Datos numéricos y logs

### Archivos de Salida

Cada ejecución genera:
1. **Archivos de texto (.txt)**: Logs completos con todos los prints
2. **Archivos JSON**: Datos estructurados (solo Parte 3)
3. **Imágenes**: Visualizaciones y resultados procesados

### Interpretación de Resultados

#### Parte 1 (Validación)
- **RMSE bajo (< 5 px)**: Buena precisión en el registro
- **Error angular bajo (< 1°)**: Rotación bien estimada
- **Error de escala bajo (< 2%)**: Escala bien estimada

#### Parte 2 (Registro)
- **Número de matches**: Más matches = mejor alineación (típicamente > 50)
- **Inliers**: Puntos válidos después de RANSAC (típicamente > 70% de matches)
- **Dimensiones del panorama**: Tamaño final del panorama fusionado

#### Parte 3 (Calibración)
- **Factor de escala**: Relación cm/píxel (típicamente 0.1-0.2 cm/px)
- **Incertidumbre relativa**: Precisión de las mediciones (típicamente < 1%)
- **Mediciones**: Dimensiones calculadas de todos los elementos

## 🔍 Solución de Problemas

### Error: "No se encontró el panorama"
- **Solución**: Ejecuta primero `2_register_images.py`

### Error: "SIFT no disponible"
- **Solución**: El script automáticamente usa ORB como fallback. Si quieres SIFT, instala OpenCV contrib:
  ```bash
  pip install opencv-contrib-python
  ```

### Error: "No se encontró {imagen}"
- **Solución**: Verifica que las imágenes estén en `data/original/` con los nombres correctos

### Imágenes no se registran correctamente
- **Causas posibles**:
  - Poca superposición entre imágenes
  - Iluminación muy diferente
  - Objetos en movimiento
- **Soluciones**:
  - Asegura al menos 30% de superposición
  - Usa imágenes con iluminación similar
  - Prueba diferentes detectores (SIFT, ORB, AKAZE)

### Calibración imprecisa
- **Solución**: Usa modo interactivo (`USE_INTERACTIVE = True`) para seleccionar puntos manualmente

## 📝 Notas Técnicas

### Algoritmos Utilizados

- **SIFT (Scale-Invariant Feature Transform)**: Detección de características robusta a escala y rotación
- **ORB (Oriented FAST and Rotated BRIEF)**: Alternativa más rápida a SIFT
- **RANSAC (Random Sample Consensus)**: Estimación robusta de homografía eliminando outliers
- **Proyección Cilíndrica**: Reduce distorsiones en panoramas amplios

### Parámetros Ajustables

En cada script puedes modificar:
- **Parte 1**: Ángulos y escalas para experimentos
- **Parte 2**: Longitud focal, detector, ratio de matching
- **Parte 3**: Modo interactivo, incertidumbre en píxeles

## 📄 Licencia

Este proyecto es parte de un trabajo académico.

## 👥 Autor

Proyecto desarrollado para trabajo académico sobre registro de imágenes y calibración métrica.

---

**Última actualización**: 2024

Para más información sobre los tests, consulta `tests/README.md`.

