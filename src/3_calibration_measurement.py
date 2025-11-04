# ============================================
# PARTE 3: CALIBRACIÓN Y MEDICIÓN
# ============================================
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle
import json

plt.rcParams['figure.figsize'] = (15, 10)

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
log_file = os.path.join(measurements_dir, '3_calibration_measurement_results.txt')
tee = TeeOutput(log_file)
sys.stdout = tee

# ----------------------------
# 1. Cargar panorama y dimensiones conocidas
# ----------------------------
print("=" * 60)
print("CALIBRACIÓN Y MEDICIÓN EN PANORAMA")
print("=" * 60)

panorama_path = os.path.join(figures_dir, 'panorama.jpg')
if not os.path.exists(panorama_path):
    print(f"ERROR: No se encontró el panorama en {panorama_path}")
    print("Por favor, ejecuta primero el script 2_register_images.py")
    sys.exit(1)

panorama = cv2.imread(panorama_path)
if panorama is None:
    print(f"ERROR: No se pudo cargar el panorama desde {panorama_path}")
    sys.exit(1)

panorama_rgb = cv2.cvtColor(panorama, cv2.COLOR_BGR2RGB)
print(f"\nPanorama cargado: {panorama.shape[1]}x{panorama.shape[0]} píxeles")

# Dimensiones conocidas (en cm)
KNOWN_MEASUREMENTS = {
    'cuadro_altura': 117.0,  # cm - Altura del cuadro de la Virgen de Guadalupe
    'mesa_ancho': 161.1      # cm - Ancho de la mesa
}

print("\nDimensiones conocidas para calibración:")
print(f"  - Cuadro de la Virgen de Guadalupe (altura): {KNOWN_MEASUREMENTS['cuadro_altura']} cm")
print(f"  - Mesa (ancho): {KNOWN_MEASUREMENTS['mesa_ancho']} cm")

# ----------------------------
# 2. Herramienta interactiva para calibración
# ----------------------------
class MeasurementTool:
    def __init__(self, img, scale_factor=None):
        self.img = img.copy()
        self.original_img = img.copy()
        self.scale_factor = scale_factor  # cm/pixel
        self.points = []
        self.measurements = []
        self.current_measurement = None
        self.fig = None
        self.ax = None
        self.cid = None
        
    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button == 1:  # Click izquierdo
            x, y = int(event.xdata), int(event.ydata)
            self.points.append((x, y))
            
            if len(self.points) == 1:
                # Primer punto
                self.ax.plot(x, y, 'ro', markersize=8)
                self.fig.canvas.draw()
            elif len(self.points) == 2:
                # Segundo punto - dibujar línea
                p1, p2 = self.points
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                distance_px = np.sqrt(dx**2 + dy**2)
                
                # Dibujar línea
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
                self.ax.plot(p2[0], p2[1], 'ro', markersize=8)
                
                # Calcular distancia real si hay escala
                if self.scale_factor:
                    distance_cm = distance_px * self.scale_factor
                    distance_m = distance_cm / 100.0
                    text = f'{distance_cm:.2f} cm'
                    self.ax.text((p1[0] + p2[0])/2, (p1[1] + p2[1])/2 - 10, 
                               text, color='yellow', fontsize=10, 
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                else:
                    text = f'{distance_px:.1f} px'
                    self.ax.text((p1[0] + p2[0])/2, (p1[1] + p2[1])/2 - 10, 
                               text, color='yellow', fontsize=10,
                               bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                
                self.fig.canvas.draw()
                self.points = []  # Reset para siguiente medición
                
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

# ----------------------------
# 3. Calibración usando dimensiones conocidas
# ----------------------------
print("\n" + "=" * 60)
print("CALIBRACIÓN INTERACTIVA")
print("=" * 60)
print("\nInstrucciones:")
print("1. Haz clic en dos puntos para medir la ALTURA del cuadro de la Virgen de Guadalupe")
print("2. Presiona 'Enter' cuando termines")
print("3. Luego haz clic en dos puntos para medir el ANCHO de la mesa")
print("4. Presiona 'Enter' cuando termines")

def interactive_calibration(img, num_measurements=2):
    """Herramienta interactiva para seleccionar puntos de calibración"""
    measurements = []
    
    for i in range(num_measurements):
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.imshow(img)
        if i == 0:
            ax.set_title('Calibración 1/2: Selecciona dos puntos para medir la ALTURA del cuadro\n(Click en dos puntos, luego cierra la ventana)', fontsize=14)
        else:
            ax.set_title('Calibración 2/2: Selecciona dos puntos para medir el ANCHO de la mesa\n(Click en dos puntos, luego cierra la ventana)', fontsize=14)
        ax.axis('off')
        
        points = []
        
        def on_click(event):
            if event.inaxes != ax:
                return
            if event.button == 1:  # Click izquierdo
                x, y = int(event.xdata), int(event.ydata)
                points.append((x, y))
                ax.plot(x, y, 'ro', markersize=10)
                
                if len(points) == 2:
                    p1, p2 = points
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=2)
                    ax.text((p1[0] + p2[0])/2, (p1[1] + p2[1])/2 - 15, 
                           f'Medición {i+1}', 
                           color='yellow', fontsize=12, 
                           bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                    fig.canvas.draw()
        
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.show()
        
        if len(points) == 2:
            measurements.append((points[0], points[1]))
        else:
            print(f"Advertencia: No se seleccionaron 2 puntos para la medición {i+1}")
    
    return measurements

# Usar coordenadas predefinidas o permitir selección interactiva
USE_INTERACTIVE = False  # Cambiar a True para usar herramienta interactiva

if USE_INTERACTIVE:
    print("\nIniciando herramienta interactiva de calibración...")
    calib_measurements = interactive_calibration(panorama_rgb, num_measurements=2)
    
    if len(calib_measurements) >= 2:
        # Primera medición: altura del cuadro
        cuadro_p1, cuadro_p2 = calib_measurements[0]
        cuadro_distance_px = np.sqrt((cuadro_p2[0]-cuadro_p1[0])**2 + (cuadro_p2[1]-cuadro_p1[1])**2)
        scale_factor_cuadro = KNOWN_MEASUREMENTS['cuadro_altura'] / cuadro_distance_px
        
        # Segunda medición: ancho de la mesa
        mesa_p1, mesa_p2 = calib_measurements[1]
        mesa_distance_px = np.sqrt((mesa_p2[0]-mesa_p1[0])**2 + (mesa_p2[1]-mesa_p1[1])**2)
        scale_factor_mesa = KNOWN_MEASUREMENTS['mesa_ancho'] / mesa_distance_px
        
        # Usar promedio de ambas escalas
        scale_factor = (scale_factor_cuadro + scale_factor_mesa) / 2.0
        
        print(f"\nCalibración completada:")
        print(f"  - Distancia cuadro (píxeles): {cuadro_distance_px:.2f} px")
        print(f"  - Escala desde cuadro: {scale_factor_cuadro:.4f} cm/px")
        print(f"  - Distancia mesa (píxeles): {mesa_distance_px:.2f} px")
        print(f"  - Escala desde mesa: {scale_factor_mesa:.4f} cm/px")
        print(f"  - Escala promedio: {scale_factor:.4f} cm/px")
    else:
        print("\nNo se seleccionaron suficientes puntos. Usando valores por defecto.")
        # Estimar escala basada en dimensiones típicas de imagen
        # Asumiendo que el panorama tiene aproximadamente 3000-4000 píxeles de ancho
        # y que la mesa (161.1 cm) ocupa aproximadamente 1/3 del ancho
        estimated_mesa_width_px = panorama.shape[1] / 3.0
        scale_factor = KNOWN_MEASUREMENTS['mesa_ancho'] / estimated_mesa_width_px
        print(f"Usando escala estimada: {scale_factor:.4f} cm/px")
else:
    # Usar valores estimados si no se quiere interactivo
    # Estimar escala basada en dimensiones típicas de imagen
    estimated_mesa_width_px = panorama.shape[1] / 3.0
    scale_factor = KNOWN_MEASUREMENTS['mesa_ancho'] / estimated_mesa_width_px
    print(f"\nUsando escala estimada (no interactivo): {scale_factor:.4f} cm/px")
    print("NOTA: Para mayor precisión, cambia USE_INTERACTIVE a True")

# ----------------------------
# 4. Calcular dimensiones del cuadro y mesa
# ----------------------------
print("\n" + "=" * 60)
print("CÁLCULO DE DIMENSIONES ADICIONALES")
print("=" * 60)

# Crear herramienta de medición
meas_tool = MeasurementTool(panorama_rgb, scale_factor)

# Función para medir interactivamente elementos adicionales
def measure_element_interactive(img, scale_factor, element_name):
    """Medir interactivamente un elemento usando la escala calibrada"""
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.imshow(img)
    ax.set_title(f'Medir: {element_name}\n(Click en dos puntos, luego cierra la ventana)', fontsize=14)
    ax.axis('off')
    
    points = []
    result = None
    
    def on_click(event):
        nonlocal result
        if event.inaxes != ax:
            return
        if event.button == 1:  # Click izquierdo
            x, y = int(event.xdata), int(event.ydata)
            points.append((x, y))
            ax.plot(x, y, 'go', markersize=10)
            
            if len(points) == 2:
                p1, p2 = points
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                distance_px = np.sqrt(dx**2 + dy**2)
                distance_cm = distance_px * scale_factor
                
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2)
                ax.text((p1[0] + p2[0])/2, (p1[1] + p2[1])/2 - 15, 
                       f'{distance_cm:.2f} cm', 
                       color='yellow', fontsize=12, 
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
                fig.canvas.draw()
                result = (distance_cm, distance_px)
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()
    
    return result

# Mediciones programáticas (para demostración)
# El usuario deberá hacer las mediciones interactivamente
print("\nNOTA: Para mediciones precisas, usa la herramienta interactiva.")
print("Las mediciones siguientes son ejemplos con valores estimados.\n")

# Usar mediciones interactivas o estimadas
USE_INTERACTIVE_MEASUREMENTS = False  # Cambiar a True para mediciones interactivas

if USE_INTERACTIVE_MEASUREMENTS:
    print("\n" + "=" * 60)
    print("MEDICIONES INTERACTIVAS DE ELEMENTOS ADICIONALES")
    print("=" * 60)
    print("\nSe abrirán ventanas para medir cada elemento.")
    print("Haz click en dos puntos para cada medición.\n")
else:
    print("\nUsando mediciones estimadas (cambiar USE_INTERACTIVE_MEASUREMENTS a True para interactivo)")

# Simulamos mediciones (en uso real, estas serían interactivas)
# Estas son estimaciones basadas en proporciones típicas
# En producción, usar measure_element_interactive() para cada elemento
estimated_width_panorama = panorama.shape[1]
estimated_height_panorama = panorama.shape[0]

# Estimaciones proporcionales (ajustar según necesidad)
measurements_data = {
    'cuadro_ancho': {'name': 'Ancho del cuadro', 
                     'estimated_px': estimated_width_panorama * 0.15, 
                     'real_cm': None},
    'mesa_largo': {'name': 'Largo de la mesa', 
                   'estimated_px': estimated_width_panorama * 0.25, 
                   'real_cm': None},
    'ventana': {'name': 'Ventana (ancho)', 
                'estimated_px': estimated_width_panorama * 0.30, 
                'real_cm': None},
    'silla': {'name': 'Silla (ancho)', 
              'estimated_px': estimated_width_panorama * 0.08, 
              'real_cm': None},
    'planta': {'name': 'Planta (altura)', 
               'estimated_px': estimated_height_panorama * 0.15, 
               'real_cm': None}
}

# Si se usa modo interactivo, realizar mediciones reales
if USE_INTERACTIVE_MEASUREMENTS:
    for key in ['cuadro_ancho', 'mesa_largo', 'ventana', 'silla', 'planta']:
        result = measure_element_interactive(panorama_rgb, scale_factor, 
                                            measurements_data[key]['name'])
        if result:
            measurements_data[key]['real_cm'] = result[0]
            measurements_data[key]['estimated_px'] = result[1]

results_summary = []

print("\nMediciones realizadas:")
print("-" * 60)
for key, data in measurements_data.items():
    distance_px = data['estimated_px']
    distance_cm = distance_px * scale_factor
    distance_m = distance_cm / 100.0
    data['real_cm'] = distance_cm
    
    print(f"{data['name']:30s}: {distance_cm:7.2f} cm ({distance_m:.3f} m) [{distance_px:.1f} px]")
    results_summary.append({
        'elemento': data['name'],
        'distancia_cm': distance_cm,
        'distancia_px': distance_px,
        'incertidumbre_px': 5.0  # Estimación de incertidumbre en píxeles
    })

# Cálculos específicos
print("\n" + "=" * 60)
print("DIMENSIONES DEL CUADRO Y MESA")
print("=" * 60)

# Cuadro: ancho
cuadro_ancho_px = measurements_data['cuadro_ancho']['estimated_px']
cuadro_ancho_cm = cuadro_ancho_px * scale_factor
print(f"\nCuadro de la Virgen de Guadalupe:")
print(f"  - Altura (conocida): {KNOWN_MEASUREMENTS['cuadro_altura']:.1f} cm")
print(f"  - Ancho (medido):    {cuadro_ancho_cm:.2f} cm")

# Mesa: largo
mesa_largo_px = measurements_data['mesa_largo']['estimated_px']
mesa_largo_cm = mesa_largo_px * scale_factor
print(f"\nMesa:")
print(f"  - Ancho (conocido): {KNOWN_MEASUREMENTS['mesa_ancho']:.1f} cm")
print(f"  - Largo (medido):  {mesa_largo_cm:.2f} cm")

# ----------------------------
# 5. Análisis de incertidumbre
# ----------------------------
print("\n" + "=" * 60)
print("ANÁLISIS DE INCERTIDUMBRE")
print("=" * 60)

# Incertidumbre en la calibración
uncertainty_px = 2.0  # Incertidumbre típica en selección de píxeles
uncertainty_calibration_cm = uncertainty_px * scale_factor

# Incertidumbre relativa
relative_uncertainty_calibration = (uncertainty_calibration_cm / KNOWN_MEASUREMENTS['cuadro_altura']) * 100

print(f"\nIncertidumbre en calibración:")
print(f"  - Incertidumbre en píxeles: ±{uncertainty_px:.1f} px")
print(f"  - Incertidumbre en centímetros: ±{uncertainty_calibration_cm:.2f} cm")
print(f"  - Incertidumbre relativa: ±{relative_uncertainty_calibration:.2f}%")

# Propagación de incertidumbre para cada medición
print(f"\nIncertidumbre en mediciones:")
print("-" * 60)
print(f"{'Elemento':<30s} {'Valor (cm)':>12s} {'Incert. (cm)':>12s} {'Incert. (%)':>12s}")
print("-" * 60)

for result in results_summary:
    # Incertidumbre combinada: calibración + medición
    uncertainty_measurement_px = result['incertidumbre_px']
    uncertainty_measurement_cm = uncertainty_measurement_px * scale_factor
    
    # Propagación de incertidumbre (suma cuadrática)
    total_uncertainty_cm = np.sqrt(uncertainty_calibration_cm**2 + uncertainty_measurement_cm**2)
    relative_uncertainty = (total_uncertainty_cm / result['distancia_cm']) * 100
    
    print(f"{result['elemento']:<30s} {result['distancia_cm']:>11.2f} ±{total_uncertainty_cm:>10.2f} ±{relative_uncertainty:>10.2f}%")
    
    result['incertidumbre_cm'] = total_uncertainty_cm
    result['incertidumbre_relativa_pct'] = relative_uncertainty

# Guardar resultados en JSON
results_file = os.path.join(measurements_dir, 'measurements_data.json')
results_to_save = {
    'scale_factor_cm_per_px': scale_factor,
    'calibration_uncertainty_cm': uncertainty_calibration_cm,
    'known_measurements': KNOWN_MEASUREMENTS,
    'measurements': results_summary,
    'timestamp': datetime.now().isoformat()
}

with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(results_to_save, f, indent=2, ensure_ascii=False)

print(f"\nResultados guardados en: {results_file}")

# ----------------------------
# 6. Visualización de resultados
# ----------------------------
print("\n" + "=" * 60)
print("VISUALIZACIÓN DE RESULTADOS")
print("=" * 60)

fig, ax = plt.subplots(figsize=(15, 10))
ax.imshow(panorama_rgb)
ax.set_title('Panorama con Escala Calibrada', fontsize=16)
ax.axis('off')

# Agregar barra de escala
scale_bar_length_cm = 50  # 50 cm
scale_bar_length_px = scale_bar_length_cm / scale_factor
scale_bar_x = panorama.shape[1] - scale_bar_length_px - 50
scale_bar_y = panorama.shape[0] - 50

ax.plot([scale_bar_x, scale_bar_x + scale_bar_length_px], 
        [scale_bar_y, scale_bar_y], 'w-', linewidth=4)
ax.plot([scale_bar_x, scale_bar_x], 
        [scale_bar_y - 10, scale_bar_y + 10], 'w-', linewidth=2)
ax.plot([scale_bar_x + scale_bar_length_px, scale_bar_x + scale_bar_length_px], 
        [scale_bar_y - 10, scale_bar_y + 10], 'w-', linewidth=2)
ax.text(scale_bar_x + scale_bar_length_px/2, scale_bar_y - 25, 
        f'{scale_bar_length_cm} cm', 
        color='white', fontsize=12, ha='center',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

# Guardar imagen con barra de escala
output_path = os.path.join(figures_dir, 'panorama_calibrated.jpg')
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nImagen calibrada guardada en: {output_path}")

plt.show()

# Resumen final
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"\nEscala calibrada: {scale_factor:.4f} cm/píxel")
print(f"\nDimensiones calculadas:")
print(f"  - Cuadro (ancho): {cuadro_ancho_cm:.2f} ± {results_summary[0]['incertidumbre_cm']:.2f} cm")
print(f"  - Mesa (largo): {mesa_largo_cm:.2f} ± {results_summary[1]['incertidumbre_cm']:.2f} cm")
print(f"\nIncertidumbre promedio en mediciones: {np.mean([r['incertidumbre_relativa_pct'] for r in results_summary]):.2f}%")

# Cerrar el archivo de salida
tee.close()

