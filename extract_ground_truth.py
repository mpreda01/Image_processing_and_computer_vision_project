import json
import cv2
import numpy as np

def calculate_polygon_area(points):
    """
    Calcola l'area di un poligono dati i suoi punti.
    
    Args:
        points: Lista di coordinate [[x1, y1], [x2, y2], ...]
    
    Returns:
        Area del poligono in pixel
    """
    # Converti in formato numpy array per cv2.contourArea
    points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    area = cv2.contourArea(points_array)
    return abs(area)  # Assicurati che sia positivo

def extract_ground_truth_areas(json_file_path):
    """
    Estrae le aree totali per ogni scena dal file JSON labels.
    
    Args:
        json_file_path: Path al file labels.json
    
    Returns:
        Dizionario {scene_index: total_area}
    """
    # Carica il JSON
    with open(json_file_path, 'r') as f:
        labels = json.load(f)
    
    ground_truth_areas = {}
    
    # Processa ogni scena
    for scene_name, scene_data in labels.items():
        # Estrai l'indice della scena dal nome (es: "scene_1.jpg" -> 1)
        scene_index = int(scene_name.split('_')[1].split('.')[0]) - 1
        
        total_area = 0
        
        # Calcola l'area per ogni regione nella scena
        for region in scene_data['regions']:
            points = region['points']
            area = calculate_polygon_area(points)
            total_area += area
        
        ground_truth_areas[scene_index] = total_area
    
    return ground_truth_areas

def print_ground_truth_dict(ground_truth_areas):
    """
    Stampa il dizionario in formato Python copiabile.
    """
    print("ground_truth_areas = {")
    
    # Ordina per chiave (scene index)
    for scene_idx in sorted(ground_truth_areas.keys()):
        area = ground_truth_areas[scene_idx]
        print(f"    {scene_idx}: {area:.0f},  # Scene {scene_idx}: area totale attesa in pixel")
    
    print("}")

# UTILIZZO
if __name__ == "__main__":
    # Path al file JSON
    json_file_path = "labels.json"
    
    # Estrai le aree
    ground_truth_areas = extract_ground_truth_areas(json_file_path)
    
    # Stampa statistiche
    print(f"Trovate {len(ground_truth_areas)} scene nel file labels.json\n")
    
    # Stampa il dizionario formattato
    print_ground_truth_dict(ground_truth_areas)
    
    # Statistiche aggiuntive
    print(f"\n=== STATISTICHE ===")
    print(f"Area minima: {min(ground_truth_areas.values()):.0f} pixels")
    print(f"Area massima: {max(ground_truth_areas.values()):.0f} pixels")
    print(f"Area media: {sum(ground_truth_areas.values()) / len(ground_truth_areas):.0f} pixels")
    
    # Conta modelli per scena
    with open(json_file_path, 'r') as f:
        labels = json.load(f)
    
    print(f"\n=== MODELLI PER SCENA ===")
    for scene_name in sorted(labels.keys(), key=lambda x: int(x.split('_')[1].split('.')[0])):
        scene_idx = int(scene_name.split('_')[1].split('.')[0])
        count_info = labels[scene_name]['count']
        total_models = sum(count_info.values())
        print(f"Scene {scene_idx}: {total_models} libri totali - {count_info}")