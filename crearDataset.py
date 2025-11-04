import os
import shutil
import glob

ruta_base = r'C:\Lucas\Ingenieria_Informatica\Inteligencia_Computacional\TP\Codigo\POP909-Dataset\POP909'
ruta_destino = os.path.join(ruta_base, 'midi_files')

os.makedirs(ruta_destino, exist_ok=True)

# Buscar en todas las carpetas numeradas
for i in range(1, 910):
    nombre_carpeta = f"{i:03d}"
    ruta_carpeta = os.path.join(ruta_base, nombre_carpeta)
    
    if os.path.exists(ruta_carpeta):
        # Buscar cualquier archivo MIDI en la carpeta
        patron_midi = os.path.join(ruta_carpeta, "*.mid*")
        archivos_midi = glob.glob(patron_midi)
        
        if archivos_midi:
            for archivo in archivos_midi:
                nombre_archivo = os.path.basename(archivo)
                archivo_destino = os.path.join(ruta_destino, nombre_archivo)
                shutil.copy2(archivo, archivo_destino)
                print(f"Copiado: {nombre_archivo}")
        else:
            print(f"No se encontraron archivos MIDI en: {nombre_carpeta}")

print("Proceso completado!")