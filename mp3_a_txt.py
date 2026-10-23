import os
from faster_whisper import WhisperModel

def transcribir_mp3(ruta_mp3: str, ruta_salida_txt: str, ruta_modelo_local: str, idioma: str = "es"):
    """
    Transcribe un archivo MP3 a texto usando faster-whisper con aceleracion por GPU.
    """
    
    if not os.path.isfile(ruta_mp3):
        print(f"Error: El archivo de audio '{ruta_mp3}' no existe en la carpeta actual.")
        return

    print(f"Archivo de audio: {ruta_mp3}")
    print(f"Cargando modelo desde: {ruta_modelo_local}")
    print("Esto puede tardar un poco la primera vez...")

    try:
        # --- CONFIGURACION PARA GPU ---
        # device="cuda" fuerza el uso de la tarjeta grafica
        # compute_type="float16" usa media precision (ideal para GPU)
        model = WhisperModel(ruta_modelo_local, device="cuda", compute_type="float16")
        
        print("Modelo cargado en GPU exitosamente. Iniciando transcripcion...")

        segments, info = model.transcribe(ruta_mp3, beam_size=5, language=idioma)
        
        print(f"Idioma detectado: '{info.language}' (Probabilidad: {info.language_probability:.2f})")
        print(f"Duracion del audio: {info.duration:.2f} segundos")
        print("-" * 50)

        with open(ruta_salida_txt, "w", encoding="utf-8") as f:
            for segmento in segments:
                linea = f"[{segmento.start:.2f}s -> {segmento.end:.2f}s] {segmento.text}\n"
                f.write(linea)
                print(linea, end="")

        print("-" * 50)
        print(f"Transcripcion completada. Archivo guardado en: {ruta_salida_txt}")

    except Exception as e:
        print(f"Ocurrio un error durante la transcripcion: {e}")
        print("Sugerencia: Verifica que las librerias CUDA esten instaladas y copiadas correctamente.")

if __name__ == "__main__":
    archivo_entrada = "7.mp3" 
    archivo_salida = "7_transcripcion.txt"
    ruta_modelo = r"C:\Users\ESTIVEN MARTINEZ\Desktop\modelo_whisper_small"
    idioma_audio = "es" 
    
    transcribir_mp3(
        ruta_mp3=archivo_entrada, 
        ruta_salida_txt=archivo_salida, 
        ruta_modelo_local=ruta_modelo,
        idioma=idioma_audio
    )