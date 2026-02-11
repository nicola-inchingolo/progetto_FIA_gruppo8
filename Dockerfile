# Usa un'immagine base ufficiale di Python
FROM python:3.12-slim

# Imposta la directory di lavoro all'interno del container
WORKDIR /app

# Copia il file dei requisiti per installare le dipendenze
COPY requirements.txt .

# Installa le dipendenze di sistema necessarie per alcune librerie grafiche e le dipendenze Python
RUN apt-get update && apt-get install -y \
    && pip install --no-cache-dir -r requirements.txt


# Copia tutto il resto del codice sorgente nel container
COPY . .

# Crea le cartelle per l'input e l'output (verranno usate come punti di montaggio)
RUN mkdir -p /app/data /app/output_plots /app/output_result

# Comando per avviare l'applicazione. 
# Si usa -u per forzare l'output non bufferizzato (utile per vedere i log in tempo reale)
CMD ["python", "-u", "main.py"]
