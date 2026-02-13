# Progetto Classificazione Tumori

Il progetto ha l'obiettivo di sviluppare un modello di **apprendimento automatico** per la classificazione dei tumori, basandosi su un dataset fornito.  
L'analisi permette di distinguere tra tumori benigni e maligni utilizzando diverse caratteristiche dei campioni.

L'algoritmo principale utilizzato è **KNN (K-Nearest Neighbors)**, e il progetto prevede l'uso di diversi tipi di split per la **validazione del dataset**.

---

## Indice
- [Installazione](#installazione)
- [Uso](#uso)
- [Funzionalità](#funzionalità)
- [Architettura del progetto](#architettura-del-progetto)
- [Utilizzo con Docker](#Utilizzo-con-Docker)
- [Contributi](#contributi)
- [Licenza](#licenza)

---

## Installazione

1. Clona il repository:
```bash
   git clone <URL_DEL_REPO>
```
2. Installa le dipendenze:
```bash
   pip install -r requirements.txt
```

## Uso

Per avviare il sistema, eseguire il file principale del progetto:
```bash
   python main.py
```
Durante l'esecuzione verranno richieste alcune scelte:

- Metriche da calcolare:
  - Accuracy
  - Error Rate
  - Sensitivity
  - Specificity
  - Geomettric Mean
  - Area under the curve 
  - All the above

- Strategia di distanza per KNN:
  - Manhattan
  - Euclidean
  
- Numero di vicini k

- Tipo di validazione:
  - Holdout
  - K-Fold
  - Leave-One-Out

##  Funzionalità

Il sistema offre una suite completa per l'analisi e la classificazione, strutturata in quattro macro-aree:

1. Gestione e Preprocessing dei Dati
    - **Pipeline Automatizzata**: Caricamento, pulizia e normalizzazione del dataset clinico.
    - **Data Imputation**: Gestione avanzata dei valori mancanti e rimozione degli outlier per garantire la qualità del training set.
    - **Suddivisione Dinamica**: Split automatico dei dati in base alla strategia di validazione scelta.

2. Configurazione del Modello (KNN)
    - **Algoritmo K-Nearest Neighbors**: Implementazione flessibile con parametro **k** (vicini) personalizzabile.
    - **Strategie di Distanza**: Supporto duale per il calcolo della similarità:
        - **Distanza Euclidea**
        - **Distanza di Manhattan**

3. Strategie di Validazione:
    - **Holdout**: Divisione classica Training/Test con percentuale personalizzabile (es. 70/30).
    - **K-Fold Cross Validation**: Validazione incrociata a $K$ segmenti per ridurre la varianza.
    - **Leave-One-Out**: Validazione esaustiva, ideale per dataset di dimensioni contenute.

4. Metriche e Visualizzazione delle Performance:
    - **Indicatori Numerici**: Calcolo di *Accuracy, Error Rate, Sensitivity, Specificity, Geometric Mean* e *AUC (Area Under Curve)*.
    - **Grafici Generati**:
        - **Confusion Matrix**: Per visualizzare falsi positivi/negativi.
        - **ROC Curve**: Per analizzare la capacità discriminante del classificatore.

## Architettura del progetto

Architettura del progetto

Il progetto è stato sviluppato da tre persone con suddivisione del lavoro in tre aree principali:

Data Processing
  - Preparazione del dataset
  - Pipeline di pulizia (cleaning) e normalizzazione dei dati

Model Development
  - Implementazione dell'algoritmo **KNN (K-Nearest Neighbors)**.
  - Sviluppo delle funzioni ottimizzate per l'addestramento e la predizione.

Model Evaluation
  - Implementazione di  **Evaluator (Classe Astratta)**: Definisce l'interfaccia base con attributi e metodi generici.
  - Implementazione di 3 **Sottoclassi Concrete**: Ogni classe implementa la logica specifica per una strategia di validazione:
        - holdout_evaluator
        - k_fold_evaluator
        - leave_one_out_evaluator
  - Implementazione di **EvaluatorFactory**: Classe factory che istanzia dinamicamente l'oggetto valutatore specifico richiesto dall'utente.

Ogni sottoclasse implementa metodi specifici per la rispettiva strategia di validazione

EvaluatorFactory per creare l’oggetto specifico richiesto dall’utente

Questo approccio permette di mantenere il codice modulare e facilmente estendibile per altre strategie di validazione o metriche future.

##  Utilizzo con Docker

Il progetto è containerizzato per garantire la massima riproducibilità senza problemi di dipendenze.

1.  **Costruisci l'immagine**:
```bash
   docker build -t gruppo8-tumor-classifier .
```

2.  **Esegui il container**:
```bash
   docker run -it gruppo8-tumor-classifier
```
**Nota: L'opzione `-it` è fondamentale per interagire con il menu del programma.**

## Contributi
Progetto sviluppato dal Gruppo 8:

- Data Processing: Preparazione dataset, pulizia e normalizzazione.

- Model Development: Implementazione algoritmo KNN e logica di predizione.

- Model Evaluation: Implementazione del Factory Method, metriche e visualizzazione.

## Licenza
**Questo progetto è distribuito a scopo accademico.**

