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
- [Contributi](#contributi)
- [Licenza](#licenza)

---

## Installazione

1. Clona il repository:
   ```bash
   git clone <URL_DEL_REPO>


## Uso

Esempio di esecuzione del programma principale:
   ```bash
   python model_evaluation/evaluators_test.py
  ```
Durante l'esecuzione verranno richieste alcune scelte:

- Metriche da calcolare (Accuracy, Sensitivity, AUC, ecc.)
- Strategia di distanza per KNN (Manhattan o Euclidean)
- Numero di vicini k
- Tipo di validazione:
  - Holdout
  - K-Fold
  - Leave-One-Out

## Funzionalità

- Lettura e gestione del dataset di tumori.
- Suddivisione automatica dei dati per validazione.

Valutazione del modello KNN tramite varie metriche:
- Accuracy, Error Rate, Sensitivity, Specificity, Geometric Mean, AUC.

Visualizzazione grafica dei risultati:
- Confusion matrix
- ROC curve

Supporto a più strategie di validazione:
- Holdout
- K-Fold
- Leave-One-Out

## Architettura del progetto

Architettura del progetto

Il progetto è stato sviluppato da tre persone con suddivisione del lavoro in tre aree principali:

Data Processing
  - Preparazione del dataset
  - Pulizia e normalizzazione dei dati

Model Development
  - Implementazione dell'algoritmo KNN
  - Funzioni di addestramento e predizione

Model Evaluation
  - Applicazione del Factory Method Pattern
  - Creazione di una classe astratta Evaluator con attributi e metodi generici

Tre sottoclassi:

  - holdout_evaluator
  - k_fold_evaluator
  - leave_one_out_evaluator

Ogni sottoclasse implementa metodi specifici per la rispettiva strategia di validazione

EvaluatorFactory per creare l’oggetto specifico richiesto dall’utente

Questo approccio permette di mantenere il codice modulare e facilmente estendibile per altre strategie di validazione o metriche future.
