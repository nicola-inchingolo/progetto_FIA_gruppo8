import pandas as pd
import numpy as np

df = pd.read_csv('data/dati_fia.csv')

print("Riepilogo informazioni del DataFrame:")
# fornisce il riepilogo delle informazioni
df.info()
print("\n" + "="*50 + "\n")

print("Colonne con valori mancanti:")
# conta quanti record sono null in una colonna
valori_nulli_per_colonna = df.isnull().sum()
# colonne con valori nulli
colonne_con_nulli = valori_nulli_per_colonna[valori_nulli_per_colonna > 0]
print(colonne_con_nulli)
print("\n" + "="*50 + "\n")

print("### **Descrizione statistica del Dataframe:**")
# stampa i dati statistici sia delle colonne numeriche che di quelle categoriche
print(df.describe(include='all'))
print("\n" + "="*50 + "\n")

print("Riassunto:")
dimensioni = df.shape
tipi_di_dati = df.dtypes.unique()
print(f"""Il DataFrame contiene {dimensioni[1]} colonne e {dimensioni[0]} righe.
I possibili tipi di dati sono: {tipi_di_dati}.
Sono presenti {colonne_con_nulli.sum()} valori nulli.""")

print("\n" + "="*50 + "\n")

print("Stampa delle prime 5 righe:")
# stampa le prime righe del dataframe (giÃ  5 sono di default)
print(df.head())





# Ottenere una lista delle colonne con dtype 'O'
colonne_oggetto = df.select_dtypes(include=['object']).columns

# DataFrame solo con le colonne filtrate
df_oggetti = df[colonne_oggetto]

# Visualizzare il risultato
print("Colonne filtrate (dtype='O'):")
print(colonne_oggetto)
print("\nPrime righe del nuovo DataFrame con sole colonne 'object':")
print(df_oggetti.head())