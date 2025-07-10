import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Passo 1: Ler o Dataset
dataset_url = "https://archive.ics.uci.edu/static/public/176/data.csv"
df = pd.read_csv(dataset_url)

# Exibir as primeiras linhas para verificar o carregamento
print("\nPrimeiras linhas do dataset:")
print(df.head())

# Passo 2: Análise Exploratória dos Dados (EDA)
print("\nResumo estatístico do dataset:")
print(df.describe())

# Contar doadores (1) e não doadores (0)
print("\nDistribuição da variável alvo (Donated_Blood):")
print(df['Donated_Blood'].value_counts())

# Passo 3: Visualizar os Dados
plt.figure(figsize=(12, 6))
df.hist(bins=20, figsize=(12, 6), edgecolor='black')
plt.suptitle("Distribuições das variáveis", fontsize=14)
plt.show()

# Boxplot de Monetary por Donated_Blood
plt.figure(figsize=(6, 4))
sns.boxplot(x='Donated_Blood', y='Monetary', data=df)
plt.title("Boxplot de Monetary por Donated_Blood")
plt.show()

# Passo 4: Análise Estatística
# ANOVA para verificar diferenças na Frequency entre doadores e não doadores
model = ols('Frequency ~ C(Donated_Blood)', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print("\nResultados da ANOVA para Frequency:")
print(anova_table)

# Teste t para comparar Monetary entre doadores e não doadores
doadores = df[df['Donated_Blood'] == 1]['Monetary']
nao_doadores = df[df['Donated_Blood'] == 0]['Monetary']
t_stat, p_value = stats.ttest_ind(doadores, nao_doadores)
print("\nTeste t para Monetary entre doadores e não doadores:")
print(f"Estatística t: {t_stat:.4f}, Valor p: {p_value:.4f}")

# Se ANOVA for significativa, realizar teste de Tukey
tukey = sm.stats.multicomp.pairwise_tukeyhsd(df['Frequency'], df['Donated_Blood'])
print("\nTeste de Tukey para Frequency:")
print(tukey)

# Passo 5: Interpretação dos Resultados
# Os valores de p indicarão se há diferenças estatisticamente significativas entre os grupos
if anova_table['PR(>F)'][0] < 0.05:
    print("\nA ANOVA indicou uma diferença significativa em Frequency entre doadores e não doadores.")
else:
    print("\nA ANOVA não indicou diferença significativa em Frequency entre os grupos.")

if p_value < 0.05:
    print("O teste t indicou uma diferença significativa em Monetary entre os grupos.")
else:
    print("O teste t não indicou diferença significativa em Monetary entre os grupos.")
