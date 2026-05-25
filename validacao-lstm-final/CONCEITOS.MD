# Explicação dos parâmetros de comparação

## Colunas de Identificação

(DE ONDE FOI RETIRADO ESSE MODELO DE LSTM)

### `Variante`
O pipeline de pré-processamento testado. Cada variante é **idêntica** em arquitetura de rede (LSTM 32 units, dropout 0.1, lookback 20) — a única coisa que muda é como os dados de entrada são preparados. Isso é um **ablation study de pré-processamento**.

### `Normalizer`
Como os valores das 18 features foram escalonados antes de entrar na rede:
- **Z-score**: subtrai a média e divide pelo desvio padrão do **treino** → cada feature fica com média 0, std 1
- **Min-Max**: escala para [0, 1] usando mín/máx do **treino**
- **Rolling Z-score**: Z-score causal — usa apenas os 252 dias anteriores a cada ponto t (1 ano de trading)
- **MinMax/ch**: Min-Max calculado **por canal** individualmente (variante F com 21 features)

> Em todas as variantes, os parâmetros de normalização são calculados **somente no treino** para evitar vazamento.

---

## Colunas de Acurácia

### `Acc Train` (Accuracy no treino)
% de dias acertados nos ~5.205 dias de treino (2000–2020). **Não é o indicador mais importante** — um modelo pode memorizar o treino e ainda assim ser inútil no futuro.

### `Acc Val` (Accuracy na validação)
% de dias acertados nos ~503 dias de validação (2021–2022). Usado pelo early stopping para decidir quando parar o treino. **Mais honesto que o treino**, mas ainda não é o número final.

### `Acc Test` (Accuracy no teste)
**O único número que importa para avaliar o modelo.** % de dias acertados nos ~482 dias de teste (2023–2024) — dados que o modelo nunca viu durante o treino ou validação. O S&P500 sobe ~55% dos dias, então:
- **55%** = chute (dizer "sobe todo dia" dá 55%)
- **60%** = sinal fraco, mas real
- **67%** (variante B) = resultado sólido

---

## Colunas de Métricas Finais (Test Set)

### `F1 Test`
**Média harmônica entre precisão e recall**, calculada para a classe 1 (sobe).

```
Precisão = TP / (TP + FP)   → dos dias que previu "sobe", quantos realmente subiram?
Recall   = TP / (TP + FN)   → dos dias que realmente subiram, quantos previu corretamente?
F1       = 2 * (P * R) / (P + R)
```

**Armadilha**: a variante F tem F1=0.720 mas MCC=−0.039. Isso acontece porque o F1 só mede a classe 1 — se o modelo chuta "sobe" para **todos** os dias, o recall é 100% e o F1 fica alto, mesmo sem aprender nada. **Por isso F1 sozinho é enganoso para dados desbalanceados.**

### `MCC Test` — **a métrica mais importante**
**Matthews Correlation Coefficient** — correlação entre o que o modelo previu e o que realmente aconteceu, levando em conta as 4 células da matriz de confusão (TP, TN, FP, FN).

```
MCC = (TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
```

| Valor | Interpretação |
|---|---|
| **+1.0** | Perfeito — acertou tudo |
| **+0.3** | Sinal real moderado (como variante B) |
| **+0.1 a +0.2** | Sinal fraco mas presente |
| **≈ 0** | Equivalente a chute aleatório |
| **< 0** | Pior que chute — modelo invertido |

É a **única métrica que penaliza igualmente errar "sobe" e errar "cai"**. Para o S&P500 onde 55% dos dias sobem, um modelo que chuta sempre sobe terá Acc=55%, F1 alto, mas **MCC=0**.

### `AUC Test` (Area Under ROC Curve)
Mede a **qualidade da probabilidade** P(sobe) antes de aplicar o threshold de 0.5. Representa: se pegar um dia aleatório que subiu e um que caiu, qual a probabilidade de o modelo atribuir P(sobe) maior ao dia que subiu?

| Valor | Interpretação |
|---|---|
| **1.0** | Perfeito — sempre distingue as classes |
| **0.724** (B) | 72% das vezes o ranking está correto |
| **0.5** | Aleatório — probabilidades sem informação |
| **< 0.5** | Modelo invertido |

AUC é útil porque avalia o **ranking das probabilidades**, não só a decisão binária. Um modelo com bom AUC pode ser calibrado para diferentes thresholds (ex: só operar quando P(sobe) > 0.7).

A Curva ROC é construída varrendo todos os possíveis thresholds de decisão de P(sobe) (de 0 a 1) e medindo, para cada threshold, dois números:

- **TPR (True Positive Rate / Recall / Sensibilidade)**: dos dias que realmente subiram, qual fração o modelo identificou corretamente?
  ```
  TPR = TP / (TP + FN)
  ```
- **FPR (False Positive Rate / 1 − Especificidade)**: dos dias que realmente caíram, qual fração o modelo classificou erroneamente como "sobe"?
  ```
  FPR = FP / (FP + TN)
  ```

**Exemplo com threshold variando**:

| Threshold | Prevê "sobe" quando | TPR | FPR | Interpretação |
|---|---|---|---|---|
| 0.0 | sempre | 1.0 | 1.0 | Diz "sobe" pra tudo — captura tudo, mas alarma tudo |
| 0.5 | P > 0.5 | ~0.85 | ~0.45 | Ponto de operação padrão |
| 0.8 | P > 0.8 | ~0.40 | ~0.05 | Conservador — só aposta quando muito confiante |
| 1.0 | nunca | 0.0 | 0.0 | Nunca diz "sobe" — não captura nada |

Cada threshold gera um ponto (FPR, TPR). Ligando todos esses pontos forma-se a **curva ROC**. A **AUC é a área sob essa curva**, calculada numericamente pela regra trapezoidal:

```
AUC = Σ (FPR[i+1] − FPR[i]) × (TPR[i+1] + TPR[i]) / 2
```

**Interpretação geométrica**:
- Diagonal (AUC=0.5): modelo aleatório — curva fica na diagonal
- Curva acima da diagonal (AUC>0.5): modelo aprende algo
- Canto superior esquerdo (AUC=1.0): perfeito — TPR=1 com FPR=0

**Interpretação probabilística equivalente**: AUC = P(score(dia_sobe) > score(dia_cai)) para um par aleatório de dias. Com AUC=0.724 (variante B), em 72.4% dos pares aleatórios (um dia que subiu, um que caiu), o modelo atribui P(sobe) maior ao dia correto.

---

## Colunas de Diagnóstico

### `McNemar p` — teste estatístico
Testa se o modelo é **estatisticamente diferente de chutar sempre a classe majoritária** (dizer "sobe" todo dia). A hipótese nula é "o modelo acerta tanto quanto o chute".

```
p < 0.05  → rejeita hipótese nula → modelo é genuinamente melhor que chute
p ≥ 0.05  → não há evidência estatística de que o modelo supera o chute
```
O teste de McNemar compara dois classificadores em **exatamente os mesmos exemplos**. Aqui compara o nosso modelo contra o classificador "chuta sempre sobe" (majoritário).

**Passo 1 — Tabela de discordância (2×2)**:

Para cada dia do test set, classifica em 4 categorias:

| | Majoritário acertou | Majoritário errou |
|---|---|---|
| **Modelo acertou** | (a) ambos certos | (b) só modelo certo |
| **Modelo errou** | (c) só majoritário certo | (d) ambos errados |

Só as células **b** e **c** importam — são os dias em que os dois classificadores **discordam**.

**Passo 2 — Estatística do teste**:

```
χ² = (|b − c| − 1)² / (b + c)
```

A correção de continuidade `−1` é aplicada porque a distribuição é discreta.

**Passo 3 — p-valor**:

```
p = 1 − CDF_chi2(χ², df=1)
```

A estatística χ² segue uma distribuição qui-quadrado com 1 grau de liberdade.

**Exemplo concreto para variante B** (b=?, c=? → p=0.0001):
- b grande (modelo acerta muito mais que o majoritário nos dias difíceis)
- c pequeno (poucos dias em que só o chute acertaria)
- χ² alto → p muito baixo → diferença é real, não acaso

**Por que usar McNemar e não um teste t simples?**
- McNemar é o teste correto para comparar dois classificadores nos **mesmos dados** (dados pareados)
- Não assume normalidade
- É exato mesmo com amostras pequenas
- A hipótese nula "b = c" equivale a "ambos os classificadores têm a mesma taxa de erro"

---

### Loss Function, Overfitting, Early Stopping e Épocas

#### O que é a Loss Function (BCE Loss)

A loss function é a **função que o modelo minimiza durante o treinamento**. É a medida numérica de "quão errado" o modelo está em cada iteração — o gradiente descendente ajusta os pesos da rede na direção que reduz esse número.

Para classificação binária (sobe/cai), usamos a **Binary Cross-Entropy (BCE)**:

```
BCE = −[y·log(p) + (1−y)·log(1−p)]
```

Onde:
- `y` = 1 se o dia realmente subiu, 0 se caiu
- `p` = P(sobe) prevista pelo modelo ∈ [0, 1]

**Por que logaritmo?** A função log penaliza erros com alta confiança de forma muito severa:

| Realidade (y) | P(sobe) prevista | BCE | Interpretação |
|---|---|---|---|
| 1 (subiu) | 0.95 | 0.05 | Modelo confiante e certo — perda baixa |
| 1 (subiu) | 0.50 | 0.69 | Modelo incerto — perda média |
| 1 (subiu) | 0.05 | 3.00 | Modelo confiante e **errado** — perda altíssima |
| 0 (caiu) | 0.05 | 0.05 | Modelo confiante e certo — perda baixa |

BCE médio sobre todos os exemplos do batch → o otimizador Adam calcula o gradiente ∂BCE/∂pesos e atualiza os pesos para reduzir esse valor.

**Valores típicos de BCE**:
- `0.693` = log(2) = modelo completamente aleatório (P=0.5 sempre)
- `0.60–0.65` = modelo aprendeu algo mas longe do ótimo
- `< 0.60` = modelo com sinal real (como variante B com val_loss=0.607)

---

### O que são as Épocas

Uma **época** é uma passagem completa por **todos os exemplos do conjunto de treino**, organizada em batches.

```
1 época = ceil(N_treino / batch_size) atualizações de pesos

Neste projeto:
  N_treino ≈ 5.205 sequências
  batch_size = 32
  → 1 época = ceil(5.205 / 32) = 163 atualizações de pesos
```

Ao final de cada época:
1. O modelo faz predições em todo o treino → calcula `loss` (BCE médio)
2. O modelo faz predições em todo o val → calcula `val_loss` (BCE médio, **sem atualizar pesos**)
3. Esses dois valores são os pontos plotados no gráfico de loss por época

---

### O que é Overfitting

**Overfitting** é quando o modelo **memoriza os dados de treino em vez de aprender padrões generalizáveis**. É análogo a decorar as respostas de uma prova específica sem entender o conteúdo — funciona naquela prova, falha em qualquer outra.

**Sintoma visual**: no gráfico de loss por época, o overfitting aparece quando:

```
loss (treino) continua caindo  →  modelo melhora no treino
val_loss (val) para de cair ou sobe  →  modelo piora em dados novos
```

Isso ocorre porque, ao treinar demais, a rede começa a **ajustar os pesos para as idiossincrasias específicas** dos dados de treino (ruído, outliers, padrões espúrios do período 2000–2020) em vez dos padrões reais que se repetem em 2023–2024.

---

### Leitura do Gráfico `overfitting_variantes.png`

Cada subplot mostra a variante com o número de épocas treinadas no título. O eixo X é a época, o eixo Y é a BCE Loss. A **linha vermelha tracejada** marca a época com o menor `val_loss` (o modelo que foi salvo pelo early stopping).

**Variante A (21 épocas) e C (21 épocas) — overfitting moderado típico**:
- Treino (azul) cai suavemente de 0.69 até ~0.59
- Val (laranja) cai até ~0.63 na época 10 (best@10/11), depois **sobe levemente**
- A subida do val após o best é o sinal clássico de overfitting iniciando
- Early stopping para na época 21 (10 épocas após o best sem melhora)
- Resultado: bom modelo capturado no ponto ótimo antes do overfitting se agravar

**Variante B (25 épocas) — overfitting presente, melhor modelo geral**:
- Val cai mais agressivamente que A/C até ~0.607 na época 14 (best@14)
- Após a época 14, val sobe de volta para ~0.64 — overfitting mais pronunciado
- O treino continua descendo até ~0.58 (brecha maior entre treino e val = mais overfitting)
- Porém: o val_loss mínimo de B (0.607) é **menor que o val_loss mínimo de A (0.631)**, o que significa que B encontrou um mínimo genuinamente melhor antes de overfit
- É por isso que B tem melhor MCC no test set — o denoising DWT entrega um sinal mais limpo, o modelo aprende padrões mais generalizáveis

**Variante D_corrigida (26 épocas) — sinal quase inexistente**:
- Val (laranja) começa em 0.688 e **mal se move** — oscila entre 0.673–0.688
- Treino (azul) cai para ~0.655, mas val não acompanha
- A brecha enorme treino/val com val estagnada próximo de `log(2)=0.693` indica que o sinal nos log-returns denoised pelo MODWT é insuficiente para classificação direcional
- O best é salvo na época 15 mas não representa aprendizado real (val_loss=0.675 vs. 0.693 aleatório)

**Variante E (16 épocas) — early stopping agressivo**:
- Val atinge seu mínimo na época 5 (best@5) muito cedo
- Após isso, val sobe de ~0.635 para ~0.645 enquanto treino continua caindo
- O rolling Z-score causal introduz um "edge effect" nas primeiras 252 observações — a normalização é menos consistente nas bordas, o que pode tornar o sinal mais ruidoso e o overfitting mais rápido

**Variante F (11 épocas) — não aprende nada**:
- Escala do eixo Y: 0.6875–0.6895 — amplitude de apenas **0.002**
- Ambas as curvas oscilam caoticamente sem tendência de descida
- A linha best@0 na época 0 indica que o modelo nunca melhorou após a inicialização
- Os 3 canais DWT adicionados (fb_approx, fb_detail1, fb_detail2) são redundantes com as médias móveis já presentes nas 18 features — o modelo recebe informação duplicada que gera gradientes contraditórios

**Variante G (62 épocas) — aprendizado lento e estável**:
- As duas curvas descem **juntas** por mais épocas antes de divergir — o menor gap treino/val indica menos overfitting que B
- Treino e val ficam muito próximos até ~época 40 (gap ≈ 0.01), depois abre levemente
- MODWT causal no Close suaviza as variações de alta frequência, os gradientes são menores, o aprendizado é mais gradual
- Porém: val_loss mínimo de G (~0.640) é pior que B (~0.607) — aprendeu mais devagar e não chegou tão longe

---

### Por que o Early Stopping é Necessário

Sem early stopping, o modelo continuaria treinando por todas as `epochs=80` (configuração deste projeto), passando pelo ponto ótimo e cada vez piorando no val. O resultado seria um modelo com **val_loss alto** e **MCC baixo no test set**.

```
Sem early stopping (hipotético para variante B):
  Época 14: val_loss = 0.607  ← melhor ponto (o que usamos)
  Época 30: val_loss = 0.630  ← já overfit
  Época 80: val_loss = 0.660  ← fortemente overfit

Com early stopping (patience=10):
  Época 24: sem melhora por 10 épocas → para e restaura pesos da época 14
```

O parâmetro `restore_best_weights=True` faz o Keras **descartar os pesos das últimas 10 épocas** e restaurar os pesos do momento de melhor `val_loss`. Ou seja, o número de "épocas treinadas" na tabela inclui as 10 épocas de paciência, mas o modelo salvo corresponde ao `best_val_epoch`.

**Resumo do mecanismo**:

```
Para cada época:
  1. Treina (atualiza pesos com dados de treino)
  2. Avalia val_loss (sem atualizar pesos)
  3. Se val_loss < melhor_val_loss até agora:
       → salva os pesos atuais como "melhor modelo"
       → contador_paciencia = 0
     Senão:
       → contador_paciencia += 1
       → Se contador_paciencia == 10: para o treinamento
  4. Restaura pesos do "melhor modelo" salvo
```

**Por que patience=10 e não patience=1?**
O val_loss tem ruído natural — pode subir por 1–2 épocas mesmo com o modelo ainda melhorando. Uma paciência muito baixa pararia cedo demais. Com patience=10, damos margem para oscilações temporárias antes de concluir que o modelo realmente parou de melhorar.