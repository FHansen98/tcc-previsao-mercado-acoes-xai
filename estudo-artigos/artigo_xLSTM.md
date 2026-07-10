Ao analisar o código (`xlstm_ts.py`) em comparação com o artigo (`xlstm.pdf`), nota-se que o autor implementou o núcleo da arquitetura matemática exatamente como descrito no texto, mas introduziu **adaptações rigorosas na engenharia de dados (a "Variante B")** para evitar vazamento de dados (data leakage) durante o treinamento.

### **O que se manteve igual (Conformidade com o Artigo)**

A estrutura fundamental da rede neural e a maioria dos hiperparâmetros de treinamento estão perfeitamente alinhados com as tabelas do artigo.

#### **Arquitetura Base (xLSTM-TS):**
* A rede mantém o `input_size` igual a 1, `embedding_dim` em 64 e `output_size` igual a 1.

* O empilhamento usa exatamente 4 blocos (`num_blocks = 4`), combinando camadas sLSTM e mLSTM.

* **mLSTM:** Configurado com `conv1d_kernel_size = 4`, `qkv_proj_blocksize = 2` e `num_heads = 2`.

* **sLSTM:** Configurado com `conv1d_kernel_size = 2`, `num_heads = 2` e fator de projeção (`proj_factor`) de 1.1.

#### **Hiperparâmetros de Treinamento:**

* O tamanho da sequência temporal (Lookback) se manteve em 150 dias.

* O tamanho do lote (`batch_size`) permaneceu em 16 para contornar limites de memória.

* A taxa de aprendizado inicial é `0.0001` com o otimizador Adam e função de perda MSE.

* O escalonador de taxa de aprendizado (`ReduceLROnPlateau`) continua cortando o valor pela metade (`factor=0.5`) após 10 épocas sem melhoria.

* O *Gradient Clipping* (norma máxima) está cravado em 1.0 para evitar a explosão de gradientes.

#### **Fundamento do Denoising:**
* A filtragem de ruído continua utilizando a Transformada Wavelet Discreta (DWT) com a família Daubechies 4 (`db4`) e a técnica de *soft thresholding*.

---

### **O que está diferente (Alterações no Código)**

As diferenças no código revelam que esta é uma versão refatorada e mais defensiva, voltada para garantir que o modelo seja testado de forma realista, sem "espiar" o futuro temporal.

* **Prevenção de Vazamento de Dados (A "Variante B"):**
* 
*No artigo:* O texto descreve o preenchimento (padding) dos dados para mitigar efeitos de borda durante a aplicação do Wavelet.


* *No código:* A função `wavelet_denoise_series` foi adaptada para ser estritamente **causal** (`mode='zero'`), calculando o *threshold* de ruído **apenas** com os dados de treino (`train_mask`). Além disso, o dimensionamento (`MinMaxScaler`) é ajustado (`fit`) única e exclusivamente no conjunto de treino. Isso corrige uma falha comum em artigos de séries temporais, garantindo que métricas futuras não influenciem o passado.


* **Paciência do Early Stopping:**
* 
*No artigo:* O treinamento é interrompido se não houver melhoria na perda de validação por 30 épocas consecutivas.


* *No código:* A paciência (`patience`) foi aumentada para 40 épocas na função `_patched_train_model`.


* **Divisão Temporal do Conjunto de Dados (S&P 500 Daily):**
* *No artigo:* A divisão foi Treino: 2000 até final de 2020; Validação: Janeiro de 2021 até 30 de junho de 2022; Teste: Julho de 2022 até o final de 2023.


* *No código:* A janela foi simplificada por anos completos. Treino: 2000 a 2020; Validação: 2021 a final de 2022; Teste: 2023 a 2024.


* **Hardware / Backend de Execução:**
* 
*No artigo:* Modelos de aprendizado profundo baseados em LSTM e redes grandes são treinados visando a eficiência em GPU.


* *No código:* Há um *monkey-patch* forçando o backend da camada sLSTM para `"vanilla"` em vez de `"cuda"`. Os comentários do autor indicam que isso foi feito para manter compatibilidade com processadores (CPU), alertando que essa versão pode apresentar ligeiras diferenças numéricas em relação ao kernel CUDA original.


* **Foco da Tarefa de Avaliação:**
* 
*No artigo:* Há uma grande ênfase em métricas de classificação direcional (prever se a tendência será de alta ou baixa), reportando *Accuracy*, *F1 Score* e *Precision* em tabelas detalhadas.


* *No código:* O *docstring* do autor declara explicitamente: *"Tarefa: regressão de preço (não classificação direcional como nossa LSTM)"*. Embora gere gráficos que avaliam acertos de direção indiretamente, o script foca em avaliar o distanciamento numérico absoluto via MAE, MAPE e R².