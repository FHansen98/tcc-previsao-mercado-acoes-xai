Diogo:
- menos texto nos slides da apresentação
- melhorar texto do título
- lista de abreviaturas em ordem alfabética
- explicar como que será analisado o SHAP. Após a previsão, entender as variáveis utilizadas para ser realizada essa previsão
- verificar se o tratamento de ruído nos feriados, foi definido um valor aleatório
- Alterar a figura do ensemble, possui erro de somatório. Criar uma imagem
- inves de regressão somente indicar alta ou baixa

Rafael:
-* Objetivos: inves de comprovar, escrever "avaliar"
-* Revisão bibliográfica: explicar exatamente cada um dos campos das medições (preço máximo negociado)
- analise quantitativa: explicar o que vai ser feito
- como fazer o tratamento de dados de grandes ruidos, crashes
- explicar melhor a verificação de lucro no cronograma

verificação simples de algoritmo:
- fazer uma classificação dos dados (filme bom ruim/ previsão alta e baixa) 
- GRU e transformers:
    - quais ferramentes utilizadas para esse tipo de análise
    - qualquer repositório para analise simples, ex: filmes (não utilizar o da bolsa de valores)


------------- anotados na hora;
rafael:
- objetivo não é comprovar é avaliar
- preço negociado na tabela com as explicações das variáveis 
- indices são calculados por outros mecanismos, não são da bolsa
- cometar sobre a analise quantitativa
- obter alguns indicadores 
- não pegar todo o intervalo para fazer a normalização MinMax
- treinamento com walk forward
- análise de rentabilidade financeira

diogo:
- tirar siglas
- deixar claro no texto como será utilizado a explicabilidade da xAI
- unix time, tratamento de ruido

teste para o inicio do tcc2
- fluxo melhor definido com implementação de testes