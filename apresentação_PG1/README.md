# Apresentação LaTeX (apresentação_PG1)

Ambiente Docker para compilar e pré-visualizar a apresentação LaTeX usando a mesma abordagem do projeto `template-latex-tcc`.

O arquivo principal da apresentação é `Feathertheme.tex` e o PDF gerado será `Feathertheme.pdf`.

## Dependências

- [Docker](https://docs.docker.com/engine/install/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Compilar e pré-visualizar em tempo real

Dentro da pasta `apresentação_PG1`, execute:

```bash
docker compose up --build
```

Isso irá:

- Construir a imagem Docker (se necessário).
- Iniciar um container com:
  - `latexmk -pvc -pdf Feathertheme.tex` observando o arquivo principal;
  - `nginx` servindo o PDF gerado.

Após a execução, o PDF estará disponível em `Feathertheme.pdf` na raiz do projeto e poderá ser visualizado no navegador em:

- http://localhost:8081/Feathertheme.pdf

Qualquer alteração em `Feathertheme.tex` (ou arquivos incluídos) fará com que o PDF seja recompilado automaticamente. Basta atualizar a página no navegador.

## Encerrando o ambiente

Para parar e remover os serviços:

```bash
docker compose down
```

## Observações

- O diretório atual (`apresentação_PG1`) é montado dentro do container em `/home/latex`.
- Caso queira mudar o nome do arquivo principal ou do PDF, atualize:
  - O comando em `run.sh` (linha com `latexmk -pvc -pdf ...`);
  - A diretiva `index` no arquivo `nginx.conf`.
