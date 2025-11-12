# Template TCC FGA-UnB

Template para Trabalhos de Conclusão de Cursos (TCC) na Faculdade do
Gama (FGA) em Latex.

Licenciado em Creative Commons Atribuição 3.0:
http://creativecommons.org/licenses/by/3.0/

Desenvolvido e adaptado pelo professor Edson Alves <edsomjr@gmail.com>.

## Dependências
Para utilizar, certifique-se de ter instalados no seu ambiente o [Docker](https://docs.docker.com/engine/install/) e o [Docker-compose](https://docs.docker.com/compose/install/).

## Compilando e Pré-visualizando em Tempo Real

Para compilar o projeto e iniciar um servidor de pré-visualização que se atualiza automaticamente, execute o comando:
```
docker compose up --build
```

Este comando irá construir a imagem Docker (se for a primeira vez ou se houver alterações) e iniciar o servidor. Após a execução, o PDF do seu projeto estará disponível em `latex/tcc.pdf` e também poderá ser visualizado diretamente no seu navegador através do endereço:

[http://localhost:8080/tcc.pdf](http://localhost:8080/tcc.pdf)

Qualquer alteração nos arquivos `.tex` na pasta `latex/` fará com que o projeto seja recompilado automaticamente. Basta atualizar a página no navegador para ver a versão mais recente.

## Encerrando o ambiente

Para parar e encerrar todos os serviços, execute o comando:
```
docker compose down
```
### FAQ

Caso você encontre algum erro na execução dos comandos acima, siga as instruções abaixo:

1. Adicione o grupo `docker` ao seu usuário com o comando
```
sudo usermod -a -G docker $USER
```

1. Altere as permissões do arquivo `/var/run/docker.sock` com o comando:
```
sudo chown $USER /var/run/docker.sock
```

1. Caso o Docker não esteja rodando, inicie o serviço (e agende o início automático no _boot_) com os comandos:
```
sudo systemctl enable docker
sudo systemctl start docker
```
