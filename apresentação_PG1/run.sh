#!/bin/bash

# Inicia o servidor Nginx em segundo plano
nginx

# Garante que estamos na pasta de trabalho
cd /home/latex

# Observa o arquivo principal da apresentação e recompila o PDF ao salvar
# O PDF será gerado como Feathertheme.pdf em /home/latex
latexmk -pvc -pdf Feathertheme.tex
