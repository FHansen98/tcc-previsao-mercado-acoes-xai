#!/bin/bash

# Inicia o servidor Nginx em segundo plano
nginx

# Inicia o latexmk para observar mudanças e recompilar o PDF
# O PDF será gerado em /home/latex/tcc.pdf
latexmk -pvc -pdf tcc.tex
