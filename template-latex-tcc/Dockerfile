FROM debian:bullseye-slim

RUN apt-get update && apt-get install -y ghostscript texlive-publishers texlive-lang-portuguese texlive-lang-french texlive-lang-spanish texlive-latex-extra texlive-fonts-recommended make nginx latexmk texlive-font-utils

COPY nginx.conf /etc/nginx/sites-available/default
COPY run.sh /usr/local/bin/run.sh

RUN chmod +x /usr/local/bin/run.sh

WORKDIR /home/latex
