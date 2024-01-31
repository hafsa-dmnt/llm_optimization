#!/bin/bash

# Fonction pour compter les lignes d'un fichier et afficher le résultat
count_lines() {
  local file="$1"
  local lines=$(wc -l < "$file")
  echo "$file : $lines"
}

# Chemin vers le répertoire à analyser
directory="/Users/hafsa/Documents/M2 - Reims/projet/Marathon"

# Fichier de sortie
output_file="./resultats.txt"

# Parcours récursif des fichiers .c dans le répertoire spécifié
find "$directory" -type f -name "*.c" -print0 |
while IFS= read -r -d '' file; do
  count_lines "$file"
done | sort -t ':' -k 2 -n > "$output_file"
