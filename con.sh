#!/bin/bash

SOURCE="/home/ldena/Bureau/PIR"
DEST="millot@dormammu:~/stage"

echo "Surveillance de $SOURCE..."

while inotifywait -r -e modify,create,delete,move "$SOURCE"; do
    rsync -avP --exclude='.git' "$SOURCE" "$DEST"
done
