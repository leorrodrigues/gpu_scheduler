#!/bin/bash

for d in ./*/ ; do
	cd "$d"
	for file in ./*.pdf; do
  		pdfcrop "$file" "$file"
	done
done
