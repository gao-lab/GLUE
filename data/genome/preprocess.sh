#!/bin/bash

set -e

python extract_genes_promoters.py --input-gtf gencode.v19.chr_patch_hapl_scaff.annotation.gtf.gz  --output-bed gencode.v19.chr_patch_hapl_scaff.genes_with_promoters.bed
python extract_genes_promoters.py --input-gtf gencode.v35.chr_patch_hapl_scaff.annotation.gtf.gz  --output-bed gencode.v35.chr_patch_hapl_scaff.genes_with_promoters.bed
python extract_genes_promoters.py --input-gtf gencode.vM10.chr_patch_hapl_scaff.annotation.gtf.gz --output-bed gencode.vM10.chr_patch_hapl_scaff.genes_with_promoters.bed
python extract_genes_promoters.py --input-gtf gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz --output-bed gencode.vM25.chr_patch_hapl_scaff.genes_with_promoters.bed
LC_ALL=C sort -k1,1 -k2,2n -k3,3n gencode.v19.chr_patch_hapl_scaff.genes_with_promoters.bed  > gencode.v19.chr_patch_hapl_scaff.genes_with_promoters.sorted.bed
LC_ALL=C sort -k1,1 -k2,2n -k3,3n gencode.v35.chr_patch_hapl_scaff.genes_with_promoters.bed  > gencode.v35.chr_patch_hapl_scaff.genes_with_promoters.sorted.bed
LC_ALL=C sort -k1,1 -k2,2n -k3,3n gencode.vM10.chr_patch_hapl_scaff.genes_with_promoters.bed > gencode.vM10.chr_patch_hapl_scaff.genes_with_promoters.sorted.bed
LC_ALL=C sort -k1,1 -k2,2n -k3,3n gencode.vM25.chr_patch_hapl_scaff.genes_with_promoters.bed > gencode.vM25.chr_patch_hapl_scaff.genes_with_promoters.sorted.bed
