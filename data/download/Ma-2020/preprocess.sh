#!/bin/bash

set -e

tar xf GSE140203_RAW.tar
zcat GSM4156597_skin.late.anagen.atac.fragments.bed.gz | LC_ALL=C sort -k1,1 -k2,2n -k3,3n > GSM4156597_skin.late.anagen.atac.fragments.sorted.bed
bedmap --ec --delim "\t" --echo --echo-map-id ../../genome/gencode.vM25.chr_patch_hapl_scaff.genes_with_promoters.sorted.bed GSM4156597_skin.late.anagen.atac.fragments.sorted.bed | gzip > GSM4156597_skin.late.anagen.atac.fragments.bedmap.gz
