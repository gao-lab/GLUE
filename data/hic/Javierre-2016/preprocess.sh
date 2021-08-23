#!/bin/bash

set -e

unzip 1-s2.0-S0092867416313228-mmc4.zip  # Produces: DATA_S1.zip
unzip DATA_S1.zip  # Produces: ActivePromoterEnhancerLinks.tsv, PCHiC_peak_matrix_cutoff5.tsv, PCHiC_vs_rCHiC_peak_matrix.tsv
tar xf human_PCHiC_hg19_HindIII_design.tar.gz  # Produces: Human_hg19

if [ ! -d Human_hg38 ]; then mkdir Human_hg38; fi
awk '{print "chr" $0}' Human_hg19/Digest_Human_HindIII.rmap | liftOver /dev/stdin ../../genome/hg19ToHg38.over.chain.gz Human_hg38/Digest_Human_HindIII.rmap Human_hg38/Digest_Human_HindIII.unmapped
awk '{print "chr" $0}' Human_hg19/Digest_Human_HindIII_baits_e75_ID.baitmap | liftOver /dev/stdin ../../genome/hg19ToHg38.over.chain.gz Human_hg38/Digest_Human_HindIII_baits_e75_ID.baitmap Human_hg38/Digest_Human_HindIII_baits_e75_ID.unmapped
