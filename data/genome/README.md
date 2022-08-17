# Genomes and annotations

## Genomes

|filename|url|date|comment|
|:------:|:-:|:--:|:-----:|
|GRCh37.p13.genome.fa.gz|https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/GRCh37.p13.genome.fa.gz|Aug 12, 2022|GENCODE V19|
|GRCh38.p13.genome.fa.gz|https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_35/GRCh38.p13.genome.fa.gz|Aug 12, 2022|GENCODE V35|
|NCBIM37.genome.fa.gz|https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M1/NCBIM37.genome.fa.gz|Aug 12, 2022|GENCODE VM1|
|GRCm38.p6.genome.fa.gz|https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.p6.genome.fa.gz|Aug 10, 2022|GENCODE VM25|

## Annotations

|filename|url|date|comment|
|:------:|:-:|:--:|:-----:|
|gencode.v19.chr_patch_hapl_scaff.annotation.gtf.gz|ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_19/gencode.v19.chr_patch_hapl_scaff.annotation.gtf.gz|Apr 17, 2021|GENCODE V19|
|gencode.v35.chr_patch_hapl_scaff.annotation.gtf.gz|ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_35/gencode.v35.chr_patch_hapl_scaff.annotation.gtf.gz|Oct 7, 2020|GENCODE V35|
|gencode.vM1.annotation.gtf.gz|https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M1/gencode.vM1.annotation.gtf.gz|Aug 12, 2022|GENCODE VM1|
|gencode.vM10.chr_patch_hapl_scaff.annotation.gtf.gz|ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M10/gencode.vM10.chr_patch_hapl_scaff.annotation.gtf.gz|Jan 24, 2021|GENCODE VM10|
|gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz|ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz|Nov 1, 2020|GENCODE VM25|

## Blacklist regions

```bash
git clone git@github.com:Boyle-Lab/Blacklist.git
git checkout c4b5e42b0c4d77ed4f1d1acd6bccd1297f163069  # Latest commit at the time
```

## Ortholog mapping

|filename|url|date|comment|
|:------:|:-:|:--:|:-----:|
|human-mouse-orthologs.csv.gz||Aug 6, 2022|Ensembl Genes 107 on BioMart|
