# Functional annotations

## Ontologies

| File name | URL | Access date | MD5SUM | Remark |
|:----------|:----|:------------|:-------|:-------|
|go.obo|http://purl.obolibrary.org/obo/go.obo|Jan 4, 2021|e41f4aecef8075b2492689f16211be1f||
|go.owl|http://purl.obolibrary.org/obo/go.owl|Jan 4, 2021|c8763c2fda07f9f352864d1d5a62f48d||
|go-basic.obo|http://purl.obolibrary.org/obo/go/go-basic.obo|May 18, 2021|0e676a0dd749a3d430470273f4aa7c9f||
|goslim_generic.obo|http://current.geneontology.org/ontology/subsets/goslim_generic.obo|Jan 4, 2021|be6e575d5093c1c92f137b9b46e91912||
|goslim_generic.owl|http://current.geneontology.org/ontology/subsets/goslim_generic.owl|Jan 4, 2021|65f5e45733ab8877edc3ccf30a427261||

## Annotations

| File name | URL | Access date | MD5SUM | Remark |
|:----------|:----|:------------|:-------|:-------|
|goa_human.gaf.gz|http://geneontology.org/gene-associations/goa_human.gaf.gz|Jan 4, 2021|e9958a55aed4fd826aa70bdaf8c39611||
|mgi.gaf.gz|http://release.geneontology.org/2021-05-01/annotations/mgi.gaf.gz|May 18, 2021|acc0effde1f1229cf65f7877574f80f4||

Gaf files were decompressed via:

```{sh}
gunzip -k goa_human.gaf.gz
gunzip -k mgi.gaf.gz
```

Goslim annotations were generated via:

```{sh}
owltools go.obo --gaf goa_human.gaf --map2slim -s goslim_generic -u goa_human_goslim_unmapped.gaf -w goa_human_goslim.gaf
```
