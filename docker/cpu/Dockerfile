FROM mambaorg/micromamba
RUN micromamba install -y -n base -c conda-forge -c bioconda \
    faiss-cpu pytorch-cpu plotly scglue && \
    micromamba clean -y -a
