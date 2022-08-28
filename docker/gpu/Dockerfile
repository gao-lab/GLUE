FROM mambaorg/micromamba
RUN micromamba install -y -n base -c conda-forge -c bioconda \
    faiss-gpu pytorch-gpu plotly scglue && \
    micromamba clean -y -a