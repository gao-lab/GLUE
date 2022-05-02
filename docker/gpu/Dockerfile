FROM gpuci/miniconda-cuda:11.0-base-ubuntu20.04
RUN conda install -c conda-forge -c bioconda -y faiss plotly pytorch-gpu scglue
RUN conda clean -a -y
