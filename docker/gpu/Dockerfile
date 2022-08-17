FROM gpuci/miniconda-cuda:11.5-base-ubuntu20.04
RUN conda install -c conda-forge -c bioconda -y faiss-gpu pytorch-gpu plotly scglue
RUN conda clean -a -y
