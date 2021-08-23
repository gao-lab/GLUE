#!/bin/bash

set -e

Rscript export_data.R  # Produces: F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.mtx, F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.rownames, F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.colnames, F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.cell_cluster_outcomes.csv
gzip F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.mtx  # Produces: gzip F_GRCm38.81.P60Cortex_noRep5_FRONTALonly.raw.dge.mtx.gz
