#!/bin/bash

set -e

Rscript wnn.r  # Produces: wnn_meta_data.csv
Rscript doubletfinder.r  # Produces: doubletfinder_inference.csv
