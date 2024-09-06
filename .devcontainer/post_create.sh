#!/bin/bash

cd /workspaces/CityGaussian
mkdir output
conda run -n citygs --no-capture-output pip install --upgrade-strategy only-if-needed -r requirements.txt
cd LargeLightGaussian
conda run -n citygs --no-capture-output pip install submodules/compress-diff-gaussian-rasterization
ln -sf /workspaces/CityGaussian/data /workspaces/CityGaussian/LargeLightGaussian/data
ln -sf /workspaces/CityGaussian/output /workspaces/CityGaussian/LargeLightGaussian/output