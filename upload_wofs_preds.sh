#!/usr/bin/bash

export AZCOPY_AUTO_LOGIN_TYPE="MSI"
cd /datadrive2/wofs-preds/2023/
azcopy copy --recursive --block-blob-tier cool --check-length=false "./$1" "https://wofsdltornado.blob.core.windows.net/wofs-preds/2023/"
