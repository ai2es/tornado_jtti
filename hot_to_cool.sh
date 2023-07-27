#!/usr/bin/bash
export AZCOPY_AUTO_LOGIN_TYPE="MSI"
python -u hot_to_cool.py  \
--date=20230508
