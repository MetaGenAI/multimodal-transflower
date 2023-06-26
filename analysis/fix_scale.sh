#!/bin/bash

#find $1 -name "*.bvh" -print0 | xargs -0 -I{} python3 analysis/fix_scale.py {}
find $1 -name "*.bvh" -print0 | parallel -0 -I{} python3 analysis/fix_scale.py {}
