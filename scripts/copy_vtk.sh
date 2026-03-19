#!/bin/bash

BASE=/gscratch/matsulab/sim/pinn-exploration-model/outputs/logs/biconcave/data_0123
DEST=$HOME

for lambda_2 in 0 10
do
    for m in m00 m06 m07 m08 m09
    do
        if [ "$m" = "m00" ]; then
            a=a000
        else
            a=a005
        fi

        SRC="${BASE}/${a}_${m}/lambda_100000_${lambda_2}_100000_cont/verts_10000.vtk"
        OUT="${DEST}/verts_${a}_${m}_${lambda_2}.vtk"

        echo "Copying $SRC -> $OUT"
        cp "$SRC" "$OUT"
    done
done
