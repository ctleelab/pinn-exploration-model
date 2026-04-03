#!/bin/bash

BASE=/gscratch/matsulab/sim/pinn-exploration-model/outputs/logs
DEST=$HOME

# for lambda_2 in 0 10
# do
#     for m in m00 m06 m07 m08 m09
#     do
#         if [ "$m" = "m00" ]; then
#             a=a000
#         else
#             a=a005
#         fi

#         SRC="${BASE}/${a}_${m}/lambda_100000_${lambda_2}_100000_cont/verts_10000.vtk"
#         OUT="${DEST}/verts_${a}_${m}_${lambda_2}.vtk"

#         echo "Copying $SRC -> $OUT"
#         cp "$SRC" "$OUT"
#     done
# done

lambda_1=100000
lambda_2=1
lambda_3=100000
lambda_4=10

a=a000
m=m00
g=g128
shape="bud_04"

# for m in m06
for g in g64 g32 g16
do
    for phase in 0 1 2
    do
        if [ $phase -eq 0 ]; then
            _lambda_2=0
            _lambda_4=0
        elif [ $phase -eq 1 ]; then
            _lambda_2=$lambda_2
            _lambda_4=0
        elif [ $phase -eq 2 ]; then
            _lambda_2=$lambda_2
            _lambda_4=$lambda_4
        fi

        SRC="${BASE}/${shape}/data_0321/${a}_${m}_${g}/phase${phase}_${lambda_1}_${_lambda_2}_${lambda_3}_${_lambda_4}/verts_10000.vtk"
        OUT="${DEST}/${shape}_${a}_${m}_${g}_phase${phase}.vtk"

        echo "Copying $SRC -> $OUT"
        cp "$SRC" "$OUT"
    done
done

