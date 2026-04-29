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

a=a010
m=m04
g=g128
# shape="biconcave"
shape="bud_04"

# for shape in "biconcave" "bud_04"
# for gz in gz64 gz32 gz16
for w in wz80 wz70 wz60
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

        # SRC="${BASE}/${shape}/data_0321/${a}_${m}_${g}/phase${phase}_${lambda_1}_${_lambda_2}_${lambda_3}_${_lambda_4}/verts_10000.vtk"
        # OUT="${DEST}/${shape}_${a}_${m}_${g}_phase${phase}.vtk"

        # SRC="${BASE}/${shape}/data_0321/${a}_${m}_${g}_${gz}/phase${phase}_${lambda_1}_${_lambda_2}_${lambda_3}_${_lambda_4}/verts_10000.vtk"
        # OUT="${DEST}/${shape}_${a}_${m}_${g}_${gz}_phase${phase}.vtk"

        SRC="${BASE}/${shape}/data_0321/${a}_${m}_${w}/phase${phase}_${lambda_1}_${_lambda_2}_${lambda_3}_${_lambda_4}/verts_10000.vtk"
        OUT="${DEST}/${shape}_${a}_${m}_${w}_phase${phase}.vtk"


        echo "Copying $SRC -> $OUT"
        cp "$SRC" "$OUT"
    done
done


# lambda_1=100000
# lambda_2=1
# lambda_3=100000
# lambda_4=10
# phase=2
# data_id="jrc_cos7-1b/unscaled"


# for shape in "bud-07" "bud-09" "bud-10" "bud-11" "bud-12"
# do
#     if [ $phase -eq 0 ]; then
#         _lambda_2=0
#         _lambda_4=0
#     elif [ $phase -eq 1 ]; then
#         _lambda_2=$lambda_2
#         _lambda_4=0
#     elif [ $phase -eq 2 ]; then
#         _lambda_2=$lambda_2
#         _lambda_4=$lambda_4
#     fi

#     SRC="${BASE}/${data_id}/${shape}/phase${phase}_${lambda_1}_${_lambda_2}_${lambda_3}_${_lambda_4}/stack_${shape}.vtk"
#     OUT="${DEST}/"

#     echo "Copying $SRC -> $OUT"
#     cp "$SRC" "$OUT"
# done


# lambda_1=100000
# lambda_2=1
# lambda_3=100000
# phase=2
# # hidden_dim=128

# # data_id="czii_27042022"
# # shape="golgi_01"
# # lambda_4=0.1

# data_id="mendelsohn_2021"
# shape="mito_01"
# lambda_4=0.01


# # for phase in 0 1 2
# for hidden_dim in 8 16 32 64
# do
#     if [ $phase -eq 0 ]; then
#         _lambda_2=0
#         _lambda_4=0
#     elif [ $phase -eq 1 ]; then
#         _lambda_2=$lambda_2
#         _lambda_4=0
#     elif [ $phase -eq 2 ]; then
#         _lambda_2=$lambda_2
#         _lambda_4=$lambda_4
#     fi

#     SRC="${BASE}/${data_id}/${shape}_${hidden_dim}/phase${phase}_${lambda_1}_${_lambda_2}_${lambda_3}_${_lambda_4}/verts_10000.vtk"
#     OUT="${DEST}/${shape}_${hidden_dim}_${phase}.vtk"

#     echo "Copying $SRC -> $OUT"
#     cp "$SRC" "$OUT"
# done


