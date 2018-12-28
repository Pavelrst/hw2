#!/bin/bash
# just run this script from HW dir



## EXPERIMENT 1.1
##K=32 fixed, with L=2,4,8,16 varying per run
#echo "sending 1 job"
#./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_1_K32_L2 -K 32 -L 2 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
#echo "sending 2 job"
#./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_1_K32_L4 -K 32 -L 4 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
#echo "sending 3 job"
#./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_1_K32_L8 -K 32 -L 8 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
#echo "sending 4 job"
#./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_1_K32_L16 -K 32 -L 16 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
#
##K=64 fixed, with L=2,4,8,16 varying per run
#echo "sending 5 job"
#./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_1_K64_L2 -K 64 -L 2 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
#echo "sending 6 job"
#./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_1_K64_L4 -K 64 -L 4 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
#echo "sending 7 job"
#./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_1_K64_L8 -K 64 -L 8 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
#echo "sending 8 job"
#./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_1_K64_L16 -K 64 -L 16 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001


# EXPERIMENT 1.2

#L=2 fixed, with K=[32],[64],[128],[258] varying per run.
echo "sending 9 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K32_L2 -K 32 -L 2 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 10 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K64_L2 -K 64 -L 2 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 11 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K128_L2 -K 128 -L 2 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 12 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K258_L2 -K 258 -L 2 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001


#L=4 fixed, with K=[32],[64],[128],[258] varying per run.
echo "sending 13 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K32_L2 -K 32 -L 4 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 14 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K64_L2 -K 64 -L 4 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 15 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K128_L2 -K 128 -L 4 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 16 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K258_L2 -K 258 -L 4 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001

#L=8 fixed, with K=[32],[64],[128],[258] varying per run.
echo "sending 17 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K32_L2 -K 32 -L 8 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 18 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K64_L2 -K 64 -L 8 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 19 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K128_L2 -K 128 -L 8 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 20 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_2_K258_L2 -K 258 -L 8 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001


# EXPERIMENT 1.3

#K=[64, 128, 256] fixed with L=1,2,3,4 varying per run.
echo "sending 21 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_3_L1_K64-128-256 -K 64 128 256 -L 1 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 22 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_3_L2_K64-128-256 -K 64 128 256 -L 2 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 23 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_3_L3_K64-128-256 -K 64 128 256 -L 3 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
echo "sending 24 job"
./py-sbatch.sh -m hw2.experiments run-exp --run-name exp1_3_L4_K64-128-256 -K 64 128 256 -L 4 -P 4 -H 128 256 128 --early-stopping 5 --lr 0.001
