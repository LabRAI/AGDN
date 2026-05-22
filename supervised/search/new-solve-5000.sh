#!/bin/bash
# author: 
rm test
rm code/*.o 
make
STARTTIME=$(date +%s)
model="agd"  # agd, gcn
type="distribution"  # normal, distribution
infernum=20  # 128, 2000, 200, 20
tsp=$(printf "./heatmap/heatmap_%s_%s_TSP5000_%s.txt" "$model" "$type" "$infernum")
instancenum=(5000)
j=0
sources5000=("./results/5000/result_1.txt" "./results/5000/result_2.txt" "./results/5000/result_3.txt" "./results/5000/result_4.txt" "./results/5000/result_5.txt" "./results/5000/result_6.txt" "./results/5000/result_7.txt" "./results/5000/result_8.txt" "./results/5000/result_9.txt" "./results/5000/result_10.txt" "./results/5000/result_11.txt" "./results/5000/result_12.txt" "./results/5000/result_13.txt" "./results/5000/result_14.txt" "./results/5000/result_15.txt" "./results/5000/result_16.txt" "./results/5000/result_17.txt" "./results/5000/result_18.txt" "./results/5000/result_19.txt" "./results/5000/result_20.txt" "./results/5000/result_21.txt" "./results/5000/result_22.txt" "./results/5000/result_23.txt" "./results/5000/result_24.txt" "./results/5000/result_25.txt" "./results/5000/result_26.txt" "./results/5000/result_27.txt" "./results/5000/result_28.txt" "./results/5000/result_29.txt" "./results/5000/result_30.txt" "./results/5000/result_31.txt" "./results/5000/result_32.txt")
threads=32
#threads=16
use_rec=1
rec_only=$1
mcn=$2
md=$3
alpha=$4
beta=$5
ph=$6
retart=$7
retart_rec=$8
for ((i=0;i<$threads;i++));do
{
	./test $i ${sources5000[i]} ${tsp[j]} ${instancenum[j]} ${use_rec} ${rec_only} ${mcn} ${md} ${alpha} ${beta} ${ph} ${retart} ${retart_rec}
}&
done
wait


ENDTIME=$(date +%s)
echo "It takes $(($ENDTIME-$STARTTIME)) seconds to complete this task..."
echo "Search for $model in $tsp"
echo "Done."
