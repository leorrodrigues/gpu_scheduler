#echo "Multicriteria method;Fat Tree Size;Number of containers;Time" >> times.txt
#for m in ahp_clusterized; do
cd ./../../scheduler/;
ft=20
for g in none; do # mcl none; do
	for dt in bw frag flat; do
		for m in ahpg topsis; do
			echo "Fat Tree k=$ft Algoritmo $m Cluster $g"
        	       ./gpuscheduler.out --test 4 -s $ft -m $m -c $g --request-size 50 --bw 50 --data-type $dt
		done;
	done;
done;
