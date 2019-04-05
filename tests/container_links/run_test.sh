#echo "Multicriteria method;Fat Tree Size;Number of containers;Time" >> times.txt
#for m in ahp_clusterized; do
cd ./../../scheduler/;
ft=44
for g in none mcl; do
	for dt in bw frag flat; do
		for m in ahpg topsis; do
			echo "Fat Tree k=$ft Algoritmo $m Cluster $g"
        	       ./gpuscheduler.out --test 4 -s $ft -m $m -c $g --request-size 50 --bw 50 --data-type $dt
		done;
	done;
#done;	
#mv logs/*.txt ../tests/container_links/new_results
#for g in none mcl; do
	for sm in best_fit worst_fit; do # first_fit; do
		echo "Fat Tree k=$ft Algoritmo $sm Cluster $g"
       		./gpuscheduler.out --test 4 -s $ft --standard-allocation $sm -c $g --request-size 50 --bw 50 --data-type flat
	done;
done;
