#echo "Multicriteria method;Fat Tree Size;Number of containers;Time" >> times.txt
#for m in ahp_clusterized; do
cd ./../../scheduler/;
ft=4
for dt in flat; do
	for g in none; do
		for p in 0 50 100; do
	        	for bw in 0 25 50; do
				for sm in best_fit worst_fit; do # first_fit; do
                			echo "Fat Tree k=$ft Algoritmo $sm Cluster $g Request-size $rs"
		               		./gpuscheduler.out --test 4 -s $ft --standard-allocation $sm -c $g --request-size $p --bw $bw --data-type $dt
		       	 	done;
		       done;
		done;
	done;
done;	
#mv logs/*.txt ../tests/container_links/new_results
