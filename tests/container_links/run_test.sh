#echo "Multicriteria method;Fat Tree Size;Number of containers;Time" >> times.txt
#for m in ahp_clusterized; do
cd ./../../scheduler/;
for dt in flat bw; do
	for g in none mcl; do
		for p in 0 50 100; do
	        	for bw in 0 25 50; do
				for m in ahpg topsis; do
					echo "Fat Tree k= 44 Algoritmo $m Cluster $g Request-size $rs"
		        	       ./gpuscheduler.out --test 4 -s 44 -m $m -c $g --request-size $p --bw $bw --data-type $dt
		      		done;
	        	   	for sm in best_fit worst_fit first_fit; do
                			echo "Fat Tree k= 44 Algoritmo $sm Cluster $g Request-size $rs"
		               		./gpuscheduler.out --test 4 -s 44 --standard-allocation $sm -c $g --request-size $p --bw $bw --data-type $dt
		       	 	done;
		       done;
		done;
	done;
done;	
#mv logs/*.txt ../tests/container_links/new_results
