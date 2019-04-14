#echo "Multicriteria method;Fat Tree Size;Number of containers;Time" >> times.txt
#for m in ahp_clusterized; do
cd ./../../scheduler/;
ft=38
for sm in best_fit worst_fit; do # first_fit; do
	echo "Fat Tree k=$ft Algoritmo $sm"
	./gpuscheduler.out --test 4 -s $ft --standard-allocation $sm -c none --request-size 50 --bw 50
done;
