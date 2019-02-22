#echo "Multicriteria method;Fat Tree Size;Number of containers;Time" >> times.txt
#for m in ahp_clusterized; do
cd ./../../scheduler/;
for c in none mcl; do
    for rs in 10 150 1100 250 2550 25100 500 5050 50100; do
        for m in ahpg topsis; do
            echo "Fat Tree k= 48 Algoritmo $m Cluster $c Request-size $rs"
            ./gpuscheduler.out --test 4 -s 48 -m $m -c $c --request-size $rs
        done;
        for sm in bf wf ff; do
            echo "Fat Tree k= 48 Algoritmo $sm Cluster $c Request-size $rs"
            ./gpuscheduler.out --test 4 -s 48 --standard-allocation $sm -c $c --request-size $rs
        done;
    done;
done;
mv logs/*.txt ../tests/container_links/
