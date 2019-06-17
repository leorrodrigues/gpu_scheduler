#echo "Multicriteria method;Fat Tree Size;Number of containers;Time" >> times.txt
#for m in ahp_clusterized; do
cd ./../../scheduler/;
#for round in 1; do
    for k in 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46; do
        for i in 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536 131072 262144; do
            for m in ahpg topsis ; do
                echo "Fat Tree k=$k Container Size $i Algoritmo $m Request Size $rs round $round MCL"
                timeout 15m ./gpuscheduler.out --test 1 -s $k -m $m --data-type flat -c mcl --request-size $i
                if [ $? -eq 124 ]
                then
                    echo "$m;$k;$i;NA" >> ./logs/test1/dc-test1.log
                fi
            done;
            echo "Fat Tree k=$k Container Size $i Algoritmo pure_mcl"
            timeout 15m ./gpuscheduler.out --test 1 -s $k -m $m --data-type flat -c pure_mcl --request-size $i
            if [ $? -eq 124 ]
            then
                echo "pure_mcl;$k;$i;NA" >> ./logs/test1/dc-test1.log
            fi

        done;
    done;
#done;
