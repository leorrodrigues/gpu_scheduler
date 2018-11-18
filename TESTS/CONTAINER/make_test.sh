#echo "Multicriteria method;Fat Tree Size;Number of containers;Time" >> times.txt
for m in mcl mcl_ahp mcl_ahpg ahp ahpg; do
	for k in 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48; do
        for i in 1 2 4 8 16 32 64 128 256 512 1024 2048; do
            #for round in 1 2 3 4 5 6 7 8 9 10; do
            #sync; echo 3 > /proc/sys/vm/drop_caches #clear all ram memory
            #swapoff -a && swapon -a #clear all swap
            su -c "echo 3 >'/proc/sys/vm/drop_caches' && swapoff -a && swapon -a && printf '\n%s\n' 'Ram-cache and Swap Cleared'" root
            rabbitmqadmin purge queue name=test_scheduler
            sleep 3;
            echo "Fat Tree k=$k Container Size $i Algoritmo $m round $round"
            cd ./../../simulator/;
            echo "Execunting simulator"
            ./simulator.out -s $i -d 1;
            cd ./../TESTS/CONTAINER/;
            sleep 5;
            cd ./../../gpuScheduler/;
            echo "Executing scheduler"
            timeout 10m ./gpuscheduler.out --test 1 -s $k -m $m >> ./../TESTS/CONTAINER/times_$m.txt
            if [ $? -eq 124 ]
            then
                echo "timeouted" >> ./../TESTS/CONTAINER/times_$m.txt && cd ./../TESTS/CONTAINER/ && break;
            else
                cd ./../TESTS/CONTAINER/;
            fi
            #done;
            #if [ $TEST_ERROR -eq 1 ]
            #then
            #	break;
            #fi
        done;
    done;
done;
