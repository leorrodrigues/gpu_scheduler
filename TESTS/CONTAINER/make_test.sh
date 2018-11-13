#echo "Multicriteria method;Fat Tree Size;Number of containers;Time" >> times.txt
for k in 4 16 32 40; do
    for i in 1 2 4 8 16 32 64 128 256 512 1024; do
        echo "Fat Tree k=$k Container Size $i Algoritmo $1"
        cd ./../../simulator/;
        echo "Execunting simulator"
        ./simulator.out -s $i -d 1;
        cd ./../TESTS/CONTAINER/;
        sleep 5;
        cd ./../../gpuScheduler/;
        echo "Executing scheduler"
        timeout 15m ./gpuscheduler.out --test 1 -a dc -s $k -m $1 >> times_mcl_$1.txt;
        cd ./../TESTS/CONTAINER/;
    done;
done;
