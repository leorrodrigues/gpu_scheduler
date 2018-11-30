#echo "Multicriteria method;Fat Tree Size;Number of containers;Time" >> times.txt
su -c "echo 3 >'/proc/sys/vm/drop_caches' && swapoff -a && swapon -a && printf '\n%s\n' 'Ram-cache and Swap Cleared'" root
rabbitmqadmin purge queue name=test_scheduler
sleep 3;
echo "Fat Tree k=48 Algoritmo ahp_clusterized"

cd ./../../simulator/;
echo "Execunting simulator"
./simulator.out -s 0 -d 2;
cd ./../TESTS/OBJECTIVE_FUNCTION/;

sleep 5;

cd ./../../gpuScheduler/;
echo "Executing scheduler"
./gpuscheduler.out --test 2 -s 48 -m ahp_clusterized >> ./../TESTS/OBJECTIVE_FUNCTION/obj.txt
