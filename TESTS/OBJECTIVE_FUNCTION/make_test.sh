#echo "Multicriteria method;Fat Tree Size;Number of containers;Time" >> times.txt
su -c "echo 3 >'/proc/sys/vm/drop_caches' && swapoff -a && swapon -a && printf '\n%s\n' 'Ram-cache and Swap Cleared'" root
#rabbitmqadmin purge queue name=test_scheduler
sleep 3;
SIZE=20
#echo "Fat Tree k=$SIZE Algoritmo ahpg_clusterized"

#cd ./../../simulator/;
#echo "Execunting simulator"
#./simulator.out -s 1 -d 2;
#cd ./../TESTS/OBJECTIVE_FUNCTION/;

#sleep 5;

cd ./../../gpuScheduler/;
echo "Executing scheduler"
./gpuscheduler.out --test 2 -s $SIZE -m ahp_clusterized --request-size $1 >> ./../TESTS/OBJECTIVE_FUNCTION/obj.txt
