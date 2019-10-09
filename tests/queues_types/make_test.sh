cd ./../../bin/;
onoff="online"
size="50k"
for queue in {"fcfs","spf","sqfmin","safmin"}; do
    for site in "lyon"; do
        echo "Executing scheduler with file $size, type $onoff and queue $queue"
        ./gpu_scheduler.out --test 5 --data-type flat -c none --file_name $site-$size -r ahpg --scheduling_type $onoff --debug critical --cmp $queue;
    done;
done;
