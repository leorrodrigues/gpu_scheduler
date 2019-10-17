cd ./../../../bin/;

for site in "lyon"; do
    for onoff in {"online","offline"}; do
        for queue in {"fcfs","spf","sqfmin","safmin"}; do
            for size in {"1k","10k","50k","100k","150k"}; do
                echo "Executing scheduler with file $site-$size, type $onoff and queue $queue"
                ./gpu_scheduler.out --test 5 --data-type flat -c none --file_name $site-$size -r ahpg --scheduling_type $onoff --debug critical --cmp $queue;
            done;
        done;
    done;
done;
