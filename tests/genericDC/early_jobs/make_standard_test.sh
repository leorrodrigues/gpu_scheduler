cd ./../../../bin/;

DC="LLNL_Thunder"

for onoff in {"offline","online"}; do
    for site in $DC; do
        echo "Executing scheduler with file $size and type $onoff"
        ./gpu_scheduler.out --test 5 --data-type flat -c none --file_name $site-$size-early -r ahpg --scheduling_type $onoff --debug critical;
    done;
done;
