cd ./../../../bin/;

DC="LLNL_Thunder"

for onoff in {"offline","online"}; do
    for site in $DC; do
        echo "Executing scheduler with file $size and type $onoff on site $site"
        ./gpu_scheduler.out --test 5 --data-type flat -c none --file_name $site -r ahpg --scheduling_type $onoff --debug info;
    done;
done;
