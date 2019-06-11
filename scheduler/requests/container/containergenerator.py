
#for name in ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024', '2048', '4096', '8192', '16384', '32768' ]:
for name in ['65536','131072','262144']:
    contend=''

    contend = '{"tasks": ['

    for total in range(1,int(name)+1):
        contend+='{ "duration": 1.0, "links": [], "containers": [ { "epc_min": 0, "name": '+ str(int(total)) +', "ram_min": 1.0, "vcpu_min": 1.0, "vcpu_max": 8.0, "ram_max": 8.0, "pod": 0, "epc_max": 0 } ], "id":' + str(int(total)) +', "submission": '+ str(total) +'},'

    contend=contend[:-1]

    contend+=']}'
    out = open ('data-'+name+".json","w")
    out.write(contend)
    out.close()
