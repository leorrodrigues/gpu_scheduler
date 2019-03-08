import sys

a="host"

level=-1
print ('Number of arguments:'+str(len(sys.argv))+ 'arguments.')
print ('Argument List:', str(sys.argv))

type=str(sys.argv[1])
qnt=int(str(sys.argv[2]))
if len(sys.argv)==4:
    level=int(str(sys.argv[3]))


filename="datacenter/json/"+type+"/"+str(qnt)+".json"

file=open(filename,mode="w")

data='{"topology":{"type":'
data+='"'+type+'",'
data+='"size":'+str(qnt)
if(level!=-1):
    data+=',"level:'+str(level)
data+='},"hosts":['

#The memory is in Gb
#The Storage is in Gb
#The bandwidth is in Mb
for i in range(int((qnt**3)/4)):
    data+='{"name":"'+a+str(i+1)+'","vcpu": 20,"storage":1000,"memory":1,"bandwidth":1000,"security":false}\n,'

data=data[:-1]
data+="]}"
file.write(data)
file.close()
