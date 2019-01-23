name = "MCL_AHP"

contend = "size;time;\n"
size = 0

file = open ("MCL.txt","r")

size2=0

cont=0

lines = file.readlines()

for line in lines:
    line_split=line.split(" ")
    if line_split[0]=="FAT":
        size = int(line_split[3])
    else:
        if line_split[0]=="Cluster:" or line_split[0]=="Multicriteria:":
            cont+=float(line_split[1])
            size2+=1
    if size2==20:
        contend+=str(size)+";"+str(cont/10)+";\n"
        cont=0
        size2=0

file.close()

out = open (name+".data","w")
out.write(contend)
out.close()
