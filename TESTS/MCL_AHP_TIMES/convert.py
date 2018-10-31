name = input()

contend = "size;time;\n"
size = 0

file = open (name+".txt","r")

lines = file.readlines()

for line in lines:
    line_split=line.split(" ")
    if line_split[0]=="FAT":
        size = int(line_split[3])
    else:
        if line_split[0]=="All:":
            contend+=str(size)+";"+line_split[1]+";\n"

file.close()

out = open (name+".data","w")
out.write(contend)
out.close()

contend = "size;time\n"
size = 4
total = 0
new_size = 0
quantidade = 0
file = open (name+".data","r")

lines = file.readlines()

for line in lines:
    line_split = line.split(";")
    if(line_split[0] != "size"):
        new_size=int(line_split[0])
        if(new_size != size):
            contend += str(size) + ";" + str(total/quantidade) + "\n"
            quantidade = 0
            total = 0
            size = new_size
        total += float ( line_split[1] )
        quantidade += 1
while(size<50):
    contend += str(size) + ";" + "999999999\n"
    size+=2

out = open (name+".data","w")
out.write(contend)
out.close()
