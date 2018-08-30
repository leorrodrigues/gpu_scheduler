ahp = open ("AHP.data","r")
mcl = open ("MCL_AHP.data","r")

contend = "size;AHP Time;AHP + MCL Time\n"
size = 0

alines = ahp.readlines()
mlines = mcl.readlines()

for al,ml in zip (alines,mlines):
    a=al.split("\n")
    m=ml.split(";")
    if m[0]!="size":
        contend+=a[0]+";"+m[1]

ahp.close()
mcl.close()

out = open ("merge.data","w")
out.write(contend)
out.close()
