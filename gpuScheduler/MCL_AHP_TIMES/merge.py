ahp = open ("AHP.data","r")
mcl = open ("MCL.data","r")
ahpg = open ("AHPG.data","r")
ahp_mcl = open("MCL_AHP.data","r")

contend = "Size;Time;Algorithm\n"
size = 0

alines = ahp.readlines()
mlines = mcl.readlines()
aglines = ahpg.readlines()
amlines = ahp_mcl.readlines()


for al,ml,ag,am in zip (alines,mlines,aglines,amlines):
    if size>0:
        a=al.split("\n")
        m=ml.split(";")
        print(m)
        print(size)
        g=ag.split("\n")
        c=am.split(";")
        contend+=a[0]+";AHP\n"+g[0]+";AHP GPU\n"+m[0]+";"+m[1]+";MCL\n"+c[0]+";"+c[1]+";AHP + MCL\n"+c[0]+";"+str(float(c[1])-float(m[1]))+";AHP CLUSTERING\n"
    size+=1

ahp.close()
mcl.close()

out = open ("merge.data","w")
out.write(contend)
out.close()
