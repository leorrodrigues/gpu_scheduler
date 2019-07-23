name = input("nome do arquivo:\n::> ")

with open(name,"r+") as f:
    old=""
    for line in f:
        if line.count(";")==3:
            line=line[:-1]
            line+=";NA\n"
        old+=line
    f.seek(0)
    f.write(old)
