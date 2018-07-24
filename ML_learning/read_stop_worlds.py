
def return_stop_wrolds():
    worldslist=[]
    with open('./stopworldlist.txt','r') as f:
        for line in f:
            world=f.readline().strip()
            worldslist.append(world)

    # print(worldslist)
    return worldslist