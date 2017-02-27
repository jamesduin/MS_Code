

def printDataInstance(instance, file):
    for dimension in instance[:-1]:
        file.write(str(dimension) + " ")
    file.write(str(instance[-1]))
    file.write("\n")
    return
