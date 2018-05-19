import os

def check_directory(directory_name):
    if os.path.exists(directory_name) != True:
        os.mkdir(directory_name)
        print("Directory create : {0}".format(directory_name,))
    return True
