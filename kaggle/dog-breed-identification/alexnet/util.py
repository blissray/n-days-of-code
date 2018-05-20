import os


def check_and_make_directory(directory_name):
    if os.path.exists(directory_name) is not True:
        os.mkdir(directory_name)
        print("Directory create : {0}".format(directory_name))
