import glob

PATH_PEKORA = './Train_Data/pekora/*'

files = glob.glob(PATH_PEKORA)
for file in files:
    print(file)