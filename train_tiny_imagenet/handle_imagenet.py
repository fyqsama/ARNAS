import glob
import os
from shutil import move
from os import rmdir
import zipfile

fz = zipfile.ZipFile('/home/yuqi/data/tiny-imagenet-200.zip', 'r')
for file in fz.namelist():
    fz.extract(file, '/home/yuqi/data')

target_folder = '/home/yuqi/data/tiny-imagenet-200/val/'

val_dict = {}
with open('/home/yuqi/data/tiny-imagenet-200/val/val_annotations.txt', 'r') as f:
    for line in f.readlines():
        split_line = line.split('\t')
        val_dict[split_line[0]] = split_line[1]

paths = glob.glob('/home/yuqi/data/tiny-imagenet-200/val/images/*')
for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    if not os.path.exists(target_folder + str(folder)):
        os.mkdir(target_folder + str(folder))
        os.mkdir(target_folder + str(folder) + '/images')

for path in paths:
    file = path.split('/')[-1]
    folder = val_dict[file]
    dest = target_folder + str(folder) + '/images/' + str(file)
    move(path, dest)

rmdir('/home/yuqi/data/tiny-imagenet-200/val/images')