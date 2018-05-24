import os
import shutil
import sys
import requests
from contextlib import closing
import torch
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader
import random
from PIL import Image


def download_url(url, file_name):
    os.system('wget -O %s %s' % (file_name, url))
#    print('downloading %s to %s ...' % (url, file_name))
#    with closing(requests.get(url, stream=True)) as response:
#        chunk_size = 4096
#        content_size = int(response.headers['content-length'])
#        have_size = 0
#        with open(file_name, "wb") as f:
#            for data in response.iter_content(chunk_size=chunk_size):
#                f.write(data)
#                have_size += len(data)
#                print('download progress %.1f%%' % (float(have_size) / content_size), end='\r')
#                sys.stdout.flush()
#    print('\ndownload %s finish')


def extract_tar(fname, dname):
    print('extracting %s to %s ...' % (fname, dname))
    os.system('mkdir -p %s' % dname)
    os.system('bar %s | tar xz -C %s' % (fname, dname))
    print('extract %s finish' % fname)


imgtrans = transforms.Compose([
    transforms.Resize(101),
    transforms.ToTensor()])
pose_mean = torch.Tensor([6.4316, 16.2672, 24.5792])
pose_std = torch.Tensor([0.7120, 4.8770, 107.5853])
patch_size0 = 192

#stat_n = 0
#stat_sum = torch.Tensor([0, 0, 0])
#stat_sum2 = torch.Tensor([0, 0, 0])

def read_data(dname, fname):
    #print('read image data %s' % fname)
    img_path = os.path.join(dname, fname + ".jpg")
    txt_path = os.path.join(dname, fname + ".txt")
    img = Image.open(img_path)
    f = open(txt_path, 'r')
    lines = f.readlines()
    f.close()
    d = lines[0].split() # Google Meta Data
    #a = lines[1].split() if len(lines) >= 2 else None # Alignment Data
    a = None # Alignment Data (already aligned)
    #d[5:8] # Target Point
    #d[10:13] # Street View Location
    distance = float(d[13]) # Distance to Target
    camera_pose = tuple((float(i) for i in d[14:17])) # Heading, Pitch, Roll
    camera_pose = camera_pose[:-1] # Roll is always zero
    if a is None:
        patch_center = (img.width // 2, img.height // 2)
    else:
        patch_center = (int(a[0]), int(a[1]))
    patch_size_ = patch_size0 // 2
    img = img.crop((patch_center[0]-patch_size_, patch_center[1]-patch_size_, patch_center[0]+patch_size_, patch_center[1]+patch_size_))
    img = imgtrans(img)
    pose = torch.Tensor([distance, camera_pose[0], camera_pose[1]])
    pose = (pose - pose_mean) / pose_std

    #global stat_n, stat_sum, stat_sum2
    #stat_n += 1
    #stat_sum += pose
    #stat_sum2 += pose ** 2
    #xmean = stat_sum / stat_n
    #s2 = (stat_sum2 - stat_n * xmean ** 2) / (stat_n - 1)
    #print(xmean, s2 ** 0.5)
    return (img, pose)


class SceneDataset(Dataset):

    def __init__(self, datasetID, keepTar=True):
        super(SceneDataset, self).__init__()
        self.datasetID = datasetID
        self.keepTar = keepTar
        self.path = "../data/%04d.tar" % datasetID
        self.dname = os.path.dirname(os.path.abspath(self.path))

        if not os.access(self.path, os.R_OK):
            download_url('https://storage.googleapis.com/streetview_image_pose_3d/dataset_aligned/%04d.tar' % datasetID, self.path)

        extract_tar(self.path, self.dname)
        if not self.keepTar:
            print('delete %s' % self.path)
            os.remove(self.path)

        self.dname = os.path.join(self.dname, '%04d' % datasetID)

        data = {}
        for i in os.listdir(self.dname):
            bname, ext = os.path.splitext(i)
            if ext == ".jpg":
                tid = int(bname[bname.rfind('_')+1:])
                if data.get(tid) is None:
                    data[tid] = [bname]
                else:
                    data[tid].append(bname)

        match_pairs = []
        unmatch_pairs = []
        unmatch_data = []
        for bnames in data.values():
            random.shuffle(bnames)
            if len(bnames) % 2 == 1:
                unmatch_data.append(bnames.pop())
            for i in range(0, len(bnames), 2):
                match_pairs.append((bnames[i], bnames[i + 1]))
        if len(unmatch_data) % 2 == 1:
            unmatch_data.pop()
        for i in range(0, len(unmatch_data), 2):
            unmatch_pairs.append((unmatch_data[i], unmatch_data[i + 1]))

        random.shuffle(match_pairs)
        random.shuffle(unmatch_pairs)
        while len(match_pairs) > len(unmatch_pairs) + 4 and len(match_pairs) > 2:
            p1 = match_pairs.pop()
            p2 = match_pairs.pop()
            unmatch_data.append((p1[0], p2[1]))
            unmatch_data.append((p1[1], p2[0]))

        self.pairs = []
        for i in match_pairs:
            self.pairs.append((1, i))
        for i in unmatch_pairs:
            self.pairs.append((0, i))

        print('load dataset %04d finish' % datasetID)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        match = self.pairs[idx][0]
        name1, name2 = self.pairs[idx][1]
        img1, pose1 = read_data(self.dname, name1)
        img2, pose2 = read_data(self.dname, name2)
        imgc = torch.cat((img1, img2))
        pose = pose1 - pose2
        return (imgc, (pose, match))

    def __del__(self):
        if not self.keepTar:
            print('delete %s' % self.dname)
            shutil.rmtree(self.dname)


class TrainLoader:

    def __init__(self, shuffleTar=True, keepTar=False):
        self.shuffleTar = shuffleTar
        self.keepTar = keepTar
        f = open('../data/list.txt', 'r')
        self.ids = [int(i) for i in f.readlines()]
        f.close()
        if shuffleTar:
            random.shuffle(self.ids)
        self.loader = None
        self.it = None

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            try:
                if self.it is None:
                    raise StopIteration
                return next(self.it)
            except StopIteration:
                if len(self.ids) == 0:
                    raise StopIteration
                dataset = SceneDataset(self.ids.pop(), keepTar=self.keepTar)
                #self.loader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=0)
                self.loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=2)
                self.it = iter(self.loader)
