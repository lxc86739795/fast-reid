# encoding: utf-8
"""
@author:  xingyu liao
@contact: liaoxingyu5@jd.com
"""

from fastreid.data.common import CommDataset
from fastreid.data.data_utils import read_image


class KdDataset(CommDataset):
    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)

        if self.transform is not None: img = self.transform(img)

        if self.relabel: pid = self.pid_dict[pid]

        return {
            'images': img,
            'targets': pid,
            'camid': camid,
            'img_path': img_path,
            'index': index,
        }
