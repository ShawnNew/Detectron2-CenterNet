import os
import re

from .register_coco import register_coco_instances


def check_path(path):
    if os.path.exists(path):
        return path
    else:
        path = re.sub("^/autox-sz", "/autox", path)
        assert os.path.exists(path), "{} does not exist!"
        return path


def register_scale():
    root = check_path("/autox-sz/users/dongqixu/share/xdataset/detection/coco_format")

    def _register_coco_instances(name, json_file, image_root):
        register_coco_instances(name, {}, os.path.join(root, json_file), os.path.join(root, image_root))

    # scale_20190827
    _register_coco_instances("scale_20190827_train_9", "20190827/annotations/d_train_2020_10_classes.json", "20190827")
    _register_coco_instances("scale_20190827_val_9", "20190827/annotations/d_val_2020_10_classes.json", "20190827")
    # scale_20191209_20200317
    _register_coco_instances("scale_20191209_20200317_train_9",
                             "20191209-20200317/annotations/d_train_2020_10_classes.json", "20191209-20200317")
    _register_coco_instances("scale_20191209_20200317_val_9",
                             "20191209-20200317/annotations/d_val_2020_10_classes.json", "20191209-20200317")
    # coco_2017_train_9
    _register_coco_instances("coco_2017_train_9",
                             "coco_cleanup/annotations/coco_instances_2017_cleanup_autox_10_classes.json",
                             "coco_cleanup/images/train2017")
