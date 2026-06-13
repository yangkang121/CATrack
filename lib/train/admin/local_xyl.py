class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/home/xyl/newdrive/xyl-code2/0.my_trackers/FMTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/home/xyl/newdrive/xyl-code2/0.my_trackers/FMTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/xyl/newdrive/xyl-code2/0.my_trackers/FMTrack/pretrained_networks'
        self.lasot_dir = '/public/dataset/lasot'
        self.got10k_dir = '/home/xyl/newdrive/GOT-10k/train'
        self.got10k_val_dir = '/public/dataset/got10k/val'
        self.lasot_lmdb_dir = '/public/dataset/lasot_lmdb'
        self.got10k_lmdb_dir = '/public/dataset/got10k_lmdb'
        self.trackingnet_dir = '/public/dataset/trackingnet'
        self.trackingnet_lmdb_dir = '/public/dataset/trackingnet_lmdb'
        self.coco_dir = '/public/dataset/coco'
        self.coco_lmdb_dir = '/public/dataset/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/public/dataset/vid'
        self.imagenet_lmdb_dir = '/public/dataset/vid_lmdb'
        self.imagenetdet_dir = ''
        # self.lasher_train_dir = '/public/dataset/lasher/trainingset'
        # self.lasher_test_dir = '/public/dataset/lasher/testingset'
        self.depthtrack_train_dir = '/public/dataset/depthtrack/train'
        self.depthtrack_test_dir = '/public/dataset/depthtrack/test'
        self.visevent_train_dir = '/public/dataset/visevent/train'
        self.visevent_test_dir = '/public/dataset/visevent/test'
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.lasher_train_dir = '/public/datasets/LasHeR/trainingset'  # xyl 修改Lasher数据集的路径
        self.lasher_test_dir = '/public/datasets/LasHeR/testingset'  # xyl 修改Lasher数据集的路径
        self.vtuav_train_dir = '/public/datasets/VTUAV/trainingset'  # xyl 修改VTUAV数据集的路径
        self.vtuav_test_dir = '/public/datasets/VTUAV/testingset'  # xyl 修改VTUAV数据集的路径
        