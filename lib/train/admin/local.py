class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/root123/date-5T/YK-Track_code/FMTrack'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/root123/date-5T/YK-Track_code/FMTrack/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/media/root123/date-5T/YK-Track_code/FMTrack/pretrained_networks'
        self.lasot_dir = '/media/root123/date-5T/FMTrack_dataset/lasot'
        self.got10k_dir = '/media/root123/date-5T/FMTrack_dataset/got10k/train'
        self.got10k_val_dir = '/media/root123/date-5T/FMTrack_dataset/got10k/val'
        self.lasot_lmdb_dir = '/media/root123/date-5T/FMTrack_dataset/lasot_lmdb'
        self.got10k_lmdb_dir = '/media/root123/date-5T/FMTrack_dataset/got10k_lmdb'
        self.trackingnet_dir = '/media/root123/date-5T/FMTrack_dataset/trackingnet'
        self.trackingnet_lmdb_dir = '/media/root123/date-5T/FMTrack_dataset/trackingnet_lmdb'
        self.coco_dir = '/media/root123/date-5T/FMTrack_dataset/coco'
        self.coco_lmdb_dir = '/media/root123/date-5T/FMTrack_dataset/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/media/root123/date-5T/FMTrack_dataset/vid'
        self.imagenet_lmdb_dir = '/media/root123/date-5T/FMTrack_dataset/vid_lmdb'
        self.imagenetdet_dir = ''
        # self.lasher_train_dir = '/media/root123/date-5T/FMTrack_dataset/lasher/trainingset'
        # self.lasher_test_dir = '/media/root123/date-5T/FMTrack_dataset/lasher/testingset'
        self.depthtrack_train_dir = '/media/root123/date-5T/FMTrack_dataset/depthtrack/train'
        self.depthtrack_test_dir = '/media/root123/date-5T/FMTrack_dataset/depthtrack/test'
        self.visevent_train_dir = '/media/root123/date-5T/FMTrack_dataset/visevent/train'
        self.visevent_test_dir = '/media/root123/date-5T/FMTrack_dataset/visevent/test'
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''

        # LasHeR and VTUAV
        self.lasher_train_dir = '/media/root123/date-5T/FMTrack_dataset/LasHeR/trainingset'  # xyl 修改Lasher数据集的路径
        self.lasher_test_dir = '/media/root123/date-5T/FMTrack_dataset/LasHeR/testingset'  # xyl 修改Lasher数据集的路径
        self.vtuav_train_dir = '/media/root123/date-5T/FMTrack_dataset/VTUAV/trainingset'  # xyl 修改VTUAV数据集的路径
        self.vtuav_test_dir = '/media/root123/date-5T/FMTrack_dataset/VTUAV/testingset'  # xyl 修改VTUAV数据集的路径
        # 新增数据集必须添加
        self.hial_train_dir = '/media/root123/date-5T/FMTrack_dataset/HiAL/trainingset'  # xyl 修改HiAl数据集的路径
        self.hial_test_dir = '/media/root123/date-5T/FMTrack_dataset/HiAL/testingset'  # xyl 修改HiAl数据集的路径
