from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/media/root123/date-5T/FMTrack_dataset/got10k_lmdb'
    settings.got10k_path = '/media/root123/date-5T/FMTrack_dataset/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.gtot_dir = '/media/root123/date-5T/FMTrack_dataset/gtot'
    settings.itb_path = '/media/root123/date-5T/FMTrack_dataset/itb'    
    settings.lasot_extension_subset_path = '/media/root123/date-5T/FMTrack_dataset/lasot_extension_subset'
    settings.lasot_lmdb_path = '/media/root123/date-5T/FMTrack_dataset/lasot_lmdb'
    settings.lasot_path = '/media/root123/date-5T/FMTrack_dataset/lasot'
    settings.network_path = '/media/root123/date-5T/YK-Track_code/FMTrack/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/media/root123/date-5T/FMTrack_dataset/nfs'
    settings.otb_path = '/media/root123/date-5T/FMTrack_dataset/otb'
    settings.prj_dir = '/media/root123/date-5T/YK-Track_code/FMTrack'

    settings.result_plot_path = '/media/root123/date-5T/YK-Track_code/FMTrack/output/test/result_plots'
    settings.results_path = '/media/root123/date-5T/YK-Track_code/FMTrack/output/test/tracking_results' # Where to store tracking results
    # /home/a806/newdriver/YK-Track_code/FMTrack/lib/test/parameter/odtrack.py中9-29行来判断加载哪个数据集的权重
    settings.save_dir = '/media/root123/date-5T/YK-Track_code/FMTrack/output'

    # yk LasHeR/VTUAV/HiAL测试路径
    settings.lasher_path = '/media/root123/date-5T/FMTrack_dataset/LasHeR'
    settings.lasher_test_dir = '/media/root123/date-5T/FMTrack_dataset/LasHeR/testingset'  # 
    settings.vtuav_path = '/media/root123/date-5T/FMTrack_dataset/VTUAV'
    settings.vtuav_test_dir = '/media/root123/date-5T/FMTrack_dataset/VTUAV/testingset'
    settings.hial_path = '/media/root123/date-5T/FMTrack_dataset/HiAL'
    settings.hial_test_dir = '/media/root123/date-5T/FMTrack_dataset/HiAL/testingset'
    # yk RGBT210/234测试
    settings.rgbt210_dir = '/media/root123/date-5T/FMTrack_dataset/RGBT-210/RGBT_T210'
    settings.rgbt234_dir = '/media/root123/date-5T/FMTrack_dataset/RBGT-234/RGBT_T234'

    # settings.save_dir = '/home/a806/newdriver/YK-Track_code/FMTrack/output'
    settings.segmentation_path = '/media/root123/date-5T/YK-Track_code/FMTrack/output/test/segmentation_results'
    settings.tc128_path = '/media/root123/date-5T/FMTrack_dataset/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/media/root123/date-5T/FMTrack_dataset/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/root123/date-5T/FMTrack_dataset/trackingnet'
    settings.uav_path = '/media/root123/date-5T/FMTrack_dataset/uav'
    settings.vot18_path = '/media/root123/date-5T/FMTrack_dataset/vot2018'
    settings.vot22_path = '/media/root123/date-5T/FMTrack_dataset/vot2022'
    settings.vot_path = '/media/root123/date-5T/FMTrack_dataset/VOT2019'
    settings.youtubevos_dir = ''



    settings.lasher_path = '/media/root123/date-5T/FMTrack_dataset/LasHeR'  # xyl 修改Lasher数据集的路径
    settings.vtuav_path = '/media/root123/date-5T/FMTrack_dataset/VTUAV'  # xyl 修改VTUAV数据集的路径
    # settings.webuavrgbt_path = '/home/xyl/pysot-master/testing_dataset/WebUAV-RGBT'  # xyl 修改Lasher数据集的路径
    
    return settings
