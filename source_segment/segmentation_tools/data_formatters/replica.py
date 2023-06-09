import source.segmentation_tools.segmentation_config as seg_cf
import source.config as cf
import source.segmentation_tools.data_formatters.sun_rgbd as sun_rgbd
import source.segmentation_tools.data_formatters.matterport_hmp3d as matterport_hmp3d


if __name__ == '__main__':
    # Run only with CHOSEN_MASK_TYPE = PIXEL_LEVEL_MASK_TYPE. This is to get paths for the original datasets

    # assert seg_cf.CHOSEN_MASK_TYPE == seg_cf.PIXEL_LEVEL_MASK_TYPE
    matterport_hmp3d.get_all_paths_recursively(folder_name=cf.ORG_DATASET_FOLDERS[seg_cf.REPLICA], ensure_min_pixels=True, prog_folder_minus=-2)
    ####################################################################################################################
    # Run with CHOSEN_MASK_TYPE = PIXEL_LEVEL_MASK_TYPE. This copies the original dataset to the training dataset folder

    matterport_hmp3d.get_all_paths_for_pixel(
        target_folder=cf.ORG_DATASET_FOLDERS[seg_cf.REPLICA],
        save_folder_name=cf.TRAINING_FOLDER_PATHS[seg_cf.REPLICA],
        make_masks_visible=False,
        resize_to=(seg_cf.WIDTH, seg_cf.HEIGHT)
    )
    ####################################################################################################################
    # Run with CHOSEN_MASK_TYPE = BORDER_LEVEL_MASK_TYPE. This is to get paths for training datasets

    # assert seg_cf.CHOSEN_MASK_TYPE == seg_cf.BORDER_MASK_TYPE
    # sun_rgbd.get_borders_for_all_masks(target_folder=cf.ORG_DATASET_FOLDERS[seg_cf.REPLICA],
    #                                    save_folder_name=cf.TRAINING_FOLDER_PATHS[seg_cf.REPLICA],
    #                                    make_masks_visible=False,
    #                                    resize_to=(seg_cf.WIDTH, seg_cf.HEIGHT))
    ####################################################################################################################
    # Run with CHOSEN_MASK_TYPE of your choice. This is to get paths for training datasets

    matterport_hmp3d.get_all_paths_recursively(folder_name=cf.TRAINING_FOLDER_PATHS[seg_cf.REPLICA], ensure_min_pixels=True)
