#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------

"""
Detection Training Script for MPViT.
"""

import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from ditod import add_vit_config

from ditod import MyTrainer

import pandas as pd
import imagesize
import numpy as np

from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

from detectron2.data import DatasetCatalog, MetadataCatalog

import imagesize


def prepare_data():

    main_dir_path = "./old_data"

    file_dirs = os.listdir(main_dir_path)
    all_csv_files = []
    all_csv_file_paths = []

    for fdir in file_dirs:
        
        temp_dir = main_dir_path + "/" + fdir
        temp_all_files = os.listdir(temp_dir)
        for file in temp_all_files:
            if file.endswith('.csv'):
                all_csv_file_paths.append(main_dir_path+"/"+fdir+"/"+file)   



    df = pd.read_csv(all_csv_file_paths[0])
    df['file_path'] = df['file_name'].apply(lambda x: '/'.join(all_csv_file_paths[0].split('/')[:-1])+'/'+x)

    for i in all_csv_file_paths[1:]:
        temp = pd.read_csv(i)
        temp['file_path'] = temp['file_name'].apply(lambda x: '/'.join(i.split('/')[:-1])+'/'+x)

        df = pd.concat([df,temp], ignore_index = True)

        df = df[df['classes'] == 'Question_option_answer']

        res = []
        for i in df['file_path'].values:
            try:
                p = imagesize.get(i)
                res.append(p)
            except:
                res.append(np.nan)

    df['wh'] = res

    df['width'] = df['wh'].apply(lambda x: x[0] if isinstance(x,tuple) else 0)
    df['height'] = df['wh'].apply(lambda x: x[1] if isinstance(x,tuple) else 0)

    df  = df[df['width'] > 0]

    df.replace({'Question_option_answer':0},inplace = True)

    unique_files = set(df['file_name'].values)
    mapping = dict(zip(unique_files,range(len(unique_files))))

    df.rename(columns = {'classes':'category_id','xmin':'x_min','ymin':'y_min','xmax':'x_max','ymax':'y_max'}, inplace = True)

    df['image_id'] = df['file_name'].apply(lambda x: mapping[x])

    df['category_id'] = 0 # just 1 class to detect
    thing_classes= ['text'] # cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    train_df = df.loc[:,["image_id", "category_id",	"x_min", "y_min", "x_max", "y_max"]]
    train_meta = df.loc[:,["image_id", "file_path", "width",	"height"]]

    def get_PL_data_dicts(
    _train_df: pd.DataFrame,
    _train_meta: pd.DataFrame,
    target_indices = None,
    debug: bool = False,):
  
        if debug:
                train_meta = train_meta.iloc[:100]  # For debug...
        dataset_dicts = []
        for index, train_meta_row in _train_meta.iterrows():
                        record = {}
                        image_id,file_name, width,height = train_meta_row.values
                        record["file_name"] = file_name
                        record["image_id"] = image_id
                        record["width"] = width
                        record["height"] = height
                        objs = []
                        for _, row in _train_df.query("image_id == @image_id").iterrows():
                            class_id = int(row["category_id"])
                            bbox_resized = [
                                float(row["x_min"]),
                                float(row["y_min"]),
                                float(row["x_max"]),
                                float(row["y_max"]),
                            ]
                            obj = {
                                "bbox": bbox_resized,
                                "bbox_mode": BoxMode.XYXY_ABS,
                                "category_id": class_id,
                            }
                            objs.append(obj)
                        record["annotations"] = objs
                        dataset_dicts.append(record)
                        
        if target_indices is not None:
            dataset_dicts = [dataset_dicts[i] for i in target_indices]

        return dataset_dicts
    
    debug=False

    Data_Resister_training="PL_data_train"
    Data_Resister_valid="PL_data_valid"
    
    n_dataset = len(train_meta)
    n_train = int(n_dataset * 0.95)
    # print("n_dataset", n_dataset, "n_train", n_train)
    rs = np.random.RandomState(12)
    inds = rs.permutation(n_dataset)
    train_inds, valid_inds = inds[:n_train], inds[n_train:]
    DatasetCatalog.register(
        Data_Resister_training,
        lambda: get_PL_data_dicts(
            train_df,
            train_meta,
            target_indices=train_inds,
            debug=debug,
        ),
    )
    MetadataCatalog.get(Data_Resister_training).set(thing_classes=thing_classes)
    

    DatasetCatalog.register(
        Data_Resister_valid,
        lambda: get_PL_data_dicts(
            train_df,
            train_meta,
            target_indices=valid_inds,
            debug=debug,
            ),
        )
    MetadataCatalog.get(Data_Resister_valid).set(thing_classes=thing_classes)
    
    dataset_dicts_train = DatasetCatalog.get(Data_Resister_training)
    metadata_dicts_train = MetadataCatalog.get(Data_Resister_training)

    dataset_dicts_valid = DatasetCatalog.get(Data_Resister_valid)
    metadata_dicts_valid = MetadataCatalog.get(Data_Resister_valid)
    
  
    print("Training size: ",len(dataset_dicts_train)," Valid size: ",len(dataset_dicts_valid), '\n')


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 64 # 128
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # 256
    cfg.SOLVER.IMS_PER_BATCH = 4

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    prepare_data()

    cfg = setup(args)

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    print("Command Line Args:", args)

    if args.debug:
        import debugpy

        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
