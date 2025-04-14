#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import gc
import time
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.dataset_readers import read_extrinsics_binary, read_intrinsics_binary, readColmapCameras
from utils.camera_utils import cameraList_from_camInfos


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, dataset, white_bg_flag):
    if white_bg_flag:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    else:
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders-b")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt-b")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 真实数据集的目录
    # merge_GT_dir = dataset.source_path
    # # 此处的dataset仅用作参数传递，没有实际用处
    # merge_GT_images = load_gts(merge_GT_dir, dataset, white_bg_flag)
    # for idx, view in enumerate(tqdm(merge_GT_images, desc="GT progress")):
    #     gt = view.original_image[0:3, :, :]
    #     torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    #     gc.collect()
    #     torch.cuda.empty_cache()
    all_time = 0
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        start_time = time.time()
        rendering = render(view, gaussians, pipeline, background)["render"]
        all_time += time.time() - start_time
        torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))
        gc.collect()
        torch.cuda.empty_cache()

    print("Average render time: ", all_time / len(views))
    print("Average FPS: ", 1 / (all_time / len(views)))


def load_gts(images_dir, args, white_bg_flag):
    # 给定路径 绕过scene得到和camera信息绑定的images信息
    # 只需要替换image的真实内容，其他相机内外参等信息可以不变，其实也无所谓，最终取的是image
    # 输入：gt的路径，sparse目录和参考的那个脑区一致
    cameras_extrinsic_file = os.path.join(images_dir, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(images_dir, "sparse/0", "cameras.bin")
    cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
    cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)

    if white_bg_flag:
        reading_dir = "images-w"
    else:
        reading_dir = "images"

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(images_dir, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    train_cam_infos = cam_infos
    resolution_scale = 1.0
    train_cameras = cameraList_from_camInfos(train_cam_infos, resolution_scale, args)

    return train_cameras


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, white_bg_flag):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, override_quantization=True)

        bg_color = [1, 1, 1] if white_bg_flag else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset, white_bg_flag)

        if not skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset, white_bg_flag)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--calculate_image_w", action="store_true", help="If set, it will render white bg result.")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    if args.calculate_image_w == True:
        print("Calculate black bg res, default dir name is gt/renders-b.")
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, False)
        print("Calculate white bg res, default dir name is gt/renders.")
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, True)
    else:
        print("Calculate black bg res, default dir name is gt/renders-b.")
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, False)
