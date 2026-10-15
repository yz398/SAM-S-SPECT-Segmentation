#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 15:24:50 2026

@author: yiza
"""

from utils import *
from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from prepare_dataset_zhonglin import zhonglindataloader, create_dataset
import transcript

import argparse
import os
import torch
import numpy as np


# ============================================================
# TEST ONE PATIENT
# ============================================================

def test(
    testdataset,
    savedir,
    bbox_threshold,
    prompt_mode,
    patient_idx,
    original_num_slices,
    split_idx
):

    assert prompt_mode in {
        "point",
        "box",
        "point+box"
    }

    print()
    print("============================================")
    print(f"Testing patient {patient_idx}")
    print("============================================")

    # ========================================================
    # SAM processor
    # ========================================================

    print("Loading SAM processor...")

    processor = SamProcessor.from_pretrained(
        "facebook/sam-vit-base"
    )

    # ========================================================
    # SAM dataset
    # ========================================================

    test_dataset = SAMDataset(
        dataset=testdataset,
        processor=processor,
        bbox_threshold=bbox_threshold
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )

    print(
        f"Number of 2D samples for patient "
        f"{patient_idx}: {len(test_dataset)}"
    )

    # ========================================================
    # Load SAM model
    # ========================================================

    print("Loading SAM model...")

    model = SamModel.from_pretrained(
        "facebook/sam-vit-base"
    )

    # ========================================================
    # Load trained checkpoint
    # ========================================================

    checkpoint_path = os.path.join(
        savedir,
        "checkpoint",
        "best_checkpoint.pytorch"
    )

    print(
        f"Loading checkpoint: "
        f"{checkpoint_path}"
    )

    if not os.path.isfile(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint
    )

    model.eval()

    # ========================================================
    # Device
    # ========================================================

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Using device: {device}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

    model.to(device)

    # ========================================================
    # Storage
    # ========================================================

    all_seg = []
    all_true = []
    all_bbox = []

    # ========================================================
    # TEST LOOP
    # ========================================================

    for batch_idx, batch in enumerate(
        tqdm(
            test_dataloader,
            desc=f"Patient {patient_idx}"
        )
    ):

        # ====================================================
        # Save ACTUAL bbox used by SAM
        # ====================================================

        if "input_boxes" in batch:

            bbox = (
                batch["input_boxes"]
                .detach()
                .cpu()
                .numpy()
            )

            all_bbox.append(
                bbox
            )

        # ====================================================
        # Forward pass
        # ====================================================

        outputs = forward_pass(
            model,
            batch,
            prompt_mode,
            device
        )

        # ====================================================
        # SAM logits -> probability
        # ====================================================

        predicted_masks = torch.sigmoid(
            outputs.pred_masks.squeeze(1)
        )

        predicted_masks = (
            predicted_masks
            .detach()
            .cpu()
        )

        # ====================================================
        # Probability -> binary mask
        # ====================================================

        predicted_masks = (
            predicted_masks > 0.5
        ).to(torch.uint8)

        # ====================================================
        # Ground truth
        # ====================================================

        ground_truth_masks = (
            batch["ground_truth_mask"]
            .detach()
            .cpu()
        )

        # ====================================================
        # Store
        # ====================================================

        all_seg.append(
            predicted_masks
        )

        all_true.append(
            ground_truth_masks
        )

    # ========================================================
    # CHECK RESULTS
    # ========================================================

    if len(all_seg) == 0:

        raise RuntimeError(
            f"No predictions were generated "
            f"for patient {patient_idx}."
        )

    # ========================================================
    # CONCATENATE
    #
    # IMPORTANT:
    # Do NOT use:
    #
    #     torch.stack(...).squeeze()
    #
    # because squeeze() can remove the sample dimension.
    # ========================================================

    all_seg_stack = torch.cat(
        all_seg,
        dim=0
    ).numpy()

    all_true_stack = torch.cat(
        all_true,
        dim=0
    ).numpy()

    # ========================================================
    # Remove ONLY channel dimension if present
    #
    # Desired:
    #
    #     (N, 256, 256)
    #
    # not:
    #
    #     (256, 256)
    # ========================================================

    if (
        all_seg_stack.ndim == 4
        and
        all_seg_stack.shape[1] == 1
    ):

        all_seg_stack = (
            all_seg_stack[:, 0, :, :]
        )

    if (
        all_true_stack.ndim == 4
        and
        all_true_stack.shape[1] == 1
    ):

        all_true_stack = (
            all_true_stack[:, 0, :, :]
        )

    # ========================================================
    # BBOX
    # ========================================================

    if len(all_bbox) > 0:

        all_bbox_stack = np.concatenate(
            all_bbox,
            axis=0
        )

    else:

        all_bbox_stack = None

        print(
            "WARNING: no input_boxes "
            "found in dataset."
        )

    # ========================================================
    # SANITY CHECK BEFORE PUT-BACK
    # ========================================================

    n_pred_expanded = all_seg_stack.shape[0]
    n_gt_expanded = all_true_stack.shape[0]

    if n_pred_expanded != n_gt_expanded:
        raise RuntimeError(
            f"Prediction/GT sample mismatch before put-back: "
            f"{n_pred_expanded} vs {n_gt_expanded}"
        )

    if all_bbox_stack is not None:
        n_bbox = all_bbox_stack.shape[0]
        if n_bbox != n_pred_expanded:
            raise RuntimeError(
                f"BBox/prediction sample mismatch before put-back: "
                f"{n_bbox} vs {n_pred_expanded}"
            )

    # ========================================================
    # PUT EXTRA LESION SAMPLES BACK INTO ORIGINAL SLICES
    #
    # convert_data() keeps the first connected component at
    # the original slice index and appends every additional
    # component to the end of the dataset. split_idx stores
    # [extra_sample_index, original_slice_index].
    # ========================================================

    print()
    print("============================================")
    print("PUTTING SPLIT LESIONS BACK")
    print("============================================")
    print("Expanded samples:", n_pred_expanded)
    print("Original slices:", original_num_slices)
    print("Extra split samples:", len(split_idx))

    expected_expanded = original_num_slices + len(split_idx)
    if n_pred_expanded != expected_expanded:
        raise RuntimeError(
            f"Unexpected expanded sample count: got {n_pred_expanded}, "
            f"expected {original_num_slices} + {len(split_idx)} "
            f"= {expected_expanded}."
        )

    pred_putback = all_seg_stack[:original_num_slices].copy()
    gt_putback = all_true_stack[:original_num_slices].copy()

    for extra_idx, original_idx in split_idx:
        extra_idx = int(extra_idx)
        original_idx = int(original_idx)

        if extra_idx >= n_pred_expanded:
            raise IndexError(
                f"split_idx extra sample {extra_idx} is outside "
                f"the expanded prediction array of length {n_pred_expanded}."
            )
        if original_idx >= original_num_slices:
            raise IndexError(
                f"split_idx original slice {original_idx} is outside "
                f"the original volume of length {original_num_slices}."
            )

        print(f"  sample {extra_idx} -> original slice {original_idx}")

        # Binary union. This avoids values of 2 in overlapping regions.
        pred_putback[original_idx] = np.maximum(
            pred_putback[original_idx],
            all_seg_stack[extra_idx]
        )
        gt_putback[original_idx] = np.maximum(
            gt_putback[original_idx],
            all_true_stack[extra_idx]
        )

    all_seg_stack = pred_putback
    all_true_stack = gt_putback

    # ========================================================
    # PRINT FINAL OUTPUT SHAPES
    # ========================================================

    print()
    print("============================================")
    print("FINAL OUTPUT SHAPES AFTER PUT-BACK")
    print("============================================")
    print("Prediction shape:", all_seg_stack.shape)
    print("Ground truth shape:", all_true_stack.shape)
    if all_bbox_stack is not None:
        print("BBox shape (lesion-level, not put back):", all_bbox_stack.shape)

    if all_seg_stack.shape[0] != original_num_slices:
        raise RuntimeError(
            f"Final prediction has {all_seg_stack.shape[0]} slices; "
            f"expected {original_num_slices}."
        )

    if all_true_stack.shape[0] != original_num_slices:
        raise RuntimeError(
            f"Final GT has {all_true_stack.shape[0]} slices; "
            f"expected {original_num_slices}."
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = os.path.join(
        savedir,
        "test_results"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ========================================================
    # SAVE PREDICTION
    # ========================================================

    pred_path = os.path.join(
        output_dir,
        f"pred_patient{patient_idx:03d}.npy"
    )

    np.save(
        pred_path,
        all_seg_stack
    )

    # ========================================================
    # SAVE GROUND TRUTH
    # ========================================================

    gt_path = os.path.join(
        output_dir,
        f"gt_patient{patient_idx:03d}.npy"
    )

    np.save(
        gt_path,
        all_true_stack
    )

    # ========================================================
    # SAVE BBOX
    # ========================================================

    if all_bbox_stack is not None:

        bbox_path = os.path.join(
            output_dir,
            f"bbox_patient{patient_idx:03d}.npy"
        )

        np.save(
            bbox_path,
            all_bbox_stack
        )

        # Save mapping from extra lesion samples to original slices.
        split_idx_path = os.path.join(
            output_dir,
            f"split_idx_patient{patient_idx:03d}.npy"
        )
        np.save(
            split_idx_path,
            np.asarray(split_idx, dtype=np.int32)
        )

        print(
            f"Split-index mapping saved to: "
            f"{split_idx_path}"
        )

        print(
            f"BBox saved to: "
            f"{bbox_path}"
        )

    # ========================================================
    # FINISHED
    # ========================================================

    print(
        f"Prediction saved to: "
        f"{pred_path}"
    )

    print(
        f"Ground truth saved to: "
        f"{gt_path}"
    )

    print(
        f"[Patient {patient_idx}] finished."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # ARGUMENTS
    # ========================================================

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--test_data",
        type=str,
        required=True,
        help="Path to test_data.mat"
    )

    parser.add_argument(
        "--savedir",
        type=str,
        required=True,
        help=(
            "Directory containing "
            "checkpoint/best_checkpoint.pytorch"
        )
    )

    parser.add_argument(
        "--bbox_threshold",
        type=int,
        default=5
    )

    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="box",
        choices=[
            "point",
            "box",
            "point+box"
        ]
    )

    parser.add_argument(
        "--patient_idx",
        type=int,
        default=-1,
        help=(
            ">=0: run one patient; "
            "-1: run all patients"
        )
    )

    args = parser.parse_args()

    # ========================================================
    # INITIALIZE RANDOM SEED
    # ========================================================

    init_env(
        seed_value=42
    )

    # ========================================================
    # CHECK MODEL DIRECTORY
    # ========================================================

    if not os.path.exists(
        args.savedir
    ):

        raise FileNotFoundError(
            f"Directory "
            f"{args.savedir} "
            f"not found!"
        )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = os.path.join(
        args.savedir,
        "test_results"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ========================================================
    # LOG FILE
    #
    # Separate log for each SLURM array task.
    # ========================================================

    if args.patient_idx >= 0:

        log_path = os.path.join(
            output_dir,
            f"test_patient"
            f"{args.patient_idx:03d}.log"
        )

    else:

        log_path = os.path.join(
            output_dir,
            "test_all.log"
        )

    # ========================================================
    # START
    # ========================================================

    with torch.no_grad():

        transcript.start(
            log_path,
            mode="a"
        )

        print(
            "################# "
            "START TESTING "
            "###################"
        )

        print(
            f"Test data: "
            f"{args.test_data}"
        )

        print(
            f"Model directory: "
            f"{args.savedir}"
        )

        print(
            f"BBox threshold: "
            f"{args.bbox_threshold}"
        )

        print(
            f"Prompt mode: "
            f"{args.prompt_mode}"
        )

        print(
            f"Patient index: "
            f"{args.patient_idx}"
        )

        # ====================================================
        # LOAD ORIGINAL DATA
        #
        # IMPORTANT:
        #
        # DO NOT call convert_data() here.
        #
        # We need to preserve:
        #
        #     patient x H x W x Z
        #
        # so that patient_idx refers to an actual patient.
        # ====================================================

        print()
        print(
            "Loading original Zhonglin dataset..."
        )

        full_loader = zhonglindataloader(
            datasetdir=args.test_data,
            delete_zeros=False
        )

        # ====================================================
        # Original shapes
        # ====================================================

        print()
        print(
            "Original SPECT shape:",
            full_loader.meta["spect"].shape
        )

        print(
            "Original segmentation shape:",
            full_loader.meta["seg"].shape
        )

        # ====================================================
        # Number of patients
        # ====================================================

        P = (
            full_loader
            .meta["spect"]
            .shape[0]
        )

        print(
            f"Total number of patients: "
            f"{P}"
        )

        # ====================================================
        # SELECT PATIENT INDICES
        # ====================================================

        if args.patient_idx >= 0:

            if args.patient_idx >= P:

                raise IndexError(
                    f"patient_idx="
                    f"{args.patient_idx} "
                    f"is out of range. "
                    f"Dataset contains {P} "
                    f"patients "
                    f"(valid indices: "
                    f"0-{P-1})."
                )

            patient_indices = [
                args.patient_idx
            ]

        else:

            patient_indices = list(
                range(P)
            )

        print(
            f"Patients to process: "
            f"{patient_indices}"
        )

        # ====================================================
        # PROCESS PATIENTS
        # ====================================================

        for pi in patient_indices:

            print()
            print(
                "============================================"
            )

            print(
                f"[Patient {pi}] START"
            )

            print(
                "============================================"
            )

            # ================================================
            # STEP 1
            #
            # SELECT PATIENT BEFORE convert_data()
            #
            # Original:
            #
            #     (P,H,W,Z)
            #
            # Patient:
            #
            #     (1,H,W,Z)
            # ================================================

            patient_spect = (
                full_loader
                .meta["spect"][pi:pi+1]
                .clone()
            )

            patient_seg = (
                full_loader
                .meta["seg"][pi:pi+1]
                .clone()
            )

            # Number of physical z-slices before convert_data()
            original_num_slices = int(patient_seg.shape[-1])

            print(
                "Original number of physical slices:",
                original_num_slices
            )

            print()
            print(
                "Before convert_data:"
            )

            print(
                "Patient SPECT shape:",
                patient_spect.shape
            )

            print(
                "Patient segmentation shape:",
                patient_seg.shape
            )

            # ================================================
            # STEP 2
            #
            # Create temporary loader.
            #
            # It initially loads the full file, but we
            # immediately replace spect/seg with the selected
            # patient BEFORE calling convert_data().
            # ================================================

            patient_loader = (
                zhonglindataloader(
                    datasetdir=args.test_data,
                    delete_zeros=False
                )
            )

            patient_loader.meta[
                "spect"
            ] = patient_spect

            patient_loader.meta[
                "seg"
            ] = patient_seg

            # ================================================
            # STEP 3
            #
            # NOW convert only this patient.
            # ================================================

            patient_loader.convert_data()

            # Mapping created by convert_data():
            # [extra_sample_index, original_slice_index]
            split_idx = patient_loader.split_idx.copy()

            print(
                "Number of additional lesion samples:",
                len(split_idx)
            )

            print()
            print(
                "After convert_data:"
            )

            print(
                "Patient SPECT shape:",
                patient_loader
                .meta["spect"]
                .shape
            )

            print(
                "Patient segmentation shape:",
                patient_loader
                .meta["seg"]
                .shape
            )

            # ================================================
            # STEP 4
            #
            # Create HuggingFace dataset
            # ================================================

            testdataset = create_dataset(
                images=(
                    patient_loader
                    .meta["spect"]
                ),
                labels=(
                    patient_loader
                    .meta["seg"]
                )
            )

            print(
                "Number of test samples:",
                len(testdataset)
            )

            # ================================================
            # STEP 5
            #
            # Run SAM
            # ================================================

            test(
                testdataset=testdataset,
                savedir=args.savedir,
                bbox_threshold=(
                    args.bbox_threshold
                ),
                prompt_mode=(
                    args.prompt_mode
                ),
                patient_idx=pi,
                original_num_slices=original_num_slices,
                split_idx=split_idx
            )

            print()
            print(
                f"[Patient {pi}] DONE"
            )

        # ====================================================
        # FINISHED
        # ====================================================

        print()
        print(
            "################# "
            "TESTING FINISHED "
            "###################"
        )

        transcript.stop()