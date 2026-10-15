from utils import *
from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import monai
from tqdm import tqdm
from prepare_dataset_zhonglin import zhonglindataloader, create_dataset
import transcript
from test_only import test
import argparse
import os
import torch
import numpy as np


def train_test(traindataset, testdataset, savedir, bbox_threshold, prompt_mode, num_epoch, batch_size, lr):
    assert prompt_mode in {'point', 'box', 'point+box'}

    writer = SummaryWriter(savedir)

    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    train_dataset = SAMDataset(dataset=traindataset, processor=processor,
                               bbox_threshold=bbox_threshold)
    test_dataset = SAMDataset(dataset=testdataset, processor=processor,
                              bbox_threshold=bbox_threshold)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    model = SamModel.from_pretrained("facebook/sam-vit-base")

    for name, param in model.named_parameters():
        if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
            param.requires_grad_(False)

    optimizer = Adam(model.mask_decoder.parameters(), lr=lr, weight_decay=0)
    scheduler = ReduceLROnPlateau(optimizer, mode='min')

    seg_loss = monai.losses.DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        reduction='mean'
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    valid_loss_history = []

    for epoch in range(num_epoch):
        print("############## START TRAINING ###############")
        model.train()

        train_loss = 0
        niter = 0

        for batch in tqdm(train_dataloader):
            outputs = forward_pass(model, batch, prompt_mode, device)

            predicted_masks = outputs.pred_masks.squeeze(1)
            ground_truth_masks = batch["ground_truth_mask"].float().to(device)

            loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            niter += 1

        train_loss /= niter
        writer.add_scalar("Loss/train", train_loss, epoch)

        print("############## START VALIDATION ###############")
        model.eval()

        val_loss = 0
        niter = 0

        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                outputs = forward_pass(model, batch, prompt_mode, device)

                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)

                loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

                val_loss += loss.item()
                niter += 1

        val_loss /= niter
        scheduler.step(val_loss)
        valid_loss_history.append(val_loss)

        writer.add_scalar("Loss/val", val_loss, epoch)

        print(
            f"Epoch: {epoch+1:03d}/{num_epoch:03d} || "
            f"Training loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}"
        )

        if (epoch + 1) % 5 == 0:
            print("Save checkpoint")
            save_checkpoint(
                model.state_dict(),
                is_best=False,
                checkpoint_dir=f"{savedir}/checkpoint"
            )

        if epoch == np.argmin(valid_loss_history):
            print("Save best model")
            save_checkpoint(
                model.state_dict(),
                is_best=True,
                checkpoint_dir=f"{savedir}/checkpoint"
            )

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)

    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--bbox_threshold", type=int, default=5)
    parser.add_argument("--prompt_mode", type=str, default="box")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    init_env(seed_value=42)

    os.makedirs(args.savedir, exist_ok=True)

    #copytree_code(os.getcwd(), args.savedir)

    transcript.start(f"{args.savedir}/logfile.log", mode="a")

    train_loader = zhonglindataloader(datasetdir=args.train_data, delete_zeros=True)
    train_loader.convert_data()
    traindataset = create_dataset(
        images=train_loader.meta["spect"],
        labels=train_loader.meta["seg"]
    )

    val_loader = zhonglindataloader(datasetdir=args.val_data, delete_zeros=True)
    val_loader.convert_data()
    valdataset = create_dataset(
        images=val_loader.meta["spect"],
        labels=val_loader.meta["seg"]
    )

    train_test(
        traindataset=traindataset,
        testdataset=valdataset,
        savedir=args.savedir,
        bbox_threshold=args.bbox_threshold,
        prompt_mode=args.prompt_mode,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        lr=args.lr
    )

    transcript.stop()

    # with torch.no_grad():
    #     transcript.start(f"{args.savedir}/logfile.log", mode="a")

    #     print("################# START TESTING ###################")

    #     test_loader = zhonglindataloader(datasetdir=args.test_data, delete_zeros=False)
    #     test_loader.convert_data()
    #     testdataset = create_dataset(
    #         images=test_loader.meta["spect"],
    #         labels=test_loader.meta["seg"]
    #     )

    #     test(
    #         testdataset=testdataset,
    #         savedir=args.savedir,
    #         bbox_threshold=args.bbox_threshold,
    #         prompt_mode=args.prompt_mode
    #     )

    #     transcript.stop()