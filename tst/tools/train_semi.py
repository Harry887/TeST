import argparse
import importlib
import os
import sys
import subprocess
import os.path as osp
import platform
import random
import shutil
import time
import warnings

import numpy as np
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
import torch
import torch.distributed as dist

from tqdm import tqdm

from tst.utils import DictAction
from tst.utils.env import collect_env_info
from tst.utils import AvgMeter, accuracy, parse_devices
from tst.utils.log import setup_logger
from tst.utils.torch_dist import configure_nccl, synchronize, reduce_tensor_sum
from torch.nn.parallel import DistributedDataParallel as DDP


parser = argparse.ArgumentParser(description="PyTorch FixMatch Training")
parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
parser.add_argument("--dist-url", default=None, type=str, help="url used to set up distributed training")

parser.add_argument("-b", "--batch-size", type=int, default=None, help="batch size")
parser.add_argument("-e", "--max_epoch", type=int, default=None, help="max_epoch for training")
parser.add_argument("-d", "--devices", default="0-7", type=str, help="device for training")
parser.add_argument("-f", "--exp_file", default=None, type=str, help="pls input your expriment description file")
parser.add_argument("--amp", action="store_true", help="enable automatic mixed precision training")
parser.add_argument(
    "--opt_level",
    type=str,
    default="O1",
    help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html",
)
parser.add_argument(
    "--exp-options",
    nargs="+",
    action=DictAction,
    help="override some settings in the exp, the key-value pair in xxx=yyy format will be merged into exp. "
    'If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b '
    'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
    "Note that the quotation marks are necessary and that no white space is allowed.",
)
args = parser.parse_args()


# ---------------------------------------- processing the experiment description file -------------------- #
if not args.exp_file:
    from tst.exps import fixmatch_exp

    if args.max_epoch:
        exp = fixmatch_exp.Exp(batch_size=args.batch_size, max_epoch=args.max_epoch)
    else:
        exp = fixmatch_exp.Exp(batch_size=args.batch_size)
else:
    sys.path.insert(0, os.path.dirname(args.exp_file))
    current_exp = importlib.import_module(os.path.basename(args.exp_file).split(".")[0])

    if args.max_epoch:
        exp = current_exp.Exp(batch_size=args.batch_size, max_epoch=args.max_epoch)
    else:
        exp = current_exp.Exp(batch_size=args.batch_size)

best_acc, best_mean_acc = 0, 0
best_acc_epoch, best_mean_acc_epoch = -1, -1


def save_best_checkpoint(state, epochs, out_dir):
    # filename = ("model_best_{}.pth").format(epochs)
    filepath = osp.join(out_dir, "model_best.pth")
    with open(filepath, "wb") as f:
        torch.save(state, f)
        f.flush()


def save_checkpoint(state, epochs, out_dir, save_latest=False, is_best=False, iters=None, create_symlink=False):
    def symlink(src, dst, overwrite=True, **kwargs):
        if os.path.lexists(dst) and overwrite:
            os.remove(dst)
        os.symlink(src, dst, **kwargs)

    if save_latest:
        filename = "latest_epoch.pth"
    elif iters:
        filename = "epoch_{}_iter_{}.pth".format(epochs, iters)
    else:
        filename = "epoch_{}.pth".format(epochs)
    filepath = osp.join(out_dir, filename)
    with open(filepath, "wb") as f:
        torch.save(state, f)
        f.flush()
    if is_best:
        shutil.copyfile(filepath, osp.join(out_dir, "model_best.pth"))
    if create_symlink:
        dst_file = osp.join(out_dir, "latest.pth")
        if platform.system() != "Windows":
            symlink(filename, dst_file)
        else:
            shutil.copy(filepath, dst_file)


def main():
    args.devices = parse_devices(args.devices)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    nr_gpu = len(args.devices.split(","))

    if exp.seed is not None:
        random.seed(exp.seed)
        np.random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        torch.cuda.manual_seed_all(exp.seed)
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior when restarting "
            "from checkpoints."
        )

    nr_machine = int(os.getenv("RLAUNCH_REPLICA_TOTAL", "1"))
    if nr_gpu > 1:
        args.world_size = nr_gpu * nr_machine
        processes = []
        for rank in range(nr_gpu):
            p = mp.Process(target=main_worker, args=(rank, nr_gpu, args))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        # args.world_size = nr_gpu * nr_machine
        # mp.spawn(main_worker, nprocs=nr_gpu, args=(nr_gpu, args))
    else:
        main_worker(0, nr_gpu, args)


def main_worker(gpu, nr_gpu, args):
    configure_nccl()
    update_cfg_msg = exp.update(args.exp_options)
    # ------------ set environment variables for distributed training ------------------------------------- #
    rank = gpu
    if nr_gpu > 1:
        rank += int(os.getenv("RLAUNCH_REPLICA", "0")) * nr_gpu

        if args.dist_url is None:
            master_ip = subprocess.check_output(["hostname", "--fqdn"]).decode("utf-8")
            master_ip = str(master_ip).strip()
            args.dist_url = "tcp://{}:23456".format(master_ip)

            # ------------------------hack for multi-machine training -------------------- #
            if args.world_size > 8:
                ip_add_file = "./" + exp.exp_name + "ip_add.txt"
                if rank == 0:
                    with open(ip_add_file, "w") as ip_add:
                        ip_add.write(args.dist_url)
                else:
                    while not os.path.exists(ip_add_file):
                        time.sleep(0.5)

                    with open(ip_add_file, "r") as ip_add:
                        dist_url = ip_add.readline()
                    args.dist_url = dist_url
        else:
            args.dist_url = "tcp://{}:23456".format(args.dist_url)

        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=rank
        )
        print("Rank {} initialization finished.".format(rank))
        synchronize()

        if rank == 0:
            if os.path.exists("./" + exp.exp_name + "ip_add.txt"):
                os.remove("./" + exp.exp_name + "ip_add.txt")

    logger = setup_logger(exp.out, distributed_rank=rank, mode="w")

    if rank == 0:
        logger.warning(f"Process rank: {rank}, 16-bits training: {args.amp}")
        logger.opt(ansi=True).info(
            "<yellow>Used experiment configs</yellow>:\n<blue>{}</blue>".format(exp.get_cfg_as_str())
        )
        if update_cfg_msg:
            logger.opt(ansi=True).info(
                "<yellow>List of override configs</yellow>:\n<blue>{}</blue>".format(update_cfg_msg)
            )
        logger.opt(ansi=True).info("<yellow>Environment info:</yellow>\n<blue>{}</blue>".format(collect_env_info()))

    data_loader = exp.get_data_loader()
    labeled_trainloader, unlabeled_trainloader, test_loader = (
        data_loader["train_labeled"],
        data_loader["train_unlabeled"],
        data_loader["eval"],
    )
    model = exp.get_model()
    optimizer = exp.get_optimizer()
    update_lr_func = exp.get_lr_scheduler().update_lr

    torch.cuda.set_device(gpu)
    model.to(gpu)

    if nr_gpu > 1:
        model = DDP(model, device_ids=[gpu], find_unused_parameters=True)

    if exp.use_ema:
        from tst.models.ema import ModelEMA

        no_ema = ["max_probs"]
        ema_model = ModelEMA(gpu, model, exp.ema_decay, no_ema=no_ema)

    if exp.resume:
        logger.info("==> Resuming from checkpoint..")
        assert os.path.isfile(exp.resume), "Error: no checkpoint directory found!"
        if torch.cuda.is_available():
            checkpoint = torch.load(exp.resume)
        else:
            checkpoint = torch.load(exp.resume, map_location=torch.device("cpu"))
        exp.start_epoch = checkpoint["epoch"] - 1
        if nr_gpu > 1:
            model.module.load_state_dict(checkpoint["state_dict"], strict=False)
        else:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
        if exp.use_ema:
            if nr_gpu > 1:
                ema_model.ema.module.load_state_dict(checkpoint["ema_state_dict"], strict=False)
            else:
                ema_model.ema.load_state_dict(checkpoint["ema_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer"])

    if args.amp:
        from apex import amp

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level)

    model.zero_grad()
    train(
        args,
        rank,
        labeled_trainloader,
        unlabeled_trainloader,
        test_loader,
        model,
        optimizer,
        ema_model,
        logger,
        exp,
        update_lr_func,
    )


def train(
    args,
    local_rank,
    labeled_trainloader,
    unlabeled_trainloader,
    test_loader,
    model,
    optimizer,
    ema_model,
    logger,
    exp,
    update_lr_func,
):
    if args.amp:
        from apex import amp
    global best_acc, best_mean_acc, best_acc_epoch, best_mean_acc_epoch
    test_accs = []
    batch_time = AvgMeter()
    data_time = AvgMeter()
    losses = AvgMeter()
    losses_x = AvgMeter()
    losses_u = AvgMeter()
    mask_probs = AvgMeter()
    end = time.time()

    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)
    iter_count = exp.start_epoch * exp.eval_step
    for epoch in range(exp.start_epoch, exp.max_epoch):
        if not exp.no_progress:
            p_bar = tqdm(range(exp.eval_step), disable=local_rank not in [-1, 0])
        for batch_idx in range(exp.eval_step):
            model.train()
            iter_count += 1

            _, inputs_x, targets_x = labeled_iter.next()
            unlabel_idxs, (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()

            data_time.update(time.time() - end)
            inputs_x = inputs_x.cuda()
            targets_x = targets_x.cuda()
            inputs_u_w = inputs_u_w.cuda()
            inputs_u_s = inputs_u_s.cuda()

            inputs = unlabel_idxs, inputs_x, inputs_u_w, inputs_u_s, targets_x
            Lx, Lu, mask, extra_dict = model(inputs, is_train=True)
            loss = Lx + exp.lambda_u * Lu

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())

            optimizer.step()
            lr = update_lr_func(iter_count)
            if exp.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.float().mean().item())

            if not exp.no_progress:
                p_bar.set_description(
                    "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. "
                    "Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. "
                    "Mask: {mask:.2f}. ".format(
                        epoch=epoch + 1,
                        epochs=exp.max_epoch,
                        batch=batch_idx + 1,
                        iter=exp.eval_step,
                        lr=lr,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg,
                    )
                )
                p_bar.update()

            if local_rank in [-1, 0] and batch_idx + 1 == exp.eval_step:
                logger.info(
                    "Train Epoch: {epoch}/{epochs:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. "
                    "Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}".format(
                        epoch=epoch + 1,
                        epochs=exp.max_epoch,
                        lr=lr,
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        loss_x=losses_x.avg,
                        loss_u=losses_u.avg,
                        mask=mask_probs.avg,
                    )
                )

        if local_rank == 0:
            if extra_dict:
                for k, v in extra_dict.items():
                    logger.info("{}: {}".format(k, v))

        if not exp.no_progress:
            p_bar.close()

        if exp.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model

        # ---------------------------------- eval model ------------------------------- #
        _, test_acc = test(test_loader, test_model, logger, exp.no_progress, local_rank)

        model_to_save = model.module if hasattr(model, "module") else model
        if exp.use_ema:
            ema_to_save = ema_model.ema.module if hasattr(ema_model.ema, "module") else ema_model.ema

        if local_rank == 0:
            epoch_save = torch.arange(0, 1024, 100).tolist()
            if epoch + 1 in epoch_save:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model_to_save.state_dict(),
                        "ema_state_dict": ema_to_save.state_dict() if exp.use_ema else None,
                        "optimizer": optimizer.state_dict(),
                    },
                    epoch + 1,
                    exp.out,
                    save_latest=False,
                )
            test_accs.append(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_acc_epoch = epoch
                save_best_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model_to_save.state_dict(),
                        "ema_state_dict": ema_to_save.state_dict() if exp.use_ema else None,
                        "optimizer": optimizer.state_dict(),
                    },
                    epoch + 1,
                    exp.out,
                )
            if np.mean(test_accs[-20:]) > best_mean_acc:
                best_mean_acc = np.mean(test_accs[-20:])
                best_mean_acc_epoch = epoch

            logger.info("Best top-1 acc: {:.2f}".format(best_acc))
            logger.info("Best top-1 epoch: {}".format(best_acc_epoch))
            logger.info("Best Mean top-1 acc: {:.2f}".format(best_mean_acc))
            logger.info("Best Mean top-1 epoch: {}".format(best_mean_acc_epoch))
            logger.info("Mean top-1 acc: {:.2f}\n".format(np.mean(test_accs[-20:])))


def test(test_loader, model, logger, no_progress, local_rank):
    batch_time = AvgMeter()
    data_time = AvgMeter()
    losses = AvgMeter()
    top1 = AvgMeter()
    top5 = AvgMeter()
    end = time.time()

    if not no_progress:
        test_loader = tqdm(test_loader, disable=local_rank not in [-1, 0])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
            outputs, loss = model((inputs, targets), is_train=False)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            prec1, prec5 = (
                reduce_tensor_sum(prec1) / dist.get_world_size(),
                reduce_tensor_sum(prec5) / dist.get_world_size(),
            )
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            if not no_progress and local_rank == 0:
                test_loader.set_description(
                    "Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. "
                    "top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                        batch=batch_idx + 1,
                        iter=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        loss=losses.avg,
                        top1=top1.avg,
                        top5=top5.avg,
                    )
                )
        if not exp.no_progress:
            test_loader.close()

    if local_rank == 0:
        logger.info("top-1 acc: {:.2f}".format(top1.avg))
        logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


if __name__ == "__main__":
    main()
