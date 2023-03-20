import torch.nn as nn
from networks import get_model
from torch.utils.data import Dataset, DataLoader
from utils import *
from loss import supcon_loss

import time
import numpy as np
from torchvision import transforms, datasets
import argparse

from datasets.supcon_dataset import FaceDataset, DEVICE_INFOS

from datasets import get_datasets, TwoCropTransform

torch.backends.cudnn.benchmark = True

def log_f(f, console=True):
    def log(msg):
        with open(f, 'a') as file:
            file.write(msg)
            file.write('\n')
        if console:
            print(msg)
    return log


def binary_func_sep(model, feat, scale, label, UUID, ce_loss_record_0, ce_loss_record_1, ce_loss_record_2):
    ce_loss = nn.BCELoss().cuda()
    indx_0 = (UUID == 0).cpu()
    label = label.float()
    correct_0, correct_1, correct_2 = 0, 0, 0
    total_0, total_1, total_2 = 1, 1, 1

    if indx_0.sum().item() > 0:
        logit_0 = model.fc0(feat[indx_0], scale[indx_0]).squeeze()
        cls_loss_0 = ce_loss(logit_0, label[indx_0])
        predicted_0 = (logit_0 > 0.5).float()
        total_0 += len(logit_0)
        correct_0 += predicted_0.cpu().eq(label[indx_0].cpu()).sum().item()
    else:
        logit_0 = []
        cls_loss_0 = torch.zeros(1).cuda()

    indx_1 = (UUID == 1).cpu()
    if indx_1.sum().item() > 0:
        logit_1 = model.fc1(feat[indx_1], scale[indx_1]).squeeze()
        cls_loss_1 = ce_loss(logit_1, label[indx_1])
        predicted_1 = (logit_1 > 0.5).float()
        total_1 += len(logit_1)
        correct_1 += predicted_1.cpu().eq(label[indx_1].cpu()).sum().item()
    else:
        logit_1 = []
        cls_loss_1 = torch.zeros(1).cuda()

    indx_2 = (UUID == 2).cpu()
    if indx_2.sum().item() > 0:
        logit_2 = model.fc2(feat[indx_2], scale[indx_2]).squeeze()
        cls_loss_2 = ce_loss(logit_2, label[indx_2])
        predicted_2 = (logit_2 > 0.5).float()
        total_2 += len(logit_2)
        correct_2 += predicted_2.cpu().eq(label[indx_2].cpu()).sum().item()
    else:
        logit_2 = []
        cls_loss_2 = torch.zeros(1).cuda()

    ce_loss_record_0.update(cls_loss_0.data.item(), len(logit_0))
    ce_loss_record_1.update(cls_loss_1.data.item(), len(logit_1))
    ce_loss_record_2.update(cls_loss_2.data.item(), len(logit_2))
    return (cls_loss_0 + cls_loss_1 + cls_loss_2) / 3, (correct_0, correct_1, correct_2, total_0, total_1, total_2)

def main(args):

    if args.pretrain == 'imagenet':
        normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    else:
        normalizer = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    train_transform_list = [
        transforms.RandomResizedCrop(256, scale=(args.train_scale_min, 1.), ratio=(1., 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer
    ]

    if args.train_rotation:
        train_transform_list = [transforms.RandomRotation(degrees=(-180, 180))] + train_transform_list

    train_transform = transforms.Compose(train_transform_list)

    test_transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(args.test_scale, args.test_scale), ratio=(1., 1.)),
        transforms.ToTensor(),
        normalizer
    ])

    data_name_list_train, data_name_list_test = protocol_decoder(args.protocol)

    train_set = get_datasets(args.data_dir, FaceDataset, train=True, protocol=args.protocol, img_size=args.img_size, map_size=32, transform=train_transform, debug_subset_size=args.debug_subset_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_set = get_datasets(args.data_dir, FaceDataset, train=False, protocol=args.protocol, img_size=args.img_size, map_size=32, transform=test_transform, debug_subset_size=args.debug_subset_size)
    test_loader = DataLoader(test_set[data_name_list_test[0]], batch_size=args.batch_size, shuffle=False, num_workers=4)

    live_cls_list = []
    spoof_cls_list = []
    for dataset in data_name_list_train:
        live_cls_list += DEVICE_INFOS[dataset]['live']
        spoof_cls_list += DEVICE_INFOS[dataset]['spoof']
    total_cls_num = 2

    device2idx = {pattern: idx for idx, pattern in enumerate(spoof_cls_list)}

    max_iter = args.num_epochs*len(train_loader)
    # make dirs
    model_root_path = os.path.join(args.result_path, args.result_name, "model")
    check_folder(model_root_path)
    score_root_path = os.path.join(args.result_path, args.result_name, "score")
    check_folder(score_root_path)
    csv_root_path = os.path.join(args.result_path, args.result_name, "csv")
    check_folder(csv_root_path)
    log_path = os.path.join(args.result_path, args.result_name, "log.txt")
    print = log_f(log_path)

    if args.pretrain == 'imagenet':
        model = get_model(args.model_type, max_iter, total_cls_num, pretrained=True, normed_fc=args.normfc, use_bias=args.usebias, simsiam=True if args.feat_loss == 'simsiam' else False)
    else:
        model = get_model(args.model_type, max_iter, total_cls_num, pretrained=False, normed_fc=args.normfc, use_bias=args.usebias, simsiam=True if args.feat_loss == 'simsiam' else False)
        model_path = os.path.join('pretrained', args.pretrain, 'model', "{}_best.pth".format(args.pretrain))
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])

    # model = nn.DataParallel(model).cuda()
    model = model.cuda()
    # def optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer_linear = torch.optim.SGD(model.fc.parameters(), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # def scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)


    if args.resume:
        model_path = os.path.join(model_root_path, "{}_p{}_best.pth".format(args.model_type, args.protocol))
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        # args.start_epoch = ckpt['epoch']
        scheduler = ckpt['scheduler']

    # metrics
    eva = {
        "best_epoch": -1,
        "best_HTER": 100,
        "best_auc": -100
    }

    ce_loss = nn.BCELoss().cuda()


    for epoch in range(args.start_epoch, args.num_epochs):

        ce_loss_record_0 = AvgrageMeter()
        ce_loss_record_1 = AvgrageMeter()
        ce_loss_record_2 = AvgrageMeter()
        feat_loss_record = AvgrageMeter()
        ########################### train ###########################
        model.train()
        correct = 0
        total = 0

        for i, sample_batched in enumerate(train_loader):
            lr = optimizer.param_groups[0]['lr']

            image_x_v1, image_x_v2, label, UUID = sample_batched["image_x_v1"].cuda(), sample_batched["image_x_v2"].cuda(), sample_batched["label"].cuda(), sample_batched["UUID"].cuda()

            image_x = torch.cat([image_x_v1, image_x_v2])
            feat, scale = model(image_x, out_type='feat', scale=args.scale)
            UUID2 = torch.cat([UUID, UUID])
            label2 = torch.cat([label, label])
            cls_loss, stat = binary_func_sep(model, feat, scale, label2, UUID2, ce_loss_record_0, ce_loss_record_1, ce_loss_record_2)
            correct_0, correct_1, correct_2, total_0, total_1, total_2 = stat

            feat_normed = F.normalize(feat)
            f1, f2 = torch.split(feat_normed, [len(image_x_v1), len(image_x_v1)], dim=0)
            if args.feat_loss == 'supcon':
                feat_loss = supcon_loss(torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1), UUID * 10 + label, temperature=args.temperature)
            else:
                feat_loss = torch.zeros(1).cuda()

            loss_all = cls_loss + args.feat_loss_weight * feat_loss

            if args.align:
                model.snapshot_weight()

            model.zero_grad()
            loss_all.backward()
            optimizer.step()
            feat_loss_record.update(feat_loss.data.item(), len(image_x_v1))

            if epoch >= args.align_epoch and args.align == 'v4':
                angle = model.update_weight_v4(alpha=args.alpha)
            else:
                angle = -1.0

            log_info = "epoch:{:d}, mini-batch:{:d}, lr={:.4f}, angle={:.4f}, feat_loss={:.4f}, ce_loss_0={:.4f}, ce_loss_1={:.4f}, ce_loss_2={:.4f}, ACC_0={:.4f}, ACC_1={:.4f}, ACC_2={:.4f}".format(
                epoch + 1, i + 1, lr, angle, feat_loss_record.avg, ce_loss_record_0.avg, ce_loss_record_1.avg,
                ce_loss_record_2.avg, 100. * correct_0 / total_0, 100. * correct_1 / total_1, 100. * correct_2 / total_2)

            if i % args.print_freq == args.print_freq - 1:
                print(log_info)


        # whole epoch average
        print("epoch:{:d}, Train: lr={:f}, Loss={:.4f}".format(epoch + 1, lr, ce_loss_record_0.avg))
        scheduler.step()

        ############################ test ###########################
        if args.protocol  == "I_C_M_to_O":
            epoch_test = 5
        else:
            epoch_test = args.eval_preq

        if epoch % epoch_test == epoch_test-1:

            score_path = os.path.join(score_root_path, "epoch_{}".format(epoch+1))
            check_folder(score_path)

            model.eval()
            with torch.no_grad():
                start_time = time.time()
                scores_list = []
                for i, sample_batched in enumerate(test_loader):
                    image_x, live_label, UUID = sample_batched["image_x_v1"].cuda(), sample_batched["label"].cuda(), sample_batched["UUID"].cuda()
                    _, penul_feat, logit = model(image_x, out_type='all', scale=args.scale)

                    for i in range(len(logit)):
                        scores_list.append("{} {}\n".format(logit.squeeze()[i].item(), live_label[i].item()))


            map_score_val_filename = os.path.join(score_path, "{}_score.txt".format(data_name_list_test[0]))
            print("score: write test scores to {}".format(map_score_val_filename))
            with open(map_score_val_filename, 'w') as file:
                file.writelines(scores_list)

            test_ACC, fpr, FRR, HTER, auc_test, test_err, tpr = performances_val(map_score_val_filename)
            print("## {} score:".format(data_name_list_test[0]))
            print("epoch:{:d}, test:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}, TPR={:.4f}".format(
                epoch + 1, test_ACC, HTER, auc_test, test_err, test_ACC, tpr))
            print("test phase cost {:.4f}s".format(time.time() - start_time))

            if auc_test-HTER>=eva["best_auc"]-eva["best_HTER"]:
                eva["best_auc"] = auc_test
                eva["best_HTER"] = HTER
                eva["tpr95"] = tpr
                eva["best_epoch"] = epoch+1
                model_path = os.path.join(model_root_path, "{}_p{}_best.pth".format(args.model_type, args.protocol))
                torch.save({
                    'epoch': epoch+1,
                    'state_dict':model.state_dict(),
                    'optimizer':optimizer.state_dict(),
                    'scheduler':scheduler,
                    'args':args,
                    'eva': (HTER, auc_test)
                }, model_path)
                print("Model saved to {}".format(model_path))

            print("[Best result] epoch:{}, HTER={:.4f}, AUC={:.4f}".format(eva["best_epoch"],  eva["best_HTER"], eva["best_auc"]))

            model_path = os.path.join(model_root_path, "{}_p{}_recent.pth".format(args.model_type, args.protocol))
            torch.save({
                'epoch': epoch+1,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler,
                'args':args,
                'eva': (HTER, auc_test)
            }, model_path)
            print("Model saved to {}".format(model_path))


    epochs_saved = np.array([int(dir.replace("epoch_", "")) for dir in os.listdir(score_root_path)])
    epochs_saved = np.sort(epochs_saved)
    last_n_epochs = epochs_saved[::-1][:10]

    HTERs, AUROCs, TPRs = [], [], []
    for epoch in last_n_epochs:
        map_score_val_filename = os.path.join(score_root_path, "epoch_{}".format(epoch), "{}_score.txt".format(data_name_list_test[0]))
        test_ACC, fpr, FRR, HTER, auc_test, test_err, tpr = performances_val(map_score_val_filename)
        HTERs.append(HTER)
        AUROCs.append(auc_test)
        TPRs.append(tpr)
        print("## {} score:".format(data_name_list_test[0]))
        print("epoch:{:d}, test:  val_ACC={:.4f}, HTER={:.4f}, AUC={:.4f}, val_err={:.4f}, ACC={:.4f}, TPR={:.4f}".format(
                epoch + 1, test_ACC, HTER, auc_test, test_err, test_ACC, tpr))

    os.makedirs('summary', exist_ok=True)
    file = open(f"summary/{args.result_name_no_protocol}.txt", "a")
    L = [f"{args.summary}\t\t{eva['best_epoch']}\t{eva['best_HTER']*100:.2f}\t{eva['best_auc']*100:.2f}" +
         f"\t{np.array(HTERs).mean()*100:.2f}\t{np.array(HTERs).std()*100:.2f}\t{np.array(AUROCs).mean()*100:.2f}\t{np.array(AUROCs).std()*100:.2f}\t"+
         f"{np.array(TPRs).mean()*100:.2f}\t{np.array(TPRs).std()*100:.2f}\n"]
    file.writelines(L)
    file.close()


def parse_args():
    parser = argparse.ArgumentParser()
    # build dirs
    parser.add_argument('--data_dir', type=str, default="datasets/FAS", help='YOUR_Data_Dir')
    parser.add_argument('--result_path', type=str, default='./results', help='root result directory')
    parser.add_argument('--protocol', type=str, default="O_C_I_to_M", help='O_C_I_to_M, O_M_I_to_C, O_C_M_to_I, I_C_M_to_O, O_to_O')
    # training settings
    parser.add_argument('--model_type', type=str, default="ResNet18_lgt", help='model_type')
    parser.add_argument('--eval_preq', type=int, default=1, help='batch size')
    parser.add_argument('--img_size', type=int, default=256, help='img size')

    parser.add_argument('--pretrain', type=str, default='imagenet', help='imagenet')
    parser.add_argument('--batch_size', type=int, default=96, help='batch size')
    parser.add_argument('--align', type=str, default='v4')
    parser.add_argument('--align_epoch', type=int, default=20)
    parser.add_argument('--normfc', type=str2bool, default=False)
    parser.add_argument('--usebias', type=str2bool, default=True)
    parser.add_argument('--train_rotation', type=str2bool, default=True, help='batch size')
    parser.add_argument('--train_scale_min', type=float, default=0.2, help='batch size')
    parser.add_argument('--test_scale', type=float, default=0.9, help='batch size')
    parser.add_argument('--base_lr', type=float, default=0.005, help='base learning rate')
    parser.add_argument('--alpha', type=float, default=0.995, help='')
    parser.add_argument('--scale', type=str, default='1', help='')
    parser.add_argument('--feat_loss', type=str, default='supcon', help='')
    parser.add_argument('--feat_loss_weight', type=float, default=0.1, help='')
    parser.add_argument('--seed', type=int, default=0, help='batch size')
    parser.add_argument('--temperature', type=float, default=0.1, help='')

    parser.add_argument('--device', type=str, default='0', help='device id, format is like 0,1,2')
    parser.add_argument('--start_epoch', type=int, default=0, help='start epoch')
    parser.add_argument('--num_epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--print_freq', type=int, default=3, help='print frequency')
    parser.add_argument('--resume', type=bool, default=False, help='print frequency')

    parser.add_argument('--step_size', type=int, default=40, help='how many epochs lr decays once')
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--trans', type=str, default="p", help="different pre-process")
    # optimizer
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    # debug
    parser.add_argument('--debug_subset_size', type=int, default=None)
    return parser.parse_args()


def str2bool(x):
    return x.lower() in ('true')


if __name__ == '__main__':
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    pretrain_alias = {
        "imagenet": "img",
    }
    args.result_name_no_protocol = f"pre({pretrain_alias[args.pretrain]})_pgirm({args.align}-{args.align_epoch})_normfc({args.normfc})_bsz({args.batch_size})_rot({args.train_rotation})" + \
                       f"_smin({args.train_scale_min})_tscl({args.test_scale})_lr({args.base_lr})_alpha({args.alpha})_scale({args.scale})"+\
                       f"_floss({args.feat_loss})_flossw({args.feat_loss_weight})_tmp({args.temperature})_seed({args.seed})"

    args.result_name = f"{args.protocol}_" + args.result_name_no_protocol

    info_list = [args.protocol, pretrain_alias[args.pretrain], args.align, args.align_epoch, args.batch_size, args.train_rotation, args.train_scale_min,
                 args.test_scale, args.base_lr, args.alpha, args.scale, args.feat_loss, args.feat_loss_weight, args.temperature, args.seed]

    args.summary = "\t".join([str(info) for info in info_list])
    print(args.result_name)
    print(args.summary)

    if args.protocol == "I_C_M_to_O":
        args.num_epochs *= 3
        args.step_size *= 3

    if args.scale.lower() == 'none':
        args.scale = None
    else:
        args.scale = float(args.scale)

    main(args=args)