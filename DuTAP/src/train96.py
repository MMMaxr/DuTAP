
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import random
from tqdm import tqdm
from test import test
import utils.tools as tools
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
import os
from datetime import datetime
from tqdm import tqdm
import logging

current_datetime = datetime.now()
current_date = current_datetime.date()
logging.basicConfig(filename=f'../console_output/{current_date}_96.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()


def log_code_content(file_path,logger):
    with open(file_path, 'r') as file:
        code_content = file.read()
    logger.info(code_content)
    return code_content


def ContrastiveLoss(features, label, device):
    loss = torch.zeros(1).to(device)
    zero_feature = features[label == 0]
    zero_feature = zero_feature / zero_feature.norm(dim=-1, keepdim=True)
    one_feature = features[label == 1]
    one_feature = one_feature / one_feature.norm(dim=-1, keepdim=True)
    for i in range(zero_feature.shape[0]):
        for j in range(i + 1, one_feature.shape[0]):
            distance = torch.abs(zero_feature[i].view(-1) @ one_feature[j].view(-1))
            loss += distance
    if zero_feature.shape[0] * one_feature.shape[0] == 0:
        return loss
    else:
        return loss/(zero_feature.shape[0]*one_feature.shape[0])



def Task_Contra(cls1,cls2,cls3):
    cls1 = cls1 / cls1.norm(dim=-1, keepdim=True)
    cls2 = cls2 / cls2.norm(dim=-1, keepdim=True)
    cls3 = cls3 / cls3.norm(dim=-1, keepdim=True)

    sim_12 = torch.sum(torch.abs(cls1 * cls2), dim=-1)  # cls1 和 cls2 之间的相似度
    sim_13 = torch.sum(torch.abs(cls1 * cls3), dim=-1)  # cls1 和 cls3 之间的相似度
    sim_23 = torch.sum(torch.abs(cls2 * cls3), dim=-1)  # cls2 和 cls3 之间的相似度

    total_similarity = sim_12.sum() + sim_13.sum() + sim_23.sum()

    return total_similarity/(cls2.shape[0]*3)


def train(model,train_loader, test_loader, writer, args, device):

    model.to(device)
    print("CUDA_VISIBLE_DEVICES: ", os.environ.get('CUDA_VISIBLE_DEVICES'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = MultiStepLR(optimizer, args.scheduler_milestones, args.scheduler_rate)
    acc_best = 0.88
    min_loss = 100
    criterion = nn.BCELoss()
    if args.use_checkpoint == True:
        checkpoint = torch.load(os.path.join(args.checkpoint_path, args.checkpoint_name))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        acc_best = checkpoint['acc']
        print("checkpoint info:")
        print("epoch:", epoch+1, " acc:", acc_best)
    for e in range(args.epochs)   :
        model.train()
        step = 0
        loader = tqdm(train_loader)
        bi_loss = 0
        total_loss = 0
        contra_loss = 0
        all_preds = []
        all_labels = []
        sigmoid_pred = []
        for phrase1,phrase2,phrase3,label in loader:
            phrase1, phrase2, phrase3, label = phrase1.to(device).to(torch.float32), phrase2.to(
                device).to(torch.float32), phrase3.to(device).to(torch.float32), label.to(device).to(
                torch.float32)
            optimizer.zero_grad()
            predict,signal,feature,cls1,cls2,cls3 = model(phrase1, phrase2, phrase3)
            bi_label =torch.tensor([tools.score_to_class(s.item()) for s in label]).to(device).view(-1).to(torch.float32)
            one_hot_label = torch.nn.functional.one_hot(bi_label.to(torch.int64), num_classes=2).to(torch.float32)

            bi_predict = torch.argmax(predict,dim=1).to(torch.float32)
            #loss1 biloss
            loss1 = criterion(predict,one_hot_label)
            bi_loss += loss1.item()

            # task contrastive loss
            loss4 = Task_Contra(cls1,cls2,cls3)

            all_preds.extend(bi_predict.cpu().detach().numpy())
            all_labels.extend(bi_label.cpu().detach().numpy())
            sigmoid_pred.extend(predict.cpu().detach().numpy())

            #loss3 global contrastive loss
            loss3 = ContrastiveLoss(feature,bi_label,device)
            contra_loss += loss3.item()

            #loss
            loss = loss1+loss3+loss4*0.1
            total_loss += loss.item()
            loss.backward()

            optimizer.step()
            # scheduler.step()
            step += 1
            loader.set_description("Epoch:{} Step:{} Loss:{:.4f} loss1:{:.4f} loss3:{:.4f}".format(e, step, loss.item(),loss1.item(),loss3.item()))
        bi_loss /= step
        total_loss /= step
        contra_loss /= step

        recall = recall_score(all_labels, all_preds,average='weighted')
        precision = precision_score(all_labels, all_preds,average='weighted')
        f1 = f1_score(all_labels, all_preds,average='weighted')
        accuracy = accuracy_score(all_labels, all_preds)

        model.eval()
        test_loss,test_accuracy, test_recall, test_precision, test_f1 = test(model, test_loader,  criterion, logger,device,e)
        writer.log_train(bi_loss, test_loss, e)
        if test_accuracy >= acc_best :
            if test_loss<= min_loss:
                torch.save({'epoch': e, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'acc': test_accuracy}, os.path.join(args.checkpoint_path, f'{current_date}_{str(test_accuracy)[2:5]}.pth'))
                print('Save model!,Accuracy Improve:{:.2f}'.format(test_accuracy-acc_best))
                acc_best = test_accuracy


        print('Train\tloss:{:.4f}\tbi_loss:{:.4f}\t contra_loss:{:.4f}\tacc:{:.4f}\t recall:{:.4f}\t precision:{:.4f}\t f1:{:.4f}'.format(total_loss,bi_loss, contra_loss,accuracy, recall, precision, f1))
        if e ==10:
            log_code_content('model.py', logger)

        logger.info('Train{}\tbi_loss:{:.4f}\t acc:{:.4f}\t recall:{:.4f}\t precision:{:.4f}\t f1:{:.4f}'.format(e,bi_loss, accuracy, recall, precision, f1))
        logger.info('Test{}\tbi_loss:{:.4f}\t acc:{:.4f}\t recall:{:.4f}\t precision:{:.4f}\t f1:{:.4f}'.format(e,test_loss, test_accuracy, test_recall, test_precision, test_f1))






