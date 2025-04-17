import torch
from tqdm import tqdm
import utils.tools as tools
import torch.nn as nn
from sklearn.metrics import f1_score, precision_score, recall_score,accuracy_score
import logging
from datetime import datetime
from dataset import FNIRSData
from torch.utils.data import DataLoader
def test(model,test_loader, criterion, logger,device,e):
    with torch.no_grad():
        loader = tqdm(test_loader)
        step = 0
        bi_loss = 0
        all_preds = []
        all_labels = []
        sigmoid_pred = []
        for phrase1,phrase2,phrase3,label in loader:
            phrase1, phrase2, phrase3, label = phrase1.to(device).to(torch.float32), phrase2.to(
                device).to(torch.float32), phrase3.to(device).to(torch.float32), label.to(device).to(
                torch.float32)
            predict,signal,feature,cls1,cls2,cls3 = model(phrase1, phrase2, phrase3)
            # predict = predict.view(-1)
            bi_label =torch.tensor([tools.score_to_class(s.item()) for s in label]).to(device).view(-1).to(torch.float32)
            one_hot_label = torch.nn.functional.one_hot(bi_label.to(torch.int64), num_classes=2).to(torch.float32)
            bi_predict = torch.argmax(predict,dim=1).to(torch.float32)

            all_preds.extend(bi_predict.cpu().detach().numpy())
            all_labels.extend(bi_label.cpu().detach().numpy())
            sigmoid_pred.extend(signal.cpu().detach().numpy())

            #loss1 biloss
            loss1 = criterion(predict,one_hot_label)
            bi_loss+= loss1.item()
            step += 1
            loader.set_description('test step:{} loss:{:.4f}'.format(step, loss1.item()))
        bi_loss /= step

        #metrics
        recall = recall_score(all_labels, all_preds,average='macro')
        precision = precision_score(all_labels, all_preds,average='macro')
        f1 = f1_score(all_labels, all_preds,average='macro')
        accuracy = accuracy_score(all_labels, all_preds)

        print(f'pred',sigmoid_pred)
        print(f'pred',all_preds)
        print(f'label',list(map(int,all_labels)))
        print('Test\tbi_loss:{:.4f}\t acc:{:.4f}\t recall:{:.4f}\t precision:{:.4f}\t f1:{:.4f}'.format(bi_loss, accuracy, recall, precision, f1))
        return bi_loss, accuracy, recall, precision, f1


# dataset_test = FNIRSData('/home/maxinran/FNIRS/data/', 'test')
# test_loader = DataLoader(dataset=dataset_test, batch_size=8, shuffle=False, pin_memory=True,
#                                   drop_last=False)


