# import
import os
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

from transformers import BertTokenizer
from transformers import BertConfig, BertModel

PTM = "hfl/chinese-bert-wwm-ext"
tokenizer = BertTokenizer.from_pretrained(PTM)

log_dir = '../XX/'
ren_train_file = '../ren_train.txt'
ren_test_file = '../ren_test.txt'
EPOCHS = 999
BATCH = 8
MAX_LEN = 160
CLIP = 1.0
LR = 1e-5

# data
# Love,Anxiety,Sorrow,Joy,Expect,Hate,Anger,Surprise,Neutral
def data_set(catagory='test', save=False):
    if catagory == 'test':
        ren_file = ren_test_file
    else:
        ren_file = ren_train_file
    data = []
    with open(ren_file, 'r') as rf:
        lines = rf.readlines()[1:]
        for line in lines:
            text = line.strip().split('|')[0]
            label = line.strip().split('|')[1:]
            data.append([text, label])
    return data

train_set = data_set('train')
test_set = data_set('test')

def prompting(text, label):
    prom_1, prom_0 = [0,0,0,0,0,0,0,0,0], [0,0,0,0,0,0,0,0,0]
    cho_list = ['喜爱吗？[MASK]。','喜欢吗？[MASK]。']
    cho = random.choice(cho_list)
    prom_1[0], prom_0[0] = cho, list(set(cho_list)-set([cho]))[0]
    cho_list = ['忧愁吗？[MASK]。','焦虑吗？[MASK]。']
    cho = random.choice(cho_list)
    prom_1[1], prom_0[1] = cho, list(set(cho_list)-set([cho]))[0]
    cho_list = ['伤心吗？[MASK]。','悲伤吗？[MASK]。']
    cho = random.choice(cho_list)
    prom_1[2], prom_0[2] = cho, list(set(cho_list)-set([cho]))[0]
    cho_list = ['喜悦吗？[MASK]。','高兴吗？[MASK]。']
    cho = random.choice(cho_list)
    prom_1[3], prom_0[3] = cho, list(set(cho_list)-set([cho]))[0]
    cho_list = ['期盼吗？[MASK]。','期待吗？[MASK]。']
    cho = random.choice(cho_list)
    prom_1[4], prom_0[4] = cho, list(set(cho_list)-set([cho]))[0]
    cho_list = ['怨恨吗？[MASK]。','讨厌吗？[MASK]。']
    cho = random.choice(cho_list)
    prom_1[5], prom_0[5] = cho, list(set(cho_list)-set([cho]))[0]
    cho_list = ['愤怒吗？[MASK]。','生气吗？[MASK]。']
    cho = random.choice(cho_list)
    prom_1[6], prom_0[6] = cho, list(set(cho_list)-set([cho]))[0]
    cho_list = ['惊讶吗？[MASK]。','吃惊吗？[MASK]。']
    cho = random.choice(cho_list)
    prom_1[7], prom_0[7] = cho, list(set(cho_list)-set([cho]))[0]
    cho_list = ['中立吗？[MASK]。','中性吗？[MASK]。']
    cho = random.choice(cho_list)
    prom_1[8], prom_0[8] = cho, list(set(cho_list)-set([cho]))[0]

    mask = [0,0,0,0,0,0,0,0,0]  
    for i in range(9):
        if label[i] == '1':
            mask[i] = 1  # 对2190 不679 是3221 否1415
    rand = list(range(9))
    state = np.random.get_state()
    np.random.shuffle(rand)
    np.random.set_state(state)
    np.random.shuffle(prom_1)
    np.random.set_state(state)
    np.random.shuffle(prom_0)
    np.random.set_state(state)
    np.random.shuffle(mask)
    text_0 = ''.join(prom_0) + text
    text_1 = ''.join(prom_1) + text
    return text_0, text_1, mask, rand

def data_loader(data_set, batch_size):
    random.shuffle(data_set)
    count = 0
    while count < len(data_set):
        batch = []
        size = min(batch_size, len(data_set) - count)
        for _ in range(size):
            text_0, text_1, mask, rand = prompting(data_set[count][0], data_set[count][1])
            batch.append((text_0, mask, rand))
            batch.append((text_0, mask, rand))
            batch.append((text_1, mask, rand))
            batch.append((text_1, mask, rand))
            count += 1
        yield batch

# model
class PCD_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained(PTM)
        self.norm = nn.LayerNorm(768)
        self.classifier = nn.Linear(768, 1)
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        output = self.classifier(self.norm(output[:,5:54:6,:]))
        output = torch.squeeze(output, -1)
        return output  # (batch, 9)

# run
def multi_circle_loss(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1], dtype = torch.float)
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()

def train(model, iterator, optimizer):
    model.train()
    epoch_loss, count = 0, 0
    iter_bar = tqdm(iterator, desc='Training')
    for _, batch in enumerate(iter_bar):
        count += 1
        optimizer.zero_grad()
        text, mask, rand = zip(*batch)
        token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
        logits_mlm = model(**token)
        mask = torch.FloatTensor(mask).to(device)
        mlm_loss = multi_circle_loss(logits_mlm, mask)
        kl_0 = F.kl_div(F.logsigmoid(logits_mlm[::4]), torch.sigmoid(logits_mlm[1::4]), reduction='batchmean')
        kl_1 = F.kl_div(F.logsigmoid(logits_mlm[1::4]), torch.sigmoid(logits_mlm[::4]), reduction='batchmean')
        kl_2 = F.kl_div(F.logsigmoid(logits_mlm[1::4]), torch.sigmoid(logits_mlm[2::4]), reduction='batchmean')
        kl_3 = F.kl_div(F.logsigmoid(logits_mlm[2::4]), torch.sigmoid(logits_mlm[1::4]), reduction='batchmean')
        kl_4 = F.kl_div(F.logsigmoid(logits_mlm[2::4]), torch.sigmoid(logits_mlm[3::4]), reduction='batchmean')
        kl_5 = F.kl_div(F.logsigmoid(logits_mlm[3::4]), torch.sigmoid(logits_mlm[2::4]), reduction='batchmean')
        kl_6 = F.kl_div(F.logsigmoid(logits_mlm[3::4]), torch.sigmoid(logits_mlm[::4]), reduction='batchmean')
        kl_7 = F.kl_div(F.logsigmoid(logits_mlm[::4]), torch.sigmoid(logits_mlm[3::4]), reduction='batchmean')
        loss = mlm_loss + (kl_0+kl_1+kl_2+kl_3+kl_4+kl_5+kl_6+kl_7) / 8
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP)  #梯度裁剪
        optimizer.step()
        iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
        epoch_loss += loss.item()
    return epoch_loss / count

def valid(model, iterator):
    model.eval()
    epoch_loss, count = 0, 0
    with torch.no_grad():
        iter_bar = tqdm(iterator, desc='Validation')
        for _, batch in enumerate(iter_bar):
            count += 1
            text, mask, rand = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            logits_mlm = model(**token)
            mask = torch.FloatTensor(mask).to(device)
            mlm_loss = multi_circle_loss(logits_mlm, mask)
            kl_0 = F.kl_div(F.logsigmoid(logits_mlm[::4]), torch.sigmoid(logits_mlm[1::4]), reduction='batchmean')
            kl_1 = F.kl_div(F.logsigmoid(logits_mlm[1::4]), torch.sigmoid(logits_mlm[::4]), reduction='batchmean')
            kl_2 = F.kl_div(F.logsigmoid(logits_mlm[1::4]), torch.sigmoid(logits_mlm[2::4]), reduction='batchmean')
            kl_3 = F.kl_div(F.logsigmoid(logits_mlm[2::4]), torch.sigmoid(logits_mlm[1::4]), reduction='batchmean')
            kl_4 = F.kl_div(F.logsigmoid(logits_mlm[2::4]), torch.sigmoid(logits_mlm[3::4]), reduction='batchmean')
            kl_5 = F.kl_div(F.logsigmoid(logits_mlm[3::4]), torch.sigmoid(logits_mlm[2::4]), reduction='batchmean')
            kl_6 = F.kl_div(F.logsigmoid(logits_mlm[3::4]), torch.sigmoid(logits_mlm[::4]), reduction='batchmean')
            kl_7 = F.kl_div(F.logsigmoid(logits_mlm[::4]), torch.sigmoid(logits_mlm[3::4]), reduction='batchmean')
            loss = mlm_loss + (kl_0+kl_1+kl_2+kl_3+kl_4+kl_5+kl_6+kl_7) / 8
            iter_bar.set_description('Iter (loss=%3.3f)'%loss.item())
            epoch_loss += loss.item()
    return epoch_loss / count

def run(model, train_list, valid_list, batch_size, learning_rate, epochs, name):
    log_file = log_dir+name+'.txt'
    with open(log_file, 'w') as log_f:
        log_f.write('epoch, train_loss, valid_loss\n')
    writer = SummaryWriter(log_dir)
    my_list = ['norm.weight','norm.bias','classifier.weight','classifier.bias']
    params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in my_list, model.named_parameters()))))
    base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in my_list, model.named_parameters()))))
    optimizer = optim.AdamW([{'params': base_params, 'lr': 1e-5}, {'params': params, 'lr': 1e-4}])
#     optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=1, verbose=True)
    stop = 0
    loss_list = []
    for epoch in range(epochs):
        print('Epoch: ' + str(epoch+1))
        train_iterator = data_loader(train_list, batch_size)
        valid_iterator = data_loader(valid_list, batch_size)
        train_loss = train(model, train_iterator, optimizer)
        valid_loss = valid(model, valid_iterator)
        writer.add_scalars(name, {'train_loss':train_loss, 'valid_loss':valid_loss}, epoch)
        scheduler.step(valid_loss)
        loss_list.append(valid_loss) 
        with open(log_file, 'a') as log_f:
            log_f.write('\n{epoch}, {train_loss: 3.3f}, {valid_loss: 3.3f}\n'.format(epoch=epoch+1, train_loss=train_loss, valid_loss=valid_loss))
        if valid_loss == min(loss_list) and valid_loss > 0.009:
            stop = 0
            torch.save(model.state_dict(), os.path.join(log_dir, name+'_'+str(valid_loss)[:4]+'.pt'))
        else:
            stop += 1
            if stop >= 3:
                break
    writer.close()

# train
random.shuffle(train_set)
model_1 = PCD_model().to(device)
valid_list = train_set[:6720]
train_list = train_set[6720:]
run(model_1, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='XX')

model_2 = PCD_model().to(device)
valid_list = train_set[6720:13440]
train_list = train_set[:6720] + train_set[13440:]
run(model_2, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='XX')

model_3 = PCD_model().to(device)
valid_list = train_set[13440:20160]
train_list = train_set[:13440] + train_set[20160:]
run(model_3, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='XX')

model_4 = PCD_model().to(device)
valid_list = train_set[20160:26880]
train_list = train_set[:20160] + train_set[26880:]
run(model_4, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='XX')

# evaluation
def prompting(text, label):
    prom_0 = ['喜爱吗？[MASK]。','忧愁吗？[MASK]。','伤心吗？[MASK]。','喜悦吗？[MASK]。','期盼吗？[MASK]。','怨恨吗？[MASK]。',\
            '愤怒吗？[MASK]。','惊讶吗？[MASK]。','中立吗？[MASK]。']  # 喜爱 忧愁 伤心 喜悦 期盼 怨恨 愤怒 惊讶
    prom_1 = ['喜欢吗？[MASK]。','焦虑吗？[MASK]。','悲伤吗？[MASK]。','高兴吗？[MASK]。','期待吗？[MASK]。','讨厌吗？[MASK]。',\
            '生气吗？[MASK]。','吃惊吗？[MASK]。','中性吗？[MASK]。']  # 喜爱 忧愁 伤心 喜悦 期盼 怨恨 愤怒 惊讶
    mask = [0,0,0,0,0,0,0,0,0]
    for i in range(9):
        if label[i] == '1':
            mask[i] = 1  # 对2190 不679 是3221 否1415
    rand = list(range(9))
    text_0 = ''.join(prom_0) + text
    text_1 = ''.join(prom_1) + text
    return text_0, text_1, mask, rand

def data_loader(data_set, batch_size):
    count = 0
    while count < len(data_set):
        batch = []
        size = min(batch_size, len(data_set) - count)
        for _ in range(size):
            text_0, text_1, mask, rand = prompting(data_set[count][0], data_set[count][1])
            batch.append((text_0, mask, rand))
            batch.append((text_1, mask, rand))
            count += 1
        yield batch

def outputs():
    pred_1, pred_2, pred_3, pred_4, label_1 = [], [], [], [], []
    
    model_1 = PCD_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'XX.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, mask, rand = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            logits_mlm = (model_1(**token)).detach().cpu()
            logits_mlm = logits_mlm[::2] + logits_mlm[1::2]
            pred_1.append(logits_mlm)
            label_1.append(torch.tensor(mask[::2]))
    pred_1 = torch.cat(pred_1, dim=0)
    label_1 = torch.cat(label_1, dim=0)
    
    model_1 = PCD_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'XX.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, mask, rand = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            logits_mlm = (model_1(**token)).detach().cpu()
            logits_mlm = logits_mlm[::2] + logits_mlm[1::2]
            pred_2.append(logits_mlm)
    pred_2 = torch.cat(pred_2, dim=0)
    
    model_1 = PCD_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'XX.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, mask, rand = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            logits_mlm = (model_1(**token)).detach().cpu()
            logits_mlm = logits_mlm[::2] + logits_mlm[1::2]
            pred_3.append(logits_mlm)
    pred_3 = torch.cat(pred_3, dim=0)
    
    model_1 = PCD_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'XX.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, mask, rand = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            logits_mlm = (model_1(**token)).detach().cpu()
            logits_mlm = logits_mlm[::2] + logits_mlm[1::2]
            pred_4.append(logits_mlm)
    pred_4 = torch.cat(pred_4, dim=0)

    return pred_1+pred_2+pred_3+pred_4, label_1

import math
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, coverage_error, label_ranking_loss, label_ranking_average_precision_score

def sigmoids(x, t):
    return 1/(1 + math.exp(-x+t))

def test(pred, label):
    zero = torch.zeros_like(pred)
    one = torch.ones_like(pred)

    label_all = []
    for j in range(label.shape[0]):
        label_all.append(label[j][:8].int().tolist())

    temp_max = -99.0
    for temp in tqdm(range(140)):
        th = float(temp) / 10.0 - 10.0
        sigm_all, bina_all = [], []
        bina = torch.where(pred > (th), one, zero)

        for j in range(pred.shape[0]):
            sigm_all.append([sigmoids(pred[j][0], th), sigmoids(pred[j][1], th), sigmoids(pred[j][2], th), sigmoids(pred[j][3], th),\
                             sigmoids(pred[j][4], th), sigmoids(pred[j][5], th), sigmoids(pred[j][6], th), sigmoids(pred[j][7], th)])
            temp_bina = [0,0,0,0,0,0,0,0]
            temp_bina[0] = int(bina[j][0])
            temp_bina[1] = int(bina[j][1])
            temp_bina[2] = int(bina[j][2])
            temp_bina[3] = int(bina[j][3])
            temp_bina[4] = int(bina[j][4])
            temp_bina[5] = int(bina[j][5])
            temp_bina[6] = int(bina[j][6])
            temp_bina[7] = int(bina[j][7])
            bina_all.append(temp_bina)

        f1 = f1_score(label_all, bina_all, average='micro')/0.65+f1_score(label_all, bina_all, average='macro')/0.53
        ap = label_ranking_average_precision_score(label_all, sigm_all)/0.81
        ce = coverage_error(label_all, sigm_all)/1.91
        rl = label_ranking_loss(label_all, sigm_all)/0.09
        if f1+ap-ce-rl > temp_max:
            temp_max = f1+ap-ce-rl
            print(th, temp_max)
    return 0

def outputs():
    pred_1, pred_2, pred_3, pred_4, label_1 = [], [], [], [], []
    
    model_1 = PCD_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'XX.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, mask, rand = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            logits_mlm = (model_1(**token)).detach().cpu()
            logits_mlm = logits_mlm[::2] + logits_mlm[1::2]
            pred_1.append(logits_mlm)
            label_1.append(torch.tensor(mask[::2]))
    pred_1 = torch.cat(pred_1, dim=0)
    label_1 = torch.cat(label_1, dim=0)
    
    model_1 = PCD_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'XX.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, mask, rand = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            logits_mlm = (model_1(**token)).detach().cpu()
            logits_mlm = logits_mlm[::2] + logits_mlm[1::2]
            pred_2.append(logits_mlm)
    pred_2 = torch.cat(pred_2, dim=0)
    
    model_1 = PCD_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'XX.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, mask, rand = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            logits_mlm = (model_1(**token)).detach().cpu()
            logits_mlm = logits_mlm[::2] + logits_mlm[1::2]
            pred_3.append(logits_mlm)
    pred_3 = torch.cat(pred_3, dim=0)
    
    model_1 = PCD_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'XX.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, mask, rand = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            logits_mlm = (model_1(**token)).detach().cpu()
            logits_mlm = logits_mlm[::2] + logits_mlm[1::2]
            pred_4.append(logits_mlm)
    pred_4 = torch.cat(pred_4, dim=0)

    return pred_1+pred_2+pred_3+pred_4, label_1

pred, label = outputs()

def test(pred, label):
    zero = torch.zeros_like(pred)
    one = torch.ones_like(pred)

    label_all = []
    for j in range(label.shape[0]):
        label_all.append(label[j][:8].int().tolist())

    th = XX
    sigm_all, bina_all = [], []
    bina = torch.where(pred > (th), one, zero)

    for j in range(pred.shape[0]):
        sigm_all.append([sigmoids(pred[j][0], th), sigmoids(pred[j][1], th), sigmoids(pred[j][2], th), sigmoids(pred[j][3], th),\
                         sigmoids(pred[j][4], th), sigmoids(pred[j][5], th), sigmoids(pred[j][6], th), sigmoids(pred[j][7], th)])
        temp_bina = [0,0,0,0,0,0,0,0]
        temp_bina[0] = int(bina[j][0])
        temp_bina[1] = int(bina[j][1])
        temp_bina[2] = int(bina[j][2])
        temp_bina[3] = int(bina[j][3])
        temp_bina[4] = int(bina[j][4])
        temp_bina[5] = int(bina[j][5])
        temp_bina[6] = int(bina[j][6])
        temp_bina[7] = int(bina[j][7])
        bina_all.append(temp_bina)

    print('micro_precision: ', precision_score(label_all, bina_all, average='micro'))
    print('micro_recall: ', recall_score(label_all, bina_all, average='micro'))
    print('micro_f1: ', f1_score(label_all, bina_all, average='micro'))
    print('macro_precision: ', precision_score(label_all, bina_all, average='macro'))
    print('macro_recall: ', recall_score(label_all, bina_all, average='macro'))
    print('macro_f1: ', f1_score(label_all, bina_all, average='macro'))
    print('ap: ', label_ranking_average_precision_score(label_all, sigm_all))
    print('ce: ', coverage_error(label_all, sigm_all))
    print('rl: ', label_ranking_loss(label_all, sigm_all))
    return 0

test(pred, label)
