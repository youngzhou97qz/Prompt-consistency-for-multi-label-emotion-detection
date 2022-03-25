import os
import jieba
import pickle
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

dict_dir = '/home/dango/multimodal/symbolic/dict/'
log_dir = '/home/dango/multimodal/symbolic/ren_otherss_log/'
ren_train_file = '/home/dango/multimodal/symbolic/data/ren_train.txt'
ren_test_file = '/home/dango/multimodal/symbolic/data/ren_test.txt'
ren_train_comet = '/home/dango/multimodal/symbolic/data/ren_train_comet.txt'
ren_test_comet = '/home/dango/multimodal/symbolic/data/ren_test_comet.txt'
ren_train_clst = '/home/dango/multimodal/symbolic/data/ren_train_cluster.txt'
ren_test_clst = '/home/dango/multimodal/symbolic/data/ren_test_cluster.txt'
ren_clst_dict = '/home/dango/multimodal/symbolic/data/ren_cluster_dict.txt'
EPOCHS = 999
BATCH = 16
MAX_LEN = 188
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
            label = []
            temp_label = line.strip().split('|')[-9:]
            for i in range(9):
                label.append(int(temp_label[i]))
            data.append([text, label])
    return data

train_set = data_set('train')
test_set = data_set('test')

def add_comet(catagory='test'):
    if catagory == 'test':
        file = ren_test_comet
        temp_data = test_set
    else:
        file = ren_train_comet
        temp_data = train_set
    data = []
    count = 0
    with open(file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            temp = ''
            attr = line.strip().split('|')[4]
            if attr != '' and attr != 'none':
                temp += attr + '，'
            reac = line.strip().split('|')[8]
            if reac != '' and reac != 'none':
                temp += reac + '，'
            want = line.strip().split('|')[9]
            if want != '' and want != 'none':
                temp += want + '。'
            data.append([temp+temp_data[count][0], temp_data[count][1]])
            count += 1
    return data

train_set = add_comet('train')
test_set = add_comet('test')

clst_dict = {}
with open(ren_clst_dict, 'r') as f:
    lines = f.readlines()[1:]
    for line in lines:
        clst_dict[line.strip().split('|')[0]] = line.strip().split('|')[1]
            
def add_clst(catagory='test'):
    if catagory == 'test':
        file = ren_test_clst
        temp_data = test_set
    else:
        file = ren_train_clst
        temp_data = train_set
    data = []
    count = 0
    with open(file, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            clst = line.strip().split('|')[0]
            data.append([clst + '：' + clst_dict[clst] + '。', temp_data[count][0], temp_data[count][1]])
            count += 1
    return data

train_set = add_clst('train')
test_set = add_clst('test')

def read_dict(emo):
    file = dict_dir + emo + '.pkl'
    with open(file, 'rb') as f:
        return pickle.load(f)
    
dict_hao = read_dict('好')  # love
dict_le = read_dict('乐')  # joy
dict_ai = read_dict('哀')  # sorrow
dict_nu = read_dict('怒')  # anger
dict_ju = read_dict('惧')  # fear
dict_wu = read_dict('恶')  # hate
dict_jing = read_dict('惊')  # surprise

def dict_emo(text):
    words = jieba.lcut(text)
    # text|Love,Anxiety,Sorrow,Joy,Expect,Hate,Anger,Surprise,Neutral
    love, joyy, sorr, ange, anxi, hate, surp = [], [], [], [], [], [], []
    for w in words:
        if w in dict_hao:
            love.append(w)
        elif w in dict_le:
            joyy.append(w)
        elif w in dict_ai:
            sorr.append(w)
        elif w in dict_nu:
            ange.append(w)
        elif w in dict_ju:
            anxi.append(w)
        elif w in dict_wu:
            hate.append(w)
        elif w in dict_jing:
            surp.append(w)
    return love, anxi, sorr, joyy, hate, ange, surp

def pre_pro(text):
    love, anxi, sorr, joyy, hate, ange, surp = dict_emo(text)
    sum_list = love+anxi+sorr+joyy+hate+ange+surp
    if len(sum_list) <= 0:
        return '中立。', text, [0,0,0,0,0,0,0,0,1]
    else:
        emo_num = [0,0,0,0,0,0,0,0,0]
        emo_num[0] = len(love)
        emo_num[1] = len(anxi)
        emo_num[2] = len(sorr)
        emo_num[3] = len(joyy)
        emo_num[4] = len(hate)
        emo_num[5] = len(ange)
        emo_num[6] = len(surp)
        prefix = '，'.join(sum_list)
        return prefix + '。', text, emo_num

def data_loader(data_set, batch_size):
    random.shuffle(data_set)
    count = 0
    while count < len(data_set):
        batch = []
        size = min(batch_size, len(data_set) - count)
        for _ in range(size):
            prefix, text, emo_num = pre_pro(data_set[count][1])
            # print(prefix + data_set[count][0] + text)
            batch.append((prefix + data_set[count][0] + text, emo_num, data_set[count][2]))
            batch.append((prefix + data_set[count][0] + text, emo_num, data_set[count][2]))
            count += 1
        yield batch

# model
class Dict_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BertModel.from_pretrained(PTM)
        self.norm = nn.LayerNorm(768+9)
        self.classifier = nn.Linear(768+9, 9)
    def forward(self, input_ids, attention_mask, token_type_ids, emo_num):
        output = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state  # (B, LEN, 768)
        output = torch.cat((output[:,0,:], emo_num), -1)  # cls
        return self.classifier(self.norm(output))  # (batch, 9)

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
        text, emo_num, label = zip(*batch)
        token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
        emo_num = torch.FloatTensor(emo_num).to(device)
        input_ids, attention_mask, token_type_ids = token['input_ids'], token['attention_mask'], token['token_type_ids']
        logits_mlm = model(input_ids, attention_mask, token_type_ids, emo_num)
        label = torch.FloatTensor(label).to(device)
        mlm_loss = multi_circle_loss(logits_mlm, label)
        kl_0 = F.kl_div(F.logsigmoid(logits_mlm[::2]), torch.sigmoid(logits_mlm[1::2]), reduction='batchmean')
        kl_1 = F.kl_div(F.logsigmoid(logits_mlm[1::2]), torch.sigmoid(logits_mlm[::2]), reduction='batchmean')
        loss = mlm_loss + (kl_0+kl_1) / 2
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
            text, emo_num, label = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            emo_num = torch.FloatTensor(emo_num).to(device)
            input_ids, attention_mask, token_type_ids = token['input_ids'], token['attention_mask'], token['token_type_ids']
            logits_mlm = model(input_ids[::2], attention_mask[::2], token_type_ids[::2], emo_num[::2])
            label = torch.FloatTensor(label).to(device)
            loss = multi_circle_loss(logits_mlm, label[::2])
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

model_1 = Dict_model().to(device)
valid_list = train_set[:6720]
train_list = train_set[6720:]
run(model_1, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='clst+dict+AWR_1')

model_2 = Dict_model().to(device)
valid_list = train_set[6720:13440]
train_list = train_set[:6720] + train_set[13440:]
run(model_2, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='clst+dict+AWR_2')

model_3 = Dict_model().to(device)
valid_list = train_set[13440:20160]
train_list = train_set[:13440] + train_set[20160:]
run(model_3, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='clst+dict+AWR_3')

model_4 = Dict_model().to(device)
valid_list = train_set[20160:26880]
train_list = train_set[:20160] + train_set[26880:]
run(model_4, train_list, valid_list, batch_size=BATCH, learning_rate=LR, epochs=EPOCHS, name='clst+dict+AWR_4')

def data_loader(data_set, batch_size):
    count = 0
    while count < len(data_set):
        batch = []
        size = min(batch_size, len(data_set) - count)
        for _ in range(size):
            prefix, text, emo_num = pre_pro(data_set[count][1]+data_set[count][2])
            batch.append((prefix + data_set[count][0] + data_set[count][1], emo_num, data_set[count][3]))
            count += 1
        yield batch

def outputs():
    pred_1, pred_2, pred_3, pred_4, label_1 = [], [], [], [], []
    
    model_1 = Dict_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'wotext_4_2.41.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, emo_num, label = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            emo_num = torch.FloatTensor(emo_num).to(device)
            input_ids, attention_mask, token_type_ids = token['input_ids'], token['attention_mask'], token['token_type_ids']
            logits_mlm = model_1(input_ids, attention_mask, token_type_ids, emo_num)
            pred_1.append(logits_mlm)
            label_1.append(torch.tensor(label))
    pred_1 = torch.cat(pred_1, dim=0)
    label_1 = torch.cat(label_1, dim=0)
    
    model_1 = Dict_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'wotext_3_2.40.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, emo_num, label = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            emo_num = torch.FloatTensor(emo_num).to(device)
            input_ids, attention_mask, token_type_ids = token['input_ids'], token['attention_mask'], token['token_type_ids']
            logits_mlm = model_1(input_ids, attention_mask, token_type_ids, emo_num)
            pred_2.append(logits_mlm)
    pred_2 = torch.cat(pred_2, dim=0)
    
    model_1 = Dict_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'wotext_2_2.38.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, emo_num, label = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            emo_num = torch.FloatTensor(emo_num).to(device)
            input_ids, attention_mask, token_type_ids = token['input_ids'], token['attention_mask'], token['token_type_ids']
            logits_mlm = model_1(input_ids, attention_mask, token_type_ids, emo_num)
            pred_3.append(logits_mlm)
    pred_3 = torch.cat(pred_3, dim=0)
    
    model_1 = Dict_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'wotext_1_2.38.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(train_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, emo_num, label = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            emo_num = torch.FloatTensor(emo_num).to(device)
            input_ids, attention_mask, token_type_ids = token['input_ids'], token['attention_mask'], token['token_type_ids']
            logits_mlm = model_1(input_ids, attention_mask, token_type_ids, emo_num)
            pred_4.append(logits_mlm)
    pred_4 = torch.cat(pred_4, dim=0)
    
    return pred_1+pred_2+pred_3+pred_4, label_1

pred, label = outputs()

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
    for temp in tqdm(range(20)):
        th = float(temp) / 10.0 - 3.4
        sigm_all, bina_all = [], []
        bina = torch.where(pred > (th), one, zero)
        
        for j in range(pred.shape[0]):
            sigm_all.append([sigmoids(pred[j][0], th),sigmoids(pred[j][1], th),sigmoids(pred[j][2], th),sigmoids(pred[j][3], th),\
                             sigmoids(pred[j][4], th),sigmoids(pred[j][5], th),sigmoids(pred[j][6], th),sigmoids(pred[j][7], th)])
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

test(pred, label)

def outputs():
    pred_1, pred_2, pred_3, pred_4, label_1 = [], [], [], [], []
    
    model_1 = Dict_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'wotext_4_2.41.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, emo_num, label = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            emo_num = torch.FloatTensor(emo_num).to(device)
            input_ids, attention_mask, token_type_ids = token['input_ids'], token['attention_mask'], token['token_type_ids']
            logits_mlm = model_1(input_ids, attention_mask, token_type_ids, emo_num)
            pred_1.append(logits_mlm)
            label_1.append(torch.tensor(label))
    pred_1 = torch.cat(pred_1, dim=0)
    label_1 = torch.cat(label_1, dim=0)
    
    model_1 = Dict_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'wotext_3_2.40.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, emo_num, label = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            emo_num = torch.FloatTensor(emo_num).to(device)
            input_ids, attention_mask, token_type_ids = token['input_ids'], token['attention_mask'], token['token_type_ids']
            logits_mlm = model_1(input_ids, attention_mask, token_type_ids, emo_num)
            pred_2.append(logits_mlm)
    pred_2 = torch.cat(pred_2, dim=0)
    
    model_1 = Dict_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'wotext_2_2.38.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, emo_num, label = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            emo_num = torch.FloatTensor(emo_num).to(device)
            input_ids, attention_mask, token_type_ids = token['input_ids'], token['attention_mask'], token['token_type_ids']
            logits_mlm = model_1(input_ids, attention_mask, token_type_ids, emo_num)
            pred_3.append(logits_mlm)
    pred_3 = torch.cat(pred_3, dim=0)
    
    model_1 = Dict_model().to(device)
    model_1.load_state_dict(torch.load(log_dir + 'wotext_1_2.38.pt'))
    model_1.eval()
    with torch.no_grad():
        test_iterator = data_loader(test_set, batch_size=BATCH)
        for _, batch in tqdm(enumerate(test_iterator)):
            text, emo_num, label = zip(*batch)
            token = tokenizer(list(text), padding=True, max_length=MAX_LEN, truncation=True, return_tensors="pt").to(device)
            emo_num = torch.FloatTensor(emo_num).to(device)
            input_ids, attention_mask, token_type_ids = token['input_ids'], token['attention_mask'], token['token_type_ids']
            logits_mlm = model_1(input_ids, attention_mask, token_type_ids, emo_num)
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

    th = -0.0
    sigm_all, bina_all = [], []
    bina = torch.where(pred > (th), one, zero)

    for j in range(pred.shape[0]):
        sigm_all.append([sigmoids(pred[j][0], th),sigmoids(pred[j][1], th),sigmoids(pred[j][2], th),sigmoids(pred[j][3], th),\
                         sigmoids(pred[j][4], th),sigmoids(pred[j][5], th),sigmoids(pred[j][6], th),sigmoids(pred[j][7], th)])
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
