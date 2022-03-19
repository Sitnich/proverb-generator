import os
import pickle

import torch
from tqdm import tqdm

root_dir = os.path.abspath("..")


def train(model, dataloader, config_name='config',
          root_dir=root_dir, out_path=root_dir + '/reports/train_info.txt'):
    with open(root_dir + '/model/' + config_name, "rb") as fp:
        config = pickle.load(fp)

    batch_size = config['batch_size']
    epochs = config['epochs']
    learning_rate = config['learning_rate']
    max_seq_len = config['max_seq_len']

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    tmp_items_tens = None
    for epoch in range(epochs):
        proc_seq_count = 0
        sum_loss = 0.0
        for _, item in tqdm(enumerate(dataloader), total=len(dataloader)):
            # хотим запихнуть как можно больше токенизированных итемов в последовательность длины max_seq_len
            item_tens = torch.tensor(item['input_ids'])

            # пропускаем если он длиннее max_seq_len
            if item_tens.size()[0] > max_seq_len:
                continue

            # кладем во временный накопительный тензор первый элемент
            if not torch.is_tensor(tmp_items_tens):
                tmp_items_tens = item_tens
                continue
            else:
                # если новый элемент не помещается в накопительный тензор, то мы кладем его во временный
                # а продолжаем работать с заполненным
                if tmp_items_tens.size()[0] + item_tens.size()[0] > max_seq_len:
                    work_items_tens = tmp_items_tens
                    tmp_items_tens = item_tens
                else:
                    # иначе кладем в накопительный тензор
                    tmp_items_tens = torch.cat([tmp_items_tens, item_tens[1:]], dim=0)
                    continue

            # обучаем модель
            outputs = model(work_items_tens, labels=work_items_tens)
            loss, logits = outputs[:2]
            loss.backward()
            sum_loss += loss.detach().data

            if proc_seq_count % batch_size == 0:
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()
            proc_seq_count += 1
        print(f"Epoch {epoch + 1} | Train loss: {sum_loss}")

        # сохраняем чекпоинты
        if epoch % 2 == 1:
            torch.save(model.state_dict(), root_dir + f"\model\\rugpt2_proverb_{epoch+1}.pt")
        with open(out_path, "a") as file:
            file.write(f"Epoch {epoch + 1} | Train loss: {sum_loss}\n")
