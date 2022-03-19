import os
import pickle
import sys

import click
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

root_dir = os.path.abspath("")
sys.path.append(root_dir)

if os.path.basename(os.path.normpath(root_dir)) == 'scripts':
    root_dir = os.path.abspath("..\..")
    sys.path.append(root_dir)

import src.data.prepare as pr
import src.models.train as tr
import src.models.generate_text as gen
import src.analysis.metrics as met


@click.group()
def cli():
    pass




@cli.command()
@click.option("--config", '-c', default='config', type=click.Path(),
              help='имя файла с конфигурацией')
@click.option("--path", '-p', default=root_dir + '/reports/train_info.txt', type=click.Path(),
              help='путь для сохранения вывода обучения')
def train(config, path):
    """
    обучает модель по датасету proverbs.csv
    на заданной конфигурации config
    и выводит процесс обучения в /reports/train_info.txt (path)
    """
    X_train, X_test, y_train, y_test = pr.prepare_data(root_dir=root_dir)
    dataloader = pr.prepare_dataloader(X_train, y_train)
    model = AutoModelForCausalLM.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
    tr.train(model, dataloader, config_name=config, root_dir=root_dir, out_path=path)


@cli.command()
@click.option("--generator", '-g', default='torch',
              help="тип функции генерации: 'torch' или 'mine'")
@click.option("--path_in", '-i', default=root_dir + '/data/input/tags.txt', type=click.Path(),
              help='путь для чтения названий предметов')
@click.option("--path_out", '-o', default=root_dir + '/data/output/proverbs.txt', type=click.Path(),
              help='путь для сохранения описаний предметов')
@click.option("--count", '-c', default=1,
              help='количество описаний на один предмет')
def generate(generator, path_in, path_out, count):
    """
    генерирует описания для предметов из файла data/input/tags.txt (path_in)
    и записывает их в data/output/proverbs.txt (path_out)
    """

    tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
    model = AutoModelForCausalLM.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

    with open(path_in, 'r', encoding = 'utf-8') as f:
        data = f.read().splitlines()
    # загружаем обученную ранее модель
    model.load_state_dict(torch.load(root_dir + f"\model\\rugpt2_proverb_8_dot.pt",
                                 map_location=torch.device('cpu')))

    labels, descriptions = gen.text_generation(
        data, model, tokenizer, gen_func=generator, entry_count=count)
    if os.path.exists(path_out):
        os.remove(path_out)
    with open(path_out, 'a', encoding = 'utf-8') as fo:
        for i in range(len(labels)):
            fo.write(f'Tag:\n{labels[i]}\nProverb(s):\n')
            for j in range(len(descriptions[i])):
                fo.write(f'{descriptions[i][j]}\n')
            fo.write('\n')


@cli.command()
@click.option("--generator", '-g', default='torch',
              help="тип функции генерации: 'torch' или 'mine'")
def generate_test(generator):
    """
    генерирует описания для тестовой выборки
    на rugpt2 с finetune
    """
    tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')

    _, _, _, y_test = pr.prepare_data(root_dir=root_dir)
    y_test_list = y_test.tolist()

    model = AutoModelForCausalLM.from_pretrained('sberbank-ai/rugpt3small_based_on_gpt2')
    model.load_state_dict(torch.load(root_dir + f"\model\\rugpt2_proverb_8_dot.pt",
                                     map_location=torch.device('cpu')))
    if generator == 'mine':
        folder = 'mine-gen-func'
    else:
        folder = 'torch-gen-func'
    ft_gen = gen.text_generation(
        y_test_list, model, tokenizer, gen_func=generator, entry_count=1)

    ft_gen[1] = [desc[0] for desc in ft_gen[1]]

    with open(root_dir + "/reports/" + folder + "/test_generation.txt", "w+", encoding="utf-8") as fp:
        fp.write("Generated descriptions on rugpt2+finetune:\n")
        for num in range(len(ft_gen[0])):
            fp.write(f"{ft_gen[0][num]}: {ft_gen[1][num]}\n")

    with open(root_dir + "/reports/" + folder + "/test_generation", "wb") as fp:
        pickle.dump(ft_gen[1], fp)


@cli.command()
@click.option("--generator", '-g', default='torch',
              help="тип функции генерации: 'torch' или 'mine'")
def analysis_results(generator):
    """
    выводит информацию о величинах метрик bleu и rouge
    на тестовой выборке для distilgpt2 с finetune и без
    """
    if generator == 'mine':
        folder = 'mine-gen-func'
    else:
        folder = 'torch-gen-func'

    _, X_test, _, _ = pr.prepare_data(root_dir=root_dir)
    rouge_score = met.rogue_analysis(X_test, folder=folder, root_dir=root_dir)
    mean_score = np.mean(met.bleu_analysis(X_test, folder=folder, root_dir=root_dir))

    print('BLEU scores on test dataset: \nrugpt2 with finetune = {} \
     \n'.format(mean_score))

    print('Rouge scores on test dataset: \nrugpt2 with finetune: \n{} \
     \n'.format(rouge_score))
    with open(root_dir + "/reports/analysis_results.txt", "w") as fp:
        fp.write('BLEU scores on test dataset: \ndistilgpt2 with finetune = {} \
     \n'.format(mean_score))
        fp.write('Rouge scores on test dataset: \ndistilgpt2 with finetune: \n{} \
     \n'.format(rouge_score))


if __name__ == "__main__":
    cli()
