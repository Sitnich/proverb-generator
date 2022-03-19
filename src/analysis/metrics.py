import os
import pickle

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge

root_dir = os.path.abspath("..")


# считаем среднее значение метрики BLEU на тестовом датасете

def bleu_analysis(test_data, folder='mine-gen-func', root_dir=root_dir):
    smoothie = SmoothingFunction().method4
    scores = []

    with open(root_dir + "/reports/" + folder + "/test_generation", "rb") as fp:
        generated_list = pickle.load(fp)

    for num, item in enumerate(test_data):
        reference = item
        candidate = generated_list[num]
        scores.append(sentence_bleu([reference], candidate, smoothing_function=smoothie))

    with open(root_dir + "/reports/" + folder + "/test_generation_score", "wb") as fp:
        pickle.dump(scores, fp)
    return scores


def rogue_analysis(test_data, folder='mine-gen-func', root_dir=root_dir):
    rouge = Rouge()
    with open(root_dir + "/reports/" + folder + "/test_generation", "rb") as fp:
        generated_list = pickle.load(fp)

    rouge_score = rouge.get_scores(generated_list, test_data.tolist(), avg=True, ignore_empty=True)
    return rouge_score
