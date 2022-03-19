# Generating DS-like descriptions

1) Activate virtualenv ```.\venv\Scripts\activate```
2) Install requirements ```pip install -r requirements.txt```
3) Run CLI ```python cli.py```
```
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  analysis-results  выводит информацию о величинах метрик bleu и rouge на...
  generate          генерирует описания для предметов из файла...
  generate-test     генерирует описания для тестовой выборки на...
  train             обучает модель по датасету proverbs.csv на заданной...
```

Before generating the descriptions you should:
- download finetuned rugpt2 model from ... to directory **\model** 
- or train it with ```python cli.py train```
