import os
import re

root_dir = os.path.abspath("..")


# генерация предложений
def generate(prompt_text, model, tokenizer, n_seqs=1, max_length=35, min_length=10):
    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")

    output_sequences = model.generate(
        input_ids=encoded_prompt,
        max_length=max_length + len(encoded_prompt),
        min_length=min_length + len(encoded_prompt),
        temperature=0.8,
        top_p=0.8,
        repetition_penalty=1.2,
        do_sample=True,
        num_return_sequences=n_seqs)

    # детокенизируем получившиеся последовательности в строку
    generated_list = []
    for seq in output_sequences:
        seq = seq.tolist()
        text = tokenizer.decode(seq)
        decoded_prompt = tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)
        total_sequence = (prompt_text + text[len(decoded_prompt):])
        generated_list.append(total_sequence)
    return generated_list


# генерация текста для списка названий предметов
def text_generation(test_data, model, tokenizer, gen_func='torch', entry_count=1):
    generated_descriptions = []
    for i in range(len(test_data)):
        if gen_func == 'torch':
            prompt = test_data[i] + '. '
            x = generate(prompt, model, tokenizer, n_seqs=entry_count)
        else:
            prompt = f'<|startoftext|>' + test_data[i] + f':'
            x = generate(model, tokenizer, prompt, n_seqs=entry_count)
        for j in range(0, entry_count):
            x[j] = x[j].replace(prompt, '')
            x[j] = x[j].replace('\n', ' ')
            x[j] = x[j][: x[j].find("<")]
            x[j] = x[j][: x[j].find(">")]
            x[j] = x[j][: x[j].find("|")]
            x[j] = x[j][: x[j].find("endoftext")]
            x[j] = x[j][: x[j].find("startoftext")]
            result = x[j].replace(x[j].split('.')[-1], '')
            if len(result.split()) < 5:
                result = x[j].replace("," + x[j].split(',')[-1], '.')
            x[j] = result

        generated_descriptions.append(x)
    return [test_data, generated_descriptions]
