import nltk
from nltk.stem import WordNetLemmatizer
import json
from json.decoder import JSONDecodeError
import spacy
from tqdm import tqdm
import warnings
import argparse

nltk.data.path.append('/mnt/sdc1/yahan/nltk_data')  # 替换为你的路径
#nltk.download('averaged_perceptron_tagger', download_dir='/mnt/sdc1/yahan/nltk_data')
#nltk.download('punkt_tab', download_dir='/mnt/sdc1/yahan/nltk_data')
# nltk.data.path.append('/mnt/sdc1/yahan/nltk_data')  # 替换为你的路径

nlp = spacy.load("en_core_web_lg")
warnings.filterwarnings("ignore", category=UserWarning)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_association", type=str, default='data/relation.json')
    parser.add_argument("--safe_words", type=str, default='data/safe_words.txt')
    parser.add_argument("--inference_data", type=str)
    parser.add_argument("--annotation", type=str, default='data/annotations_updated.json') #'zanshi_anno...'
    parser.add_argument("--metrics", type=str, default='data/metrics.txt')
    parser.add_argument("--similarity_score", type=float, default=0.8)
    parser.add_argument('--evaluation_type', choices=['a', 'g', 'd', 'de', 'da', 'dr'], help='a: all tasks and dimensions    g: generative task    d: descriminative task    de, da, dr: existence, attribute, relation')
    parser.add_argument('--output_file', type=str, default='model_output/dis_results.json', help='Path to the output JSON file')
    args = parser.parse_args()
    return args


def check_synonyms_word(word1, word2, similarity_score):
    token1 = nlp(word1)
    token2 = nlp(word2)
    similarity = token1.similarity(token2)
    return similarity > similarity_score


def extract_nouns(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
    return nouns


def init():
    metrics = {}
    with open(args.metrics, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split('=')
        if len(parts) == 2:
            variable_name = parts[0].strip()
            variable_value = eval(parts[1].strip())
            metrics[variable_name] = variable_value
            
    return metrics

def save_results(output_file, inference_data, eval_type, results):
    try:
        with open(output_file, "r", encoding='utf-8') as file:
            try:
                existing_data = json.load(file)
            except JSONDecodeError:
                existing_data = []
    except FileNotFoundError:
        existing_data = []

    existing_data.append({
        "name": inference_data,
        "type": eval_type,
        "result": results
    })

    with open(output_file, "w", encoding='utf-8') as file:
        file.write(json.dumps(existing_data, ensure_ascii=False, indent=4))
        # file.write(inference_data + "\n")
        # # Write eval_type as the second line
        # file.write(eval_type + "\n")
        # # Write results as the third line
        # file.write(json.dumps(results, ensure_ascii=False) + "\n\n")
    
    print("finshed: ", inference_data)

def main(args):
    # 初始化指标
    metrics = init()

    # 读取幻觉词和安全词关联
    association = json.load(open(args.word_association, 'r', encoding='utf-8'))
    hallucination_words = []
    for word1 in association.keys():
        hallucination_words.append(word1)
        for word2 in association[word1]:
            hallucination_words.append(word2)

    global_safe_words = []
    with open(args.safe_words, 'r', encoding='utf-8') as safe_file:
        for line in safe_file:
            # line = line.strip()
            line = line.split('\n')[0]
            global_safe_words.append(line)

    # 设置评估维度
    dimension = {'g': False, 'de': False, 'da': False, 'dr': False}
    if args.evaluation_type == 'a':
        for key in dimension.keys():
            dimension[key] = True
    elif args.evaluation_type == 'g':
        dimension['g'] = True
    elif args.evaluation_type == 'de':
        dimension['de'] = True
    elif args.evaluation_type == 'da':
        dimension['da'] = True
    else:
        dimension[args.evaluation_type] = True

    # 读取推理数据和标注数据
    inference_data = json.load(open(args.inference_data, 'r', encoding='utf-8'))
    ground_truth = json.load(open(args.annotation, 'r', encoding='utf-8'))

    for i in tqdm(range(len(inference_data))):
        id = inference_data[i]['id']
        # if id >= 9001:
        #     id = 1783 + id - 9001 #1783 加Common 1550原来的
        # else:
        #     continue
        # 分ee/ev
        # if ground_truth[id-1]['info_type'] == 'ee':
        #     continue

        if dimension['g']:
            # 生成任务
            nouns = extract_nouns(inference_data[i]['response'])
            after_process_nouns = [noun for noun in nouns if noun in hallucination_words]

            safe_words = []
            safe_list = []

            truth_list = None
            for item in ground_truth:
                if item['id'] == id:
                    truth_list = item['truth']
                    break

            for idx, word in enumerate(truth_list):
           # for idx, word in enumerate(ground_truth[id - 1]['truth']):
                if word in association:
                    safe_words += association[word]
                    safe_list += [idx] * len(association[word])

            ha_words = []
            ha_list = []

            hallu_list = None
            for item in ground_truth:
                if item['id'] == id:
                    hallu_list = item['hallu']
                    break
            for idx, word in enumerate(hallu_list):
           # for idx, word in enumerate(ground_truth[id - 1]['hallu']):
                if word in association:
                    ha_words += association[word]
                    ha_list += [idx] * len(association[word])

            safe_words += truth_list
            safe_len = len(truth_list)
            #safe_words += ground_truth[id - 1]['truth']
            #safe_len = len(ground_truth[id - 1]['truth'])
            safe_list += [0] * safe_len
            safe_flag_list = [0] * len(after_process_nouns)
            
            ha_words += hallu_list
            ha_len = len(hallu_list)
            #ha_words += ground_truth[id - 1]['hallu']
            #ha_len = len(ground_truth[id - 1]['hallu'])
            ha_list += [0] * ha_len

            # for idx, noun in enumerate(after_process_nouns):
            #     if noun in global_safe_words:
            #         continue

            #     if noun in safe_words:
            #         for j in range(len(safe_words)):
            #             if noun == safe_words[j]:
            #                 safe_list[j] = 1
            #                 break
            #         continue

            #     if noun in ha_words:
            #         for j in range(len(ha_words)):
            #             if noun == ha_words[j]:
            #                 ha_list[j] = 1
            #                 break

            #     for j, check_word in enumerate(ha_words):
            #         if check_synonyms_word(noun, check_word, args.similarity_score):
            #             ha_list[j] = 1
            #             break

            #     flag = False
            #     for j, check_word in enumerate(safe_words):
            #         if check_synonyms_word(noun, check_word, args.similarity_score):
            #             flag = True
            #             safe_list[j] = 1
            #             break
            #     if flag:
            #         continue

            #     safe_flag_list[idx] = 1

            # chair_score = 1 - len(set(after_process_nouns) & set(ground_truth[id - 1]['hallu'])) / len(nouns) if nouns else 0    
            # #chair_score = 1 - len([noun for noun in after_process_nouns if noun in ground_truth[id - 1]['hallu']]) / len(nouns) if nouns else 0
            # #避免after_process_nouns包含重复项
            # cover_score = len(set(after_process_nouns) & set(ground_truth[id - 1]['coco']['truth'])) / len(ground_truth[id - 1]['coco']['truth']) if ground_truth[id - 1]['coco']['truth'] else 0
            # #cover_score = len([noun for noun in after_process_nouns if noun in ground_truth[id - 1]['truth']]) / len(ground_truth[id - 1]['coco']['truth']) if ground_truth[id - 1]['coco']['truth'] else 0
            # hal_score = 1 if chair_score != 0 else 0
            # #cog_score = len([noun for noun in after_process_nouns if noun in ground_truth[id - 1]['hallu']]) / len(after_process_nouns) if after_process_nouns else 0
            # cog_score = len(set(after_process_nouns) & set(ground_truth[id - 1]['hallu'])) / len(after_process_nouns) if after_process_nouns else 0

            # metrics['chair_score'] += chair_score
            # metrics['chair_num'] += 1
            # metrics['safe_cover_score'] += cover_score
            # metrics['safe_cover_num'] += 1
            # metrics['hallu_cover_score'] += hal_score
            # metrics['hallu_cover_num'] += 1
            # if hal_score == 0:
            #     metrics['non_hallu_score'] += 1
            # metrics['non_hallu_num'] += 1
            for idx, noun in enumerate(after_process_nouns):
                if noun in global_safe_words:
                    continue
                
                if noun in safe_words:
                    for j in range(len(safe_words)):
                        if noun == safe_words[j]:
                            if j < (len(safe_list) - safe_len):
                                safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                            else:
                                safe_list[j] = 1
                            break
                    continue
                
                if noun in ha_words:
                    for j in range(len(ha_words)):
                        if noun == ha_words[j]:
                            if j < (len(ha_list) - ha_len):
                                ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                            else:
                                ha_list[j] = 1
                            break
                
                for j, check_word in enumerate(ha_words):
                    if check_synonyms_word(noun, check_word, args.similarity_score):
                        if j < (len(ha_list) - ha_len):
                                ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                        else:
                            ha_list[j] = 1
                        break
                
                flag = False
                for j, check_word in enumerate(safe_words):
                    if check_synonyms_word(noun, check_word, args.similarity_score):
                        flag = True
                        if j < (len(safe_list) - safe_len):
                                safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break
                if flag == True:
                    continue
            
                safe_flag_list[idx] = 1

            metrics['chair_score'] += sum(safe_flag_list)
            metrics['chair_num'] += len(safe_flag_list)
            metrics['safe_cover_score'] += sum(safe_list[-safe_len:])
            metrics['safe_cover_num'] += len(safe_list[-safe_len:])
            metrics['hallu_cover_score'] += sum(ha_list[-ha_len:])
            metrics['hallu_cover_num'] += len(ha_list[-ha_len:])
            if sum(safe_flag_list) == 0:
                metrics['non_hallu_score'] += 1
            metrics['non_hallu_num'] += 1

        else:  # 判别任务
            type = next(item['dis-type'] for item in ground_truth if item['id'] == id)
            # print(type)
            # if type != "dis-attribute-action":
            #     continue
            metrics['qa_correct_num'] += 1
            metrics['ha_qa_correct_num'] += 1

            # print("id:",id)
            # print("i:", i)
            truth = next(item['dis_truth'] for item in ground_truth if item['id'] == id)
            # print(id)
            # print(truth)
            # print(inference_data[i]['response'])
            #truth = ground_truth[id - 1]['dis_truth']
            response = inference_data[i]['response'].split(',')[0]
            if response not in ['Yes', 'yes', 'No', 'no']:
                response = inference_data[i]['response'].split('\n')[0]
            if response not in ['Yes', 'yes', 'No', 'no']:
                response = inference_data[i]['response'].split('.')[0]
            if response not in ['Yes', 'yes', 'No', 'no']:
                if 'No' in inference_data[i]['response'] or 'not' in inference_data[i]['response'] or 'no' in inference_data[i]['response']:
                    response = 'no'
                
            if truth == 'yes':
                if response == 'Yes' or response == 'yes':
                    metrics['qa_correct_score'] += 1
                    metrics['ha_qa_correct_score'] += 1
   
            else:
                metrics['qa_no_num'] += 1
                metrics['ha_qa_no_num'] += 1

                if response == 'No' or response == 'no':
                    # print('!!!!!!!!!')
                    metrics['qa_correct_score'] += 1
                    metrics['qa_no_score'] += 1
                    metrics['ha_qa_correct_score'] += 1
                    metrics['ha_qa_no_score'] += 1

            if response == 'No' or response == 'no':
                metrics['qa_ans_no_num'] += 1
                metrics['ha_qa_ans_no_num'] += 1

                if truth == 'no' :
                    metrics['qa_ans_no_score'] += 1
                    metrics['ha_qa_ans_no_score'] += 1

    if dimension['g']:
        # CHAIR = round(metrics['chair_score'] / metrics['chair_num'] * 100, 1) if metrics['chair_num'] > 0 else 0
        # Cover = round(metrics['safe_cover_score'] / metrics['safe_cover_num'] * 100, 1) if metrics['safe_cover_num'] > 0 else 0
        # Hal = round(100 - metrics['non_hallu_score'] / metrics['non_hallu_num'] * 100, 1) if metrics['non_hallu_num'] > 0 else 0
        # Cog = round(metrics['hallu_cover_score'] / metrics['hallu_cover_num'] * 100, 1) if metrics['hallu_cover_num'] > 0 else 0
        # results = {}
        # print("Generative Task:")
        # print("CHAIR:\t\t", CHAIR)
        # print("Cover:\t\t", Cover)
        # print("Hal:\t\t", Hal)
        # print("Cog:\t\t", Cog, "\n")
        # eval_type = "Generative"
        # results["CHAIR"] = CHAIR
        # results["Cover"] = Cover
        # results["Hal"] = Hal
        # results["Cog"] = Cog
        # save_results(args.output_file, args.inference_data, eval_type, results)
        CHAIR = round(metrics['chair_score'] / metrics['chair_num'] * 100, 1)
        Cover = round(metrics['safe_cover_score'] / metrics['safe_cover_num'] * 100, 1)
        Ha = round(metrics['hallu_cover_score'] / metrics['hallu_cover_num'] * 100, 1)
        Ha_p = round(100 - metrics['non_hallu_score'] / metrics['non_hallu_num'] * 100, 1)
        print("Generative Task:")
        print("CHAIR:\t\t", CHAIR)
        print("Cover:\t\t", Cover)
        print("Hal:\t\t", Ha_p)
        print("Cog:\t\t", Ha, "\n")

        results = {}
        print("Generative Task:")
        print("CHAIR:\t\t", CHAIR)
        print("Cover:\t\t", Cover)
        print("Hal:\t\t", Ha_p)
        print("Cog:\t\t", Ha, "\n")
        eval_type = "Generative"
        results["CHAIR"] = CHAIR
        results["Cover"] = Cover
        results["Hal"] = Ha_p
        results["Cog"] = Ha
        save_results(args.output_file, args.inference_data, eval_type, results)

    if dimension['de']:
        # print("ha_qa_correyahact_score:", metrics['ha_qa_correct_score'])
        # print("ha_qa_correct_num:", metrics['ha_qa_correct_num'])
        # print("ha_qa_ans_no_score:", metrics['ha_qa_ans_no_score'])
        # print("ha_qa_ans_no_num:", metrics['ha_qa_ans_no_num'])
        # print("ha_qa_no_score:", metrics['ha_qa_no_score'])
        # print("ha_qa_no_num:", metrics['ha_qa_no_num'])

        results = {}

        hallucination_Accuracy = round(metrics['ha_qa_correct_score'] / metrics['ha_qa_correct_num'] * 100, 1) if metrics['ha_qa_correct_num'] > 0 else 0
        hallucination_Precision = round(metrics['ha_qa_ans_no_score'] / metrics['ha_qa_ans_no_num'] * 100, 1) if metrics['ha_qa_ans_no_num'] > 0 else 0
        hallucination_Recall = round(metrics['ha_qa_no_score'] / metrics['ha_qa_no_num'] * 100, 1) if metrics['ha_qa_no_num'] > 0 else 0
        hallucination_F1 = round(2 * (hallucination_Precision / 100) * (hallucination_Recall / 100) / ((hallucination_Precision / 100) + (hallucination_Recall / 100) + 0.001) * 100, 1) if (hallucination_Precision + hallucination_Recall) > 0 else 0
        print("Existence:")
        print("Accuracy:\t", hallucination_Accuracy)
        print("Precision:\t", hallucination_Precision)
        print("Recall:\t\t", hallucination_Recall)
        print("F1 Score:\t", hallucination_F1)
        eval_type = "Existence:"
        results["Accuracy"] = hallucination_Accuracy
        results["Precision"] = hallucination_Precision
        results["Recall"] = hallucination_Recall
        results["F1Score"] = hallucination_F1
        
        save_results(args.output_file, args.inference_data, eval_type, results)
        
    if dimension['da']:
        print("ha_qa_correyahact_score:", metrics['ha_qa_correct_score'])
        print("ha_qa_correct_num:", metrics['ha_qa_correct_num'])
        print("ha_qa_ans_no_score:", metrics['ha_qa_ans_no_score'])
        print("ha_qa_ans_no_num:", metrics['ha_qa_ans_no_num'])
        print("ha_qa_no_score:", metrics['ha_qa_no_score'])
        print("ha_qa_no_num:", metrics['ha_qa_no_num'])

        results = {}

        hallucination_Accuracy = round(metrics['ha_qa_correct_score'] / metrics['ha_qa_correct_num'] * 100, 1) if metrics['ha_qa_correct_num'] > 0 else 0
        hallucination_Precision = round(metrics['ha_qa_ans_no_score'] / metrics['ha_qa_ans_no_num'] * 100, 1) if metrics['ha_qa_ans_no_num'] > 0 else 0
        hallucination_Recall = round(metrics['ha_qa_no_score'] / metrics['ha_qa_no_num'] * 100, 1) if metrics['ha_qa_no_num'] > 0 else 0
        hallucination_F1 = round(2 * (hallucination_Precision / 100) * (hallucination_Recall / 100) / ((hallucination_Precision / 100) + (hallucination_Recall / 100) + 0.001) * 100, 1) if (hallucination_Precision + hallucination_Recall) > 0 else 0
        print("Attribute:")
        print("Accuracy:\t", hallucination_Accuracy)
        print("Precision:\t", hallucination_Precision)
        print("Recall:\t\t", hallucination_Recall)
        print("F1 Score:\t", hallucination_F1)
        eval_type = "Attribute:"
        results["Accuracy"] = hallucination_Accuracy
        results["Precision"] = hallucination_Precision
        results["Recall"] = hallucination_Recall
        results["F1Score"] = hallucination_F1
        
        save_results(args.output_file, args.inference_data, eval_type, results)

if __name__ == "__main__":
    args = get_args()
    main(args)
