import random
import time
import requests
import json
import sys




def llm(input_text, stop=["\n"]):
    url = "http://47.88.8.18:8088/api/ask"
    HTTP_LLM_API_KEY='eyJ0eXAiOiJqd3QiLCJhbGciOiJIUzI1NiJ9.eyJ1c2VybmFtZSI6IjM5NDc3MyIsInBhc3N3b3JkIjoiMzk0NzczMTIzIiwiZXhwIjoyMDIxNjE4MzE3fQ.oQx2Rh-GJ_C29AfHTHE4x_2kVyy7NamwQRKRA4GPA94'
    headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + HTTP_LLM_API_KEY
                }
    data = {
            "model": 'gpt-4',
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": input_text}
            ],
            "n": 1,
            "temperature": 0.0
            }
    response = requests.post(url, json=data, headers=headers)
    response = response.json()
    new_response = response['data']['response']
    return new_response["choices"][0]["message"]["content"]




FEVER_SPLIT_FILE = {
  "train": "train.jsonl",
  "dev": "paper_dev.jsonl",
}
split="dev"
data_path = f"./data/{FEVER_SPLIT_FILE[split]}"
with open(data_path, "r") as json_file:
    json_list = list(json_file)


def webthink(idx=None):
    info = {}
    question = json.loads(json_list[idx])['claim']
    info['question']=question
    info['gt_answer']=json.loads(json_list[idx])['label']
    prompt_template = "Determine if there is Observation that SUPPORTS or REFUTES a Claim, or if there is NOT ENOUGH INFO.\nClaim: {q}\nAnswer: "
    response = llm(prompt_template.format(q=question))
    info['answer']=response
    info['em']=(response==info['gt_answer'])
    
    return info


idxs = list(range(7405))

rs = []
infos = []
old_time = time.time()
for i in idxs[:100]:
    info = webthink(i)
    print(f"info{i}:{info}")
    infos.append(info)
    file_path = 'output/feverGPT.json'
    with open(file_path, 'w') as json_file:
        json.dump(infos, json_file)

