import re
import json

pattern = re.compile(r"\{.*?\}")
text_file = 'experiments/pool_3c_t/kitti_new/log/test_ppo_10_2025-03-25-23-45-17.log'
with open(text_file,'r') as f:
    text = f.readlines()[-5:]
text = ''.join(text)
results = pattern.findall(text)
time_list = []
if results:
    for result in results:
        result_json = re.sub(r"'",r'"',result)
        time_list.append(json.loads(result_json)['time'])
    print("%0.2f"%(sum(time_list)/len(time_list)*1000))
else:
    print("no match")