import os 
import json 
import datetime
import hashlib

def make_dir(dir): 
    if os.path.exists(dir) == False: 
        os.mkdir(dir)

def save_result(summary, model, save_Dir, args): 
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]
    save_file = os.path.join(save_Dir, f'{args.exp_name}_{hash_key}.json')

    result = {}
    result.update(vars(args))
    result.update(summary)
    result['params'] = model 
    with open(save_file, 'w') as f: 
        json.dump(result,f)
    
