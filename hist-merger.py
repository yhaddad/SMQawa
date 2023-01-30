import gzip 
import pickle
import argparse
import collections
import functools
import itertools
import os, glob
from tqdm import tqdm

def merger():
    parser = argparse.ArgumentParser(description='Famous Submitter')
    parser.add_argument("-t", '--tag', type=str, default="algiers", help="")
    parser.add_argument('--era', type=str, default="2018"   , help="")
    options = parser.parse_args()
    

    def updated(c, items):
        c.update(items)
        return c


    all_hists = []
    for filename in tqdm(glob.glob(f'*{options.tag}*_{options.era}_*/*.pkl.gz'), desc="reading", ascii=False, ncols=75):
        if os.path.getsize(filename) == 0: 
            print(f"{filename} is empty !! ")
        else:
            with gzip.open(filename, 'rb') as f:
                _data = pickle.load(f)
                all_hists.append(_data)
                f.close()
                del f 
                del _data
            
    combined_hist = {}
    combined_sumw = {}
    
    for i in tqdm(all_hists, desc="format", ascii=False, ncols=75):
        for s, v in i.items():
            v_hist = dict(filter(lambda n: not isinstance(n[1], dict), v['hist'].items()))
            if s in combined_hist:
                combined_hist[s].append(v_hist)
                combined_sumw[s].append(v['sumw'])
            else:
                combined_hist[s] = [v_hist]
                combined_sumw[s] = [v["sumw"]]
    
    del all_hists

    combined_dict = {}
    for s, h in tqdm(combined_hist.items(), desc="merging", ascii=False, ncols=75):
        hist_ = dict(functools.reduce(updated, h, collections.Counter()))
        sumw_ = sum(combined_sumw[s])
        combined_dict[s] = {"hist": hist_, "sumw": sumw_}

    with gzip.open(f"merged-histogram-{options.tag}-{options.era}.pkl.gz", "wb") as f:
        pickle.dump(combined_dict, f)
        f.close()

if __name__ == "__main__":
    merger()
