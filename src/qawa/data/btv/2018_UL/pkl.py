import pickle
import gzip

# 打开压缩的 pkl 文件
with gzip.open("eff-btag-2018.pkl.gz", "rb") as f:
    eff_maps = pickle.load(f)

# 打印读取到的数据结构
print(type(eff_maps))
print(eff_maps)
