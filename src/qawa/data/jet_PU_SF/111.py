import json
data='jmar_2018_UL.json'

offset = 1421
window_size = 20  # number of characters to display around the offset
with open(data, 'r', encoding='utf-8') as f:
    content = f.read()

start = max(0, offset - window_size)
end = min(len(content), offset + window_size)

print(content[start:end])
# 假设 data 是 JSON 字符串
