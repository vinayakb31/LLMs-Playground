import re
import emoji

with open("Sample Data/self_healing_waf.txt", "r", encoding='utf-8') as f:
    text = f.read()

print(type(text))
print(text)