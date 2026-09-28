import json

d = {"role":"system", "content":"you a a sysadmin"}
s = json.dumps(d)
print(s)
print(type(s))

for i in s:
    print(i)