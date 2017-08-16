#cell 0
import sys, json

f = open('SpeechLSTM.ipynb','r')
j = json.load(f)
of = open('SpeechLSTM.py','w')
if j["nbformat"] >= 4:
    for i,cell in enumerate(j["cells"]):
        of.write("#cell " + str(i) + "\n")
        for line in cell["source"]:
            of.write(line)
        of.write('\n\n')
else:
    for i,cell in enumerate(j["worksheets"][0]["cells"]):
        of.write("#cells " + str(i) + "\n")
        for line in cell["input"]:
            of.write(line)
        of.write('\n\n')
of.close()

