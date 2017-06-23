import csv
import os

pathnames= []
for x in range(ord('A'), ord('H')+1):
    dirs = os.listdir('base5/'+chr(x))
    for directory in dirs:
        pathnames.append('base5/'+chr(x)+'/'+directory)
# pathnames = ['base']#["base3/"+chr(x) for x in range(ord('A'),ord('H')+1) ]
pathnames.sort()
print(pathnames)
with open('results.csv', 'w') as f:
    for path in pathnames:
        print(path)
        with open('results'+'/neuralNN'+path.replace('/','_')+'.csv', 'r') as resultFile:
            f.write(resultFile.read())
            f.write('\n')