import os

dirs = ['Balanced_NoPreprocess', 'Balanced_Stop_Stem']
high_average = 0
file_high_average = ''
for dir in dirs:
    for file in os.listdir(dir):
        if '_report' in file:
            content = []
            with open(dir+'/'+file) as f:
                for line in f:
                    content.append([c for c in line.strip().replace(' / ', '/').split(' ') if c][:-1])
                    if not content[-1]:
                        del content[-1]
            average = content[-1][1:]
            content = content[1:-1]
            if high_average < float(average[2]):
                high_average = float(average[2])
                file_high_average = file
    print(dir, file_high_average, high_average)