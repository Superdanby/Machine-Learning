from pathlib import Path
from PIL import Image
# from skimage import io
import sys
import numpy as np
# import locale
import matplotlib.pyplot as plt
plt.style.use('seaborn')  # since default style is ugly


path = Path('CroppedYale')
path.resolve()
test_num = 35
test = []
truth = []
# locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

l1 = sorted(list(path.glob('*')))
# l1 = sorted([str(x) for x in l1], key=locale.strxfrm)
# l1 = list(path.glob('*'))

for _ in range(len(l1)):
    if Path(l1[_]).is_dir():
        l2 = sorted(list(Path(l1[_]).glob('*.pgm')))
        # l2 = sorted([str(x) for x in l2], key=locale.strxfrm)
        # l2 = list(Path(l1[_]).glob('*.pgm'))
        print(l2[:5])
        print(type(l2[-1]))
        test.append([np.array(Image.open(x), dtype=int) for x in l2[test_num:]])
        truth.append([np.array(Image.open(x), dtype=int) for x in l2[:test_num]])

temp = truth
truth = np.asarray(truth)
for i in range(len(temp)):
    for j in range(len(temp[i])):
        assert(np.sum(temp[i][j] - truth[i][j]) == 0)
print("pass")
# print(len(test[1]))
print(truth[1])
print(len(truth[1]))
# sys.exit()

SAD_cnt = 0
SSD_cnt = 0
Total_cnt = 0
for i in range(len(l1)):
    for j in range(len(test[i])):
        SAD = sys.maxsize
        SSD = sys.maxsize
        SAD_hit = False
        SSD_hit = False
        for k in range(len(l1)):
            SAD_local = np.min(np.sum(np.absolute(test[i][j] - truth[k]), axis=1))
            if SAD > SAD_local:
                SAD = SAD_local
                if i == k:
                    SAD_hit = True
                else:
                    SAD_hit = False
            SSD_local = np.min(np.sum((test[i][j] - truth[k])**2, axis=1))
            if SSD > SSD_local:
                SSD = SSD_local
                if i == k:
                    SSD_hit = True
                else:
                    SSD_hit = False
        Total_cnt = Total_cnt + 1
        if SAD_hit:
            SAD_cnt = SAD_cnt + 1
        if SSD_hit:
            SSD_cnt = SSD_cnt + 1
        print(f'Total: {Total_cnt}, SAD: {SAD_cnt}, SSD: {SSD_cnt}')

print(f'SAD: {SAD_cnt/Total_cnt}')
print(f'SSD: {SSD_cnt/Total_cnt}')
