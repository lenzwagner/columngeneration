from plots import *
from aggundercover import *

random.seed(0)
comb_text = str(0.06) + '_' + str(3)
file = 'perf_Plot_' + comb_text
file3 = 'comb__' + comb_text

path = f'./images/schedules/worker_schedules' + comb_text + '.svg'






combined_list = [0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 1, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 2, 2, 0, 2, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 2, 1, 0, 0, 2, 0, 1, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 3, 0, 0, 1, 0, 2, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 2, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 3, 2, 0, 0, 2, 0, 0, 3, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 3, 1, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 3, 1, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 0, 0, 2, 0, 0, 2, 2, 2, 0, 0, 3, 0, 0, 3, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 1, 2, 0, 0, 3, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 1, 2, 0, 0, 3, 0, 0, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 2, 0, 1, 2, 0, 0, 3, 0, 0, 2, 1, 0, 2, 0, 2, 0, 0, 0, 0, 3, 0, 1, 0, 2, 0, 0, 2, 2, 2, 0, 0, 2, 1, 0, 3, 0, 0, 2, 1, 0, 2, 0, 2, 0, 0, 0, 0, 3, 0, 1, 0, 2, 0, 0, 2, 2, 2, 0, 0, 2, 1, 0, 3, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 3, 0, 0, 2, 0, 0, 3, 0, 0, 2, 0, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 3, 0, 0, 2, 0, 0, 3, 0, 2, 0, 0, 0, 0, 2, 1, 2, 2, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 1, 0, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 2, 0, 2, 1, 0, 0, 0, 2, 0, 1, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 2, 0, 2, 1, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 3, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 1, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 0, 2, 2, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 2, 0, 1, 0, 0, 0, 0, 2, 2, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 2, 0, 0, 2, 0, 1, 0, 1, 1, 0, 2, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 1, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 1, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 3, 1, 0, 2, 2, 0, 0, 2, 0, 0, 0, 1, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 3, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 1, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0, 0, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 1, 2, 0, 0, 0, 1, 0, 1, 3, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 1, 0, 3, 3, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 0, 1, 2, 2, 0, 1, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 3, 3, 0, 0, 2, 2, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 3, 3, 0, 0, 2, 2, 0, 0, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 3, 3, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 0, 1, 2, 2, 0, 1, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 2, 2, 0, 1, 2, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 0, 2, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 0, 2, 0, 1, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 2, 1, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 1, 2, 1, 0, 0, 2, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1, 2, 0, 2, 0, 0, 0, 3, 2, 0, 0, 1, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 3, 2, 0, 0, 1, 0, 0, 1, 2, 2, 0, 0, 3, 0, 0, 2, 2, 0, 2, 0, 0, 0, 2, 0, 2, 0, 0, 1, 2, 0, 2, 0, 3, 0, 0, 0, 3, 2, 0, 0, 2, 0, 0, 2, 2, 1, 2, 0, 0, 0, 2, 0, 2, 0, 0, 0, 3, 0, 2, 0, 3, 0, 0, 0, 3, 2, 0, 2, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 2, 0, 0, 3, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 3, 0, 0, 0, 3, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 3, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 1, 0, 0, 2, 2, 0, 0, 3, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 3, 0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 2, 3, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0, 2, 0, 0, 0, 2, 0, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 3, 2, 0, 0, 2, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 3, 2, 0, 0, 2, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 1, 2, 3, 2, 0, 0, 2, 0, 0, 0, 2, 0, 1, 2, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 2, 3, 0, 0, 0, 2, 2, 0, 0, 0, 0, 1, 2, 3, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 2, 2, 2, 0, 1, 0, 1, 2, 2, 0, 0, 0, 0, 2, 0, 0, 2, 2, 0, 0, 0, 2, 2, 0, 1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 2, 0, 0, 0, 0, 3, 0, 2, 2, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 1, 1, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 2, 1, 0, 0, 3]


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Polygon
from matplotlib.legend_handler import HandlerBase

class HandlerDiagonalSplit(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        return [Polygon([(xdescent, ydescent+height), (xdescent+width, ydescent), (xdescent, ydescent)],
                        facecolor='#feb77e', edgecolor='black', linewidth=0.1),
                Polygon([(xdescent, ydescent+height), (xdescent+width, ydescent), (xdescent+width, ydescent+height)],
                        facecolor='#251255', edgecolor='black', linewidth=0.1)]

def visualize_schedule_dual(combined_list, days=28, num_workers=100, save_path='worker_schedules.eps'):
    sublist_length = days
    sublists = [combined_list[i:i + sublist_length] for i in range(0, len(combined_list), sublist_length)]
    data = np.array(sublists[:num_workers]).T

    fig, ax = plt.subplots(figsize=(max(8, num_workers * 0.4), max(10, days * 0.4)))

    colors = {0: 'white', 1: '#feb77e', 2: '#251255', 3: 'special'}
    border_alpha = 0.18  # Set the alpha value for the black borders

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            color = colors.get(value, 'gray')

            if color == 'special':
                ax.add_patch(Polygon([(j, i), (j+1, i+1), (j, i+1)],
                                     facecolor='#feb77e', edgecolor=(0, 0, 0, border_alpha), linewidth=0.1))
                ax.add_patch(Polygon([(j, i), (j+1, i+1), (j+1, i)],
                                     facecolor='#251255', edgecolor=(0, 0, 0, border_alpha), linewidth=0.1))
            else:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor=color, edgecolor=(0, 0, 0, border_alpha), linewidth=0.1))

    ax.set_xlim(0, num_workers)
    ax.set_ylim(days, 0)

    ax.set_xticks([0, 24, 49, 74, 99])
    ax.set_xticklabels([1, 25, 50, 75, 100])
    ax.set_yticks([0, 6, 13, 20, 27])
    ax.set_yticklabels([1, 7, 14, 21, 28])
    ax.set_xlabel("Worker", fontsize=44, fontname="Computer Modern Roman", labelpad=20)
    ax.set_ylabel("Day", fontsize=44, fontname="Computer Modern Roman", labelpad=20)
    ax.tick_params(axis='both', which='major', labelsize=34, pad=10)

    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#feb77e', edgecolor='black', label='Human Scheduling Approach'),
        Rectangle((0, 0), 1, 1, facecolor='#251255', edgecolor='black', label='Machine-Like Scheduling Approach'),
        Polygon([(0, 1), (1, 0), (0, 0)], facecolor='white', edgecolor='black', label='Both')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.18),
              fancybox=True, shadow=True, ncol=3, fontsize=44,
              handler_map={Polygon: HandlerDiagonalSplit()})

    plt.tight_layout()
    overall_avg_file = f".{os.sep}images{os.sep}schedules{os.sep}worker.svg"
    plt.savefig(overall_avg_file, bbox_inches='tight')
    plt.show()

    return fig



visualize_schedule_dual(combined_list)