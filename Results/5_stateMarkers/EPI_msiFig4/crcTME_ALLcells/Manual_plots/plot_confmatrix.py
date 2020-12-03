import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

data_mat = pd.read_csv("crcTME_ALLcells_confusion_matrix_ANNOTATED.csv")

print(data_mat)

plt.figure()
sns.heatmap(data_mat)
plt.show()

