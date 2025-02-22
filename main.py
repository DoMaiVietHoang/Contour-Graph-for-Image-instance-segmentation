import numpy as np

A = np.array([[0,1,0],[1, 0 ,1],[0 ,1 ,0]])
edge_index = np.where(A>0)
print(edge_index)   