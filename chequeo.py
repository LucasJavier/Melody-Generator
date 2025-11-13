# snippet rápido
import numpy as np
from melody_ea import load_dataset_dump
Xe, ye, _ = load_dataset_dump(r'.\dumps\dataset_easy.npz')
Xh, yh, _ = load_dataset_dump(r'.\dumps\dataset_hard.npz')
print(Xe.shape, Xh.shape)      # deberían matchear
print(np.allclose(Xe, Xh))     # debería dar False
print(ye.sum(), yh.sum())      # positivos deberían ser iguales
