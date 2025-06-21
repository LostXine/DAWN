
import matplotlib.pyplot as plt
import numpy as np

temp_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
plt.imshow(temp_img)
plt.show()