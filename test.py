import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots()
img_plot = ax.imshow(np.random.rand(1080, 1440), cmap='gray')

def testfunc(frame, img_plot):
    print(frame)
    img_plot.set_array(np.random.rand(1080,1440))
    return (img_plot,)



ani = animation.FuncAnimation(fig, testfunc, interval=100, fargs=(img_plot,))
plt.show()
