import astropy.io.fits as pf
import numpy as np
import matplotlib.pylab as plt

fig, axes = plt.subplots(3, 3, figsize=(12,8))
mags = np.arange(10.5, 16.5, 2)
stks = [1, 5, 11]

for i in range(np.size(mags)):
    mag = mags[i]
    for j in range(np.size(stks)): 
        stk = stks[j]
        ax = axes[i, j]
        for bit in range(10,17):
            fname = f"04_stacked/image.{mag:.1f}.{stk:02d}.{bit:2d}bit.fits"
            data = pf.open(fname)[0].data
            if i == 0 and j == 0:
                ax.step(np.arange(-10,10)+0.5, data[928,1018:1038],label=f'{bit:2d}bit')
            else:
                ax.step(np.arange(-10,10)+0.5, data[928,1018:1038])
#        ax.set_yscale('log')
        ax.set_title(f'{mag:.1f} mag, {stk:d} stack')
        ax.set_xlabel('x-position (px)')
        ax.set_ylabel('Normalized count (adu)')
#        ax.legend()
fig.legend(bbox_to_anchor=(.985, .96),loc='upper right', fontsize=8)
plt.tight_layout()
#plt.savefig('prg03.pdf')
plt.show()

exit()
