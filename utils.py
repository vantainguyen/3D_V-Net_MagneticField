import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from scipy import ndimage
import seaborn as sns
import numpy as np




# ------------------Visualizing function------------------------------------------

def view_slices_3d(geometry, st_axial_l, st_axial_p, slice_, title=''):

    """
    Plotting functions to compare the ground truth and prediction
    """

    fig = plt.figure(figsize=(12, 17))
    plt.suptitle(title, fontsize=14)

    # --------------Plotting geometrical sections-----------------------------------
    plt.subplot(531)
    ga = sns.heatmap(np.take(geometry, slice_, 2), cmap='YlGnBu')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)
    plt.title('Axial')
    plt.ylabel('Geo & Mag (T)', fontsize=12)

    plt.subplot(532)
    image_rot = ndimage.rotate(np.take(geometry, slice_, 1),90)
    ga = sns.heatmap(image_rot, cmap='YlGnBu')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)
    plt.title('Coronal')

    plt.subplot(533)
    image_rot = ndimage.rotate(np.take(geometry, slice_, 0),90)
    ga = sns.heatmap(image_rot, cmap='YlGnBu')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)
    plt.title('Sagittal')

    # --------------Plotting the ground truth field---------------------------------
    plt.subplot(534)
    vmaxa = np.max(np.take(st_axial_l, slice_, 2))
    vmina = np.min(np.take(st_axial_l, slice_, 2))
    ga = sns.heatmap(np.take(st_axial_l, slice_, 2), vmax=vmaxa, vmin=vmina, cmap='YlGnBu', fmt='g')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)
    plt.ylabel('Ground truth (A/m)', fontsize=12)

    plt.subplot(535)
    image_rot = ndimage.rotate(np.take(st_axial_l, slice_, 1),90)
    vmaxc = np.max(image_rot)
    vminc = np.min(image_rot)
    ga = sns.heatmap(image_rot, vmax=vmaxc, vmin=vminc, cmap='YlGnBu', fmt='g')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)

    plt.subplot(536)
    image_rot = ndimage.rotate(np.take(st_axial_l, slice_, 0),90)
    vmaxs = np.max(image_rot)
    vmins = np.min(image_rot)
    ga = sns.heatmap(image_rot, vmax=vmaxs, vmin=vmins, cmap='YlGnBu', fmt='g')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)

    # -------------Plotting predicted field--------------------------------------------
    plt.subplot(537)
    ga = sns.heatmap(np.take(st_axial_p, slice_, 2),vmax=vmaxa, vmin=vmina, cmap='YlGnBu', fmt='g')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)
    plt.ylabel('Predicted (A/m)', fontsize=12)

    plt.subplot(538)
    image_rot = ndimage.rotate(np.take(st_axial_p, slice_, 1),90)
    ga = sns.heatmap(image_rot, vmax=vmaxc, vmin=vminc, cmap='YlGnBu', fmt='g')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)

    plt.subplot(539)
    image_rot = ndimage.rotate(np.take(st_axial_p, slice_, 0),90)
    ga = sns.heatmap(image_rot, vmax=vmaxs, vmin=vmins, cmap='YlGnBu', fmt='g')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)

    # -----------Plotting difference between the ground truth and prediction------------
    err = st_axial_l - st_axial_p

    plt.subplot(5, 3, 10)
    ga = sns.heatmap(np.take(err, slice_, 2), cmap='YlGnBu')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)
    plt.ylabel('Difference (A/m)', fontsize=12)

    plt.subplot(5, 3,11)
    image_rot = ndimage.rotate(np.take(err, slice_, 1),90)
    ga = sns.heatmap(image_rot, cmap='YlGnBu')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)

    plt.subplot(5, 3,12)
    image_rot = ndimage.rotate(np.take(err, slice_, 0),90)
    ga = sns.heatmap(image_rot, cmap='YlGnBu')
    ga.set(xticklabels=[])
    ga.set(yticklabels=[])
    ga.tick_params(bottom=False, left=False)
    
    # -----------Plotting distribution of normalised error ------------------------------
    interq_991 = np.subtract(*np.percentile(np.take(err, slice_, 2).flatten(), [99, 1]))
    errors = abs(np.take(err, slice_, 2).flatten())*100/interq_991
    mean = np.mean(errors)
    std = np.std(errors)

    plt.subplot(5,3,13)
    ga = sns.distplot(errors, kde=True, hist=False, rug=False, kde_kws={"shade": True, "bw_adjust": 1, "cut":0, "bw_method": 'silverman'}, color='b')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x: .2f}')) # set yaxis to 2 decimal places
    plt.xlabel('Error (%)', fontsize=12)
    plt.legend(title='\u03BC = {:2.2}, \u03C3 = {:0.3}'.format(mean, std), fontsize=8)
    ga.set(xlim=(-10, 150))
    plt.ylabel('Density', fontsize=12)

    interq_991 = np.subtract(*np.percentile(ndimage.rotate(np.take(err, slice_, 1),90).flatten(), [99, 1]))
    errors = abs(ndimage.rotate(np.take(err, slice_, 1),90))*100/interq_991
    mean = np.mean(errors)
    std = np.std(errors)

    plt.subplot(5,3,14)
    ga = sns.distplot(errors, kde=True, hist=False, rug=False, kde_kws={"shade": True, "bw_adjust": 1, "cut":0, "bw_method": 'silverman'}, color='b')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x: .2f}')) # set yaxis to 2 decimal places
    plt.xlabel('Error (%)', fontsize=12)
    plt.legend(title='\u03BC = {:2.2}, \u03C3 = {:0.3}'.format(mean, std), fontsize=8)
    ga.set(ylabel=None)
    ga.set(xlim=(-10, 150))

    interq_991 = np.subtract(*np.percentile(ndimage.rotate(np.take(err, slice_, 0),90).flatten(), [99, 1]))
    errors = abs(ndimage.rotate(np.take(err, slice_, 0),90))*100/interq_991
    mean = np.mean(errors)
    std = np.std(errors)

    plt.subplot(5,3,15)
    ga = sns.distplot(errors, kde=True, hist=False, rug=False, kde_kws={"shade": True, "bw_adjust": 1, "cut":0, "bw_method": 'silverman'}, color='b')
    plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x: .2f}')) # set yaxis to 2 decimal places
    plt.xlabel('Error (%)', fontsize=12)
    plt.legend(title='\u03BC = {:2.2}, \u03C3 = {:0.3}'.format(mean, std), fontsize=12)
    ga.set(ylabel=None)
    ga.set(xlim=(-10, 150))

    plt.subplots_adjust()
    plt.show()