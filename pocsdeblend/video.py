import matplotlib.pyplot as plt
from celluloid import Camera


def visualize_iterations(data, vclip, cmap='gray', extent=None, titles=None,
                         figsize=(4, 4), interval=500, repeat=False,
                         videofilename=None, dpi=100):
    """Visualize a series of results (e.g., iterations of
    a deblending algorithm) in a video
    """
    fig = plt.figure()
    fig.set_size_inches(figsize[0], figsize[1], forward=False)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)
    camera = Camera(fig)

    for i in range(len(data)):
        ax.imshow(data[i], cmap=cmap,
                  extent=extent,
                  vmin=vclip[0], vmax=vclip[1],
                  interpolation="none")
        if titles is not None:
            ax.text(0.99, 0.99, titles[i], transform=ax.transAxes,
                    ha='right', va='top', color='w', fontsize=12,
                    bbox=dict(facecolor='k'))
        ax.axis('tight')
        ax.axis('off')
        camera.snap()

    animation = camera.animate(interval=interval, repeat=repeat, blit=True)
    plt.close()

    if videofilename is not None:
        animation.save(videofilename, dpi=dpi,
                       savefig_kwargs={'pad_inches': 'tight'}
                       )
    return animation