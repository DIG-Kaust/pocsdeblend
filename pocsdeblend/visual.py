import numpy as np
import matplotlib.pyplot as plt
    
from matplotlib.colors import LinearSegmentedColormap
from ipywidgets import interactive_output, IntSlider, HBox, HTML
from IPython.display import display
from pylops.utils.metrics import snr


cmap_amplitudepkdsg = \
    LinearSegmentedColormap.from_list('name', ['#33ffff', '#33adff', '#0000ff',
                                               '#666666', '#d9d9d9', '#805500',
                                               '#ff6600', '#ffdb4d', '#ffff00'])


def display_result(data, t, s, vclip, figsize=(12, 8), cmap="gray", title="CRG"):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(data, cmap=cmap,
              extent=(s[0], s[-1], t[-1], t[0]),
              vmin=vclip[0], vmax=vclip[1],
              interpolation="none")
    ax.set_title(title)
    ax.set_xlabel("#Src")
    ax.set_ylabel("t [s]")
    ax.axis("tight")
    

def display_results(data, data_pseudo, data_inv, t, s, vclip):
    fig, axs = plt.subplots(1, 4, sharey=True, figsize=(12, 8))
    axs[0].imshow(data, cmap="gray",
                  extent=(s[0], s[-1], t[-1], t[0]),
                  vmin=vclip[0], vmax=vclip[1],
                  interpolation="none")
    axs[0].set_title("CRG")
    axs[0].set_xlabel("#Src")
    axs[0].set_ylabel("t [s]")
    axs[0].axis("tight")
    axs[1].imshow(data_pseudo, cmap="gray",
                  extent=(s[0], s[-1], t[-1], t[0]),
                  vmin=vclip[0], vmax=vclip[1],
                  interpolation="none")
    axs[1].set_title("Pseudo-deblended CRG")
    axs[1].set_xlabel("#Src")
    axs[1].axis("tight")
    axs[2].imshow(data_inv, cmap="gray",
                  extent=(s[0], s[-1], t[-1], t[0]),
                  vmin=vclip[0], vmax=vclip[1],
                  interpolation="none")
    axs[2].set_xlabel("#Src")
    axs[2].set_title("Deblended CRG")
    axs[2].axis("tight")
    axs[3].imshow(data - data_inv, cmap="gray",
                  extent=(s[0], s[-1], t[-1], t[0]),
                  vmin=vclip[0], vmax=vclip[1],
                  interpolation="none")
    axs[3].set_xlabel("#Src")
    axs[3].set_title(f"Blending error {snr(data.real, data_inv.real):.2f} db")
    axs[3].axis("tight")
    plt.tight_layout()
    
    
    
def display_residuals(data_pseudo, data_inv, Bop, t, s, vclip, tlim=None, slim=None, figsize=(12, 8)):
    ns, nt = len(s), len(t)
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=figsize)
    axs[0].imshow(data_pseudo, cmap="gray",
                  extent=(s[0], s[-1], t[-1], t[0]),
                  vmin=vclip[0], vmax=vclip[1],
                  interpolation="none")
    axs[0].set_title("Pseudo-deblended CRG")
    axs[0].set_xlabel("#Src")
    axs[0].axis("tight")
    if slim is not None: axs[0].set_xlim(slim)
    axs[1].imshow((Bop.H @ Bop @ data_inv.T.ravel()).reshape(ns, nt).T.real, cmap="gray",
                  extent=(s[0], s[-1], t[-1], t[0]),
                  vmin=vclip[0], vmax=vclip[1],
                  interpolation="none")
    axs[1].set_xlabel("#Src")
    axs[1].set_title("Pseudo-deblended solution")
    axs[1].axis("tight")
    if slim is not None: axs[1].set_xlim(slim)
    axs[2].imshow(data_pseudo - (Bop.H @ Bop @ data_inv.T.ravel()).reshape(ns, nt).T.real, cmap="gray",
                  extent=(s[0], s[-1], t[-1], t[0]),
                  vmin=vclip[0], vmax=vclip[1],
                  interpolation="none")
    axs[2].set_xlabel("#Src")
    axs[2].set_title("Residual")
    axs[2].axis("tight")
    if slim is not None: axs[2].set_xlim(slim)
    if tlim is not None: axs[2].set_ylim(tlim)
    plt.tight_layout()
    

def updates_widget(data, t, s, vclip, title='Inversion Widget', figsize=(7, 6)):
    nt, ns = len(t), len(s)
    niters = len(data)
    curr_iter = niters - 1
    title = "<b>%s</b>" % title
    slider_iters = IntSlider(min=1, max=niters-1, value=curr_iter,
                             step=1, description='Iteration')
    title = HTML(value=title, placeholder="Update Widget", description='')

    def handle_iters_change(change):
        global curr_iter
        curr_iter = change.new

    slider_iters.observe(handle_iters_change, names='value')
    
    out = interactive_output(lambda iteration: display_result(data[iteration].squeeze().reshape(ns, nt).real.T, 
                                                              t, s, vclip, 
                                                              cmap=cmap_amplitudepkdsg,
                                                              figsize=figsize),
                             {"iteration": slider_iters})
    ui = HBox([title, slider_iters])

    display(ui, out)