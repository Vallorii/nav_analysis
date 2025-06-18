import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
 
def set_viz_style():
    sns.set_theme(context='notebook', style='whitegrid', font_scale=1.2)
    mpl.rcParams.update({
        'font.size': 15,
        'axes.titlesize': 15,
        'axes.labelsize': 15,
        'xtick.labelsize': 15,
        'ytick.labelsize': 15,
        'legend.fontsize': 15,
        'axes.grid': True,  # Enable gridlines
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': False,
        'axes.spines.bottom': False,
        'figure.titlesize': 15,
        'savefig.transparent': True,
        'axes.edgecolor': 'white',
        'axes.facecolor': 'white',
        'figure.facecolor': 'white',
    })
    # Set Offshore Wind color globally for consistency
    sns.set_palette(sns.color_palette(["#1E88E5"]))  # Blue for Offshore Wind