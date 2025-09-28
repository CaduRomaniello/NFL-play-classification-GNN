import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def createFootballField(linenumbers=True,
                        endzones=True,
                        highlight_line=False,
                        highlight_line_number=50,
                        highlighted_name='Line of Scrimmage',
                        fifty_is_los=False,
                        figsize=(12, 6.33)):
    """
    Function that plots the football field for viewing plays.
    Allows for showing or hiding endzones.
    
    Returns:
        fig, ax: matplotlib figure and axis objects
    """

    # create figure
    rect = patches.Rectangle((0, 0), 120, 53.3, linewidth=0.1,
                             edgecolor='r', facecolor='gray', zorder=0)

    # create axis
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.add_patch(rect)

    # plot field lines
    ax.plot([10, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 60, 60, 70, 70, 80,
              80, 90, 90, 100, 100, 110, 110, 120, 0, 0, 120, 120],
             [0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3,
              53.3, 0, 0, 53.3, 53.3, 0, 0, 53.3, 53.3, 53.3, 0, 0, 53.3],
             color='white')
    
    # plot line of scrimmage at 50 yd line if fifty_is_los is True
    if fifty_is_los:
        ax.plot([60, 60], [0, 53.3], color='gold')
        ax.text(62, 50, '<- Player Yardline at Snap', color='gold')

    # Endzones
    if endzones:
        ez1 = patches.Rectangle((0, 0), 10, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='black',
                                alpha=0.2,
                                zorder=0)
        ez2 = patches.Rectangle((110, 0), 120, 53.3,
                                linewidth=0.1,
                                edgecolor='r',
                                facecolor='brown',
                                alpha=0.2,
                                zorder=0)
        ax.add_patch(ez1)
        ax.add_patch(ez2)

    # set axis limits
    ax.set_xlim(0, 120)
    ax.set_ylim(-5, 58.3)
    ax.axis('off')

    # plot line numbers
    if linenumbers:
        for x in range(20, 110, 10):
            numb = x
            if x > 50:
                numb = 120 - x
            ax.text(x, 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white')
            ax.text(x - 0.95, 53.3 - 5, str(numb - 10),
                     horizontalalignment='center',
                     fontsize=20,  # fontname='Arial',
                     color='white', rotation=180)
            
    # checking the size of image to plot hash marks for each yd line
    if endzones:
        hash_range = range(11, 110)
    else:
        hash_range = range(1, 120)

    # plot hash marks
    for x in hash_range:
        ax.plot([x, x], [0.4, 0.7], color='white')
        ax.plot([x, x], [53.0, 52.5], color='white')
        ax.plot([x, x], [22.91, 23.57], color='white')
        ax.plot([x, x], [29.73, 30.39], color='white')

    # highlight line of scrimmage
    if highlight_line:
        hl = highlight_line_number + 10
        ax.plot([hl, hl], [0, 53.3], color='yellow')
        ax.text(hl + 2, 50, '<- {}'.format(highlighted_name),
                 color='yellow')
        
    return fig, ax