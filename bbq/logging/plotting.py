# from matplotlib import pyplot as plt
# import numpy as np

# from matplotlib.ticker import FixedLocator, FixedFormatter

# # Multiple Quantities
# def plot_ys(x, ys, labels, ax=None, reverse_y=False):
#     if ax is None:
#         fig, host = plt.subplots(figsize=(8,5),dpi=150)
#     else:
#         host = ax
#     ys= np.rollaxis(ys,1) # [quantities X series]
#     C = [plt.cm.Set2(c) for c in np.linspace(0,1,len(ys))][::-1] # Colors
#     pars = [host.twinx() for _ in range(len(ys[1:]))] # Create additional y scales

#     # Set axis labels
#     i=0
#     host.set_xlabel("Iterations")
#     p, = host.plot(x, ys[i], color=C[i], label=labels[i], linewidth=2, linestyle=':')
#     lns = [p]
#     host.set_ylabel('Archive Size')
#     host.spines['left'].set_color(C[i])
#     host.spines['left'].set_linewidth(3)
#     host.yaxis.label.set_color(C[i])   

#     for i, par in enumerate(pars,1):
#         par.set_ylabel(labels[i])
#         p, = par.plot(x, ys[i], color=C[i], label=labels[i], linewidth=2, linestyle='--')
#         lns.append(p)
#         par.spines['right'].set_position(('outward', i*70))
#         par.spines['right'].set_color(C[i])
#         par.spines['right'].set_linewidth(3)
#         par.yaxis.label.set_color(C[i])   
#         if reverse_y:
#             par.invert_yaxis()
#     host.legend(handles=lns, loc='upper center', ncol=len(lns), fancybox=True, 
#                 bbox_to_anchor=(0.5, -0.125), shadow=True)   
#     return host

# # -- ---------------------------------------------------------------------- -- #
# # Image
# def map_to_image(Z, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(4,4), dpi=150)        
#     img = ax.imshow(np.rollaxis(Z,1).astype(float), cmap='YlGnBu')
#     cbar = plt.colorbar(img,ax=ax)
#     ax.invert_yaxis()
#     return ax

# def view_map(Z, p, ax=None, bin_ticks=False):
#     if ax is None:
#         fig,ax = plt.subplots(figsize=(4,4),dpi=150)    
#     with plt.style.context('me_grid'):        
#         im = ax.imshow(Z, cmap='YlGnBu')

#         # Colorbar
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.1)
#         cb = plt.colorbar(im, cax=cax)
#         for t in cb.ax.get_yticklabels():
#             t.set_horizontalalignment('right')   
#             t.set_x(3.0)

#         # Grid
#         set_map_grid(ax, Z, bin_ticks=bin_ticks, **p)
        
#         ax.set(xlabel = p['desc_labels'][0], ylabel= p['desc_labels'][1])
                
#     return ax

# # Ticks
# def set_map_grid(ax, Z, bin_ticks=False, desc_bounds=0, grid_res=0, **_):
#     map_x, map_y = Z.shape[0], Z.shape[1]
#     xticks = -0.5+np.arange(map_x)
#     yticks = -0.5+np.arange(map_y)  

#     xlabels = np.linspace(desc_bounds[0][0],
#                         desc_bounds[0][1],
#                         grid_res[0])

#     ylabels = np.linspace(desc_bounds[1][0],
#                         desc_bounds[1][1],
#                         grid_res[1])
#     xlabels = np.round(xlabels,1).astype(int)
#     ylabels = np.round(ylabels,0).astype(int)    

#     abbrev_xlabels = ['']*map_x
#     abbrev_ylabels = ['']*map_y

#     if bin_ticks:
#         skip = 3
#         abbrev_xlabels[::skip] = np.arange(0, map_x, skip)
#         abbrev_ylabels[::skip] = np.arange(0, map_y, skip)
#     else:
#         abbrev_xlabels[1], abbrev_xlabels[-1] = xlabels[0], xlabels[-1]
#         abbrev_ylabels[1], abbrev_ylabels[-1] = ylabels[0], ylabels[-1]

#     x_formatter, x_locator = FixedFormatter(abbrev_xlabels), FixedLocator(xticks)
#     y_formatter, y_locator = FixedFormatter(abbrev_ylabels), FixedLocator(yticks)

#     ax.xaxis.set_major_locator(x_locator)
#     ax.xaxis.set_major_formatter(x_formatter)

#     ax.yaxis.set_major_locator(y_locator)
#     ax.yaxis.set_major_formatter(y_formatter)

#     grid_thick = 15/np.max(Z.shape)
#     #ax.grid(color="silver", alpha=.8, linewidth=grid_thick)
#     ax.grid(linewidth=grid_thick)
#     ax.tick_params(direction="in", width=grid_thick, length=grid_thick*4)
#     return ax

# from mpl_toolkits.axes_grid1 import make_axes_locatable
# def view_map(Z, p, ax=None, bin_ticks=False):
#     if ax is None:
#         fig,ax = plt.subplots(figsize=(4,4),dpi=150)    
#     with plt.style.context('bbq_gridmap'):        
#         im = ax.imshow(Z, cmap='YlGnBu')

#         # Colorbar
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.1)
#         cb = plt.colorbar(im, cax=cax)
#         for t in cb.ax.get_yticklabels():
#             t.set_horizontalalignment('right')   
#             t.set_x(3.0)

#         # Grid
#         set_map_grid(ax, Z, bin_ticks=bin_ticks, **p)        
#         ax.set(xlabel = p['desc_labels'][0], ylabel= p['desc_labels'][1])
                
#     return ax