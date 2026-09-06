import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
import sunpy.map
import pfsspy
from pfsspy import coords
from scipy.interpolate import interp2d

from smp.config import load_config
from smp.constants import DEFAULT_PFSS_RADIAL_POINTS, DEFAULT_SOURCE_SURFACE_RADIUS

# data directory
config = load_config()
obsv_dir = config.path("gong_fits")
pred_dir = config.repository_root / "postprocess_on_class" / "neaten"
swmf_dir = config.repository_root / "postprocess_on_class" / "shl_excel"
save_dir = config.repository_root / "postprocess_on_class" / "comparison"

# PFSS function
def PFSS_source_surface(gong_file):
    gong_map = sunpy.map.Map(gong_file)
    norm = SymLogNorm(linthresh=5)
    gong_map = sunpy.map.Map(gong_map.data - np.mean(gong_map.data), gong_map.meta, plot_settings={'norm': norm})
    nrho, rss = DEFAULT_PFSS_RADIAL_POINTS, DEFAULT_SOURCE_SURFACE_RADIUS
    input_map = pfsspy.Input(gong_map, nrho, rss)
    output_map = pfsspy.pfss(input_map)
    ss_map = output_map.source_surface_br
    return  ss_map

# available cr
cr_lst1 = np.arange(2253,2265)
cr_lst2 = np.arange(2266,2268)
cr_lst3 = np.arange(2269,2274)
cr_lst4 = np.arange(2275,2278)
cr_lst5 = np.arange(2259,2260)
cr_avail_lst = np.concatenate((cr_lst1, cr_lst2, cr_lst3, cr_lst4))

for cr in cr_avail_lst: # 2253,2278
    # PFSS result of observation
    obsv_path = obsv_dir / f"mrzqs_c{cr}.fits"
    obsv_pfss = PFSS_source_surface(obsv_path)
    obsv_pfss_map = obsv_pfss.data

    # # PFSS result of prediction
    pred_path = pred_dir / f"cr{cr}_neaten.fits"
    pred_pfss = PFSS_source_surface(pred_path)
    pred_pfss_map = pred_pfss.data

    # SWMF result of prediction
    swmf_path = swmf_dir / f"Br_{cr}.xlsx"
    pred_swmf = pd.read_excel(swmf_path)
    pred_swmf_map = np.array(pred_swmf)

    # interplate to the same grid
    # pfss grid
    lon_pfss = np.linspace(0, 360, 360)
    lat_sin_pfss = np.linspace(-1, 1, 180)
    lat_pfss = np.degrees(np.arcsin(lat_sin_pfss))
    llon_pfss, llat_pfss = np.meshgrid(lon_pfss, lat_pfss)
    # swmf grid
    lon_swmf = np.linspace(0, 360, 121)
    lat_swmf = np.linspace(-90, 90, 60)
    llon_swmf, llat_swmf = np.meshgrid(lon_swmf, lat_swmf)
    # interpolate
    f1 = interp2d(llon_pfss, llat_pfss, obsv_pfss_map, kind='linear')
    obsv_pfss_map_std = f1(lon_swmf, lat_swmf)
    f2 = interp2d(llon_pfss, llat_pfss, pred_pfss_map, kind='linear')
    pred_pfss_map_std = f2(lon_swmf, lat_swmf)
    pred_swmf_map_std = pred_swmf_map

    # calculate correlation coefficient
    cmatrix_12 = np.corrcoef(obsv_pfss_map_std.flatten(), pred_pfss_map_std.flatten())
    cc_12 = cmatrix_12[0, 1]
    cc_12_rounded = round(cc_12, 3)
    cmatrix_13 = np.corrcoef(obsv_pfss_map_std.flatten(), pred_swmf_map_std.flatten())
    cc_13 = cmatrix_13[0, 1]
    cc_13_rounded = round(cc_13, 3)

    # plot figure
    fig, axes = plt.subplots(3,1,figsize=(6,9))
    # subfig: obsv_pfss_map
    vlim1 = 0.9*np.max(np.abs(obsv_pfss_map_std))
    p1 = axes[0].pcolor(lon_swmf,lat_swmf,obsv_pfss_map_std, cmap='RdBu', vmin=-vlim1, vmax=vlim1)
    plt.colorbar(p1, ax=axes[0])
    axes[0].set_title('PFSS result of observation')
    # subfig: pred_pfss_map
    vlim2 = 0.9*np.max(np.abs(pred_pfss_map_std))
    p2 = axes[1].pcolor(lon_swmf,lat_swmf,pred_pfss_map_std, cmap='RdBu', vmin=-vlim2, vmax=vlim2)
    plt.colorbar(p2, ax=axes[1])
    if cr < 2259:
        axes[1].set_title('PFSS result of reconstruction, cc=' + str(cc_12_rounded))
    else:
        axes[1].set_title('PFSS result of prediction, cc=' + str(cc_12_rounded))
    # subfig: pred_swmf_map
    vlim3 = np.max(np.abs(pred_swmf_map_std))
    p3 = axes[2].pcolor(lon_swmf,lat_swmf,pred_swmf_map_std, cmap='RdBu', vmin=-vlim3, vmax=vlim3)
    plt.colorbar(p3, ax=axes[2])
    if cr < 2259:
        axes[2].set_title('PFSS result of reconstruction, cc=' + str(cc_13_rounded))
    else:
        axes[2].set_title('SWMF result of prediction, cc=' + str(cc_13_rounded))
    # settings
    xticks = np.linspace(0, 360, 9)
    yticks = np.linspace(-90,90, 5)
    axes[0].set_xticks([])
    axes[1].set_xticks([])
    axes[2].set_xticks(xticks)
    axes[2].set_xlabel('Carr. Lon.')
    for ax in axes:
        ax.set_yticks(yticks)
        ax.set_ylabel('Carr. Lat.')
        ax.axhline(y=0, color='black', linewidth=0.8)
    plt.suptitle('CR ' + str(cr), fontsize=16)
    # save figure
    plt.savefig(save_dir / f"CR{cr}.png", bbox_inches='tight')
    # plt.show()
    plt.close()
