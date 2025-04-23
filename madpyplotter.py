import numpy as np
import matplotlib.pyplot as plt
import madutils as mu
import os
import imageio
from glob import glob
from IPython.display import display, Image
import pandas as pd
from ipywidgets import interact, widgets

plt.rc('figure', titlesize=10)
plt.rc('axes', titlesize=8)

scatter_style = dict(
    s = 2
)

fit_style = dict(
    ls = '--',
    color = 'k'
)

def plot_beam_info(
        beam,
        beam_name = 'beam',
        energy_GeV = 400,
        filter_outliners = True,
        max_z_score = 100,
        correct_d = False,
        correct_d_for_pars = True,
        normalize = False,
        color = 'pt',
        bins = 100,
        xlim = (None, None),
        ylim = (None, None),
        pxlim = (None, None),
        pylim = (None, None),
        show = True,
        saveto = None):
    
    beam = beam.copy()

    num_part_init = len(beam.index)
    if filter_outliners:
        mu.filter_outliners(beam, max_z_score=max_z_score, inplace=True)
    
    if energy_GeV:
        beam.e = energy_GeV
    if type(beam.index[0]) == str:
        loc = beam.index[0]
    else:
        loc = 'unknown'
        beam['name'] = loc
        beam.index = beam['name']
        
    pars = mu.calc_beam_pars_df(beam, correct_d=correct_d_for_pars)
    
    if correct_d:
        beam = mu.remove_dispersion(beam)

    ex_um = pars.loc[loc]['emitt_norm_x'] * 1e6
    ey_um = pars.loc[loc]['emitt_norm_y'] * 1e6
    sx_mm = pars.loc[loc]['sigx'] * 1e3
    sy_mm = pars.loc[loc]['sigy'] * 1e3
    spx_mrad = pars.loc[loc]['sigpx'] * 1e3
    spy_mrad = pars.loc[loc]['sigpy'] * 1e3
    s = beam.s.mean() if 's' in beam.columns else 0

    fig, ax = plt.subplots(4, 2, figsize=(7, 10))
    fig.suptitle(f'\"{beam_name}\"' + f' at {beam.index[0]}, s = {s:.2f} m, num part = {len(beam.index)}, filtered = {num_part_init - len(beam.index)}' )

    if normalize:
        beam = mu.normalize_coordinates(beam)
        beam['pt'] = beam.pt * 1e3
    else:
        beam = mu.to_human_units(beam)

    beam.plot(kind='scatter', x='x', y='px', 
              ax=ax[0, 0], 
              title=fr"$\sigma_x =$ {sx_mm:.2f} mm, $\sigma_x' =$ {spx_mrad:.2f} mrad, $\varepsilon_x = ${ex_um:.2f} um", 
              color=beam[color],
              xlabel='x [mm]' if not normalize else r'$x / \sqrt{\beta_x}$',
              ylabel=r'$p_x$ [mrad]' if not normalize else r'$p_x \sqrt{\beta_x}$',
              xlim=xlim,
              ylim=pxlim,
              **scatter_style)
    beam.plot(kind='scatter', x='y', y='py', 
              ax=ax[0, 1], 
              title=fr"$\sigma_y =$ {sy_mm:.2f} mm, $\sigma_y' =$ {spy_mrad:.2f} mrad, $\varepsilon_y = ${ey_um:.2f} um", 
              color=beam[color],
              xlabel='y [mm]' if not normalize else r'$y / \sqrt{\beta_y}$',
              ylabel=r'$p_y$ [mrad]' if not normalize else r'$p_y \sqrt{\beta_y}$',
              xlim=ylim,
              ylim=pylim,
              **scatter_style)
   
    _, values = np.histogram(beam[color], bins=bins)
    cmap = plt.cm.viridis

    ax[1, 1].set_title('Momentum distribution')
    _, _, bars = ax[1, 1].hist(beam['pt'], bins=bins)
    for i, (value, bar) in enumerate(zip(values, bars)):
        bar.set_facecolor(cmap((value - values.min())/(values.max() - values.min())))
    ax[1, 1].set_xlabel(r'$\delta_p \,/ \,10^{-3}$')

    ax[1, 0].set_title('Longitudinal distribution')
    _, _, bars = ax[1, 0].hist(beam['t'] * 1e3, bins=bins)
    for i, (value, bar) in enumerate(zip(values, bars)):
        bar.set_facecolor(cmap((value - values.min())/(values.max() - values.min())))
    ax[1, 0].set_xlabel('ct [m]')

    pt = np.linspace(beam.pt.min(), beam.pt.max(), 10)
    beam.plot(
        kind='scatter', x='pt', y='x', 
        ax=ax[2,0], 
        title=fr'$D_x =$ {pars.loc[loc].dx:.2f} m',
        xlabel=r'$\delta_p \,/ \,10^{-3}$',
        ylabel='x [mm]' if not normalize else r'$x / \sqrt{\beta_x}$',
        ylim=xlim,
        **scatter_style)
    dx = pars.loc[loc]['dx'] * (1. if not normalize else 1e-3 / pars.loc[loc]['betx']**0.5)
    ax[2,0].plot(pt, (pt - pt.mean()) * dx + beam.x.mean(), **fit_style)
    beam.plot(
        kind='scatter', x='pt', y='px', 
        ax=ax[2,1], 
        title=fr"$D'_x =$ {pars.loc[loc].dpx:.2f}",
        xlabel=r'$\delta_p \,/ \,10^{-3}$',
        ylabel=r'$p_x$ [mrad]' if not normalize else r'$p_x \sqrt{\beta_x}$',
        ylim=pxlim,
        **scatter_style)
    dpx = pars.loc[loc]['dpx'] * (1. if not normalize else 1e-3 * pars.loc[loc]['betx']**0.5)
    ax[2,1].plot(pt, (pt - pt.mean()) * dpx + beam.px.mean(), **fit_style)
    beam.plot(
        kind='scatter', x='pt', y='y', 
        ax=ax[3,0], 
        title=fr'$D_y =$ {pars.loc[loc].dy:.2f} m',
        xlabel=r'$\delta_p \,/ \,10^{-3}$',
        ylabel='y [mm]' if not normalize else r'$y / \sqrt{\beta_y}$',
        ylim=ylim,
        **scatter_style)
    dy = pars.loc[loc]['dy'] * (1. if not normalize else 1e-3 / pars.loc[loc]['bety']**0.5)
    ax[3,0].plot(pt, (pt - pt.mean()) * dy + beam.y.mean(), **fit_style)
    beam.plot(
        kind='scatter', x='pt', y='py', 
        ax=ax[3,1], 
        title=fr"$D'_y =$ {pars.loc[loc].dpy:.2f}",
        xlabel=r'$\delta_p \,/ \,10^{-3}$',
        ylabel=r'$p_y$ [mrad]' if not normalize else r'$p_y \sqrt{\beta_y}$',
        ylim=pylim,
        **scatter_style)
    dpy = pars.loc[loc]['dpy'] * (1. if not normalize else 1e-3 * pars.loc[loc]['bety']**0.5)
    ax[3,1].plot(pt, (pt - pt.mean()) * dpy + beam.py.mean(), **fit_style)

    plt.tight_layout()

    if saveto:
        os.makedirs(saveto, exist_ok=True)
        fig.savefig(os.path.join(saveto, f'{beam_name}-{loc}.png'))
    
    if show:
        plt.show()
    else:
        plt.close()

def show_beam_images(
    png_dir,
    tw,
    interactive=False,
    prefix='',
    postfix='.png',
    par_delimiter=':',
    val_delimeter='-',
    loc_key = 'beam',   
    **kwargs):

    def parse_filename(filename, par_delimiter=par_delimiter, val_delimeter=val_delimeter, prefix=prefix, postfix=postfix):
        unparced = filename.replace(prefix, '').replace(postfix, '').split(par_delimiter)
        keys = [part.split(val_delimeter)[0] for part in unparced]
        values = [part.split(val_delimeter)[1] for part in unparced]
        out = {}
        for key, value in zip(keys, values):
            try:
                out[key] = float(value)
            except:
                out[key] = value
        return out

    def show_image(**kwargs):
        suffix = par_delimiter.join([f"{key}{val_delimeter}{value:.2f}" if type(value) == float else f"{key}{val_delimeter}{value}" for key, value in kwargs.items()])
        filename = prefix + suffix + postfix
        filename = os.path.join(png_dir, filename)
        print(filename)
        display(Image(filename=filename))

    pngs = glob(png_dir + '*' + postfix)
    pngs = [os.path.basename(png) for png in pngs]
    pars = [pd.DataFrame(parse_filename(png), index=[0]) for png in pngs]
    pars = pd.concat(pars, ignore_index=True)
    pars['filename'] = pngs

    pars.index = pars[loc_key].values
    pars = pars.join(tw[['s']]).sort_values('s').drop('s', axis='columns')

    kwargs = {column: widgets.SelectionSlider(description=column, 
                                            options=sorted(pars[column].drop_duplicates()) if column != loc_key else pars[column]
                                            ) for column in pars.drop('filename', axis=1).columns
    }
    if interactive:
        interact(show_image, **kwargs)

    return pars

def make_gif(png_dir, gif_name=None, pars=None, **kwargs):
    if gif_name is None:
        gif_name = os.path.basename(os.path.dirname(png_dir)) + '.gif'
    images = []
    if pars is None:
        filenames = sorted(glob(png_dir + '*.png'))
    else:
        filenames = png_dir + pars.filename 
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(gif_name, images, **kwargs)
    return gif_name

def show_gif(gif_name):
    display(Image(filename=gif_name))