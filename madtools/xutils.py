import numpy as np
import pandas as pd

def build_particles_from_df(line, df, particle_ref):
    particles = df
    particles['number'] = np.arange(len(particles))
    if 'delta' not in particles.columns:
        particles['delta'] = particles['pt']
    particles['zeta'] = 0.0
    cols = ['x', 'px', 'y', 'py', 'zeta', 'delta']
    particles_selected = particles.loc[:, cols]
    return line.build_particles(**(particles_selected.to_dict(orient='list')), particle_ref=particle_ref)

def get_tracking_output(line, cols = ['s', 'x', 'px', 'y', 'py', 'zeta', 'delta', 'at_element']):
    out = line.record_last_track.to_dict()
    out = pd.DataFrame(out['data'])
    elements = line.get_table().to_pandas()
    elements['at_element'] = elements.index
    out = out.join(elements, on='at_element', rsuffix='_elem', how='inner')
    #out = out.join(twiss.loc[:, ['tilt']], on='name', rsuffix='_twiss', how='inner')
    out.index = out['name']
    out_selected = out.loc[:, cols]
    return out_selected

def get_elements_with_attributes(line, attr_names):
    elements = line.get_table().to_pandas()
    attrs = {}
    for name in attr_names:
        attrs[name] = []

    for name in elements.name:
        for key in attrs.keys():
            if key == 'aperture':
                try:
                    aper_name = line.element_dict[name].name_associated_aperture
                    aper = line.element_dict[aper_name]
                except KeyError:
                    aper = None
                attrs['aperture'].append(aper)
            else:
                try:
                    attr = getattr(line.element_dict[name], key)
                except AttributeError:
                    attr = None
                except KeyError:
                    attr = None
                attrs[key].append(attr)
    
    for key, value in attrs.items():
        elements[key] = value
    return elements