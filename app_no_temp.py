import intake
import numpy as np
import pandas as pd
import hvplot.pandas

import holoviews as hv
from holoviews.streams import Selection1D, Params
import panel as pn
import colorcet as cc

import geoviews as gv
import geoviews.tile_sources as gts
import cartopy.crs as ccrs
import pyproj

hv.extension('bokeh')

df = intake.open_csv('./data/bird_migration/{species}.csv').read()

def fill_day(v):
    next_year = v.assign(day=v.day + v.day.max())
    last_year = v.assign(day=v.day - v.day.max())
    surrounding_years = pd.concat([last_year, v, next_year])
    filled = surrounding_years.assign(
        lat=surrounding_years.lat.interpolate(),
        lon=surrounding_years.lon.interpolate())
    this_year = filled[filled.day.isin(v.day)]
    return this_year

g = pyproj.Geod(ellps='WGS84')

def calculate_speed(v):
    today_lat = v['lat'].values
    today_lon = v['lon'].values
    tomorrow_lat = np.append(v['lat'][1:].values, v['lat'][0])
    tomorrow_lon = np.append(v['lon'][1:].values, v['lon'][0])
    _, _, dist = g.inv(today_lon, today_lat, tomorrow_lon, tomorrow_lat)
    return v.assign(speed=dist/1000.)

df = pd.concat([calculate_speed(fill_day(v)) for k, v in df.groupby('species')])

species_cmap = dict(zip(df.species.cat.categories, cc.glasbey))

birds = df.hvplot.points('lon', 'lat', color='species', groupby='day', geo=True,
                         cmap=species_cmap, legend=False, width=400, height=600,
                         size=100).options(tools=['tap', 'hover', 'box_select'])

tiles = gts.EsriImagery()
tiles.extents = df.lon.min(), df.lat.min(), df.lon.max(), df.lat.max()

def timeseries(species=None, y='lat'):
    data = df[df.species.isin(species)] if species else df
    plots = [
        (data.groupby(['day', 'species'], observed=True)[y]
            .mean()
            .groupby('day').agg([np.min, np.max])
            .hvplot.area('day', 'amin', 'amax', alpha=0.2, fields={'amin': y}))]
    if not species or len(species) > 7:
        plots.append(data.groupby('day')[y].mean().hvplot().relabel('mean'))
    else:
        gb = data.groupby('species', observed=True)
        plots.extend([v.hvplot('day', y, color=species_cmap[k]).relabel(k) for k, v in gb])
    return hv.Overlay(plots).options(width=900, height=250, toolbar='below', legend_position='right', legend_offset=(20, 0), label_width=150)

def daily_table(species=None, day=None):
    if not species or not day:
        return hv.Table(pd.DataFrame(columns=['Species', 'Speed [km/day]'])).relabel('No species selected')
    subset = df[df.species.isin(species)]
    subset = subset[subset.day==day]
    return hv.Table(pd.DataFrame({'Species': species, 'Speed [km/day]': subset['speed']})).relabel('day: {}'.format(day))

species = pn.widgets.MultiSelect(options=df.species.cat.categories.tolist(), size=10)
day = pn.widgets.Player(value=1, start=1, end=365, step=5, loop_policy='loop', name='day', width=350)
highlight = pn.widgets.Toggle(name='Highlight Birds', active=False)

species_stream = Params(species, ['value'], rename={'value': 'species'})
day_stream = Params(day, ['value'], rename={'value': 'day'})
highlight_stream = Params(highlight, ['active'])

def reset(arg=None):
    day_stream.update(value=1)
    species_stream.update(value=[])
    highlight_stream.update(active=False)

reset_button = pn.widgets.Button(name='Reset')
reset_button.param.watch(reset, 'clicks')

def do_highlight(points, active=True):
    return points.options(line_alpha=(0.5 if active else 0), selection_line_alpha=active)

bird_dmap = hv.util.Dynamic(birds.clone(streams=[day_stream]).options(line_color='white'),
                            operation=do_highlight, streams=[highlight_stream])

ts_lat = hv.DynamicMap(lambda species: timeseries(species, 'lat'), streams=[species_stream])
ts_speed = hv.DynamicMap(lambda species: timeseries(species, 'speed'), streams=[species_stream])
table = hv.DynamicMap(daily_table, streams=[species_stream, day_stream])

def on_map_select(index):
    if index:
        species = df.species.cat.categories[index].tolist()
        if set(species_stream.contents['species']) != set(species):
            species_stream.update(value=species)

map_selected_stream = Selection1D(source=bird_dmap)
map_selected_stream.param.watch_values(on_map_select, ['index'])

dashboard = pn.Column(
    pn.Row('## Bird Migration Dashboard', pn.Spacer(width=200, height=80)),
    pn.Row(
        pn.Column(
            pn.Row(
                pn.Row(tiles * bird_dmap)[0][0],
                pn.Spacer(width=20),
                pn.Column(
                    '**Day of Year**', day,
                    '**Species**:',
                    'This selector does not affect the map. Use plot selectors.', species,
                    highlight,
                    'This reset button only resets widgets - otherwise use the plot reset 🔄',
                    reset_button
                ),
                pn.Spacer(width=100),
            ),
            pn.Row(pn.layout.Tabs(('Latitude', ts_lat), ('Speed', ts_speed))),
        ),
        pn.Column(table.options(width=300, height=850))
    )
)
dashboard.servable()