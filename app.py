import os
import intake
import numpy as np
import pandas as pd
import xarray as xr

import hvplot.pandas
import hvplot.xarray

import holoviews as hv
from holoviews.streams import Selection1D, Params
import panel as pn

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

colors = pd.read_csv('./assets/colormap.csv', header=None, names=['R', 'G', 'B'])
species_cmap = dict(zip(df.species.cat.categories,
                        ['#{row.R:02x}{row.G:02x}{row.B:02x}'.format(row=row)
                         for _, row in colors.iterrows()]))

birds = df.hvplot.points('lon', 'lat', color='species', groupby='day', geo=True,
                         cmap=species_cmap, legend=False).options(tools=['tap', 'hover', 'box_select'],
                                                                  width=500, height=600)
data_url = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep/air.day.ltm.nc'

# I downloaded the file locally because I was hitting rate limits. Just comment out this line
data_url = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep/air.day.ltm.nc'

# I downloaded the file locally because I was hitting rate limits.
local_file = './data/air.day.ltm.nc'
if os.path.isfile(local_file):
    data_url = local_file

ds = xr.open_dataset(data_url)
ds = ds.rename(time='day').sel(level=1000)
ds['day'] = list(range(1,366))

## convert to F
ds = ds.assign(air_F = (ds['air'] - 273.15) * 9/5 + 32)

ROI = ds.sel(lon=slice(205, 310), lat=slice(75, -55))

grouped_air = (ROI.hvplot.quadmesh('lon', 'lat', 'air_F', groupby='day', geo=True)
                         .options(height=600, width=500, tools=[])
                         .redim.range(air_F=(-20, 100)))


tiles = gts.EsriImagery()
tiles.extents = df.lon.min(), df.lat.min(), df.lon.max(), df.lat.max()

def timeseries(species=None, day=None, y='lat'):
    plots = [
        (df.groupby(['day', 'species'], observed=True)[y]
            .mean()
            .groupby('day').agg([np.min, np.max])
            .hvplot.area('day', 'amin', 'amax', alpha=0.2, fields={'amin': y}))]
    if not species:
        plots.append(df.groupby('day')[y].mean().hvplot().relabel('mean'))
    else:
        gb = df[df.species.isin(species)].groupby('species', observed=True)
        plots.extend([v.hvplot('day', y, color=species_cmap[k]).relabel(k) for k, v in gb])
    if day:
        plots.append(hv.VLine(day).options(color='black'))
    return hv.Overlay(plots).options(width=900, toolbar='below', legend_position='right', legend_offset=(20, 0), label_width=150)

def daily_table(species=None, day=None):
    def temp_calc(ds, row):
        lat_lon_day = row[['lat', 'lon', 'day']]
        return round(ds.sel(**lat_lon_day, method='nearest')['air_F'].item())

    if not species or not day:
        return hv.Table(pd.DataFrame(columns=['Species', 'Air [F]', 'Speed [km/day]'])).relabel('No species selected')

    subset = df[df.species.isin(species)]
    subset = subset[subset.day==day]
    temps = [temp_calc(ds, row) for _, row in subset.iterrows()]

    return hv.Table(pd.DataFrame({'Species': species, 'Air [F]': temps, 'Speed [km/day]': subset['speed']})).relabel('day: {}'.format(day))

species = pn.widgets.MultiSelect(options=df.species.cat.categories.tolist(), size=10)
day = pn.widgets.Player(value=1, start=1, end=365, step=5, loop_policy='loop', name='day', width=350)
toggle = pn.widgets.Toggle(name='Air Temperature Layer', active=True)

species_stream = Params(species, ['value'], rename={'value': 'species'})
day_stream = Params(day, ['value'], rename={'value': 'day'})
toggle_stream = Params(toggle, ['active'])

def reset(arg=None):
    day_stream.update(value=1)
    species_stream.update(value=[])
    toggle_stream.update(active=True)

reset_button = pn.widgets.Button(name='Reset')
reset_button.param.watch(reset, 'clicks')

def toggle_temp(layer, active=True):
    return layer.options(fill_alpha=int(active))

bird_dmap = birds.clone(streams=[day_stream])
air_dmap = grouped_air.clone(streams=[day_stream])
temp_layer = hv.util.Dynamic(air_dmap, operation=toggle_temp, streams=[toggle_stream])
ts_lat = hv.DynamicMap(lambda species, day: timeseries(species, day, 'lat'), streams=[species_stream, day_stream])
ts_speed = hv.DynamicMap(lambda species, day: timeseries(species, day, 'speed'), streams=[species_stream, day_stream])
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
                pn.Row(tiles * temp_layer * gv.feature.coastline * bird_dmap)[0][0],
                pn.Spacer(width=20),
                pn.Column(
                    '**Day of Year**', day,
                    '**Species**:',
                     'This selector does not affect the map. Use plot selectors.', species,
                    '**Toggle Air Temp on and off**', toggle,
                    'This reset button only resets widgets - otherwise use the plot reset 🔄',
                    reset_button
                ),
                pn.Spacer(width=120),
            ),
            pn.Row(pn.layout.Tabs(('Latitude', ts_lat), ('Speed', ts_speed))),
        ),
        pn.Column(table.options(width=300, height=900))
    )
)
dashboard.servable()