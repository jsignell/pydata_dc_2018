import intake
import numpy as np
import pandas as pd
import hvplot.pandas
import xarray as xr
import hvplot.xarray

import holoviews as hv
from holoviews.streams import Selection1D, Params
import panel as pn

import geoviews as gv
import geoviews.tile_sources as gts
import cartopy.crs as ccrs

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

df = pd.concat([fill_day(v) for k, v in df.groupby('species')])

ds = xr.open_dataset('http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/ncep/air.day.ltm.nc')
ds = ds.rename(time='day').sel(level=1000)
ds['day'] = list(range(1,366))

## convert to F
ds = ds.assign(air_F = (ds['air'] - 273.15) * 9/5 + 32)

ROI = ds.sel(lon=slice(205, 310), lat=slice(75, -55)).persist()

grouped_air = ROI.hvplot.quadmesh('lon', 'lat', 'air_F', groupby='day', geo=True).options(height=600, width=500, tools=[])
    
tiles = gts.EsriImagery()
tiles.extents = df.lon.min(), df.lat.min(), df.lon.max(), df.lat.max()

birds = df.hvplot.points('lon', 'lat', color='species', groupby='day', geo=True,
                         cmap='colorwheel', legend=False).options(tools=['tap', 'hover', 'box_select'], 
                                                                  width=500, height=600)

def temp_table(species=None, day=None):
    def func(ds, row):
        lat_lon_day = row[['lat', 'lon', 'day']]
        return round(ds.sel(**lat_lon_day, method='nearest')['air_F'].item())
    foo = pd.DataFrame(columns=['temp'], index=pd.Series(name='species'))
    if not species or not day:
        return hv.Table(pd.DataFrame(columns=['Species', 'Air [F]']))
    subset = df[df.species.isin(species)]
    subset = subset[subset.day==day]
    temps = [func(ds, row) for _, row in subset.iterrows()]
    return hv.Table(pd.DataFrame({'Species': species, 'Air [F]': temps}))
                                                                 
                                                                  
def timeseries(species=None, day=None):
    if not species:
        ts = hv.Overlay([
            df.groupby(['day', 'species'], observed=True)['lat'].mean().groupby('day').agg([np.min, np.max]).hvplot.area('day', 'amin', 'amax', alpha=0.2, fields={'amin': 'lat'}),
            df.groupby('day')['lat'].mean().hvplot().relabel('mean')
        ]).options(width=700)
    else:
        gb = df[df.species.isin(species)].groupby('species', observed=True)
        ts = hv.Overlay([v.hvplot('day', 'lat').relabel(k) for k, v in gb])
    if day:
        return ts * hv.VLine(day).options(color='black')
    return ts

day = pn.widgets.Player(value=1, interval=3, length=365, loop_policy='loop', name='day', width=400)
species = pn.widgets.MultiSelect(options=df.species.cat.categories.tolist())
baselayer_selector = pn.widgets.Select(options=['tiles', 'temp'])

day_stream = Params(day, ['value'], rename={'value': 'day'})
species_stream = Params(species, ['value'], rename={'value': 'species'})
baselayer_selector_stream = Params(baselayer_selector, ['value'])

def on_select_ts(index, species, day):
    if index:
        species = df.species.cat.categories[index].tolist()
    return timeseries(species, day)

def on_select_table(index, species, day):
    if index:
        species = df.species.cat.categories[index].tolist()
    return temp_table(species, day)

def set_baselayer(value, day):
    if value == 'tiles':
        return hv.Overlay([tiles])
    if value == 'temp':
        return hv.Overlay([tiles, grouped_air[day], gv.feature.coastline]).options(tools=[])

bird_map = birds.clone(streams=[day_stream])
map_selected_stream = Selection1D(source=bird_map)

ts = hv.DynamicMap(on_select_ts, streams=[map_selected_stream, 
                                          species_stream,
                                          day_stream])

baselayer = hv.DynamicMap(set_baselayer, streams=[baselayer_selector_stream, day_stream])
table = hv.DynamicMap(on_select_table, streams=[map_selected_stream, 
                                                species_stream,
                                                day_stream])

dashboard = pn.Column(
    pn.Row(
        pn.Row(baselayer * bird_map)[0][0], 
        pn.Spacer(width=50),
        pn.Column(
            '**Day of Year**', day, 
            '**Species**:',
             'this selector only applies no birds are selected on map', species, 
            '**Baselayers**', baselayer_selector, 
            table)
    ),
    pn.Row(ts)
)
dashboard.servable()