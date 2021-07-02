"""HWT mode  dashboard."""

import numpy as np
import pandas as pd
from datetime import date, datetime

from bokeh.plotting import figure, curdoc
from bokeh.palettes import Set1 as palette
from bokeh.models import HoverTool, BoxZoomTool, CategoricalColorMapper
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models.widgets import Slider, Select, DateRangeSlider
from bokeh.models.annotations import Title

from data import HRRRProvider
import config as cfg

TOOLTIP = """
<div class="plot-tooltip">
    <div>
        <h5>@label</h5>
    </div>
    <div>
        <span style="font-weight: bold;">Probability of Mode: 0
    </div>
</div>
"""

data_provider = HRRRProvider(cfg.DATA_SOURCE, cfg.COLS, cfg.COLS_VIEW)
data_scr = data_provider.data_ds

USA_map = figure(x_range=(-14000000, -7000000),
                 y_range=(3100000, 6300000),
                 x_axis_type="mercator",
                 y_axis_type="mercator",
                 x_axis_location=None,
                 y_axis_location=None,
                 tools=['wheel_zoom', 'box_select', 'lasso_select', 'poly_select', 'tap', 'pan', 'reset', 'save'],
                 match_aspect=True,
                 name="main_plot")

USA_map.add_tile(get_provider(Vendors.CARTODBPOSITRON))
USA_map.add_tools(BoxZoomTool(match_aspect=True))

color_mapper = CategoricalColorMapper(factors=['Supercell', 'QLCS', 'Disorganized'], palette=palette[3])
USA_map.patches("x", "y",
                fill_color='colors',
#                 fill_alpha='label_prob',
                line_alpha=0.0,
                source=data_scr,
#                 legend='label',
                view=data_provider.data_view)

USA_map.add_tools(HoverTool(tooltips=TOOLTIP))

start = pd.Timestamp(data_provider.valid_time_menu[0])
start = datetime(start.year, start.month, start.day, start.hour, start.minute)
end = pd.Timestamp(data_provider.valid_time_menu[-1])
end=datetime(end.year, end.month, end.day, end.hour, end.minute)
value_min = pd.Timestamp(data_provider.valid_time_menu[0])
value_min = datetime(value_min.year, value_min.month, value_min.day, value_min.hour, value_min.minute)
value_max = pd.Timestamp(data_provider.valid_time_menu[0]) + pd.Timedelta("5 min")
value_max = datetime(value_max.year, value_max.month, value_max.day, value_max.hour, value_max.minute)

select_valid_time = DateRangeSlider(start=start,
                                    end=end,
                                    value=(value_min, value_max),
                                    format="%x, %X",
                                    step=300000,
                                    title="Select valid datetime",
                                    name="select_valid_time")

def update():
    """Periodic callback."""
    data_provider.fetch_data()

def update_valid_time(attr, old, new):
    """Update the run date filter."""
    if new[0] != old[0]:
        data_provider.set_valid_time(datetime.fromtimestamp(new[0] / 1000))
    elif new[1] != old[1]:
        data_provider.set_valid_time(datetime.fromtimestamp(new[1] / 1000))        
    else:
        data_provider.set_valid_time(old[0])    
        
select_valid_time.on_change("value", update_valid_time)

curdoc().add_root(USA_map)
curdoc().add_root(select_valid_time)
