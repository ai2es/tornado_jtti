"""HWT mode  dashboard."""

import numpy as np
import pandas as pd
from datetime import date, datetime

from bokeh.plotting import figure, curdoc
from bokeh.palettes import Set1 as palette
from bokeh.models import HoverTool, BoxZoomTool, CategoricalColorMapper
from bokeh.tile_providers import get_provider, Vendors
from bokeh.models.widgets import Slider, Select
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
#                 fill_color={'field': 'label', 'transform': color_mapper},
#                 fill_alpha='label_prob',
#                 line_alpha=0.0,
                source=data_scr,
#                 legend='label',
                view=data_provider.data_view)

USA_map.add_tools(HoverTool(tooltips=TOOLTIP))

select_valid_time = Select(title="Select valid datetime",
                           value=data_provider.valid_time_menu[-1],
                           options=data_provider.valid_time_menu,
                           name="select_valid_time")

def update():
    """Periodic callback."""
    data_provider.fetch_data()

def update_valid_time(attr, old, new):
    """Update the run date filter."""
    if new != old:
        data_provider.set_valid_time(new)
    else:
        data_provider.set_valid_time(old)    
        
select_valid_time.on_change("value", update_valid_time)

curdoc().add_root(USA_map)
curdoc().add_root(select_valid_time)
