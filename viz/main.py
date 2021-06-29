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
        <span style="font-weight: bold;">Probability of Mode: </span>@label_prob{0.3f}
    </div>
</div>
"""

data_provider = HRRRProvider(cfg.DATA_SOURCE, cfg.COLS)
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
                fill_color={'field': 'label', 'transform': color_mapper},
                fill_alpha='label_prob',
                line_alpha=0.0,
                source=data_scr,
#                 legend='label',
                view=data_provider.data_view)

USA_map.add_tools(HoverTool(tooltips=TOOLTIP))

select_model = Select(title="Select model",
                      value="CNN",
                      options=["CNN", "GMM"],
                      name="select_model")

select_run_date = Select(title="Select run date and time",
                         value=data_provider.run_date_menu[-1],
                         options=data_provider.run_date_menu,
                         name="select_run_date")

slider_forecast_hour = Slider(start=1,
                              end=17,
                              value=1,
                              step=1,
                              name="slider_forecast_hour",
                              title="Forecast Hour")

def update():
    """Periodic callback."""
    data_provider.fetch_data()

def update_model(attr, old, new):
    """Set the model displayed in the visualization."""
    if new != old:
        data_provider.set_model(new)
        data_provider.set_run_date(data_provider.run_date)
        data_provider.set_forecast_hour(1)
        data_provider.set_valid_hour()        

def update_run_date(attr, old, new):
    """Update the run date filter."""
    if new != old:
        data_provider.set_run_date(new)
        data_provider.set_forecast_hour(1)
        data_provider.set_valid_hour()
    else:
        data_provider.set_run_date(old)
        data_provider.set_forecast_hour(1)
        data_provider.set_valid_hour()        

def update_forecast_hour(attr, old, new):
    """Update the forecast hour filter."""
    if new != old:
        data_provider.set_forecast_hour(new)
        data_provider.set_valid_hour()
        
select_model.on_change("value", update_model)
select_run_date.on_change("value", update_run_date)
slider_forecast_hour.on_change("value", update_forecast_hour)

curdoc().add_root(USA_map)
curdoc().add_root(select_model)
curdoc().add_root(select_run_date)
curdoc().add_root(slider_forecast_hour)
