from enum import Enum, auto

import dash
import pandas as pd
from dash import dcc as dcc
from dash import html as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

dt_idx = 4
qtr_fmt = "Q%q %Y"

df = pd.read_csv('/Users/soubhik1c/Downloads/MA705/personal-project/breach_report_archived.csv', parse_dates=[dt_idx])

colMap = list(enumerate(df.columns))
df['submit_qtr'] = df.iloc[:, dt_idx].dt.to_period("Q").dt.strftime(qtr_fmt)
submit_dates = pd.Series(df.iloc[:, dt_idx].dt.to_period("Q").sort_values(ascending=True).
                         dt.strftime(qtr_fmt).unique())


class ControlType(Enum):
    DropDown = auto()
    DatePickerRange = auto()
    RangeSlider = auto()

    @classmethod
    def get_id(cls, control_name, col_info, prefix=""):
        _, col_name = col_info
        return prefix + control_name.lower() + "_" + col_name.replace(" ", "_").lower()

    def wrap(self, o, label, **kwargs):
        return html.Div([
            html.Label(f"{str(label).strip()}", title=label, style=label_style)
            , o
        ],
            style=dict(width=kwargs["width"])
        )

    def get_control_obj(self, col_info, **kwargs):
        ct = self.value
        col_id, col_name = col_info
        _id = ControlType.get_id(self.name, col_info)
        col = col_id
        control_obj = None
        if ct == self.DropDown.value:
            control_obj = dcc.Dropdown(
                id=_id,
                options={
                    str(b).strip(): str(b).strip()
                    for b in sorted(df.iloc[:, col].fillna('').unique())
                },
                value=df.iloc[0, col],
                clearable=False
            )
        elif ct == self.RangeSlider.value:
            control_obj = dcc.RangeSlider(id=_id,
                                          min=0,
                                          max=len(submit_dates),
                                          value=[0, len(submit_dates) - 1],
                                          marks={each: {"label": str(date), "style": {"transform": "rotate(45deg)"}}
                                                 for each, date in enumerate(submit_dates)},
                                          step=None)
        elif ct == self.DatePickerRange.value:
            control_obj = dcc.DatePickerRange(
                id=_id,
                start_date=df.iloc[:, col].min().date(),
                end_date=df.iloc[:, col].max().date(),
                min_date_allowed=df.iloc[:, col].min().date(),
                max_date_allowed=df.iloc[:, col].max().date(),
                display_format='MMM YYYY, Do'
            )

        if control_obj is not None:
            return self.wrap(control_obj, col_name, **kwargs)

        raise (Exception, "No controls found")


input_1 = (
    (colMap[1], ControlType.DropDown, dict(width="6%"))
    , (colMap[2], ControlType.DropDown, dict(width="16%"))
    , (colMap[dt_idx], ControlType.DatePickerRange, dict(width="21%"))
    , (colMap[5], ControlType.DropDown, dict(width="16%"))
    , (colMap[6], ControlType.DropDown, dict(width="20%"))
    , (colMap[7], ControlType.DropDown, dict(width="14%"))
)

input_2 = ((colMap[dt_idx], ControlType.RangeSlider, dict(width="100%")),)

# df.iloc[:, 4] = pd.to_datetime(df.iloc[:, 4])

h1_style = {
    'fontSize': 50
    , 'textAlign': 'center'
    , 'color': 'white'
    , 'background-color': 'MidnightBlue'
}

h2_style = {
    'fontSize': 30
    , 'textAlign': 'left'
    , 'color': 'MidnightBlue'
}

label_style = {
    'fontSize': 15
    , 'textAlign': 'left'
    , 'color': 'MidnightBlue'
}

h1 = html.H1('Breach of Protected Health Information', style=h1_style)
h2 = html.H2('U.S. Department of Health and Human Services Office for Civil Rights:'
             , style=h2_style)

intro = dcc.Markdown(
    '''
        Breach Portal: Notice to the Secretary of HHS Breach of Unsecured Protected Health Information
        
        All healthcare data breaches in the U.S. up to 9/14/2021 from the U.S. Dept. of HHS Office 
        for Civil Rights portal.
    ''')
#
# set1 = {
#     # "Covered Entity": (0, ControlType.DropDown, "35%", "Name of Covered Entity")
#     "State": (1, ControlType.DropDown, "6%", "State")
#     , "Entity Type": (2, ControlType.DropDown, "16%", "Covered Entity Type")
#     , "Submission Date": (4, ControlType.DatePickerRange, "21%", "Breach Submission Date")
#     # , "Affected": (3, ControlType.DropDown, "10%", "Individuals Affected")
#     , "Breach Type": (5, ControlType.DropDown, "16%", "Type of Breach")
#     , "Location": (6, ControlType.DropDown, "11%", "Location Breached")
#     , "Assoc. Present": (7, ControlType.DropDown, "11%", "Business Associate Present")
# }
#
# set2 = {
#     "Submission Range": (4, ControlType.RangeSlider, "100%", "Breach Submission Date")
# }

app.layout = html.Div([
    h1
    # , html.Div([html.P("-" * 80)])
    , h2
    , intro
    , html.Div(children=[
        ct.get_control_obj(col, **kwargs)
        for col, ct, kwargs in input_1
    ],
        style=dict(display='flex')
    )
    , html.Br()
    , html.Div(children=[
        ct.get_control_obj(col, **kwargs)
        for col, ct, kwargs in input_2
    ],
        style=dict(display='flex')
    )
    , html.Br()
    ,

])


submit_dt_picker_id = ControlType.get_id(ControlType.DatePickerRange.name,
                                         colMap[dt_idx])

submit_rg_slider_id = ControlType.get_id(ControlType.RangeSlider.name,
                                         colMap[dt_idx])


@app.callback(
    Output(submit_dt_picker_id, 'start_date'),
    Output(submit_dt_picker_id, 'end_date'),
    Output(submit_rg_slider_id, 'value'),
    Input(submit_dt_picker_id, 'start_date'),
    Input(submit_dt_picker_id, 'end_date'),
    Input(submit_rg_slider_id, 'value'),
    prevent_initial_call=True
)
def update_date_picker_range_submit_date_end(start_date, end_date, range_slider_value):
    def get_index(x):
        search = pd.to_datetime(x).to_period("Q").strftime(qtr_fmt)
        return submit_dates[submit_dates == search].index

    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")
    trig_ctrl = trigger[0]
    trig_field = trigger[1]
    lwr_v, upr_v = start_date, end_date
    lwr_r, upr_r = range_slider_value

    if trig_ctrl == submit_rg_slider_id:
        lwr_limit = submit_dates[lwr_r]
        lwr_v = df[df.submit_qtr == lwr_limit].iloc[:, dt_idx].min().strftime("%Y-%m-%d")
        upr_limit = submit_dates[upr_r]
        upr_v = df[df.submit_qtr == upr_limit].iloc[:, dt_idx].max().strftime("%Y-%m-%d")
    elif trig_field == 'start_date':
        assert trig_ctrl == submit_dt_picker_id, f"{trig_ctrl} != {submit_dt_picker_id}"
        lwr_r = get_index(start_date).min()

    elif trig_field == 'end_date':
        assert trig_ctrl == submit_dt_picker_id, f"{trig_ctrl} != {submit_dt_picker_id}"
        upr_r = get_index(end_date).max()

    return lwr_v, upr_v, (lwr_r, upr_r)


if __name__ == '__main__':
    app.run_server(debug=True)