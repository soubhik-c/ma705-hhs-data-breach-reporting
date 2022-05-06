import os
from enum import Enum, auto

import colorlover
import dash
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import dash_table as dt
from dash import dcc as dcc
from dash import html as html
from dash.dependencies import Input, Output
from plotly.colors import n_colors

dt_idx = 4
qtr_fmt = "Q%q %Y"

df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'breach_report_archived.csv'),
                 parse_dates=[dt_idx])

qtrcol_nm = 'Submission Qtr'
submit_year = 'Submission Year'
col_typof_breach_idx = 5
col_loc_breach_idx = 6
col_cov_ent_type = 2

df.iloc[:, [col_typof_breach_idx, col_loc_breach_idx]].fillna(' ')
df['id'] = df.index
df.set_index('id')
df[qtrcol_nm] = df.iloc[:, dt_idx].dt.to_period("Q")
df[submit_year] = df.iloc[:, dt_idx].dt.strftime("%Y")
df = df.sort_values(by=qtrcol_nm, ascending=True)
df[qtrcol_nm] = df[qtrcol_nm].dt.strftime(qtr_fmt)

uniq_submit_dates = pd.Series(df[qtrcol_nm].unique())
uniq_typeof_breach = df.iloc[:, col_typof_breach_idx].str.split(',|/').explode().dropna().unique()
uniq_loc_breach_idx = df.iloc[:, col_loc_breach_idx].str.split(',|/').explode().dropna().unique()

colMap = list(enumerate(filter(lambda _c: _c not in ('id', qtrcol_nm, submit_year), df.columns)))

# dont' sort tab_df, otherwise filter logic will break!!
tab_df = df.copy()
tab_df.dropna(inplace=True)
tab_df[colMap[dt_idx][1]] = tab_df.iloc[:, dt_idx].dt.strftime("%b %d, %y")
tool_tip_df = df.iloc[:, [0, 8]]
q_bins = 10
qcut_col = "Decile Cut"
tab_df[qcut_col], labs = pd.qcut(tab_df.iloc[:, 3], q=q_bins, labels=False, retbins=True)
tab_df.set_index('id')

drop_enable = [8]

tab_cols_fmt = []
colMap_columns = list(map(lambda cm: cm[1], colMap))
for i, c in enumerate([_c for _c in tab_df.columns if _c in colMap_columns]):
    base = dict(id=c, name=c)
    # enable filters
    if i in (0, 3, 8):
        base['filter_options'] = dict(filter_action='native')

    # enable deletable
    if i in (8,):
        base['deletable'] = True
        base['hideable'] = True

    if i in (1,):
        base['presentation'] = 'dropdown'

    tab_cols_fmt.append(base)

filter_col_obj_ids = {}


class ControlType(Enum):
    DropDown = auto()
    DatePickerRange = auto()
    RangeSlider = auto()

    @classmethod
    def get_id(cls, control_name, col_info, prefix=""):
        _, col_name = col_info
        return prefix + control_name.lower() + "_" + col_name.replace(" ", "_").lower()

    @classmethod
    def wrap(cls, o, label, **kwargs):
        return html.Div([
            html.Label(f"{str(label).strip()}", title=label, style=label_style)
            , o
        ],
            style=kwargs
        )

    def get_control_obj(self, col_info, **kwargs):
        ct = self.value
        col_id, col_name = col_info
        _id = ControlType.get_id(self.name, col_info)
        col = col_id
        control_obj = None
        if col_id == col_loc_breach_idx:
            dd_opt = uniq_loc_breach_idx
        elif col_id == col_typof_breach_idx:
            dd_opt = uniq_typeof_breach
        else:
            dd_opt = df.iloc[:, col].fillna('').unique()
        if ct == self.DropDown.value:
            control_obj = dcc.Dropdown(
                id=_id,
                options={
                    str(b).strip(): str(b).strip()
                    for b in sorted(dd_opt)
                },
                # value=df.iloc[0, col],
                clearable=True,
                multi=True,
            )
            filter_col_obj_ids[_id] = "value"
        elif ct == self.RangeSlider.value:
            control_obj = dcc.RangeSlider(id=_id,
                                          min=0,
                                          max=len(uniq_submit_dates),
                                          value=[0, len(uniq_submit_dates) - 1],
                                          marks={each: {"label": str(date), "style": {"transform": "rotate(45deg)"}}
                                                 for each, date in enumerate(uniq_submit_dates)},
                                          step=None)
        elif ct == self.DatePickerRange.value:
            control_obj = dcc.DatePickerRange(
                id=_id,
                start_date=df.iloc[:, col].min().date(),
                end_date=df.iloc[:, col].max().date(),
                min_date_allowed=df.iloc[:, col].min().date(),
                max_date_allowed=df.iloc[:, col].max().date(),
                number_of_months_shown=3,
                day_size=30,
                display_format='MMM YYYY, Do'
            )
            filter_col_obj_ids[_id] = ("start_date", "end_date")

        if control_obj is not None:
            return self.wrap(control_obj, col_name, **kwargs)

        raise (Exception, "No controls found")


input_1 = (
    (colMap[1], ControlType.DropDown, dict(width="6%"))
    , (colMap[2], ControlType.DropDown, dict(width="16%"))
    , (colMap[dt_idx], ControlType.DatePickerRange, dict(width="21%"))
    , (colMap[5], ControlType.DropDown, dict(width="25%"))
    , (colMap[6], ControlType.DropDown, dict(width="20%"))
    , (colMap[7], ControlType.DropDown, dict(width="14%"))
)

input_2 = ((colMap[dt_idx], ControlType.RangeSlider, dict(width="99%")),)

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

app = dash.Dash(__name__,
                external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css']
                )


def quantile_based_table_highlight():
    col_palette = colorlover.scales[str(12)]['qual']['Set3']

    styles = []
    legend = []
    for cur_q in range(0, q_bins):
        backgroundColor = col_palette[cur_q]
        # color = 'white' if cur_q > q_bins / 2. else 'inherit'
        color = 'inherit'
        lwr_bound = tab_df[tab_df[qcut_col] == cur_q][colMap[3][1]].min()
        styles.append({
            'if': {
                'filter_query': (
                    '{{{qcut_col}}} = {qcut_val}'
                ).format(qcut_col=qcut_col, qcut_val=cur_q),
            },
            'backgroundColor': backgroundColor,
            'color': color
        })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(lwr_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return styles, html.Div(legend, style={'padding': '5px 0 5px 0'})


(styles, legend) = quantile_based_table_highlight()

app.layout = html.Div([
    h1
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
    , html.Div(children=[
        html.Div(legend, style={'float': 'right'}, title="Decile based Individuals Effected"),
        dt.DataTable(id='breaches_table',
                     css=[{
                         'selector': '.dash-spreadsheet td div',
                         'rule': '''
                                max-height: 30px; min-height: 30px; height: 30px;
                                max-width: 350px; overflow: hidden;
                                text-align: left; text-overflow: ellipsis;
                                display: block;
                            '''
                     }],
                     tooltip_data=[
                         {
                             column: {'value': str(value), 'type': 'markdown'}
                             for column, value in row.items()
                         } for row in tool_tip_df.to_dict('records')
                     ],
                     tooltip_duration=None,
                     style_table=dict(height="300px"),
                     style_data={
                         'whiteSpace': 'normal',
                         'lineHeight': '15px'
                     },
                     # style_cell={'textAlign': 'left',
                     #             'overflow': 'hidden',
                     #             'textOverflow': 'ellipsis',
                     #             'lineHeight': '15px',
                     #             'maxWidth': 100
                     #             },
                     filter_action="native",
                     sort_action="native",
                     sort_mode="multi",
                     page_action="native",
                     page_current=0,
                     page_size=25,
                     fixed_rows={'headers': True},
                     # style_table={'height': 400, 'overflowY': 'auto'},
                     columns=tab_cols_fmt,
                     data=tab_df.to_dict('records'),
                     style_data_conditional=styles
                     ),
    ],
        style=dict(height="10%")
        ,title="Decile based Individuals Effected"
    )
    , html.Br()
    , html.Div(id='graphs-placeholder-container')
    ,
    html.H5('-> References',
            style={'fontSize': 30, 'textAlign': 'left', 'color': 'MidnightBlue'}),
    html.Div([
        html.A("> HHS Breach Reported Data.",
               href='https://ocrportal.hhs.gov/ocr/breach/breach_report.jsf'),
        html.Br(),
        html.A("> Graphs",
               href='https://plotly.com/python/categorical-axes/'),
        html.Br(),
        html.A("> Dash Table",
               href="https://dash.plotly.com/datatable/interactivity"),
        html.Br(),
        html.A("> Color.",
               href="https://htmlcolorcodes.com/color-names/"),
        html.Br(),
        html.A("> Plotly Bug Tackled.",
               href="https://github.com/numpy/numpy/issues/21008"),
        html.Br(),
        html.A("> Hacking Statistics",
               href="https://review42.com/resources/hacking-statistics/")
    ])

])

submit_dt_picker_id = ControlType.get_id(ControlType.DatePickerRange.name,
                                         colMap[dt_idx])

submit_rg_slider_id = ControlType.get_id(ControlType.RangeSlider.name,
                                         colMap[dt_idx])


def get_all_filters():
    a1 = [Input(fid, "value")
          for fid in filter_col_obj_ids
          if isinstance(filter_col_obj_ids[fid], str)
          ]
    a1.append(Input(submit_dt_picker_id, "start_date"))
    a1.append(Input(submit_dt_picker_id, "end_date"))
    return a1


def get_col_from_filter_col_order(idx):
    if idx == 0:  # State
        return 1
    elif idx == 1:  # Covered Entity Type
        return col_cov_ent_type
    elif idx == 2:  # Type of breach
        return col_typof_breach_idx
    elif idx == 3:  # loc of breach
        return col_loc_breach_idx
    elif idx == 4:  # business assoc present
        return 7
    elif idx in (5, 6):  # submission date
        return dt_idx
    else:
        return -100


@app.callback(
    Output("breaches_table", "data"),
    get_all_filters()
)
def update_breaches_table(*args):
    # print(args)
    start_dt = args[5]
    end_dt = args[6]
    # print(f"start {start_dt} to {end_dt}")

    # first apply a date which is always not-null.
    filtered_data = tab_df.loc[
        (df.iloc[:, dt_idx] >= start_dt)
        & (df.iloc[:, dt_idx] < end_dt)
        ]
    # print(len(filtered_data.iloc[:, 0]))

    for idx, _a in enumerate(args):
        if idx >= 5:
            continue

        col_info = colMap[get_col_from_filter_col_order(idx)]
        cidx, col_nm = col_info

        if cidx in (col_typof_breach_idx, col_loc_breach_idx) and _a is not None:
            # print(f"{col_nm} >- {_a}")
            regex = ' | '.join(_va
                               for _va in _a if len(_va.strip()) > 0)
            if len(regex.strip()) > 0:
                filtered_data = filtered_data.loc[
                    filtered_data.iloc[:, cidx].str.contains(regex.strip(), regex=True)
                ]

        elif cidx < 5 and _a is not None:
            # print(f"{col_nm} <- {_a}")
            q_str = ' | '.join(f'`{col_nm}` == {repr(_va)}'
                               for _va in _a if len(_va.strip()) > 0)
            if len(q_str.strip()) > 0:
                filtered_data = filtered_data.query(q_str)
    # print(len(filtered_data.iloc[:, 0]))
    # print(filtered_data.columns)
    filtered_data.set_index('id')
    return filtered_data.to_dict('records')


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
        return uniq_submit_dates[uniq_submit_dates == search].index

    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")
    trig_ctrl = trigger[0]
    trig_field = trigger[1]
    lwr_v, upr_v = start_date, end_date
    lwr_r, upr_r = range_slider_value

    if trig_ctrl == submit_rg_slider_id:
        lwr_limit = uniq_submit_dates[lwr_r]
        df_qtr_str = df[qtrcol_nm]  # .dt.strftime(qtr_fmt)
        lwr_v = df[df_qtr_str == lwr_limit].iloc[:, dt_idx].min().strftime("%Y-%m-%d")
        upr_limit = uniq_submit_dates[upr_r]
        upr_v = df[df_qtr_str == upr_limit].iloc[:, dt_idx].max().strftime("%Y-%m-%d")
    elif trig_field == 'start_date':
        assert trig_ctrl == submit_dt_picker_id, f"{trig_ctrl} != {submit_dt_picker_id}"
        lwr_r = get_index(start_date).min()

    elif trig_field == 'end_date':
        assert trig_ctrl == submit_dt_picker_id, f"{trig_ctrl} != {submit_dt_picker_id}"
        upr_r = get_index(end_date).max()

    return lwr_v, upr_v, (lwr_r, upr_r)


def create_heat_map(_df):
    return go.Figure(data=go.Heatmap(
        z=_df[4],
        x=_df[dt_idx],
        y=_df[1],
        colorscale='Viridis'))


def create_grp_bar_charts(_df):
    gc = colMap[1][1]
    ac = colMap[3][1]
    yv = pd.DataFrame(
        _df.groupby([gc, qtrcol_nm]) \
            .agg({ac: ['mean', 'std']}) \
            .xs(ac, axis=1, drop_level=True)
    )
    yv = pd.DataFrame(yv.reset_index([gc, qtrcol_nm]))
    yv = yv \
        .round(2).rename(columns={"mean": ac})
    # yv[qtrcol_nm] = yv.loc[:, qtrcol_nm].dt.strftime(qtr_fmt)
    # yv[ac] = yv["mean"]
    return ControlType.wrap(dcc.Graph(
        id="x1",
        figure=px.bar(
            yv,
            x=gc,
            y=ac,
            log_y=True,
            error_y=yv["std"],
            # text=yv["mean"].map('{:,.0f}K'.format),
            color=yv[qtrcol_nm],
            # barmode="group",
            # facet_col="sex",
            color_continuous_scale=px.colors.sequential.algae,
            # color_continuous_midpoint=yv["mean"].mean(),
            orientation="v",
            title="Average Individuals Effected Per State",
        )
            .update_layout(paper_bgcolor="#AFFAFF")
            .update_xaxes(categoryorder='category ascending')
    ),
        "", width="100%", display='inline-block')


def create_annual_violin_plot(_df):
    cov_ent_nm = colMap[col_cov_ent_type][1]
    ac = colMap[3][1]

    yr_df = _df.copy()
    fig = go.Figure()

    uniq_cov_ent = pd.unique(yr_df[cov_ent_nm])
    colors = ["orange", "blue", "green", "yellow"]
    uniq_yrs = pd.unique(yr_df[submit_year])
    for _i, uce in enumerate(uniq_cov_ent):
        # print(yr_df[cov_ent_nm] == uce)
        fig.add_trace(go.Violin(x=yr_df[submit_year][yr_df[cov_ent_nm] == uce],
                                y=np.log(yr_df[ac][yr_df[cov_ent_nm] == uce]),
                                legendgroup=repr(uce), scalegroup=repr(uce), name=repr(uce)
                                )
                      )

    fig.update_traces(box_visible=True, meanline_visible=True)
    fig.update_layout(violinmode='group').update_layout(paper_bgcolor="#AFFAFF")

    return ControlType.wrap(dcc.Graph(
        id="x2",
        figure=fig),
        "", width="100%", display='inline-block')


def discard_create_annual_violin_plot(_df):
    ac = colMap[3][1]

    yr_df = _df.copy()
    yr = yr_df.groupby(['State', submit_year])[ac].sum().unstack(fill_value=0)
    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 12, colortype='rgb')

    fig = go.Figure()
    for data_line, color in zip(yr, colors):
        fig.add_trace(go.Violin(x=yr[data_line], line_color=color))

    fig.update_traces(orientation='h', side='positive', width=3, points=False)
    fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

    return ControlType.wrap(dcc.Graph(
        id="x2_1",
        figure=fig),
        "", width="100%", display='inline-block')


def create_pie_chart_1(_df):
    ac = colMap[3][1]

    yr_df = _df.copy()
    yr_df = yr_df.assign(
        loc_of_breach=yr_df.iloc[:, col_loc_breach_idx].str.split('\\/|,') \
            .fillna('').map(lambda x: [aa.strip() for aa in x])
    ).explode('loc_of_breach')

    yr_df = yr_df.groupby(['loc_of_breach'])[ac].mean()

    # issue #835. Will fail in debugger, see Bugs in the references section here.
    # yr_df = yr_df.unstack(level=1, fill_value=0)
    yr_df = pd.DataFrame(yr_df.reset_index(['loc_of_breach']))

    yr_df = yr_df.round(2).rename(columns={"Unknown": ac})
    yr_df[ac] = np.log(yr_df[ac])

    # fig = px.violin(yr_df, x="explode_loc_breach", y=ac, color="explode_typof_breach")
    fig = px.pie(yr_df, values=ac, names='loc_of_breach',
                 labels={
                     'loc_of_breach': 'Location Of Breach'
                 },
                 title="Average Individual effected Per Location of Breach")
    fig.update_layout(paper_bgcolor="#AFFAFF") \
        .update_traces(textposition='inside', textinfo='percent+label')

    return ControlType.wrap(dcc.Graph(
        id="x3",
        figure=fig),
        "", width="50%", display='inline-block')


def create_pie_chart_2(_df):
    ac = colMap[3][1]

    yr_df = _df.copy()

    yr_df = yr_df.assign(
        type_of_breach=yr_df.iloc[:, col_typof_breach_idx].str.split('\\/|,') \
            .fillna('').map(lambda x: [aa.strip() for aa in x])
    ).explode('type_of_breach')

    yr_df = yr_df.groupby(['type_of_breach'])[ac].mean()

    yr_df = pd.DataFrame(yr_df.reset_index(['type_of_breach']))

    yr_df = yr_df.round(2).rename(columns={"Unknown": ac})
    yr_df[ac] = np.log(yr_df[ac])

    # fig = px.violin(yr_df, x="explode_loc_breach", y=ac, color="explode_typof_breach")
    fig = px.pie(yr_df, values=ac, names='type_of_breach',
                 # hover_data = ['type_of_breach'], labels = {
                 labels={
                     'type_of_breach': 'Type Of Breach'
                 },
                 title="Average Individual effected Per Type of Breach")
    fig.update_layout(paper_bgcolor="#AFFAFF") \
        .update_traces(textposition='inside', textinfo='percent+label')

    return ControlType.wrap(dcc.Graph(
        id="x4",
        figure=fig),
        "", width="50%", display='inline-block')


@app.callback(
    Output('graphs-placeholder-container', 'children'),
    Input('breaches_table', 'derived_virtual_row_ids'),
    Input('breaches_table', 'selected_row_ids'),
    Input('breaches_table', 'active_cell'))
def update_graphs(row_ids, selected_row_ids, active_cell):
    selected = set(selected_row_ids or [])
    if row_ids is not None:
        dff = tab_df.copy(deep=True).loc[row_ids]
    else:
        dff = tab_df.copy(deep=True)

    cur_row = active_cell['row_id'] if active_cell else None

    return [
        create_pie_chart_1(dff),
        create_pie_chart_2(dff),
        create_grp_bar_charts(dff),
        # create_annual_violin_plot(dff),
        create_annual_violin_plot(dff)
    ]


if __name__ == '__main__':
    app.run_server(debug=True)