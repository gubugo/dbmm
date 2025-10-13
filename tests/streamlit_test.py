# import streamlit as st
# import plotly.graph_objects as go

# # if 'fig' not in st.session_state:
# st.session_state.fig = go.Figure()
# st.session_state.fig.add_trace(go.Scatter(x=[1, 2, 3], y=[1, 2.435, 1.65765], mode='markers', name='Initial Trace'))

# # Add a new trace based on user interaction
# if st.button("Add Another Trace"):
#     new_y = [y + 1 for y in st.session_state.fig.data[0].y] # Example modification
#     st.session_state.fig.add_trace(go.Scatter(x=[1, 2, 3], y=new_y, mode='markers', name='New Trace'))

# st.plotly_chart(st.session_state.fig)

import plotly.graph_objects as go

# Sample data
x = [1, 2, 3, 4, 5]
y1 = [1, 2, 3, 4, 5]
y2 = [5, 4, 3, 2, 1]

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Trace 1'))
fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Trace 2', visible=False))

fig.update_layout(
    updatemenus=[
        dict(
            type="buttons",
            direction="left",
            buttons=list([
                dict(
                    args=[{"visible": [True, False]}],
                    label="Show Trace 1",
                    method="restyle"
                ),
                dict(
                    args=[{"visible": [False, True]}],
                    label="Show Trace 2",
                    method="restyle"
                )
            ]),
            pad={"r": 10, "t": 10},
            showactive=True,
            x=0.1,
            xanchor="left",
            y=1.1,
            yanchor="top"
        ),
    ]
)

fig.show()