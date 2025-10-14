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
###########################################
# import plotly.graph_objects as go

# # Sample data
# x = [1, 2, 3, 4, 5]
# y1 = [1, 2, 3, 4, 5]
# y2 = [5, 4, 3, 2, 1]

# fig = go.Figure()

# fig.add_trace(go.Scatter(x=x, y=y1, mode='lines', name='Trace 1'))
# fig.add_trace(go.Scatter(x=x, y=y2, mode='lines', name='Trace 2', visible=False))

# fig.update_layout(
#     updatemenus=[
#         dict(
#             type="buttons",
#             direction="left",
#             buttons=list([
#                 dict(
#                     args=[{"visible": [True, False]}],
#                     label="Show Trace 1",
#                     method="restyle"
#                 ),
#                 dict(
#                     args=[{"visible": [False, True]}],
#                     label="Show Trace 2",
#                     method="restyle"
#                 )
#             ]),
#             pad={"r": 10, "t": 10},
#             showactive=True,
#             x=0.1,
#             xanchor="left",
#             y=1.1,
#             yanchor="top"
#         ),
#     ]
# )

# fig.show()
###############################
# import streamlit as st
# import plotly.express as px
# import numpy as np

# # Create some sample image data
# img_data = np.random.rand(100, 100)

# # Create the Plotly imshow figure
# fig = px.imshow(img_data, color_continuous_scale='gray')

# # Display the figure in Streamlit
# st.plotly_chart(fig)
########
# import plotly.graph_objects as go
# import streamlit as st
# from plotly.subplots import make_subplots

# # Create a figure with 1 row and 2 columns
# fig = make_subplots(rows=2, cols=2, subplot_titles=("Chart 1", "Chart 2","Chart 3", "Chart 4"))

# # Add traces to the first subplot
# fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode="lines"), row=1, col=1)

# # Add traces to the second subplot
# fig.add_trace(go.Bar(x=["A", "B", "C"], y=[10, 15, 7]), row=1, col=2)

# # Add traces to the first subplot
# fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode="lines"), row=2, col=2)

# # Add traces to the second subplot
# fig.add_trace(go.Bar(x=["A", "B", "C"], y=[10, 15, 7]), row=2, col=1)

# fig.update_layout(title_text="Multiple Charts in One Figure", width=1000,  height=800)  # Set the height in pixels)
# # fig.show()
# st.set_page_config(layout="wide")
# st.plotly_chart(fig, use_container_width=True)#
########
import streamlit as st

# Create two columns with equal width
col1, col2 = st.columns(2)

with col1:
    st.header("Column 1")
    st.write("This is the content of the first column.")
    st.button("Button in Column 1")

with col2:
    st.header("Column 2")
    st.write("This is the content of the second column.")
    st.slider("Slider in Column 2", 0, 100, 50)