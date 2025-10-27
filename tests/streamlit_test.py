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
# import streamlit as st

# # Create two columns with equal width
# col1, col2 = st.columns(2)

# with col1:
#     st.header("Column 1")
#     st.write("This is the content of the first column.")
#     st.button("Button in Column 1")

# with col2:
#     st.header("Column 2")
#     st.write("This is the content of the second column.")
#     st.slider("Slider in Column 2", 0, 100, 50)

#########

# import streamlit as st

# # Default behavior: selecting one deselects others
# st.write("## Default Radio Button")
# selection = st.radio(
#     "Choose one:",
#     ("Option 1", "Option 2", "Option 3")
# )
# st.write(f"You selected: {selection}")

# # Example with a "No selection" option (and hiding it via CSS)
# st.write("## Radio Button with \"No Selection\" (Hidden)")

# # CSS to hide the first radio button element
# st.markdown("""
# <style>
# div[role="radiogroup"] > :first-child {
#     display: none !important;
# }
# </style>
# """, unsafe_allow_html=True)

# options_with_none = ["None", "Option A", "Option B", "Option C"]
# selection_with_none = st.radio(
#     "Select an option:",
#     options_with_none,
#     index=0 # Set the default to the hidden "None" option
# )

# # You'll need to handle the case where the first option ("None") is selected
# if selection_with_none == "None":
#     st.write("No option selected yet.")
# else:
#     st.write(f"You selected: {selection_with_none}")
##############################

# import streamlit as st

# def update_radio1_options():
#     """Callback function to update radio2 options based on radio1 selection."""
#     st.session_state.radio1_value = "Option A1" # Reset selected value
#     st.session_state.radio1_key = "Option A1" # Reset selected value

# def update_radio2_options():
#     """Callback function to update radio2 options based on radio1 selection."""
#     st.session_state.radio2_value = "Option B1" # Reset selected value
#     st.session_state.radio2_key = "Option B1" # Reset selected value


# st.title("Chained Radio Buttons")

# # First radio button, with a callback to modify the second
# st.radio(
#     "Select a category:",
#     options=["Option A1", "Option A2", "Option A3"],
#     key="radio1_key", # Unique key for this widget
#     on_change=update_radio2_options,
# )

# # Update the session state variable for the first radio button after its value is selected
# # st.session_state.radio1_value = st.session_state.radio1_key

# # Second radio button, whose options and value are controlled by the first
# st.radio(
#     "Select an item:",
#     options=["Option B1", "Option B2", "Option B3"],
#     key="radio2_key", # Unique key for this widget
#     on_change=update_radio1_options,
# )

# st.write(f"You selected Category: {st.session_state.radio1_key}")
# st.write(f"You selected Item: {st.session_state.radio2_key}")

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Grid size
n_rows = 9
n_cols = 9

# Create 9x9 subplot grid
fig = make_subplots(
    rows=n_rows,
    cols=n_cols,
    vertical_spacing=0.003,
    horizontal_spacing=0.003
)

img_data = np.random.rand(100, 100, 3)

# Optional: Fill all subplots with a dummy line (or leave empty)
for row in range(1, n_rows + 1):
    for col in range(1, n_cols + 1):
        # Add a basic line in each subplot
        fig.add_trace(
            go.Image(z=255*img_data),
            row=row,
            col=col
        )

# Target subplot (e.g. row 5, col 5)
target_row = 5
target_col = 5

# Calculate paper domain for background shape
x0 = (target_col - 1) / n_cols
x1 = target_col / n_cols
y0 = 1 - (target_row / n_rows)
y1 = 1 - ((target_row - 1) / n_rows)

# Add background color behind the specific subplot
fig.update_layout(
    shapes=[
        dict(
            type="rect",
            xref="paper",
            yref="paper",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor="lightyellow",
            layer="below",
            line_width=0,
        )
    ],
    height=900,
    width=900,
    showlegend=False,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(visible=False),  # Hide x-axis
    yaxis=dict(visible=False),  # Hide y-axis
)

fig.show()