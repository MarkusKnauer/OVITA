
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import json
from src.util import convert_to_vector
import copy

# load the trajectory for nth user
user_name = "24"
st.set_page_config(layout="wide")

if "response" not in st.session_state:
    st.session_state.response = {}
if "visualized_trajectories" not in st.session_state:   
    st.session_state.visualized_trajectories = []
if "current_traj" not in st.session_state:
    st.session_state.current_traj=0
if "first" not in st.session_state:
    st.session_state.first=True
if "data" not in st.session_state:
    with open("user_study_refined_1.json","r") as file: 
        data=json.loads(file.read())[user_name]
    st.session_state.data=copy.deepcopy(data)
    st.session_state.paths=[]
    for i in range(len(data)):
        st.session_state.paths.append(data[i])
    

def save_response():
    # Response is a list of dicts
    with open(f"results_user_study/{user_name}_response.json", "w") as file: 
        file.write(json.dumps(st.session_state.response))


def get_trajectory(option):
    """ 
    Generate a 3D trajectory based on the selected option.
    The option influences the pattern and complexity of the trajectory.
    """
    try:
        file_path = st.session_state.paths[int(option)]
        if not file_path:
            raise KeyError(f"The 'path' key is missing for option '{option}'.")
        # Read the file
        with open(file_path, "r") as infile:
            data_point = json.loads(infile.read())
        # Extract zero-shot and final details
        LLM=data_point.get("LLM")
        zero_shot_details = data_point.get('zero_shot_trajectory', {})
        final_details = data_point.get('final_trajectory', {})

        # Extract individual elements with default fallbacks
        objects = zero_shot_details.get('objects', [])
        instruction = zero_shot_details.get('instruction', "No instruction available.")
        if final_details.get('global feedbacks',None) is not None:
            global_feedbacks = "ORIGINAL: ".join(
                item + " " for item in final_details.get('global feedbacks', [])
            )
        else:
            global_feedbacks=None

        if final_details.get('local feedbacks',None) is not None:
            local_feedbacks = "CURRENT: ".join(
                item + " " for item in final_details.get('local feedbacks', [])
            )
        else:
            local_feedbacks=None

        original_traj = convert_to_vector(zero_shot_details.get('trajectory', "No trajectory available."))
        zero_shot_traj = convert_to_vector(zero_shot_details.get('modified trajectory', "No modified trajectory available."))
        final_traj = final_details.get('modified trajectory', None)
        final_traj=convert_to_vector(final_traj) if final_traj else None
        zero_interpret = (
            str(zero_shot_details.get('high_level_plan', "No high-level plan.")) + "\n" +
            str(zero_shot_details.get('interpretation', "No code interpretation."))
        )
        final_interpret = (
            str(final_details.get('high_level_plan', "No high-level plan for final traj")) + "\n" +
            str(final_details.get('interpretation', "No code interpretation for final traj"))
        )
        # import ipdb; ipdb.set_trace()
        # Return the processed data
        return (
            (original_traj, zero_shot_traj, final_traj, objects),
            (instruction, global_feedbacks, local_feedbacks, zero_interpret, final_interpret,LLM)
        )

    except KeyError as e:
        st.error(f"KeyError: {e}")
        return None, None
    except FileNotFoundError as e:
        st.error(f"FileNotFoundError: {e}")
        return None, None
    except json.JSONDecodeError as e:
        st.error(f"JSONDecodeError: {e}")
        return None, None


def compare_trajectory(original_trajectory, modified_trajectory, title, points=None, elev=30, azim=45):
    """
    Helper function to visualize the trajectory using Plotly.
    Points represent critical objects in the environment.
    Red represents modified trajectory, and blue represents the original trajectory.
    """
    # Extract points and velocities
    x1, y1, z1, vel1 = map(list, zip(*original_trajectory))
    x2, y2, z2, vel2 = map(list, zip(*modified_trajectory))

    # Create 3D trajectory plot
    fig = go.Figure()

    # Original trajectory
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='lines',
        marker=dict(size=5, color='blue'),
        line=dict(color='blue'),
        name='Original Trajectory'
    ))

    # Modified trajectory
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='lines',
        marker=dict(size=5, color='red'),
        line=dict(color='red'),
        name='Modified Trajectory'
    ))

    # Start and end markers for original trajectory
    fig.add_trace(go.Scatter3d(
        x=[x1[0]], y=[y1[0]], z=[z1[0]],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=['Original Start'],
        textposition="top center",
        name='Original Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=[x1[-1]], y=[y1[-1]], z=[z1[-1]],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=['Original End'],
        textposition="top center",
        name='Original End'
    ))

    # Start and end markers for modified trajectory
    fig.add_trace(go.Scatter3d(
        x=[x2[0]], y=[y2[0]], z=[z2[0]],
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text=['Modified Start'],
        textposition="top center",
        name='Modified Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=[x2[-1]], y=[y2[-1]], z=[z2[-1]],
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text=['Modified End'],
        textposition="top center",
        name='Modified End'
    ))

    # Plot the objects in the environment
    if points is not None:
        for obj in points:
            name = obj["name"]
            x = obj["x"]
            y = obj["y"]
            z = obj["z"]
            obj_length = obj['dimensions'][0]
            obj_width = obj['dimensions'][1]
            obj_height = obj['dimensions'][2]

            # Cuboid vertices
            vertices = [
                [x - obj_length / 2, y - obj_width / 2, z - obj_height / 2],
                [x + obj_length / 2, y - obj_width / 2, z - obj_height / 2],
                [x + obj_length / 2, y + obj_width / 2, z - obj_height / 2],
                [x - obj_length / 2, y + obj_width / 2, z - obj_height / 2],
                [x - obj_length / 2, y - obj_width / 2, z + obj_height / 2],
                [x + obj_length / 2, y - obj_width / 2, z + obj_height / 2],
                [x + obj_length / 2, y + obj_width / 2, z + obj_height / 2],
                [x - obj_length / 2, y + obj_width / 2, z + obj_height / 2],
            ]

            edges = [
                        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
                        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
                        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical edges
                    ]

            # Create the lines for the edges
            x_lines, y_lines, z_lines = [], [], []
            for edge in edges:
                start, end = edge
                x_lines.extend([vertices[start][0], vertices[end][0], None])  # None separates segments
                y_lines.extend([vertices[start][1], vertices[end][1], None])
                z_lines.extend([vertices[start][2], vertices[end][2], None])

            fig.add_trace(go.Scatter3d(
                            x=x_lines,
                            y=y_lines,
                            z=z_lines,
                            mode='lines',
                            line=dict(color='blue', width=4),
                        ))

            # Add a scatter marker for the object's center
            fig.add_trace(go.Scatter3d(
                x=[x], y=[y], z=[z + obj_height / 2],
                mode='lines+text',
                marker=dict(size=5, color='grey'),
                text=[name],
                textposition="top center",
                name=name
            ))

    # Update layout for the 3D plot
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
        ),
        title=title,
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False
    )

    # Create 2D velocity profile plot
    velocity_fig = go.Figure()

    # Original velocity profile
    velocity_fig.add_trace(go.Scatter(
        x=list(range(len(vel1))),
        y=vel1,
        mode='lines',
        line=dict(color='blue'),
        name='Original Velocity'
    ))

    # Modified velocity profile
    velocity_fig.add_trace(go.Scatter(
        x=list(range(len(vel2))),
        y=vel2,
        mode='lines',
        line=dict(color='red'),
        name='Modified Velocity'
    ))

    # Update layout for the 2D plot
    velocity_fig.update_layout(
        title="Velocity Profile",
        xaxis_title="Position Index",
        yaxis_title="Velocity",
        margin=dict(l=0, r=0, t=50, b=0),
        showlegend=False
    )

    return fig, velocity_fig



trajectory_options = list(range(len(st.session_state.paths)))
with st.sidebar:
    st.header("Options")
    col1,col2=st.columns(2)
    with col1:
        next_button=st.button("Next",use_container_width=True)
        visualized_button = st.button("Mark Done",use_container_width=True)

    with col2:
        previous_button=st.button("Previous",use_container_width=True)
        save_button = st.button("Save",use_container_width=True)

# Remaining trajectories
selected_trajectory=st.session_state.current_traj
(original_traj, zero_shot_traj, final_traj, objects),(instruction, global_feedbacks, local_feedbacks, zero_interpret, final_interpret,LLM)=get_trajectory(selected_trajectory)
# Getting the data

visualized_trajectories = st.session_state.visualized_trajectories
remaining_trajectories = [t for t in trajectory_options if t not in visualized_trajectories]

# Select a trajectory to visualize
st.markdown(f"<p style='font-size:18px; font-weight:bold;'>Instructions for rating</p>",unsafe_allow_html=True)
st.write("Positive change in X value is right, Negative change in X value is left. Positive change in Y value is front, Negative change in Y value is back. Positive change in Z value is up, Negative change in Z value is down.")
st.write("User rating: 1: Completely wrong, 2: somewhat wrong 3: neutral 4: somewhat correct 5: completely correct")
col1, col2 = st.columns(2)
if selected_trajectory is not None:
    with col1:
        st.markdown(f"<p style='font-size:18px; font-weight:bold;'>Trajectory {selected_trajectory+1}</p>",unsafe_allow_html=True)  
        refresh_button=st.button("Refresh")
    with col2: 
        st.markdown(
        f"<p style='font-size:18px;'><span style='font-weight:bold; font-size:20px;'>INSTRUCTION:</span> {instruction}</p>",
        unsafe_allow_html=True
        )


if selected_trajectory is not None :
    col1, col2 = st.columns(2)
    fig1,vel1=compare_trajectory(original_traj,zero_shot_traj,"zero shot",objects)
    if final_traj is not None: 
        fig2,vel2=compare_trajectory(original_traj,final_traj,"After feedbacks",objects)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(vel1, use_container_width=True,key='zero shot speed profile')

    st.text_area("Zero-shot code explanation", zero_interpret, height=200, disabled=True)
    rating_zero_shot = st.slider(
        "Rating (1-5) zero shot:",
        min_value=1, 
        max_value=5, 
        value=3,  # Default value
        step=1
    )
        
     
    if final_traj is not None:
        temp=global_feedbacks if global_feedbacks is not None else "N/A"
        st.markdown(
        f"<p style='font-size:18px;'><span style='font-weight:bold; font-size:20px;'>GLOBAL FEEDBACKS:</span> {temp}</p>",
        unsafe_allow_html=True
        )
        temp=local_feedbacks if local_feedbacks is not None else "N/A"
        st.markdown(
        f"<p style='font-size:18px;'><span style='font-weight:bold; font-size:20px;'>LOCAL FEEDBACKS:</span> {temp}</p>",
        unsafe_allow_html=True
        )
        col1,col2=st.columns(2) 
        with col1:
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.plotly_chart(vel2, use_container_width=True,key='final speed profile')
            
        st.text_area("Final code explanation", final_interpret, height=200, disabled=True)
        rating_final = st.slider(
            "Rating (1-5) final:",
            min_value=1, 
            max_value=5, 
            value=3,  # Default value
            step=1
        )

if final_traj is None:
    st.markdown("<p style='font-size:18px; font-weight:bold;'>No feedbacks were needed for this experiment</p>", unsafe_allow_html=True)


rating_code_interpret = st.slider(
    "Rating (1-5) interpretability:",
    min_value=1, 
    max_value=5, 
    value=3,  # Default value
    step=1
)

if visualized_button:
    if selected_trajectory is None:
        st.sidebar.warning("All the trajectories have been done")
    else:
        if selected_trajectory not in st.session_state.visualized_trajectories:
            st.session_state.visualized_trajectories.append(selected_trajectory)
        visualized_trajectories = st.session_state.visualized_trajectories
        remaining_trajectories = [t for t in trajectory_options if t not in visualized_trajectories]
        response_dict={
            "path": st.session_state.paths[selected_trajectory], "rating_zero_shot": rating_zero_shot, "rating_final": rating_final if final_traj is not None else rating_zero_shot,"interpret_score":rating_code_interpret, "LLM": LLM
        }
        st.session_state.response.update({selected_trajectory:response_dict})
        st.sidebar.success(f"Trajectory {selected_trajectory+1} marked as visualized!")

if next_button:
    if st.session_state.current_traj<len(trajectory_options)-1:   
        st.session_state.current_traj+=1
    else:
        st.sidebar.warning("You have reached the end of the evaluation")
if previous_button:
    if st.session_state.current_traj>0:
        st.session_state.current_traj-=1
    else:
        st.sidebar.warning("You are at the start of the evaluation")

st.sidebar.subheader("Remaining Trajectories")
if len(remaining_trajectories)!=0:
    remaining_trajectories.sort()
st.sidebar.write("".join(str(item+1) + " " for item in remaining_trajectories))


if len(visualized_trajectories)!=0:
    visualized_trajectories.sort()
st.sidebar.subheader("Visualized Trajectories")
st.sidebar.write("".join(str(item+1) + " " for item in visualized_trajectories))
with st.sidebar:
    reset_button = st.button("Reset all the responses",use_container_width=True)

# Save functionality
if save_button:
    # Save both figures as HTML files
    save_response()
    st.sidebar.success("Responses saved")   

if reset_button:
    st.session_state.response = {}
    st.session_state.visualized_trajectories = []
    st.session_state.current_traj=0
    st.rerun()