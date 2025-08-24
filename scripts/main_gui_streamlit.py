import streamlit as st
import json
import os
import numpy as np
import plotly.graph_objects as go
from OVITA.constraints import RobotArm7DOF_constraint, Drone_and_husky_constraints, general_constraints
from OVITA.config import Config
from OVITA.util import convert_to_vector, compare_trajectory,convert_from_vector
from OVITA.agent import LLM_Agent
import copy
import tkinter as tk
from tkinter import filedialog
import re 
import time
from OVITA.CSM import *

st.set_page_config(layout="wide")
placeholder = st.empty()
cwd = os.getcwd()
if "original_trajectory" not in st.session_state:
        st.session_state.original_trajectory = []
if "modified_trajectory" not in st.session_state:
    st.session_state.modified_trajectory = []
if 'new_instruction' not in st.session_state:
    st.session_state.new_instruction = ""
if "run_count" not in st.session_state:
    st.session_state.run_count = 0
if 'interpretation_details' not in st.session_state:
    st.session_state.interpretation_details = []
if 'feedback' not in st.session_state:
        st.session_state.feedback = ""
if 'feedback_type' not in st.session_state:
    st.session_state.feedback_type = None
if 'global_feedbacks' not in st.session_state:
    st.session_state.global_feedbacks = []
if 'local_feedbacks' not in st.session_state:
    st.session_state.local_feedbacks = []
if 'zero_shot_trajectory' not in st.session_state:
    st.session_state.zero_shot_trajectory = {"modified trajectory":[]}
if 'final_trajectory' not in st.session_state:
    st.session_state.final_trajectory = {"modified trajectory":[]}
if 'modified_trajectory_list' not in st.session_state:
    st.session_state.modified_trajectory_list = []
if "toggle_state_CSM" not in st.session_state:
    st.session_state.toggle_state_CSM = True
if "toggle_state_start" not in st.session_state:
    st.session_state.toggle_state_start = False
if "toggle_state_end" not in st.session_state:
    st.session_state.toggle_state_end = False
if "toggle_state_smooth" not in st.session_state:
    st.session_state.toggle_state_smooth = True
if 'get_trajectory_param' not in st.session_state:
        st.session_state.get_trajectory_param = True


def load_trajectory(file):
    return json.load(file)
def get_trajectory(sample=True,num_points=100, dict_format=True):
    if st.session_state.get_trajectory_param:
        trajectory = copy.deepcopy(data.get('trajectory', []))
    else: 
        trajectory= copy.deepcopy(convert_to_vector(st.session_state.modified_trajectory)).tolist()
    if not trajectory:
        return []
    if sample:  
        if num_points <= len(trajectory):
            indices = np.linspace(0, len(trajectory) - 1, num_points, dtype=int)
            trajectory = [trajectory[i] for i in indices]
        else:
            trajectory = np.array(trajectory)  # Convert to numpy array for easy manipulation
            original_indices = np.linspace(0, len(trajectory) - 1, len(trajectory))
            new_indices = np.linspace(0, len(trajectory) - 1, num_points)
            interpolated_trajectory = []
            for dim in range(trajectory.shape[1]):  # Iterate over dimensions (x, y, z, velocity)
                interpolated_dim = np.interp(new_indices, original_indices, trajectory[:, dim])
                interpolated_trajectory.append(interpolated_dim)
            trajectory = np.stack(interpolated_trajectory, axis=1).tolist()
    
    if dict_format:
        trajectory=[{'x':point[0],'y':point[1],'z':point[2],'velocity':point[3]} for point in trajectory]
    return trajectory
def detect_objects():
    objs=copy.deepcopy(data['objects'])
    for item in objs:
        item['name']=item['name'].lower()
        if 'dimensions' not in item.keys():
            item.update({'dimensions': [config.DEFAULT_DIMENSION]*3})
    return objs
def plot_trajectory(fig,original, modified, objects,instruction):
    # fig = go.Figure()
    
    # Original Trajectory (Blue)
    fig.add_trace(go.Scatter3d(
        x=[p['x'] for p in original],
        y=[p['y'] for p in original],
        z=[p['z'] for p in original],
        mode='lines',
        line=dict(color='blue'),
        name='Original Trajectory'
    ))
    
    
    # Start and end markers for original trajectory
    fig.add_trace(go.Scatter3d(
        x=[original[0]["x"]], y=[original[0]["y"]], z=[original[0]["z"]],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=['Original Start'],
        textposition="top center",
        name='Original Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=[original[-1]["x"]], y=[original[-1]["y"]], z=[original[-1]["z"]],
        mode='markers+text',
        marker=dict(size=10, color='blue'),
        text=['Original End'],
        textposition="top center",
        name='Original End'
    ))
    if len(modified)!=0:
        # Modified Trajectory (Red)
        fig.add_trace(go.Scatter3d(
            x=[p['x'] for p in modified],
            y=[p['y'] for p in modified],
            z=[p['z'] for p in modified],
            mode='lines',
            line=dict(color='red'),
            name='Modified Trajectory'
        ))
        # Start and end markers for modified trajectory
        fig.add_trace(go.Scatter3d(
            x=[modified[0]["x"]], y=[modified[0]["y"]], z=[modified[0]["z"]],
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=['Modified Start'],
            textposition="top center",
            name='Modified Start'
        ))
        fig.add_trace(go.Scatter3d(
            x=[modified[-1]["x"]], y=[modified[-1]["y"]], z=[modified[-1]["z"]],
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=['Modified End'],
            textposition="top center",
            name='Modified End'
        ))
       
    
    for obj in objects:
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
        title='Trajectory Visualization',
        # margin=dict(l=0, r=0, t=50, b=0),
         margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,autosize=True
    )
    # return fig
    # st.plotly_chart(fig)

def plot_velocity(fig,original, modified):
    # fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=[p['velocity'] for p in original],
        mode='lines',
        line=dict(color='blue'),
        name='Original Velocity'
    ))
    if len(modified)!=0:
        fig.add_trace(go.Scatter(
            y=[p['velocity'] for p in modified],
            mode='lines',
            line=dict(color='red'),
            name='Modified Velocity'
        ))
    
    fig.update_layout(title='Velocity Profile', xaxis_title='Time Step', yaxis_title='Velocity', margin=dict(l=10, r=10, t=40, b=10),autosize=True)
    # return fig
# Function to open directory picker using tkinter
def select_directory():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_selected = filedialog.askdirectory()  # Open directory selection dialog
    return folder_selected
st.title('OVITA: Open-Vocabulary Interpretable Trajectory Adaptations')

uploaded_file = st.file_uploader("Upload trajectory JSON file", type=["json"], key="unique_json_uploader")  #accept_multiple_files=True

if uploaded_file:
    file_name=uploaded_file.name
    config = Config()
    data = load_trajectory(uploaded_file)
    # instruction = data['instruction'].lower()
    st.session_state.original_trajectory = get_trajectory()
    with st.sidebar:
        instruction=st.text_input("Provide instruction for adaptation")
        llm_choice = st.selectbox("Select LLM:", ["openai", "claude", "gemini"])
        robot_type = st.selectbox("Select Robot Type:", ["Drone", "Arm", "GroundRobot", "Latte"])
        # Request feedback
        # if st.session_state.run_count>0:
        if st.session_state.zero_shot_trajectory["modified trajectory"] !=[]:
            st.session_state.feedback_type=st.selectbox("Feedback Type:", ["original","current"], index=None, placeholder="Select feedback type")
            st.session_state.feedback = st.text_input("Provide Feedback (or type YES if satisfied):")
        run_adaptation=st.button("Run Adaptation")
        save_button=st.button("Save")
        browse_button=st.button("Browse for Directory")
        # Create some empty space before the toggle button
        for _ in range(2):  # Adjust the range for more/less space
            st.sidebar.text("")
        # st.session_state.toggle_state = st.toggle("Constraints Satisfaction",value=True)
        # if st.session_state.toggle_state:
        st.markdown("**CSM configs**")
        st.session_state.toggle_state_CSM = st.toggle(
            "Constraints Satisfaction", 
            value=st.session_state.toggle_state_CSM, 
            key="toggle_CSM")
        st.session_state.toggle_state_start = st.toggle(
            "Fix start position for CSM", 
            value=st.session_state.toggle_state_start, 
            key="toggle_start")
        st.session_state.toggle_state_end = st.toggle(
            "Fix goal position for CSM", 
            value=st.session_state.toggle_state_end, 
            key="toggle_end")
        st.session_state.toggle_state_smooth = st.toggle(
            "Smooth via splines", 
            value=st.session_state.toggle_state_smooth, 
            key="toggle_smooth")
    # if st.session_state.run_count>0:
    # traj_sel=st.selectbox("Trajectory View Select", ["original_trajectory","zero_shot_trajectory", "final_trajectory"],index=st.session_state.run_count if st.session_state.run_count<3 else 2)
    if st.session_state.run_count>0:
        traj_sel=st.selectbox("Trajectory View Select", ["original_trajectory","zero_shot_trajectory", "final_trajectory"] if st.session_state.run_count>1 else ["original_trajectory","zero_shot_trajectory"],index=0)
    else:
        traj_sel=st.selectbox("Trajectory View Select", ["original_trajectory"] ,index=0)
    # if st.session_state.zero_shot_trajectory["modified trajectory"] !=[]:
    if traj_sel=="zero_shot_trajectory" or traj_sel=="final_trajectory":
        st.header(f"Modified Trajectory")
        st.subheader(f"Instruction: {instruction}")
        if  traj_sel=="final_trajectory":
            # st.subheader(f"Feedback: [{st.session_state.feedback_type}] {st.session_state.feedback}")
            st.subheader("Global Feedbacks")
            st.markdown("\n".join([f"- {None if st.session_state.global_feedbacks==[] else fb}" for fb in st.session_state.global_feedbacks]))

            st.subheader("Local Feedbacks")
            st.markdown("\n".join([f"- {None if st.session_state.local_feedbacks==[] else fb}" for fb in st.session_state.local_feedbacks]))
    else:
        st.header(f"Original Trajectory")
    col1, col2 = st.columns(2) 
    with col1:
        fig_traj=go.Figure()
        plot_trajectory(fig_traj,st.session_state["original_trajectory"], st.session_state["original_trajectory"] if traj_sel=="original_trajectory" else st.session_state[traj_sel]["modified trajectory"],detect_objects(),instruction)
        st.plotly_chart(fig_traj,use_container_width=True)
        st.markdown("<br>", unsafe_allow_html=True)  # Add space
    with col2:
        fig_vel=go.Figure()
        plot_velocity(fig_vel,st.session_state["original_trajectory"],st.session_state["original_trajectory"] if traj_sel=="original_trajectory" else st.session_state[traj_sel]["modified trajectory"])
        st.plotly_chart(fig_vel,use_container_width=True)
    # if st.session_state.run_count>0:
    # if st.session_state.feedback!="" and st.session_state.feedback_type!= None:
    if traj_sel=="final_trajectory":
        # st.text_area("Modefied Intrepretation", , height=200, disabled=True)
        st.markdown("<h1 style='color:red;'>LLM Generation</h1>", unsafe_allow_html=True)
        st.header("Final Intrepretation")
    if traj_sel=="zero_shot_trajectory":
        st.markdown("<h1 style='color:red;'>LLM Generation</h1>", unsafe_allow_html=True)
        st.header("Zero Shot Intrepretation")
    if traj_sel!="original_trajectory":
        if st.session_state[traj_sel]["modified trajectory"]!=[]:
            st.subheader("High Level Plan")
            st.markdown("\n".join([f"- {fb}" for fb in [point.strip()for point in re.split(r'\d+\)', st.session_state[traj_sel]["high_level_plan"]) if point.strip()]]))

            st.subheader("Code Explanation")
            st.markdown(st.session_state[traj_sel]["interpretation"])

            st.subheader("Code As a Policy")
            st.code(st.session_state[traj_sel]["code"], language='python') 
        else:
            st.markdown("<h4 style='color:red;'>LLM Generation Pending</h4>", unsafe_allow_html=True)
            st.markdown(f"Click Run Adaptation to get the {traj_sel}")
    
    config.api_name = llm_choice
    agent = LLM_Agent(config)
    workspace_bound=True
    if robot_type == 'Drone':
        workspace_type="cuboidal" if workspace_bound else "general"
        workspace_bounds = config.workspace_bounds_drone
        # robot = Drone_and_husky_constraints(safety_distance=config.SAFETY_DISTANCE, is_ground_robot=False,lambda_smooth=st.session_state.lambda_smooth,lambda_obstacle=st.session_state.lambda_obstacle,lambda_original=st.session_state.lambda_original,lambda_reach=st.session_state.lambda_reach)
    elif robot_type == 'Arm':
        workspace_type="arm" if workspace_bound else "general"
        workspace_bounds = config.workspace_bounds_arm
        # robot = RobotArm7DOF_constraint()
    elif robot_type == 'GroundRobot':
        workspace_type="cuboidal" if workspace_bound else "general"
        workspace_bounds = config.workspace_bounds_ground_robot
        # robot = Drone_and_husky_constraints(safety_distance=config.SAFETY_DISTANCE, is_ground_robot=True,lambda_smooth=st.session_state.lambda_smooth,lambda_obstacle=st.session_state.lambda_obstacle,lambda_original=st.session_state.lambda_original,lambda_reach=st.session_state.lambda_reach)
    else:
        workspace_type="cuboidal" if workspace_bound else "general"
        workspace_bounds=config.workspace_bounds_common
        # robot = general_constraints(safety_distance=config.SAFETY_DISTANCE,lambda_smooth=st.session_state.lambda_smooth,lambda_obstacle=st.session_state.lambda_obstacle,lambda_original=st.session_state.lambda_original,lambda_reach=st.session_state.lambda_reach)
    
    # Button to open directory picker
    if browse_button:
        selected_dir = select_directory()
        if selected_dir:
            st.session_state["save_dir"] = selected_dir  # Store selected directory in session state

    # Display selected directory
    save_dir = st.session_state.get("save_dir", "Not selected")
    try:   
        if run_adaptation:
                st.session_state.run_count += 1  # Increment count
            # Get the feedback type 

                if st.session_state.feedback=="" or st.session_state.feedback is None:
                    st.session_state.new_instruction = instruction
                else:
                    
                    if st.session_state.feedback_type == "original":
                        # print("##########################################")
                        # Reset local feedbacks and related data 
                        local_feedbacks = []
                        st.session_state.get_trajectory_param = True
                        st.session_state.global_feedbacks.append(st.session_state.feedback)
                        st.session_state.new_instruction = instruction + " ".join(st.session_state.global_feedbacks)
                    elif st.session_state.feedback_type == "current":
                        st.session_state.get_trajectory_param = False  
                        st.session_state.local_feedbacks.append(st.session_state.feedback)
                        st.session_state.new_instruction = st.session_state.feedback

                try:
                    object_details=""
                    object_details += " ".join(item['name']+"," for item in detect_objects())
                    env_descp=data.get("Env_descp",None)
                    env_object_info="The exact names of the objects present in the environement are "+object_details+ "use these names only as argument for detect_objects() function"
                    if data.get("Env_descp",None) is None:
                        env_descp = env_object_info
                    else: 
                        env_descp+=" \n"+env_object_info
                    high_level_plan,generated_code = agent.generate_code(st.session_state.new_instruction,env_descp)
                    print("################################")
                    print("The new instrcution is",st.session_state.new_instruction)
                    exec(generated_code,globals())
                    variables=agent.extract_variables_and_constants(generated_code)
                    # import ipdb; ipdb.set_trace()
                    variable_values = {var: globals().get(var, None) for var in variables}
                    interpretation=agent.explain_code(generated_code,variable_values)
                    st.session_state.interpretation_details=[high_level_plan,generated_code,interpretation]
                    # print("The interpretation of the code is \n", interpretation)
                except Exception as e:
                    print(f"Error executing the generated code: {e}")
                    # Terminate loop if executable code is not produced
                    if st.session_state.feedback is None:  
                        st.session_state.zero_shot_trajectory.update({"code_executability":False})
                    else: 
                        st.session_state.final_trajectory.update({"code_executability":False})
                # if st.session_state.toggle_state:   
                #     st.session_state.modified_trajectory=robot.satisfy_constraints(modified_trajectory,detect_objects())
                # else:


                if st.session_state.toggle_state_CSM:
                    config.fix_start = st.session_state.toggle_state_start
                    config.fix_goal = st.session_state.toggle_state_end
                    config.apply_smooth_spline = st.session_state.toggle_state_smooth
                    st.session_state.modified_trajectory=convert_from_vector(constraint_satisfaction_module(
                            trajectory=np.array(convert_to_vector(modified_trajectory)),
                            workspace_type=workspace_type,
                            workspace_bounds=workspace_bounds,
                            obstacles=detect_objects(),
                            safety_margin_obstacles=config.safety_margin_obstacles,
                            safety_margin_boundary=config.safety_margin_boundary,
                            fix_start=config.fix_start,
                            fix_goal=config.fix_goal,
                            max_vel=config.max_vel+0.1,
                            apply_spline_smooth=config.apply_smooth_spline
                        ))
                else:
                    st.session_state.modified_trajectory=modified_trajectory
                


                print("Constraints satisfied")
                if st.session_state.feedback is None or st.session_state.feedback_type=="original":
                    st.session_state.original_trajectory=get_trajectory()
                elif st.session_state.feedback_type=="current":
                    print("current context, trajectory extracted from modified list")# Request feedback
                    st.session_state.original_trajectory=st.session_state.modified_trajectory_list[-1]
                
   
                # Store the zero-shot trajectory and corresponding data
                # if st.session_state.feedback =="" or st.session_state.feedback ==None: 
                import ipdb; ipdb.set_trace()
                if st.session_state.run_count==1:
                    st.session_state.zero_shot_trajectory.update({
                        "trajectory": get_trajectory(), 
                        "instruction": instruction,
                        "objects": detect_objects(),
                        "modified trajectory": st.session_state.modified_trajectory,
                        "high_level_plan": high_level_plan,
                        "code": generated_code, 
                        "interpretation": interpretation,
                        "code_executability": True
                    })
                if len(st.session_state.global_feedbacks) !=0 or len(st.session_state.local_feedbacks)!=0:
                    st.session_state.final_trajectory.update({
                        "trajectory": get_trajectory(), 
                        "instruction": instruction,
                        "objects": detect_objects(),
                        "modified trajectory": st.session_state.modified_trajectory_list[-1],
                        "high_level_plan": high_level_plan,
                        "code": generated_code, 
                        "global feedbacks": st.session_state.global_feedbacks,
                        "local feedbacks": st.session_state.local_feedbacks,
                        "interpretation": interpretation,
                        "code_executability": True
                    })

                st.session_state.modified_trajectory_list.append(st.session_state.modified_trajectory)



                
                
                placeholder.write("Generating...")
                time.sleep(5)
                st.rerun() 
    except Exception as e:
        print(f"Error {e} has occured")
        with st.expander("⚠️ Error Details"):
            st.warning(f"Error {e} has occured")
        with open(os.path.join(cwd+"/",file_name), "w") as outfile:
            json.dump({
                'zero_shot_trajectory': st.session_state.zero_shot_trajectory,
                'final_trajectory': st.session_state.final_trajectory,
                'LLM': config.api_name,
                'code_executability':False
            }, outfile, indent=4)
        # exit()
        st.rerun()
    # Save the results 
    if save_button:
        if save_dir == "Not selected":
            st.error("Please select a directory first!")
        else:
            with open(os.path.join(save_dir,file_name), "w") as outfile:
                json.dump({
                    'zero_shot_trajectory': st.session_state.zero_shot_trajectory,
                    'final_trajectory': st.session_state.final_trajectory,
                    'LLM': config.api_name,
                    'code_executability': True
                }, outfile, indent=4)
else:
    st.session_state.clear()