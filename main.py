
from src.constraints import RobotArm7DOF_constraint, Drone_and_husky_constraints, general_constraints
from src.config import Config
from src.util import convert_to_vector,compare_trajectory
from src.agent import LLM_Agent
import copy
import json
import numpy as np
import math
import os
import argparse
import ipdb

def get_trajectory(sample=True,num_points=100, dict_format=True):
    if get_trajectory_param:
        trajectory = copy.deepcopy(data.get('trajectory', []))
    else: 
        trajectory= copy.deepcopy(convert_to_vector(modified_trajectory)).tolist()
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



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Trajectory adatation using LLMs")

    parser.add_argument(
        "--trajectory_path", 
        type=str, 
        required=True, 
        help="Path to the trajectory JSON file."
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        required=True, 
        help="Directory where results will be saved."
    )
    parser.add_argument(
        "--llm", 
        type=str, 
        choices=["openai", "claude", "gemini"], 
        required=True, 
        help="The LLM to be used for processing."
    )
    parser.add_argument(
        "--save_results", 
        type=bool, 
        choices=[True, False], 
        required=True, 
        help="Whether to save the results or not."
    )
    parser.add_argument(
        "--robot_type", 
        type=str, 
        choices=["Drone", "Arm", "GroundRobot", "None"], 
        required=True, 
        help="Type of robot for which the trajectory applies."
    )
    
    args = parser.parse_args()

    trajectory_path=args.trajectory_path
    file_name = os.path.basename(trajectory_path)
    save_dir=args.save_dir
    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    with open(trajectory_path, 'r') as file:    
        data = json.load(file)

    config=Config()
    config.api_name=args.llm
    agent=LLM_Agent(config)
    get_trajectory_param=True
    # Robot or environment specific 
    if args.robot_type=='Drone':
        robot=Drone_and_husky_constraints(safety_distance=config.SAFETY_DISTANCE,is_ground_robot=False)
    elif args.robot_type=='Arm':
        robot=RobotArm7DOF_constraint()
    elif args.robot_type=='GroundRobot':
        robot=Drone_and_husky_constraints(safety_distance=config.SAFETY_DISTANCE,is_ground_robot=True)
    else: 
        robot=general_constraints(safety_distance=config.SAFETY_DISTANCE)   
    instruction=data['instruction'].lower()
    modified_trajectory_list=[]
    feedback=None
    global_feedbacks=[]
    local_feedbacks=[]
    feedback_type=None
    get_trajectory_param=True #if true return the original trajectory else return the trajectory at time t

    zero_shot_trajectory={}
    final_trajectory={}

    try:
        while True:
            # Get the feedback type 
            if feedback is None:
                new_instruction = instruction
            else:
                
                if feedback_type == "[original]":
                    # Reset local feedbacks and related data 
                    local_feedbacks = []
                    get_trajectory_param = True
                    global_feedbacks.append(feedback.replace("[original]",""))
                    new_instruction = instruction + " ".join(global_feedbacks)
                elif feedback_type == "[current]":
                    get_trajectory_param = False
                    feedback=feedback.replace("[current]","")   
                    local_feedbacks.append(feedback)
                    new_instruction = feedback
                elif feedback.lower() == "yes":
                    # Save the trajectory and results
                    get_trajectory_param = True
                    # Save the trajectory after multiple feedbacks
                    if len(global_feedbacks) !=0 or len(local_feedbacks)!=0: # there are feedbacks involved after zero shot adaptation
                        final_trajectory.update({
                            "trajectory": get_trajectory(), 
                            "instruction": instruction,
                            "objects": detect_objects(),
                            "modified trajectory": modified_trajectory,
                            "high_level_plan": high_level_plan,
                            "code": generated_code, 
                            "global feedbacks": global_feedbacks,
                            "local feedbacks": local_feedbacks,
                            "interpretation": interpretation,
                            "code_executability": True
                        })
                    break
                else:
                    print("Choose from the following only: yes, [original] feedback, [current] feedback")
                    feedback = input("If the high-level plan is right, type YES; else type the feedback ([current], [original]]): \n")
                    feedback_type = feedback.split(" ")[0].lower() if feedback else ""
                    continue

            try:
                object_details=""
                object_details += " ".join(item['name']+"," for item in detect_objects())
                env_object_info="The exact names of the objects present in the environement are "+object_details+ "use these names only as argument for detect_objects() function"
                if data.get("Env_descp",None) is None:
                    env_descp = env_object_info
                else: 
                    env_descp=env_descp+" \n"+env_object_info
                high_level_plan,generated_code = agent.generate_code(new_instruction,env_descp)
                exec(generated_code,globals())
                variables=agent.extract_variables_and_constants(generated_code)
                # import ipdb; ipdb.set_trace()
                variable_values = {var: globals().get(var, None) for var in variables}
                interpretation=agent.explain_code(generated_code,variable_values)
                print("The interpretation of the code is \n", interpretation)
            except Exception as e:
                print(f"Error executing the generated code: {e}")
                # Terminate loop if executable code is not produced
                if feedback is None:  
                    zero_shot_trajectory.update({"code_executability":False})
                else: 
                    final_trajectory.update({"code_executability":False})
                break   
            
            modified_trajectory=robot.satisfy_constraints(modified_trajectory,detect_objects())
            print("Constraints satisfied")

            if feedback is None or feedback_type=="[original]":
                original_trajectory=get_trajectory()
            elif feedback_type=="[current]":
                print("current context, trajectory extracted from modified list")
                original_trajectory=modified_trajectory_list[-1]
            
            # Visualise the trajectory
            compare_trajectory(
                original_trajectory=convert_to_vector(original_trajectory),
                modified_trajectory=convert_to_vector(modified_trajectory),
                points=detect_objects(),
                title=instruction
            )
            
            # Store the zero-shot trajectory and corresponding data
            if feedback is None: 
                zero_shot_trajectory.update({
                    "trajectory": get_trajectory(), 
                    "instruction": instruction,
                    "objects": detect_objects(),
                    "modified trajectory": modified_trajectory,
                    "high_level_plan": high_level_plan,
                    "code": generated_code, 
                    "interpretation": interpretation,
                    "code_executability": True
                })

            modified_trajectory_list.append(modified_trajectory)
            # Request feedback
            feedback = input("If the high-level plan is right, type YES; else type the feedback ([current], [original]): \n")
            feedback_type = feedback.split(" ")[0].lower() if feedback else ""

    except Exception as e:
        print(f"Error {e} has occured")
        with open(os.path.join(save_dir,file_name), "w") as outfile:
            json.dump({
                'zero_shot_trajectory': zero_shot_trajectory,
                'final_trajectory': final_trajectory,
                'LLM': config.api_name,
                'code_executability':False
            }, outfile, indent=4)
        exit()

    
        # Save the results 
    with open(os.path.join(save_dir,file_name), "w") as outfile:
        json.dump({
            'zero_shot_trajectory': zero_shot_trajectory,
            'final_trajectory': final_trajectory,
            'LLM': config.api_name,
            'code_executability': True
        }, outfile, indent=4)
