------------------------------------------------------------
OVITA: Open Vocabulary Interpretable Trajectory Adaptations
------------------------------------------------------------
- This repository implements the official code for OVITA: Open Vocabulary Interpretable Trajectory Adaptations.
----------------
INSTALLATION
----------------
- Pre-requisites:
	Miniconda (https://docs.conda.io/projects/miniconda/en/latest/index.html)

- Install with:
	conda env create -n ovita python=3.9
	conda activate ovita
	# navigate to the source folder
	pip install -r requirements.txt

Download Dataset and extract it under the 'dataset' directory.

---------------------
RUNNING THE AGENT
---------------------
- To try out the agent:

- Create a keys.py file inside the src folder and add your API keys:

	openai_key = "your_openai_api_key"
	gemini_key = "your_gemini_api_key"
	claude_key = "your_claude_api_key"

- Run the agent with:

	python main.py --trajectory_path <path_to_trajectory> --save_dir <path_to_save_directory> --llm <openai|gemini|claude> --save_results <True|False> --robot_type <robot_name_or_None>
	Example: 
	python main.py --trajectory_path ~/trajectory.json --save_dir ~/ovita_results --llm openai --save_results True --robot_type None
c

--------------
TRY IN GUI
--------------
- Run the GUI with:

	streamlit run ~/<Path to GUI File>/main_gui_streamlit.py

- Steps to Adapt Trajectory:
	1. Upload the trajectory file via the navigator.
	2. Inspect the original trajectory in the Plotly-rendered view.
	3. In the sidebar, enter adaptation instructions, choose the LLM, and set the robot type to LaTTe.
	4. Click Run Adaptation and wait for it to complete.
	5. Select Zero-Shot as the trajectory view and inspect the modified trajectory.
	6. If further adjustments are needed, provide feedback, select the context type, and click Run Adaptation.
	7. Select Final as the trajectory view. Repeat steps 6 and 7 until satisfied.
	8. Adjust lambda values of the CSM module to satisfy Smoothness, Obstacle Avoidance, Similarity, and Reachability constraints. Press Adjust to reflect the changes.
	9. To reset to the initial modified results, press Reset. Repeat step 8 until satisfied.
	10. Once satisfied, browse for the respective directory and press Save to save the results.

----------------------------
USING YOUR OWN DATAPOINT
----------------------------
- Ensure your trajectory file is a JSON file with the following structure:

	{
	    "trajectory": [[x, y, z, speed], [x, y, z, speed], ...],
	    "instruction": "your instruction here; can be given directly in the GUI too",
	    "objects": [
		{
		    "name": "person",
		    "x": 1.0,
		    "y": 0.11,
		    "z": 0.8
		},
		...
	    ],
	    "Env_descp": "Any environment description you want to give"
	}


