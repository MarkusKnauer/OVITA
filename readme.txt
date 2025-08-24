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
	conda env create -f environment.yaml
	conda activate ovita

navigate to the OVITA directory and run 
	pip install -e .

Download Dataset and extract it under the 'dataset' directory.

---------------------
VISUALIZE THE DATASET
---------------------
	python scripts\visualize_dataset.py --trajectory_path <Path-to-your-file>
	python scripts\visualize_dataset.py 

---------------------
RUNNING THE AGENT
---------------------
- To try out the agent:

- Save your your API keys as environment variables:

	OPENAI_API_KEY = "your_openai_api_key"
	GEMINI_API_KEY = "your_gemini_api_key"
	CLAUDE_API_KEY = "your_claude_api_key"

- Run the agent with:

	python scripts/main.py --trajectory_path <path_to_trajectory> --save_dir <path_to_save_directory> --llm <openai|gemini|claude> --save_results <True|False> --robot_type <robot_name_or_None>
	Example: 
	python scripts/main.py --trajectory_path ~/trajectory.json --save_dir ~/ovita_results --llm openai --save_results True --robot_type None

--------------
TRY IN GUI
--------------
- Run the GUI with:

	streamlit run scripts/main_gui_streamlit.py

- Steps to Adapt Trajectory:
	1. Upload the trajectory file via the navigator.
	2. Inspect the original trajectory in the Plotly-rendered view.
	3. In the sidebar, enter adaptation instructions, choose the LLM, and set the robot type to LaTTe.
	4. Click Run Adaptation and wait for it to complete.
	5. Select Zero-Shot as the trajectory view and inspect the modified trajectory.
	6. If further adjustments are needed, provide feedback, select the context type, and click Run Adaptation.
	7. Select Final as the trajectory view. Repeat steps 6 and 7 until satisfied.
	8. Play around with the CSM configs to achieve best results. Have a look at the config.py file for more fine-graiend control over params.
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


