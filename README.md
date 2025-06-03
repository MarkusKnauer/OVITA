# OVITA: Open Vocabulary Interpretable Trajectory Adaptations


Anurag Maurya, Tashmoy Ghosh, Ravi Prakash (2025).

<!-- [<img src="https://img.shields.io/badge/arxiv-%23B31B1B.svg?&style=for-the-badge&logo=arxiv&logoColor=white" />]() -->


<table>
  <tr>
    <td align="center">
      <img src="docs/intro_1_gif.gif" width="70%" /><br>
      Trajectory of a drone surveilling an area
    </td>
    <td align="center">
      <img src="docs/intro_2_gif.gif" width="70%" /><br>
      Instruction: "Can you approach person closely and slowly?"
    </td>
  </tr>
</table>

---
This repository implements the official code for `OVITA`: Open Vocabulary Interpretable Trajectory Adaptations.


If you have any questions please let me know -- [akmaurya7379@gmail.com](mailto:akmaurya7379@gmail.com)

---
## Installation

Pre-requisites:
- [miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html)

Clone this repository with
```bash
cd ~
git clone https://github.com/anurag1000101/OVITA.git
cd OVITA
conda create -n ovita python=3.9 anaconda
conda activate ovita
pip install -r requirements.txt
```

Download [Dataset]() and extract it under `dataset`

---
## Running the agent

To try out the agent:

Create a keys.py file inside the src folder and add your API keys:
```python
openai_key = "your_openai_api_key"
gemini_key = "your_gemini_api_key"
claude_key = "your_claude_api_key"
```

```bash
python main.py --trajectory_path <path_to_trajectory> --save_dir <path_to_save_directory> --llm <openai|gemini|claude> --save_results <True|False> --robot_type <robot_name_or_None>
```

---
### Try in GUI

```bash
streamlit run ~/<Path to GUI File>/main_gui_streamlit.py
```
> **Steps to Adapt Trajectory:**
> 1. Upload the trajectory file via the navigator.
> 2. Inspect the original trajectory in the Plotly-rendered view.
> 3. In the sidebar, enter adaptation instructions, choose the LLM, and set the robot type to **LaTTe**.
> 4. Click **Run Adaptation** and wait for it to complete.
> 5. Select **Zero-Shot** as the trajectory view and inspect the modified trajectory.
> 6. If further adjustments are needed, provide feedback, select the context type, and click **Run Adaptation**. wait for it to get completed.
> 7. Select **Final** as the trajectory view. Repeat steps 6 and 7 until satisfied.
> 8. Slide over the **lambda** values of the **CSM** module to satisfy the **Smoothness**, **Obstacle Avoidance**, **Similarity** and **Reachability** constraints. Press **Adjust** button to reflect the changes.
> 9. To reset to the initial modefied results press the **Reset** button. Repeat step 8 until satisfied.
> 10. Once satisfied **browse** for the respective directory and press **Save** buttom to save the results.
---

📌 Using Your Own Trajectory:
Ensure your trajectory file is a **JSON file** with the following structure: 

```json
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
```

---
## Citation

If you use our work or code base(s), please cite our article:
```latex
@article{ovita2025,
  title={OVITA: Open Vocabulary Interpretable Trajectory Adaptations},
  author={Anurag Maurya, Tashmoy Ghosh, Ravi Prakash},
  year={2025}
}

