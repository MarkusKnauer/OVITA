import os 
openai_key=os.getenv("OPENAI_API_KEY")
gemini_key=os.getenv("GEMINI_API_KEY")
claude_key=os.getenv("CLAUDE_API_KEY")

class Config:
    def __init__(self):
        self.DEFAULT_DIMENSION=0.05
        # self.SAFETY_DISTANCE=0.05
        self.safety_margin_obstacles = 0.01
        self.safety_margin_boundary = 0.01
        self.visualise_boundary=False
        self.workspace_bounds_drone = {
            "x": [-3, 3],
            "y": [-3, 3],
            "z": [-1, 3],
        }
        self.workspace_bounds_arm = {
            "centre": [-0.5, 0, 0.6],
            "r_max": 1.2,
            "r_min": 0.15
        }
        self.workspace_bounds_ground_robot = {
            "x": [-3, 3],
            "y": [-3, 3],
            "z": [-0.01, 0.01],
        }
        self.workspace_bounds_common={
            "x": [-3, 3],
            "y": [-3, 3],
            "z": [-3, 3],} 
        self.fix_start=False
        self.fix_goal=False
        self.max_vel=1.0
        self.apply_smooth_spline=True
        self.api_name="openai"
        self.model_gemini="gemini-1.5-pro"
        self.model_openai="gpt-4o"
        self.model_claude="claude-3-opus-20240229"
        self.temperature=0.1
        self.explain_temperature=0.3
        self.keys={
            "gemini_key":gemini_key,
            "claude_key":claude_key, 
            "openai_key":openai_key
        }
        


