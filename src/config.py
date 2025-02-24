from .keys import *
class Config:
    def __init__(self):
        self.DEFAULT_DIMENSION=0.05
        self.SAFETY_DISTANCE=0.05
        self.base_position = [0, 0, 0]  
        self.link_lengths = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
        self.max_dimensions_drone=[-10,10,-10,10,0,5]
        self.max_dimensions_husky=[-4,4,-3,3,0,0]
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
        


