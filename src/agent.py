import json 
import ast,astor 
from .prompts import SYSTEM_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE, EXAMPLE_1,EXAMPLE_2,CODE_EXPLAINER_PROMPT
from openai import OpenAI
import anthropic
import google.generativeai as genai
import math
import numpy as np
class LLM_Agent:
    def __init__(self, config):
        self.api = config.api_name
        self.temperature = config.temperature
        self.explain_temperature=config.explain_temperature
        self.model_name, self.api_key = self._initialize_model(config)

    def _initialize_model(self, config):
        api_keys_map = {
            "gemini": (config.model_gemini, config.keys['gemini_key']),     
            "claude": (config.model_claude, config.keys['claude_key']),
            "openai": (config.model_openai, config.keys['openai_key']),
        }
        if self.api not in api_keys_map:
            raise ValueError(f"Unsupported API: {self.api}")
        return api_keys_map[self.api]
    
    def generate_code(self, instruction,env_descp):
        messages = self._prepare_messages(instruction,env_descp)
        high_level_plan, code = self.get_code(messages)
        code=code.replace('\\n','\n')
        print("High level plan is ",high_level_plan)
        print("code generated is ",code)
        code=self._remove_dummy_functions(code)
        return high_level_plan, code

    def get_code(self, messages):
        if self.api == "gemini":
            return self._get_code_gemini(messages)
        elif self.api == "claude":
            return self._get_code_claude(messages)
        elif self.api == "openai":
            return self._get_code_openai(messages)

    def _get_code_gemini(self, messages):
        genai.configure(api_key=self.api_key)
        system_instruction = "\n".join(item['content'] for item in messages[:-1])
        client = genai.GenerativeModel(
            f"models/{self.model_name}",
            system_instruction=system_instruction
        )
        response = client.generate_content(
            messages[-1]['content'],
            generation_config=genai.GenerationConfig(
                temperature=self.temperature,
                response_mime_type="application/json"
            )
        )
        output = json.loads(response.text)
        return output['high_level_plan'], output['Python_code']

    def _get_code_claude(self, messages):
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            max_tokens=2048,
            temperature=self.temperature,
            model=self.model_name,
            messages=messages
        )
        temp_text = response.content[0].text.replace("\'", "'")
        high_level_plan = self._extract_json_field(temp_text, "high_level_plan")
        python_code = self._extract_json_field(temp_text, "Python_code").replace('\\n', '\n').replace('\\"', '"')
        return high_level_plan, python_code

    def _get_code_openai(self, messages):
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            response_format={"type": "json_object"},
            temperature=self.temperature,
            messages=messages
        )
        output = json.loads(response.choices[0].message.content)
        return output['high_level_plan'], output['Python_code']

    @staticmethod
    def _extract_json_field(text, field_name):
        """Extract a JSON field value from a text."""
        if field_name.lower()=="high_level_plan":
            field_start = text.index(f"\"{field_name}\"") + len(f"\"{field_name}\":")
            field_end = text.index(",", field_start) if "," in text[field_start:] else text.index("}", field_start)
            return text[field_start:field_end].strip().strip("\"")
        else:
            field_start = text.index(f"\"{field_name}\"") + len(f"\"{field_name}\":")
            field_end=text.rindex("\"")
            return text[field_start:field_end].strip().strip("\"")

    def explain_code(self, code,variable_values):
        if 'modified_trajectory' in variable_values:
            del variable_values['modified_trajectory']
        if 'trajectory' in variable_values:
            del variable_values['trajectory']
        if self.api=="claude":
            messages=[
                    {"role": "assistant", "content": CODE_EXPLAINER_PROMPT},
                    {"role": "user", "content": f"The code is {code}"},    
                    {"role": "user", "content": f"The parsed variables and their values are {str(variable_values)}"},  
                ]
        
        else:
            messages=[
                    {"role": "system", "content": CODE_EXPLAINER_PROMPT},
                    {"role": "user", "content": f"The code is {code}"},    
                    {"role": "user", "content": f"The parsed variables and their values are {str(variable_values)}"},  
                ]
        
        if self.api == "gemini":
            return self._get_explanation_gemini(messages)
        elif self.api == "claude":
            return self._get_explanation_claude(messages)
        elif self.api == "openai":
            return self._get_explanation_openai(messages)

    
    def _get_explanation_gemini(self, messages):
        genai.configure(api_key=self.api_key)
        system_instruction = "\n".join(item['content'] for item in messages[:-1])
        client = genai.GenerativeModel(
            f"models/{self.model_name}",
            system_instruction=system_instruction
        )
        response = client.generate_content(
            messages[-1]['content'],
            generation_config=genai.GenerationConfig(
                temperature=self.explain_temperature,
            )
        )
        output=response.text
        return output
    
    def _get_explanation_openai(self,messages):
        client = OpenAI(api_key=self.api_key)   
        response = client.chat.completions.create(
            model=self.model_name,
            temperature=self.explain_temperature,
            messages=messages
        )
        output = response.choices[0].message.content
        return output
    
    def _get_explanation_claude(self,messages):
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            max_tokens=2048,
            temperature=self.explain_temperature,
            model=self.model_name,
            messages=messages
        )
        return response.content[0].text

    def _prepare_messages(self, instruction,env_descp):
        system_prompt = SYSTEM_PROMPT_TEMPLATE.replace("[ENVIRONMENT DESCRIPTION]",env_descp)
        user_prompt = USER_PROMPT_TEMPLATE.replace("[INSTRUCTION]", instruction)
        messages = [
            {"role": "assistant", "content": "Return a JSON file as output. Do not output any extra text"},
            {"role": "assistant", "content": system_prompt},
            {"role": "user", "content": EXAMPLE_1},
            {"role": "user", "content": EXAMPLE_2},
            {"role": "user", "content": user_prompt},
        ] if self.api == "claude" else [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": EXAMPLE_1},
            {"role": "user", "content": EXAMPLE_2},
            {"role": "user", "content": user_prompt},
        ]
        return messages

    @staticmethod
    def _remove_dummy_functions(code):
        class DummyFunctionRemover(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                    return None
                if node.name in {'get_trajectory', 'detect_objects'}:
                    return None
                return node

        parsed_code = ast.parse(str(code))
        modified_ast = DummyFunctionRemover().visit(parsed_code)
        return astor.to_source(modified_ast)
    
    def extract_variables_and_constants(self,code_str):
        class VariableAndConstantVisitor(ast.NodeVisitor):
            def __init__(self):
                self.variables = set()
                self.constants = set()
                self.variable_assignments = {}

            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        self.variables.add(var_name)
                        # Track assignments to detect reassignments
                        if var_name in self.variable_assignments:
                            self.variable_assignments[var_name] += 1
                        else:
                            self.variable_assignments[var_name] = 1
                self.generic_visit(node)

            def visit_Constant(self, node):
                self.constants.add(node.value)

            def visit_NameConstant(self, node):
                self.constants.add(node.value)

            def visit_Num(self, node):
                self.constants.add(node.n)

            def visit_Str(self, node):
                self.constants.add(node.s)

        visitor = VariableAndConstantVisitor()
        tree = ast.parse(code_str)
        visitor.visit(tree)

        # Exclude variables that are reassigned multiple times
        non_reassigned_variables = {
            var for var, count in visitor.variable_assignments.items() if count == 1
        }
        print(non_reassigned_variables)
        return non_reassigned_variables

    def execute_and_get_values(self,code_str):
        variables= self.extract_variables_and_constants(code_str)
        exec_context = {}
        # Execute the code with a predefined global context
        exec(code_str, {"get_trajectory": get_trajectory, "detect_objects": detect_objects}, exec_context)
        # Retrieve values of the non-reassigned variables
        variable_values = {var: exec_context[var] for var in variables if var in exec_context}
        return variable_values