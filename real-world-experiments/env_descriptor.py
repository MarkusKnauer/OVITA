from openai import OpenAI
import os
import base64
class EnvironmentDescriptor:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def encode_image(self, image_path: str) -> str:
        """Encodes an image to base64 format."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def describe_environment(self, image_path: str) -> str:
        """Generates a structured environment description given an image."""
        image_base64 = self.encode_image(image_path)
        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an AI that describes environments in a structured format."},
                {"role": "user", "content": "Describe the environment in the given image in the following structured format: \n\n"
                                           "Environment Details:\n\n"
                                           "1. The objects in the scene:\n   - List of objects\n\n"
                                           "2. The objects' properties:\n   - Object Name: Description (color, shape, orientation any other properties if noted) \n\n"
                                           "3. Relative placement of all the objects with respect to each other:\n   - Placement details."},
                {"role": "user", "content": [{
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                }]}
            ]
        )
        return response.choices[0].message.content

# # Example usage:
descriptor = EnvironmentDescriptor(os.environ("OPENAI_API_KEY"))
print(descriptor.describe_environment("camera_image.png"))
