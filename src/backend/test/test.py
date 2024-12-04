import base64
import yaml
from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI


def convert_image_to_base64(image_path: str) -> str:
    """Convert an image file to a base64-encoded string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Load the prompt template from a YAML file
with open("common/prompts.yaml", "r") as file:
    prompts = yaml.safe_load(file)
template: str = prompts["image_to_text_prompt"]["template"]

# Initialize the ChatVertexAI model
model = ChatVertexAI(model="gemini-1.5-pro-002", temperature=0.1)

# List of image paths and their corresponding context texts
image_paths = [
    "/Users/dylim/Documents/projects/RAG_Langchain/src/backend/test/pdf_finaltest/img/a88d45cb-b3fa-c4ad-523f-724bafd5f6a1/10_97_table.png",
    # Add more image paths as needed
]
context_texts = [
    "A layer name consists of the following five fields as shown in Table 1.;The above five fields of a CAD layer name are to be arranged in the format as shown in Figure 1.",
    # Add corresponding context texts
]

# Prepare batch messages
batch_messages = []
for image_path, context_text in zip(image_paths, context_texts):
    image_base64 = convert_image_to_base64(image_path)
    image_data = f"data:image/png;base64,{image_base64}"
    formatted_prompt = template.format(context_text=context_text)
    message = [
        {"role": "system", "content": formatted_prompt},
        {"role": "user", "content": image_data},
    ]
    batch_messages.append(message)

# Process batch messages
responses = model.batch(batch_messages)
for response in responses:
    print(response.content)
