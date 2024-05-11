import streamlit as st
import requests
from langchain_groq import ChatGroq
from langchain import LLMChain
from langchain.prompts import PromptTemplate

# Function to initialize the LLMChain
def initialize_llm(api_key, model_name, temperature):
    return ChatGroq(
        temperature=temperature,
        api_key=api_key,
        model_name=model_name
    )

def generate_image(description):
    # Generate image based on the description
    image_url = f"https://image.pollinations.ai/prompt/{description.replace(' ', '%20')}"
    response = requests.get(image_url)
    if response.status_code == 200:
        return response.content
    else:
        return None

def main():
    st.sidebar.title("📚 Story Generator Settings")
    st.sidebar.markdown("---")
    
    # API Key input
    default_api_key = "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"
    api_key = st.sidebar.text_input("API Key", value=default_api_key)

    # Model selection dropdown
    supported_models = {
        "LLaMA3 8b": "llama3-8b-8192",
        "LLaMA3 70b": "llama3-70b-8192",
        "Mixtral 8x7b": "mixtral-8x7b-32768",
        "Gemma 7b": "gemma-7b-it",
        "Whisper": "whisper-large-v3"
    }
    selected_model = st.sidebar.selectbox("Select Model", list(supported_models.keys()))

    # Temperature input
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

    # Initialize LLMChain
    hub_llm = initialize_llm(api_key, supported_models[selected_model], temperature)

    # Prompt template for story generation
    prompt = PromptTemplate(
        input_variables=["description"],
        template="Create a story based on the following description: {description}."
    )

    # Initialize the LLMChain with the story generation prompt and model
    hub_chain = LLMChain(prompt=prompt, llm=hub_llm, verbose=True)

    # Text area for story description
    description = st.text_area("Story Description", height=150)

    # Generate story button
    if st.button("Generate Story"):
        # Generate and display the story
        st.markdown("---")
        st.subheader("Generated Story")
        story = hub_chain.run(description)
        st.write(story)

        # Generate and display image
        st.subheader("Generated Image")
        image_content = generate_image(description)
        if image_content:
            st.image(image_content, caption="Image generated based on the story description", use_column_width=True)
        else:
            st.write("Image generation failed. Please try again with a different description.")

if __name__ == "__main__":
    main()
