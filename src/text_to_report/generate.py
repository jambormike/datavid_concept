import os

from langchain_community.llms import Replicate
from getpass import getpass

# REPLICATE_API_TOKEN = ""
print("Please enter your Replicate API token.")
REPLICATE_API_TOKEN = getpass()
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN

# LLaMA2 13b parameters
llama2_13b = "meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
# LLaMA 70b parameters
llama2_70b = "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"

def llama_wrapper(temperature=0.75, top_p=1, max_new_tokens=1000):
    """
    Generates text using the Llama 2 model hosted on Replicate.
    
    Parameters:
    - temperature: Adjusts randomness of outputs. Values > 1 increase randomness, 0 is deterministic. Default is 0.75.
    - top_p: Samples from the top p percentage of most likely tokens. Lower values focus on more likely tokens. Default is 1.
    - max_new_tokens: Maximum number of tokens to generate. Default is 1000.
    
    Returns:
    A Replicate model instance configured with the specified parameters.
    """

    # Use the Llama 2 model hosted on Replicate
    # Temperature: Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value
    # top_p: When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens
    # max_new_tokens: Maximum number of tokens to generate. A word is generally 2-3 tokens
    llm = Replicate(
        model=llama2_13b,
        model_kwargs={
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": max_new_tokens
        }
    )
    return llm

if __name__ == "__main__":
    
    llm = llama_wrapper()
    
    prompt =  "Hello, I have a toothache. HAve you tried paracetamol to stop the pain. Yes I did but it did not work. WHere does it hurt. It hurts deep inside my tooth. Ok, I will prescribe you stronger painkiller callen TurboParacetamol. OK, thank you doctor."
    output = llm.invoke(prompt +  " Summarize this text as a doctor's report.")
    print(output)
    
    print("END")