import argparse
import json
import requests

from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import JSONAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.tools.render import render_text_description_and_args
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_ibm import WatsonxLLM

from modules.load_env import load_ibmcloud_env, load_watson_x_env_min
from typing import List

# ******************************************
# Functions
@tool
def weather_service(cities: List[str]) -> str:
     """weather service provides all of the weather needs and information in that api, it serves weather information.
     Args:
                cities: The parameter cities is a list e.g. [ LA, NY].
     """
     
     base_weather_url="https://wttr.in/"
     cities_input = cities
     cities = []
     
     for city in cities_input:
          # Ensure to get the JSON format: '?format=j1'
          city_temp_url = base_weather_url + city + "?format=j1"
          response = requests.get(city_temp_url)
          if (response.status_code == 200):      
              # convert from byte to text
              byte_content = response.content
              text_content = byte_content.decode("utf-8")
              
              # load json
              content = json.loads(text_content)

              # extract temperature
              temperature = content['current_condition'][0]['temp_C']
              cities.append({"city": city, "temperature":temperature})
          else:
              cities.append({"city": f"{city} ERROR", "temperature":0})
     full_text = ""
     sorted_by_temperature =  sorted(cities, key=lambda x: (x['temperature'], x['city']), reverse=True)
     print(f"{sorted_by_temperature}")
     i = 0 
     for city in sorted_by_temperature:
        if (i == 0):
             response_text =  f"The hottest city is {city['city']} with a temperature of {city['temperature']} degrees Celsius."
        else:
             response_text =  f"In the city {city['city']} the temperature is {city['temperature']} degrees Celsius."
        i = i + 1
        full_text = full_text + ' ' + response_text

     return full_text
            
def load_env():
        
        ibmcloud_apikey , validation = load_ibmcloud_env()
        if( validation == False):
            return {  "apikey" : "ERROR",
                      "project_id" : "ERROR",
                      "url": "ERROR",
                      "model_id": "ERROR"
                    }
        else:
             apikey = ibmcloud_apikey['IBMCLOUD_APIKEY']
                         
        watson_x_env, validation = load_watson_x_env_min()
        
        if validation == False:
            return { "apikey" : "ERROR",
                     "project_id" : "ERROR",
                     "url": "ERROR",
                     "model_id": "ERROR"
                    }
        else:
             project_id = watson_x_env['WATSONX_PROJECT_ID']
             url = watson_x_env['WATSONX_URL']
             model_id = watson_x_env['WATSONX_LLM_NAME']
                        
        return { "project_id" : project_id,
                 "url": url,
                 "model_id": model_id,
                 "apikey": apikey }

def agent_calling():
        
     environment = load_env()
     print(f"1. Load environment variables\n{environment}\n")      
       
     parameters = {
        "decoding_method": "greedy",
        "temperature": 0,
        "min_new_tokens": 5,
        "max_new_tokens": 250,
        "stop_sequences":['\nObservation', '\n\n']
     }
     print(f"2. Model parameters\n{parameters}\n")
        
     print(f"3. Create a WatsonxLLM instance.\n")
     watsonx_client = WatsonxLLM(
            model_id=environment['model_id'],
            url=environment['url'],
            project_id=environment['project_id'],
            apikey= environment['apikey'],
            params=parameters
        )

     prompt = PromptTemplate.from_template(system_prompt_load())
     print(f"4. Create a prompt using the PromptTemplate with a variable.\n{prompt}\n")

     print(f"5. Create a simple chain with the created prompt and the create watsonx_client.\n")
     simple_chain = prompt | watsonx_client

     print(f"6. Show the 'simple chain' graph dependencies\n")
     simple_chain.get_graph().print_ascii()

     print(f"7. Invoke the simple chain by asking a question.\n")     
     response = simple_chain.invoke({"question": "Which city is hotter today: LA or NY?"})

     print(f"8. Inspect the response.\n{response}\n") 
     
     tools = [weather_service]
     print(f"9. Define tools\n{tools}\n")

     agent_system_prompt = load_agent_system_prompt()
     print(f"10. Define agent system prompt\n{agent_system_prompt}\n")

     human_prompt = load_human_prompt()
     print(f"11. Define human prompt\n{human_prompt}\n")

     agent_chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", agent_system_prompt),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human_prompt),
        ]
     )
     print(f"12. Create an agent chat prompt from a ChatPromptTemplate\n{agent_chat_prompt}\n")

     agent_chat_incl_tools_prompt = agent_chat_prompt.partial(
        tools=render_text_description_and_args(list(tools)),
        tool_names=", ".join([t.name for t in tools]),
     )
     print(f"13. Extent the Chat Prompt Template with the given tool names as variables\n{agent_chat_incl_tools_prompt}\n")
     
     memory = ConversationBufferMemory()
     print(f"14. Setup of a ConversationBufferMemory\n{memory}\n")
     
     agent_chain = ( RunnablePassthrough.assign(
            agent_scratchpad=lambda x: format_log_to_str(x["intermediate_steps"]),
            chat_history=lambda x: memory.chat_memory.messages,
        )
        | agent_chat_incl_tools_prompt |watsonx_client | JSONAgentOutputParser()

     )
     print(f"15. Define an agent chain, including a runnable passthrough\n{agent_chain}\n")


     print(f"16. Show the 'agent chain' graph dependencies\n")
     agent_chain.get_graph().print_ascii()

     agent_executor = AgentExecutor(agent=agent_chain, tools=tools, handle_parsing_errors=True, verbose=True, memory=memory)
     print(f"17. Create an agent executor\n{agent_executor}\n")

     results = []
     
     question = "Which city is hotter today: LA or NY?"
     dict = {"input": question}
     response = agent_executor.invoke({"input":"Which city is hotter today: LA or NY?"})
     print(f"18.a Response of the agent executor\n{response}\n")
     data = { "question": question, "answer":  response['output'], "history": response['history'] }          
     results.append(data)

     question = "How hot is it today in Berlin?"
     dict = {"input": question}
     response = agent_executor.invoke(dict)
     print(f"18.b Response of the agent executor\n{response}\n")
     data = { "question": question, "answer":  response['output'], "history": response['history'] }          
     results.append(data)

     question = "What is the offical definition of the term weather?"
     dict = {"input": question}
     response = agent_executor.invoke(dict)
     print(f"18.c Response of the agent executor\n{response}\n")
     data = { "question": question, "answer":  response['output'], "history": response['history'] }          
     results.append(data)

     question = "How to win a soccer game?"
     dict = {"input": question}
     response = agent_executor.invoke(dict)
     print(f"18.d Response of the agent executor\n{response}\n")
     data = { "question": question, "answer":  response['output'], "history": response['history'] }          
     results.append(data)
     
     return results

def load_human_prompt():
    human_prompt = """{input}
    {agent_scratchpad}
    (reminder to always respond in a JSON blob)"""
    return human_prompt

def load_agent_system_prompt():
    agent_system_prompt = """Respond to the human as helpfully and accurately as possible. You have access to the following tools: {tools}    
Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).
Valid "action" values: "Final Answer" or {tool_names}
Provide only ONE action per $JSON_BLOB, as shown:"
```
{{
    "action": $TOOL_NAME,
    "action_input": $INPUT
}}
```
Follow this format:
Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond
Action:
```
{{
"action": "Final Answer",
"action_input": "Final response to human"
}}
Begin! Reminder to ALWAYS respond with a valid json blob of a single action.
Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation"""
    return agent_system_prompt

def system_prompt_load():
      system_prompt = """You are a weather expert. If the question is not about the weather, say: I don't know.
Based on the given instructions, answer the following question {question}."""
      return system_prompt

# ******************************************
# Execution
def main(args):
     
     agent_calling()

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)