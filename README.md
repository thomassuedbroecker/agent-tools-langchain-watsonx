# Implementing an agent from scratch using an IBM Granite foundation model, LangChain, and watsonx.ai 

This repository implements a weather query example application of an agent, including function calling based on the IBM Developer Blog Post [Create a LangChain AI Agent in Python using watsonx](https://developer.ibm.com/tutorials/awb-create-langchain-ai-agent-python-watsonx/). Therefore, the endpoint https://wttr.in/ is used as the resource for the weather information.

Related blog post: [Implementing LangChain AI Agent with WatsonxLLM for a Weather Queries application](https://wp.me/paelj4-2jw)

## 0. Clone the repository

```sh
git clone https://github.com/thomassuedbroecker/temp-agent-tools-langchain-watsonx.git
cd agent-tools-langchain-watsonx
```

## 1. Environment Setup

### Step 1: Generate a virtual Python environment

```sh
cd code
python3 -m venv --upgrade-deps venv
source venv/bin/activate
```

### Step 2: Install the needed libraries

```sh 
python3 -m pip install -qU langchain-ibm
python3 -m pip install -qU langchain
python3 -m pip install langchain_core
```

### Step 3: Generate a `.env` file for the needed environment variables

```sh
cat env_example_template > .env
```

Insert the values for the two environment variables: 

* `WATSONX_PROJECT_ID=YOUR_WATSONX_PROJECT_ID`
* `IBMCLOUD_APIKEY=YOUR_KEY`

Content of the environment file.

```sh
export IBMCLOUD_APIKEY=YOUR_KEY
export IBMCLOUD_URL="https://iam.cloud.ibm.com/identity/token"

# Watsonx
export WATSONX_URL="https://eu-de.ml.cloud.ibm.com"
export WATSONX_VERSION=2023-05-29
export WATSONX_PROJECT_ID=YOUR_PROJECT_ID

export WATSONX_MIN_NEW_TOKENS=1
export WATSONX_MAX_NEW_TOKENS=300
export WATSONX_LLM_NAME=ibm/granite-13b-chat-v2
export WATSONX_INSTANCE_ID=YOUR_WATSONX_INSTANCE_ID
```

## 2. Execution

### Step 5: Run the example

```sh
cd code
bash example_agent_invocation.sh
```
* Output

```sh
> Entering new AgentExecutor chain...


Action:
...json
{
    "action": "weather_service",
    "action_input": {
        "cities": ["LA", "NY"]
    }
}
...
Weather service response:
...json
{
    "result": "LA",
    "cities": ["LA", "NY"]
}
...

[{'city': 'NY', 'temperature': '14'}, {'city': 'LA', 'temperature': '12'}]
 The hottest city is NY with a temperature or 14 degrees Celsius. In the city LA the temperature is 12 degrees Celsius.
Action:
...json
{
    "action": "Final Answer",
    "action_input": "The hottest city is NY with a temperature of 14 degrees Celsius."
}
...




> Finished chain.
16.a Response of the agent executor
{'input': 'Which city is hotter today: LA or NY?', 'history': '', 'output': 'The hottest city is NY with a temperature of 14 degrees Celsius.'}



> Entering new AgentExecutor chain...

{
    "action": "Final Answer",
    "action_input": "The temperature in Berlin is 10 degrees Celsius."
}



> Finished chain.
16.b Response of the agent executor
{'input': 'How hot is it today in Berlin?', 'history': 'Human: Which city is hotter today: LA or NY?\nAI: The hottest city is NY with a temperature of 14 degrees Celsius.', 'output': 'The temperature in Berlin is 10 degrees Celsius.'}



> Entering new AgentExecutor chain...

{
    "action": "Final Answer",
    "action_input": "Weather is a phenomenon that occurs in the atmosphere, primarily near the Earth's surface, and is characterized by a combination of temperature, humidity, precipitation, and wind."
}



> Finished chain.
16.c Response of the agent executor
{'input': 'What is the offical definition of the term weather?', 'history': 'Human: Which city is hotter today: LA or NY?\nAI: The hottest city is NY with a temperature of 14 degrees Celsius.\nHuman: How hot is it today in Berlin?\nAI: The temperature in Berlin is 10 degrees Celsius.', 'output': "Weather is a phenomenon that occurs in the atmosphere, primarily near the Earth's surface, and is characterized by a combination of temperature, humidity, precipitation, and wind."}



> Entering new AgentExecutor chain...

{
    "action": "Final Answer",
    "action_input": "The only way to win a soccer game is to score more goals than the opposing team."
}



> Finished chain.
16.d Response of the agent executor
{'input': 'How to win a soccer game?', 'history': "Human: Which city is hotter today: LA or NY?\nAI: The hottest city is NY with a temperature of 14 degrees Celsius.\nHuman: How hot is it today in Berlin?\nAI: The temperature in Berlin is 10 degrees Celsius.\nHuman: What is the offical definition of the term weather?\nAI: Weather is a phenomenon that occurs in the atmosphere, primarily near the Earth's surface, and is characterized by a combination of temperature, humidity, precipitation, and wind.", 'output': 'The only way to win a soccer game is to score more goals than the opposing team.'}
```
