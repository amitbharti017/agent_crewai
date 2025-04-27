from dotenv import load_dotenv
import os
from crewai import Agent, Task, Crew, LLM
from crewai import Crew, Process
from crewai_tools import SerperDevTool
import yaml

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

with open("config_1.yaml","r") as file:
    config = yaml.safe_load(file)
#setup llm model
llm = LLM(
    model = "gemini/gemini-2.0-flash",
    temperature=0.2,
    api_key = gemini_api_key
)
serper_dev_tool = SerperDevTool()


research_agent= Agent(
    role = config["agents"]["research_agent"]["role"],
    goal = config["agents"]["research_agent"]["goal"],
    backstory = config["agents"]["research_agent"]["backstory"],
    tools=[serper_dev_tool],
    llm =llm,
    verbose=True
)

research_task = Task(
    description = config["tasks"]["research_task"]["description"],
    agent = research_agent,
    expected_output = config["tasks"]["research_task"]["expected_output"]
)

summarization_agent = Agent(
    role=config["agents"]["summarization_agent"]["role"],
    goal=config["agents"]["summarization_agent"]["goal"],
    backstory=config["agents"]["summarization_agent"]["backstory"],
    llm=llm,
    verbose=True
)

fact_checker_agent = Agent(
    role=config["agents"]["fact_checker_agent"]["role"],
    goal=config["agents"]["fact_checker_agent"]["goal"],
    backstory=config["agents"]["fact_checker_agent"]["backstory"],
    tools=[serper_dev_tool],
    llm=llm,
    verbose=True
)

summarization_task = Task(
    description=config["tasks"]["summarization_task"]["description"],
    agent=summarization_agent,
    expected_output=config["tasks"]["summarization_task"]["expected_output"],
)

fact_checking_task = Task(
    description=config["tasks"]["fact_checking_task"]["description"],
    agent=fact_checker_agent,
    expected_output=config["tasks"]["fact_checking_task"]["expected_output"],
    output_file='output/1_sequencial_agent.md'
)

research_crew = Crew(
    agents = [research_agent,summarization_agent,fact_checker_agent],
    tasks = [research_task,summarization_task,fact_checking_task],
    process= Process.sequential,
    verbose=True
)

result = research_crew.kickoff(inputs={"topic": "Water usage in training chatgpt o3"})
print("\n Final verified summary:\n",result)

