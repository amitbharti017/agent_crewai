from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, task, crew
from crewai_tools import SerperDevTool
import yaml
from dotenv import load_dotenv
import os
from pydantic import BaseModel, Field, HttpUrl
from typing import List,Optional


load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

llm = LLM(
    model = "gemini/gemini-2.0-flash",
    temperature=0.2,
    api_key = gemini_api_key
)

class SourceReference(BaseModel):
    title: str
    url: str

class WaterUsageResearchReport(BaseModel):
    title: str
    introduction: str
    key_findings: List[str]
    estimated_water_usage: List[str]
    factors_influencing_water_usage: List[str]
    mitigation_strategies: List[str]
    contradictory_information: str
    source_references: List[SourceReference]
    conclusion: str

#a decorator that allows defining modular Crew classes
@CrewBase
class ResearchCrew:
    """
    A crew for conducting research, summarizing findings, and fact-checking
    """
    #here ResearchCew is a modular Crew class
    agents_config = 'config/agents_2.yaml' #the class that loads agent definations from YAML
    tasks_config = 'config/tasks_2.yaml' #the class that loads task definations from YAML
    #we just specify the path for the YAML file. CrewAL will automatically laod it
    def __init__(self):
        self.search_tool = SerperDevTool()
        self.llm = llm

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config = self.agents_config["research_agent"],
            tools=[self.search_tool],
            llm=self.llm
        ) 

    @agent
    def summarization_agent(self) -> Agent:
        return Agent(
            config = self.agents_config['summarization_agent'],
            llm=self.llm
        )

    @agent
    def fact_checker_agent(self) -> Agent:
        return Agent(
            config = self.agents_config['fact_checker_agent'],
            tools=[self.search_tool],
            llm=self.llm
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config = self.tasks_config["research_task"],
        )

    @task
    def summarization_task(self) -> Task:
        return Task(
            config = self.tasks_config["summarization_task"]
        )

    @task
    def fact_checking_task(self) -> Task:
        return Task(
            config = self.tasks_config["fact_checking_task"],
            output_pydantic = WaterUsageResearchReport,
            output_file='output/2_agentic_class.md'
        )
    #crew defines AI workflow from crewAI
    @crew
    def crew(self) ->Crew:
        return Crew(
            agents=self.agents, #assigns all the dynamically created agents
            tasks=self.tasks, #assigns all the dynamically created tasks
            process=Process.sequential,
        )

research_crew = ResearchCrew()

result = research_crew.crew().kickoff(inputs={"topic": "Water usage in training chatgpt o3"})
print("\n Final verified summary:\n",result)