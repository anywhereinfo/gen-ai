import yaml
import argparse
from typing import Dict, Any, List , Type
from pydantic import BaseModel, PrivateAttr
from dotenv import load_dotenv, find_dotenv
from langchain import LLMChain
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.tools import BaseTool
import tiktoken
import time
from langchain.schema import HumanMessage

print(load_dotenv(find_dotenv()))

class SemanticAnalysisInput(BaseModel):
    oas_content: str
    context: str = ""

class SemanticAnalysisTool(BaseTool): 
    name: str = "SemanticAnalysis"
    description: str = (
        "Performs semantic analysis on the OAS document using domain knowledge."
    )

    args_schema: Type[BaseModel] = SemanticAnalysisInput

    #Define any private attributes
    _llm: Any = PrivateAttr()
    _max_tokens_per_batch: int = PrivateAttr()


    def __init__(self, llm_instance, max_tokens_per_batch: int = 4000, **kwargs):
        super().__init__(**kwargs)
        self._llm= llm_instance
        self._max_tokens_per_batch = max_tokens_per_batch

    def _run(self, oas_content: str, context: str = "") -> str:
 
        try:
            # Ensure `oas_content` is parsed into a dictionary
            if isinstance(oas_content, str):
                oas = yaml.safe_load(oas_content)
                #print("Parsed YAML type:", type(oas))
                #print("Parsed YAML preview:", dict(list(oas.items())[:2]) if isinstance(oas, dict) else oas)
            else:
                oas = oas_content

            if not isinstance(oas, dict):
                return "Error: Parsed OAS content is not a valid dictionary structure."

        except yaml.YAMLError as e:
            return f"Error parsing OAS content: {str(e)}"

        # step 1: Build domain context
        domain_descriptions = self._gather_domain_descriptions(oas, context)
        print(f"Domain Descriptions:\n{domain_descriptions}")
        # step 2: Define Context Using LLM
        domain_context = self._define_domain_context(domain_descriptions)
        print(f"Domain:\n{domain_context}")
        #step 3:  Verify Object and Propery Names with Batching
        issues = self._verify_names_with_llm(domain_context, oas)
        print(f"Issues:\n{issues}")
        # Combine all issues  into a report
        report = "\n".join(issues)
        return report
    
    def _gather_domain_descriptions(self, oas: Dict[str, Any], context: str) -> str:
        domain_knowledge = []

        #API description from info
        api_description = oas.get("info", {}).get("description", "")
        if api_description:
            domain_knowledge.append(api_description)

        #Endpoint descriptions
        paths = oas.get("paths", {})
        for path, methods in paths.items():
            for method_name, method in methods.items():
                if not isinstance(method, dict):
                    continue
                method_description = method.get("description", "")
                if method_description:
                    domain_knowledge.append(method_description)

        
        # Add the provided context
        if context.strip():
            domain_knowledge.append(context.strip())

        # combine all descritpions into a single text
        domain_text = "\n".join(domain_knowledge)
        return domain_text
    


    def _define_domain_context(self, domain_descriptions: str) -> str:
        prompt = f"""
        Following descriptions are from info.description and various endpoint descriptions from the OpenAPI Specification file, in addition to user supplied additional context about the API.
        Determine the business domain, API objective and its capabilities. Your analysis will be used as input to determine API boundaries, whether the data model uses intuitive names and if endpoint design is reusable. 

        Descriptions:
        {domain_descriptions}

        Domain Context: <<Provide your analysis here>>
        """
        try:
            # Create a message object
            message = HumanMessage(content=prompt)
            # Get response from LLM
            response = self._llm.invoke([message])
            # Extract content from response
            domain_context = response.content if hasattr(response, 'content') else str(response)
            return domain_context.strip()
        except Exception as e:
            print(f"LLM Error details: {type(e).__name__}: {str(e)}")  # Add debug logging
            return f"Error generating domain context: {str(e)}"
        

    def _verify_names_with_llm(self, domain_context: str, oas: Dict[str, Any]) -> List[str]:
        issues = []
        components = oas.get("components", {})
        schemas = components.get("schemas", {})

        batch_prompts = []
        current_batch = ""
        current_token_count = 0

        for  schema_name, schema in schemas.items():
            properties = schema.get("properties", {})

            properties_str = "\n".join([f"- {prop_name}: {prop_info.get("type", "unknown")}" for prop_name, prop_info in properties.items()])
            schema_text = f"Object Name: {schema_name}\nProperties:\n{properties_str}\n\n"
            schema_token_count = self.estimate_token_count(schema_text)

            if current_token_count + schema_token_count > self._max_tokens_per_batch:
                batch_prompts.append(current_batch)
                current_batch = schema_text
                current_token_count = schema_token_count
            else:
                current_batch += schema_text
                current_token_count += schema_token_count

            # add the last batch
            if current_batch:
                batch_prompts.append(current_batch)

            for batch in batch_prompts:
                prompt = f"""
                   Domain Context:
                   {domain_context}

                    Review the following object names and their properties. For each object, determine if the object name and property names are intuitive and meaningful within the domain context. If not, provide suggestions for improvement.

                    {batch}

                    Analysis:
                """
                try:
                    response = self._llm(prompt).strip()
                    issues.append(response)
                    time.sleep(0.15)
                except Exception as e:
                    issues.append(f"Error during LLM analysis: {str(e)}")
        return issues

    def _estimate_token_count(self,text: str) -> int:
        encoding = tiktoken.encoding_for_model("gpt-4-turbo")
        return len(encoding.encode(text))

def main():
        parser = argparse.ArgumentParser(description="OAS Review Tool focusing on Semantic Analysis")  
        parser.add_argument("oas_file", type=str, help="Path to the OpenAPI Specification (OAS) file (YAML or JSON)")
        parser.add_argument("--context", type=str, help="Optional context information about the API", default="")
        args = parser.parse_args()

        try:
            with open(args.oas_file, "r") as file:
                oas_content = file.read()
        except Exception as e:
            print(f"Error reading file '{args.oas_file}': {str(e)}")
            return
    
        context = args.context

        llm = ChatOpenAI(temperature=0, model="gpt-4-turbo")
        semantic_tool = SemanticAnalysisTool(llm_instance=llm)
        tool_name = semantic_tool.name
        orchestrator_prompt = PromptTemplate(
                                            input_variables=["oas_doc", "context" ],
                                            template="""
You are an API Governance Agent. Given an OAS document and optional context information, perform a semantic analysis using the tool provided.

**Important Instructions:**

- **You must use the tools to answer the question. Do not answer directly without using a tool.**
- **Available tool:** SemanticAnalysis
- **When using the SemanticAnalysis tool, you must pass ONLY the OAS content as your Action Input, WITHOUT any markdown formatting or backticks.**
- **Keep all line breaks and YAML formatting intact**
- **When responding, follow this format:**

Thought: Describe your thought process.
Action: SemanticAnalysis
Action Input: The input to the action.

- **Only after receiving the Observation should you provide a final answer, following this format:**

Thought: [Summarize what was learned  from observation]
Final Answer: [Your final answer]


OAS Document:
{oas_doc}


Context:
{context}

{agent_scratchpad}

Remember: Your Action Input must be the exact OAS content provided above, without any additional formatting or markers. Keep all line breaks and YAML formatting intact
"""
)
        llm_chain = LLMChain(llm=llm, prompt=orchestrator_prompt)
        agent = ZeroShotAgent(
            llm_chain= llm_chain,
            tools= [semantic_tool],
            allowed_tools=[semantic_tool.name],
            verbose= True
        )
        agent_executor = AgentExecutor.from_agent_and_tools(
            agent= agent,
            tools= [semantic_tool],
            verbose= True,
            handle_parsing_errors = True
        )
        try:
            result = agent_executor.run(oas_doc=oas_content, context=context, tool_name=tool_name)
        except Exception as e:
            print(f"Error during agent execution: {str(e)}")
            return
        # Print the report
        print("\n--- Semantic Analysis Report ---\n")
        print(result)

if __name__ == "__main__":
    main()
         


    
