import logging
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Type
import yaml
import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)

# Data Models
class APISpecificationInput(BaseModel):
    yaml_content: str = Field(..., description="OpenAPI 3.1 specification in YAML format")
    context: Optional[str] = Field(None, description="Additional context about the API and its usage")

# Endpoint Analysis Tool for compliance checks
class EndpointAnalysisTool(BaseTool):
    name: str = Field(default="endpoint_analysis", description="Analyzes API endpoints for governance compliance")
    description: str = Field(
        default="Analyzes endpoint URIs and their parameters in OpenAPI specifications for naming convention compliance. For URIs: validates lowercase, kebab-case, absence of file extensions and trailing slashes. For parameters: enforces snake_case in both query and path parameters. Returns detailed diagnostics with line numbers and recommendations.",
        description="A detailed description of what the endpoint analysis tool does and its validation capabilities"
    )
    args_schema: Type[BaseModel] = Field(default=APISpecificationInput)
    llm: Any = Field(default=None)
    analysis_chain: Any = Field(default=None)
    analysis_prompt: Any = Field(default=None)

    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.analysis_prompt = PromptTemplate(
            input_variables=["spec"],
            template = """You are an expert in OpenAPI specifications. Analyze the provided YAML for compliance issues and return a detailed diagnostic report. You must ONLY return a valid JSON object.

IMPORTANT: For each endpoint and parameter, check ONLY for actual violations. Do not report a violation unless you've confirmed it exists.

Compliance Rules and Their Specific Checks:
1. Lowercase Rule:
   - Check if any URI contains uppercase letters
   - ONLY report if uppercase letters are found
   - Example violation: "/Users" should be "/users"

2. File Extension Rule:
   - Check if URI ends with file extensions like .json, .xml, etc.
   - ONLY report if actual file extensions are found
   - Example violation: "/users.json" should be "/users"

3. Trailing Slash Rule:
   - Check if URI ends with "/"
   - ONLY report if an actual trailing slash exists
   - Example violation: "/users/" should be "/users"

4. Path Casing Rule (kebab-case):
   - Check if URI segments use camelCase or PascalCase instead of kebab-case
   - ONLY report if the path actually uses incorrect casing
   - Example violation: "/userProfile" should be "/user-profile"

5. Parameter Naming Rules (snake_case):
   - For query parameters: Check if they use camelCase/PascalCase instead of snake_case
   - For path parameters: Check if they use camelCase/PascalCase instead of snake_case
   - ONLY report parameters that actually violate the naming convention
   - Example violation: "userId" should be "user_id"

Return ONLY a JSON object with this exact structure:

{{
    "diagnostics": [
        {{
            "line_number": <number>,
            "content": "<exact problematic content>",
            "rule_violated": "<specific rule from above list>",
            "severity": "error",
            "message": "<clear description of the actual violation>",
            "recommendation": "<specific fix recommendation>"
        }}
    ],
    "uri_compliance": {{
        "lowercase_issues": [],
        "file_extension_issues": [],
        "trailing_slash_issues": [],
        "path_casing_issues": []
    }},
    "parameter_naming_compliance": {{
        "query_parameter_issues": [],
        "path_parameter_issues": []
    }},
    "summary": "<brief summary of actual issues found>"
}}


Before reporting any violation:
1. Verify the issue exists by checking against the specific rule
2. Only include actual violations in the diagnostics
3. Ensure the recommendation is specific to the actual violation found
4. Double-check that the line number corresponds to the actual violation

YAML to analyze:

{spec}"""
        )
        self.analysis_chain = LLMChain(llm=self.llm, prompt=self.analysis_prompt)

    def _run(self, yaml_content: str, context: Optional[str] = None) -> Dict[str, Any]:
        try:
            # Store line numbers for paths and parameters
            yaml_lines = yaml_content.splitlines()
            yaml_dict = yaml.safe_load(yaml_content)
            
            # Get analysis result
            result = self.analysis_chain.invoke({"spec": yaml_content})  # Pass original content to preserve line numbers
            
            # Handle the chain output properly
            if isinstance(result, dict) and 'text' in result:
                return json.loads(result['text'])
            elif isinstance(result, dict):
                return result
            elif isinstance(result, str):
                return json.loads(result)
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")
            
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON response: {str(e)}")
            return {
                "diagnostics": [],
                "uri_compliance": {
                    "lowercase_issues": [],
                    "file_extension_issues": [],
                    "trailing_slash_issues": [],
                    "path_casing_issues": []
                },
                "parameter_naming_compliance": {
                    "query_parameter_issues": [],
                    "path_parameter_issues": []
                },
                "summary": "Error analyzing specification"
            }
        except Exception as e:
            logging.error(f"Error in endpoint analysis: {str(e)}")
            raise






# Design Consistency Tool for checking design compliance
class DesignConsistencyTool(BaseTool):
    name: str = Field(default="design_consistency", description="Evaluates API design consistency and info section compliance")
    description: str = Field(
        default="Analyzes OpenAPI specifications for design quality and consistency. Validates OpenAPI version rules, information section completeness, API description quality, naming conventions across URIs/parameters/schemas, documentation completeness, schema design patterns, and error handling standards. Returns detailed assessment with quality scores and improvement recommendations.",
        description="A detailed description of the design consistency analyzer that evaluates OpenAPI specifications across version compliance, documentation quality, and architectural patterns"
    )
    
    args_schema: Type[BaseModel] = Field(default=APISpecificationInput)
    llm: Any = Field(default=None)
    consistency_chain: Any = Field(default=None)
    consistency_prompt: Any = Field(default=None)

    def __init__(self):
        super().__init__()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.consistency_prompt = PromptTemplate(
            input_variables=["spec"],
            template="""You are analyzing an OpenAPI Specification (OAS) document. Treat this YAML content as structured OpenAPI data, and interpret it according to OAS version-specific requirements. Beyond basic linting, evaluate design quality, documentation clarity, naming conventions, and schema design with consideration for the API’s business purpose and audience.

1. **OAS Version Awareness**: Check the OpenAPI version specified in the document and enforce relevant version-specific rules. For example:
   - OpenAPI 3.x should use camelCase for properties in schemas and snake_case for query/path parameters.

2. **Info Section Compliance**: Verify that `license`, `termsOfService`, and `contact` are omitted. Check `description` quality:
   - Assess if `description` clearly states the API’s purpose, objectives, business value, and target audience. Identify gaps in high-level information relevant to API consumers.

3. **API Description Quality**: Evaluate completeness of descriptions. Consider business domain, target audience, and clarity. Ensure that:
   - Purpose and objectives are clear and relevant to the business domain.
   - The business value is articulated, and the intended audience is identified.

4. **Naming Conventions**:
   - **URI Naming**: Ensure URIs are in kebab-case and intuitively represent business entities or actions.
   - **Parameter Naming**: Enforce snake_case for all query/path parameters.
   - **Schema and Object Property Naming**: Check that properties in object schemas use camelCase and reflect clear, intuitive names.

5. **Documentation Quality**:
   - Confirm that operation and parameter descriptions are complete and understandable.
   - Evaluate the quality and relevance of examples in the documentation. Examples should cover common, edge, and exceptional cases with realistic data.

6. **Schema Design and Reusability**:
   - Assess the design for modularity and reusability, ensuring components are logically organized for flexibility and scalability.
   - Validate consistency of data types and constraints. Check if common components are reused effectively across endpoints.

7. **Error Handling**: Verify consistent error response structure and error message patterns.

Return a **valid JSON** object without additional formatting or text:

{{
    "info_section_compliance": {{
        "license_absent": "<yes|no>",
        "terms_of_service_absent": "<yes|no>",
        "contact_absent": "<yes|no>",
        "description_compliance": {{
            "has_purpose_section": "<yes|no>",
            "format_compliance": "<percentage>",
            "improvements": ["<suggestions>"]
        }}
    }},
    "api_description_quality": {{
        "clarity_score": "<score>",
        "objectives_defined": "<yes|no>",
        "business_value_articulated": "<yes|no>",
        "target_audience_identified": "<yes|no>",
        "suggestions": ["<improvements>"]
    }},
    "naming_conventions": {{
        "uri_issues": ["<list of URI casing issues>"],
        "parameter_issues": ["<list of parameter naming issues>"],
        "schema_name_issues": ["<list of schema name issues>"],
        "property_naming_issues": ["<list of object property naming issues>"]
    }},
    "documentation_assessment": {{
        "operation_descriptions": "<coverage percentage>",
        "parameter_descriptions": "<coverage percentage>",
        "schema_descriptions": "<coverage percentage>",
        "example_quality_score": "<score>",
        "missing_use_cases": ["<missing cases>"]
    }},
    "schema_design_evaluation": {{
        "reusability_score": "<score>",
        "data_type_consistency": "<percentage>",
        "constraint_issues": ["<list of issues>"]
    }},
    "error_handling_assessment": {{
        "structure_compliance": "<percentage>",
        "status_code_usage": "<percentage>",
        "error_message_patterns": ["<pattern issues>"],
        "missing_error_scenarios": ["<scenarios>"]
    }},
    "summary": "<summary>"
}}
{spec}"""
        )
        self.consistency_chain = LLMChain(llm=self.llm, prompt=self.consistency_prompt)

    def _run(self, yaml_content: str, context: Optional[str] = None) -> Dict[str, Any]:
        try:
            result = self.consistency_chain.run(spec=yaml.dump(yaml.safe_load(yaml_content)))
            logging.info(f"Design Consistency Tool Output: {result}")
            return json.loads(result)
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON in Design Consistency Tool output.")
            raise Exception("Invalid JSON output from Design Consistency Tool.")
        except Exception as e:
            raise Exception(f"Error in design consistency analysis: {str(e)}")



# FastAPI Application Setup
app = FastAPI(
    title="API Governance Service",
    description="Service for reviewing OpenAPI specifications and providing governance recommendations",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Analyze endpoint for governance analysis
@app.post("/analyze/", response_model=Dict[str, Any])
async def analyze_api(
    file: UploadFile,
    context: Optional[str] = Form(None),
):
    try:
        # Read and validate YAML content
        content = await file.read()
        yaml_content = content.decode('utf-8')

        try:
            spec = yaml.safe_load(yaml_content)
            if not isinstance(spec, dict):
                raise HTTPException(status_code=400, detail="Invalid YAML: must be an object")
        except yaml.scanner.ScannerError as e:
            line_num = e.problem_mark.line + 1
            raise HTTPException(status_code=400, detail=f"YAML syntax error at line {line_num}")

        # Instantiate tools
        endpoint_tool = EndpointAnalysisTool()
        design_tool = DesignConsistencyTool()

        try:
            # Run endpoint analysis
            endpoint_analysis_result = endpoint_tool._run(yaml_content, context)
            
            # Run design consistency analysis
            design_consistency_result = design_tool._run(yaml_content, context)
            
            # Parse design consistency result if it's a string
            if isinstance(design_consistency_result, str):
                try:
                    design_consistency_result = json.loads(design_consistency_result)
                except json.JSONDecodeError:
                    logging.error("Failed to parse design consistency result as JSON")
                    design_consistency_result = {"error": "Failed to parse design consistency analysis"}

            # Combine results
            combined_result = {
                "file_name": file.filename,
                "review_date": datetime.now().isoformat(),
                "endpoint_analysis": endpoint_analysis_result,  # Use the entire result
                "design_consistency": design_consistency_result
            }
            
            return combined_result

        except Exception as analysis_error:
            logging.error(f"Analysis error: {str(analysis_error)}")
            return {
                "file_name": file.filename,
                "review_date": datetime.now().isoformat(),
                "error": f"Analysis failed: {str(analysis_error)}",
                "endpoint_analysis": None,
                "design_consistency": None
            }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
    finally:
        await file.close()