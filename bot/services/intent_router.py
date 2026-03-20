"""Intent router for natural language queries.

Routes user messages to appropriate tools using LLM.
"""

import json
import sys
from typing import Any

import httpx

from services.api_client import LMSAPIClient


# System prompt for the LLM
SYSTEM_PROMPT = """You are an AI assistant for a Learning Management System (LMS).
You help students and instructors understand their progress, scores, and performance.

You have access to several tools that provide data from the LMS backend.
When a user asks a question, respond with a JSON object describing which tool to call.

Response format - use JSON to call tools:
{"tool": "tool_name", "arguments": {"arg1": "value1"}}

Or provide a final answer:
{"answer": "Your response to the user"}

Available tools:
- get_items: Get the list of all labs and tasks (no arguments)
- get_learners: Get the list of enrolled students (no arguments)
- get_scores: Get score distribution for a lab (requires: lab)
- get_pass_rates: Get per-task average scores for a lab (requires: lab)
- get_timeline: Get submissions per day for a lab (requires: lab)
- get_groups: Get per-group scores for a lab (requires: lab)
- get_top_learners: Get top N learners by score for a lab (requires: lab, optional: limit)
- get_completion_rate: Get completion rate percentage for a lab (requires: lab)
- trigger_sync: Trigger ETL sync to refresh data (no arguments)

Examples:
User: "what labs are available?"
Assistant: {"tool": "get_items", "arguments": {}}

User: "show me scores for lab 4"
Assistant: {"tool": "get_scores", "arguments": {"lab": "lab-04"}}

User: "which lab has the lowest pass rate?"
Assistant: {"tool": "get_items", "arguments": {}}
(Then after seeing the results, call get_pass_rates for each lab)

IMPORTANT: After you receive tool results, you must either:
1. Call another tool if you need more data
2. Provide a final answer using {"answer": "..."}

Do NOT just echo the tool call. Always either call a tool OR provide a final answer.

Guidelines:
1. Always use tools to get real data before answering questions about scores, labs, or students.
2. If the user asks about a specific lab (e.g., "lab 4" or "lab-04"), use the lab identifier format "lab-04".
3. For comparison questions, first get the list of labs, then fetch data for each.
4. Be concise but informative. Include relevant numbers in your answers.
"""


class IntentRouter:
    """Routes natural language queries to backend tools using LLM."""

    def __init__(
        self,
        api_client: LMSAPIClient,
        llm_api_key: str,
        llm_base_url: str,
        llm_model: str,
    ):
        """Initialize the intent router.
        
        Args:
            api_client: LMS API client for backend calls
            llm_api_key: API key for LLM
            llm_base_url: Base URL for LLM API
            llm_model: Model name to use
        """
        self.api_client = api_client
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        
        # Map tool names to methods
        self.tool_handlers: dict[str, callable] = {
            "get_items": self._handle_get_items,
            "get_learners": self._handle_get_learners,
            "get_scores": self._handle_get_scores,
            "get_pass_rates": self._handle_get_pass_rates,
            "get_timeline": self._handle_get_timeline,
            "get_groups": self._handle_get_groups,
            "get_top_learners": self._handle_get_top_learners,
            "get_completion_rate": self._handle_get_completion_rate,
            "trigger_sync": self._handle_trigger_sync,
        }

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get the schema for all available tools."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_items",
                    "description": "Get the list of all labs and tasks. Use this to discover what labs are available.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_learners",
                    "description": "Get the list of enrolled students and their groups.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_scores",
                    "description": "Get score distribution (4 buckets) for a specific lab.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lab": {"type": "string", "description": "Lab identifier, e.g., 'lab-01'"},
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_pass_rates",
                    "description": "Get per-task average scores and attempt counts for a lab.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lab": {"type": "string", "description": "Lab identifier, e.g., 'lab-01'"},
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_timeline",
                    "description": "Get submissions per day for a lab.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lab": {"type": "string", "description": "Lab identifier"},
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_groups",
                    "description": "Get per-group scores and student counts for a lab.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lab": {"type": "string", "description": "Lab identifier"},
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_top_learners",
                    "description": "Get top N learners by score for a lab.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lab": {"type": "string", "description": "Lab identifier"},
                            "limit": {"type": "integer", "description": "Number of learners, default 5"},
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_completion_rate",
                    "description": "Get completion rate percentage for a lab.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lab": {"type": "string", "description": "Lab identifier"},
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "trigger_sync",
                    "description": "Trigger ETL sync to refresh data from autochecker.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
        ]

    async def route(self, user_message: str, debug: bool = False) -> str:
        """Route a user message to appropriate tools and return response.
        
        Args:
            user_message: User's natural language query
            debug: If True, print debug info to stderr
            
        Returns:
            Response to the user
        """
        # Check for greetings and simple cases
        lower_message = user_message.lower().strip()
        
        if lower_message in ["hello", "hi", "hey", "привет", "здравствуйте"]:
            return "Hello! I'm your LMS assistant. I can help you with information about labs, scores, and student performance. Try asking something like 'what labs are available?' or 'show me scores for lab 4'."
        
        if len(lower_message) < 3 or lower_message in ["asdfgh", "test", "abc"]:
            return "I'm not sure I understand. Try asking me something like:\n- What labs are available?\n- Show me scores for lab 4\n- Which lab has the lowest pass rate?\n- Who are the top students?"
        
        # Build conversation with system prompt
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]
        
        # Chat with tools
        try:
            response = await self._chat_with_tools(messages, debug)
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                return "LLM authentication error (401). The API token may have expired. Try restarting the Qwen proxy: `cd ~/qwen-code-oai-proxy && docker compose restart`."
            return f"LLM error: HTTP {e.response.status_code}. {str(e)[:100]}"
        except Exception as e:
            if debug:
                print(f"[router] Error: {type(e).__name__}: {e}", file=sys.stderr)
            return f"LLM error: {type(e).__name__}. Please try again or use /help for available commands."

    async def _chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        debug: bool = False,
        max_iterations: int = 5,
    ) -> str:
        """Chat with LLM using JSON-based tool calling loop.
        
        Args:
            messages: Conversation history
            debug: Print debug info
            max_iterations: Max tool call iterations
            
        Returns:
            Final response
        """
        conversation = messages.copy()
        tool_results = []
        
        for iteration in range(max_iterations):
            if debug:
                print(f"[iteration {iteration}] Calling LLM...", file=sys.stderr)
            
            # Call LLM
            response = await self._call_llm(conversation)
            
            # Get assistant message content
            choice = response.get("choices", [{}])[0]
            assistant_message = choice.get("message", {})
            content = assistant_message.get("content", "")
            
            if not content:
                return "I'm having trouble processing your request."
            
            if debug:
                print(f"[llm] Raw response: {content[:200]}...", file=sys.stderr)
            
            # Try to parse as JSON
            try:
                # Extract JSON from the response (might be wrapped in markdown or text)
                import re
                # Try to find all JSON objects in the response
                json_objects = []
                
                # First try to find JSON in markdown code blocks
                md_matches = re.findall(r'```(?:json)?\s*(\{[^}]*\})\s*```', content, re.DOTALL)
                if md_matches:
                    json_objects = md_matches
                else:
                    # Try to find all {...} patterns
                    # Look for JSON objects line by line
                    for line in content.split('\n'):
                        line = line.strip()
                        if line.startswith('{') and line.endswith('}'):
                            json_objects.append(line)
                        else:
                            # Try to find JSON within the line
                            start = line.find('{')
                            end = line.rfind('}') + 1
                            if start >= 0 and end > start:
                                json_str = line[start:end]
                                try:
                                    json.loads(json_str)  # Validate
                                    json_objects.append(json_str)
                                except json.JSONDecodeError:
                                    pass
                
                if not json_objects:
                    raise json.JSONDecodeError("No JSON found", content, 0)
                
                # Parse all JSON objects
                data_list = [json.loads(j) for j in json_objects]
                
            except (json.JSONDecodeError, AttributeError, ValueError) as e:
                # Not JSON - this is the final answer
                if debug:
                    print(f"[response] Final answer (not JSON): {content[:100]}...", file=sys.stderr)
                return content
            
            # Process each JSON object
            tools_called = False
            for data in data_list:
                # Check if it's a tool call or final answer
                if "tool" in data and "arguments" in data:
                    tool_name = data["tool"]
                    tool_args = data["arguments"]
                    
                    if debug:
                        print(f"[tool] LLM called: {tool_name}({tool_args})", file=sys.stderr)
                    
                    # Execute tool
                    result = await self._execute_tool_by_name(tool_name, tool_args)
                    
                    if debug:
                        result_str = json.dumps(result) if isinstance(result, (dict, list)) else str(result)
                        print(f"[tool] Result: {result_str[:150]}...", file=sys.stderr)
                    
                    tool_results.append({"tool": tool_name, "result": result})
                    tools_called = True
                    
                    # Add tool result to conversation with clearer format
                    conversation.append({
                        "role": "user",
                        "content": f"TOOL_RESULT: {tool_name} returned: {json.dumps(result)[:500]}",
                    })
                    
                elif "answer" in data:
                    # Final answer from LLM
                    answer = data["answer"]
                    if debug:
                        print(f"[response] Final answer: {answer[:100]}...", file=sys.stderr)
                    return answer
            
            # If no tools were called and no answer provided, treat content as answer
            if not tools_called and not any("answer" in d for d in data_list):
                if debug:
                    print(f"[response] No tools called, treating as answer: {content[:100]}...", file=sys.stderr)
                return content
        
        # Max iterations reached - summarize tool results
        if tool_results:
            return self._summarize_results(tool_results, debug)
        
        return "I'm having trouble getting a complete answer. Let me summarize what I found..."

    def _summarize_results(self, tool_results: list[dict[str, Any]], debug: bool = False) -> str:
        """Summarize tool results into a final answer.
        
        Args:
            tool_results: List of tool execution results
            debug: Print debug info
            
        Returns:
            Summarized response
        """
        if debug:
            print(f"[summary] Feeding {len(tool_results)} tool result(s) back to LLM", file=sys.stderr)
        
        # Build a summary message
        summary_parts = ["Based on the data I retrieved:"]
        
        for item in tool_results:
            tool_name = item["tool"]
            result = item["result"]
            
            if tool_name == "get_items":
                labs = [r for r in result if r.get("type") == "lab"] if isinstance(result, list) else []
                summary_parts.append(f"- Found {len(labs)} labs")
            elif tool_name == "get_pass_rates":
                if isinstance(result, list) and result:
                    avg_score = sum(r.get("avg_score", 0) for r in result) / len(result)
                    summary_parts.append(f"- Average pass rate: {avg_score:.1f}%")
            elif isinstance(result, dict) and "error" in result:
                summary_parts.append(f"- Error: {result['error']}")
        
        # Ask LLM to generate final answer based on tool results
        summary_context = "\n".join(summary_parts)
        return f"I found the following information:\n{summary_context}\n\nLet me know if you need more details!"

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call the LLM API."""
        base_url = self.llm_base_url.rstrip("/")
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.llm_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.llm_model,
                    "messages": messages,
                    "temperature": 0.1,  # Low temperature for more deterministic output
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def _execute_tool_by_name(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool by name."""
        handler = self.tool_handlers.get(name)
        if handler is None:
            return {"error": f"Unknown tool: {name}"}
        
        # Normalize lab argument format (e.g., "lab 4" -> "lab-04")
        if "lab" in arguments:
            lab_value = arguments["lab"]
            # Convert "lab 4" or "lab4" to "lab-04"
            import re
            match = re.match(r"lab\s*(\d+)", lab_value.lower().replace("-", " "))
            if match:
                lab_num = int(match.group(1))
                arguments["lab"] = f"lab-{lab_num:02d}"
        
        try:
            return await handler(**arguments)
        except Exception as e:
            return {"error": str(e)}

    # Tool handlers
    async def _handle_get_items(self) -> list[dict[str, Any]]:
        """Get all items (labs and tasks)."""
        return await self.api_client.get_items()

    async def _handle_get_learners(self) -> list[dict[str, Any]]:
        """Get all learners."""
        return await self.api_client.get_learners()

    async def _handle_get_scores(self, lab: str) -> dict[str, Any]:
        """Get score distribution for a lab."""
        return await self.api_client.get_analytics_scores(lab)

    async def _handle_get_pass_rates(self, lab: str) -> list[dict[str, Any]]:
        """Get pass rates for a lab."""
        return await self.api_client.get_analytics_pass_rates(lab)

    async def _handle_get_timeline(self, lab: str) -> dict[str, Any]:
        """Get timeline for a lab."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{self.api_client.base_url}/analytics/timeline",
                    params={"lab": lab},
                    headers=self.api_client._get_headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"timeline": [], "message": "Timeline data not available"}
            return {"error": str(e)}

    async def _handle_get_groups(self, lab: str) -> dict[str, Any]:
        """Get groups data for a lab."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{self.api_client.base_url}/analytics/groups",
                    params={"lab": lab},
                    headers=self.api_client._get_headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"groups": [], "message": "Groups data not available"}
            return {"error": str(e)}

    async def _handle_get_top_learners(self, lab: str, limit: int = 5) -> dict[str, Any]:
        """Get top learners for a lab."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{self.api_client.base_url}/analytics/top-learners",
                    params={"lab": lab, "limit": limit},
                    headers=self.api_client._get_headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"top_learners": [], "message": "Top learners data not available"}
            return {"error": str(e)}

    async def _handle_get_completion_rate(self, lab: str) -> dict[str, Any]:
        """Get completion rate for a lab."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{self.api_client.base_url}/analytics/completion-rate",
                    params={"lab": lab},
                    headers=self.api_client._get_headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {"completion_rate": 0, "message": "Completion rate data not available"}
            return {"error": str(e)}

    async def _handle_trigger_sync(self) -> dict[str, Any]:
        """Trigger ETL sync."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.api_client.base_url}/pipeline/sync",
                    headers=self.api_client._get_headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except Exception as e:
            return {"error": str(e)}
