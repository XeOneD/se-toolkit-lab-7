"""LLM client for intent routing.

Handles communication with the LLM API for tool calling.
"""

import json
from typing import Any

import httpx


class LLMClient:
    """Client for LLM API with tool calling support."""

    def __init__(self, api_key: str, base_url: str, model: str, timeout: float = 60.0):
        """Initialize the LLM client.
        
        Args:
            api_key: API key for authentication
            base_url: Base URL of the LLM API (e.g., http://localhost:42005/v1)
            model: Model name to use (e.g., "qwen3-coder-flash")
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get the schema for all available tools.
        
        Returns:
            List of tool schemas in OpenAI function calling format
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_items",
                    "description": "Get the list of all labs and tasks. Use this to discover what labs are available.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_learners",
                    "description": "Get the list of enrolled students and their groups. Use this to find student information.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
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
                            "lab": {
                                "type": "string",
                                "description": "Lab identifier, e.g., 'lab-01', 'lab-04'",
                            },
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_pass_rates",
                    "description": "Get per-task average scores and attempt counts for a lab. Use this to see how students performed on each task.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lab": {
                                "type": "string",
                                "description": "Lab identifier, e.g., 'lab-01', 'lab-04'",
                            },
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_timeline",
                    "description": "Get submissions per day for a lab. Use this to see activity over time.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lab": {
                                "type": "string",
                                "description": "Lab identifier, e.g., 'lab-01', 'lab-04'",
                            },
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_groups",
                    "description": "Get per-group scores and student counts for a lab. Use this to compare group performance.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lab": {
                                "type": "string",
                                "description": "Lab identifier, e.g., 'lab-01', 'lab-04'",
                            },
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_top_learners",
                    "description": "Get top N learners by score for a lab. Use this to find the best performing students.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lab": {
                                "type": "string",
                                "description": "Lab identifier, e.g., 'lab-01', 'lab-04'",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of top learners to return, default 5",
                            },
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_completion_rate",
                    "description": "Get completion rate percentage for a lab. Use this to see what percentage of students completed the lab.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "lab": {
                                "type": "string",
                                "description": "Lab identifier, e.g., 'lab-01', 'lab-04'",
                            },
                        },
                        "required": ["lab"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "trigger_sync",
                    "description": "Trigger ETL sync to refresh data from autochecker. Use this when the user asks to update data.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
        ]

    async def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_iterations: int = 5,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Chat with the LLM using tool calling.
        
        Args:
            messages: Conversation history (list of {role, content} dicts)
            tools: List of tool schemas
            max_iterations: Maximum number of tool call iterations
            
        Returns:
            Tuple of (final_response, conversation_history)
        """
        conversation = messages.copy()
        
        for iteration in range(max_iterations):
            # Call the LLM
            response = await self._call_llm(conversation, tools)
            
            # Check if LLM wants to call tools
            tool_calls = response.get("tool_calls", [])
            
            if not tool_calls:
                # LLM provided final answer
                final_message = response.get("choices", [{}])[0].get("message", {})
                content = final_message.get("content", "I don't have enough information to answer.")
                return content, conversation
            
            # Execute tool calls
            conversation.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            })
            
            for tool_call in tool_calls:
                tool_result = await self._execute_tool(tool_call)
                
                # Add tool result to conversation
                conversation.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(tool_result),
                })
        
        # Max iterations reached, summarize what we have
        return "I'm having trouble getting a complete answer. Let me summarize what I found...", conversation

    async def _call_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Make a chat completion request to the LLM.
        
        Args:
            messages: Conversation history
            tools: List of tool schemas
            
        Returns:
            LLM response
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json={
                    "model": self.model,
                    "messages": messages,
                    "tools": tools,
                    "tool_choice": "auto",
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def _execute_tool(
        self,
        tool_call: dict[str, Any],
        tool_handler: callable | None = None,
    ) -> Any:
        """Execute a tool call.
        
        Args:
            tool_call: Tool call from LLM
            tool_handler: Optional callable to execute the tool
            
        Returns:
            Tool execution result
        """
        if tool_handler:
            return await tool_handler(tool_call)
        
        # Default: return error
        return {"error": "Tool execution not configured"}
