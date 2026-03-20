"""LMS API client for the Telegram bot.

Handles HTTP requests to the LMS backend with Bearer token authentication.
"""

import httpx
from typing import Any


class LMSAPIClient:
    """Client for the LMS backend API."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 30.0):
        """Initialize the API client.
        
        Args:
            base_url: Base URL of the LMS backend (e.g., http://localhost:42002)
            api_key: API key for Bearer authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        """Get headers for API requests."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def get_items(self) -> list[dict[str, Any]]:
        """Fetch all items (labs and tasks) from the backend.
        
        Returns:
            List of items (labs and tasks)
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}/items/",
                headers=self._get_headers(),
            )
            resp.raise_for_status()
            return resp.json()

    async def get_learners(self) -> list[dict[str, Any]]:
        """Fetch all learners from the backend.
        
        Returns:
            List of learners
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}/learners/",
                headers=self._get_headers(),
            )
            resp.raise_for_status()
            return resp.json()

    async def get_analytics_pass_rates(self, lab: str) -> dict[str, Any]:
        """Fetch pass rates for a specific lab.
        
        Args:
            lab: Lab identifier (e.g., "lab-04")
            
        Returns:
            Pass rates analytics data
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}/analytics/pass-rates",
                params={"lab": lab},
                headers=self._get_headers(),
            )
            resp.raise_for_status()
            return resp.json()

    async def get_analytics_scores(self, lab: str) -> dict[str, Any]:
        """Fetch score distribution for a specific lab.
        
        Args:
            lab: Lab identifier (e.g., "lab-04")
            
        Returns:
            Score distribution data
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                f"{self.base_url}/analytics/scores",
                params={"lab": lab},
                headers=self._get_headers(),
            )
            resp.raise_for_status()
            return resp.json()

    async def health_check(self) -> dict[str, Any]:
        """Check if the backend is healthy by fetching items count.
        
        Returns:
            Dict with 'healthy' status and 'items_count'
            
        Raises:
            httpx.HTTPError: If the request fails
        """
        items = await self.get_items()
        return {"healthy": True, "items_count": len(items)}
