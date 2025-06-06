import re
from typing import Dict, Optional
import asyncio
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

class CrawlTools:
    """Tools for crawling and processing web content."""
    
    @staticmethod
    def is_url(text: str) -> bool:
        """Check if the input text is a URL."""
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)
        return bool(url_pattern.match(text))

    @staticmethod
    async def get_url_content(url: str, max_length: int = 5000) -> Dict[str, Optional[str]]:
        """
        Fetch content from a URL using crawl4ai.
        
        Args:
            url: The URL to fetch content from
            max_length: Maximum length of content to return
            
        Returns:
            dict: Dictionary containing the content and any error message
        """
        try:
            browser_config = BrowserConfig(
                headless=True,
                verbose=False
            )
            
            run_config = CrawlerRunConfig(
                cache_mode=CacheMode.BYPASS,
                output_formats=['markdown']
            )
            
            async with AsyncWebCrawler(config=browser_config) as crawler:
                result = await crawler.arun(url=url, config=run_config)
                
                if result.success:
                    content = result.markdown.fit_markdown[:max_length] if len(result.markdown.fit_markdown) > max_length else result.markdown.fit_markdown
                    return {
                        "content": content,
                        "error": None
                    }
                else:
                    return {
                        "content": None,
                        "error": result.error_message
                    }
                    
        except Exception as e:
            return {
                "content": None,
                "error": str(e)
            }

    @staticmethod
    async def process_potential_url(text: str) -> Dict[str, Optional[str]]:
        """
        Process text that might be a URL and return its content if it is.
        
        Args:
            text: Input text that might be a URL
            
        Returns:
            dict: Dictionary containing the processed result and any error message
        """
        if not CrawlTools.is_url(text):
            return {
                "content": None,
                "error": "Input is not a valid URL"
            }
            
        return await CrawlTools.get_url_content(text) 