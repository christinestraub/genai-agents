import os
import asyncio
import aiohttp
import time
import json
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from dataclasses import dataclass, asdict
from typing import List, Optional, Union, Dict
from urllib.parse import urlparse
from xml.etree import ElementTree as ET
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.eleven_labs import ElevenLabsTools
from agno.tools.firecrawl import FirecrawlTools
from agno.agent import Agent, RunResponse
from agno.utils.audio import write_audio_to_file
from agno.utils.log import logger
import streamlit as st

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


@dataclass
class BlogObject:
    """Structured blog object with title, url, content, and fetch status."""
    title: str
    url: str
    content: str
    fetch_status: str


@dataclass
class PipelineStatus:
    """Detailed status tracking for each pipeline stage."""
    url: str
    fetch_status: str = "pending"  # pending, success, failed
    fetch_error: Optional[str] = None
    summary_status: str = "pending"  # pending, success, timeout, failed
    summary_error: Optional[str] = None
    tts_status: str = "pending"  # pending, success, failed
    tts_error: Optional[str] = None
    overall_status: str = "pending"  # pending, processing, completed, failed
    directory_path: Optional[str] = None
    blog_title: Optional[str] = None
    processing_start: Optional[str] = None
    processing_end: Optional[str] = None
    processing_duration: Optional[float] = None
    last_updated: str = ""

    def __post_init__(self):
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

    def update_timestamp(self):
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now().isoformat()

    def is_completed(self) -> bool:
        """Check if all pipeline stages are completed successfully."""
        return (self.fetch_status == "success" and 
                self.summary_status == "success" and 
                self.tts_status == "success")

    def is_failed(self) -> bool:
        """Check if any pipeline stage has failed."""
        return (self.fetch_status == "failed" or 
                self.summary_status in ["failed", "timeout"] or 
                self.tts_status == "failed")


@dataclass
class ProcessedBlog:
    """Represents a fully processed blog with all generated content."""
    blog_object: BlogObject
    directory_path: str
    summary: Optional[str] = None
    script: Optional[str] = None
    audio_file: Optional[str] = None
    processing_status: str = "pending"  # pending, processing, completed, failed
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    created_at: str = ""
    pipeline_status: Optional[PipelineStatus] = None

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class MasterIndex:
    """Master index of all processed blogs."""
    blogs: List[ProcessedBlog]
    created_at: str
    updated_at: str
    total_blogs: int = 0
    successful_blogs: int = 0
    failed_blogs: int = 0

    def __post_init__(self):
        self.total_blogs = len(self.blogs)
        self.successful_blogs = len([b for b in self.blogs if b.processing_status == "completed"])
        self.failed_blogs = len([b for b in self.blogs if b.processing_status == "failed"])


class BlogListFetcher:
    """
    Asynchronously fetches content from multiple blog URLs or RSS feeds.
    
    Features:
    - Parallel HTTP requests using aiohttp
    - Retry mechanism for failed requests
    - Graceful error handling with detailed logging
    - Support for both individual URLs and RSS feeds
    """
    
    def __init__(self, max_retries: int = 3, timeout: int = 30, retry_delay: float = 1.0):
        """
        Initialize the BlogListFetcher.
        
        Args:
            max_retries: Maximum number of retry attempts for failed requests
            timeout: Request timeout in seconds
            retry_delay: Delay between retry attempts in seconds
        """
        self.max_retries = max_retries
        self.timeout = timeout
        self.retry_delay = retry_delay
    
    async def fetch_single_blog(self, session: aiohttp.ClientSession, url: str) -> BlogObject:
        """
        Fetch a single blog post with retry logic.
        
        Args:
            session: aiohttp ClientSession
            url: Blog URL to fetch
            
        Returns:
            BlogObject with fetched content and status
        """
        attempt = 0
        last_error = None
        
        while attempt <= self.max_retries:
            try:
                logger.info(f"Fetching blog from {url} (attempt {attempt + 1})")
                
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                    if response.status == 200:
                        content = await response.text()
                        title = self._extract_title(content, url)
                        
                        # Clean up content - remove HTML tags if BeautifulSoup is available
                        clean_content = self._clean_content(content)
                        
                        logger.info(f"Successfully fetched blog from {url}")
                        return BlogObject(
                            title=title,
                            url=url,
                            content=clean_content,
                            fetch_status="success"
                        )
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP {response.status}"
                        )
                        
            except asyncio.TimeoutError as e:
                last_error = f"Timeout after {self.timeout}s"
                logger.warning(f"Timeout fetching {url} on attempt {attempt + 1}: {e}")
                
            except aiohttp.ClientResponseError as e:
                if e.status == 404:
                    logger.error(f"Blog not found (404) at {url}")
                    return BlogObject(
                        title="Not Found",
                        url=url,
                        content="",
                        fetch_status=f"error_404"
                    )
                last_error = f"HTTP {e.status}: {e.message}"
                logger.warning(f"HTTP error fetching {url} on attempt {attempt + 1}: {e}")
                
            except Exception as e:
                last_error = f"Unexpected error: {str(e)}"
                logger.warning(f"Error fetching {url} on attempt {attempt + 1}: {e}")
            
            attempt += 1
            if attempt <= self.max_retries:
                await asyncio.sleep(self.retry_delay * attempt)  # Exponential backoff
        
        # All retries failed
        logger.error(f"Failed to fetch {url} after {self.max_retries + 1} attempts. Last error: {last_error}")
        return BlogObject(
            title="Failed to Fetch",
            url=url,
            content="",
            fetch_status=f"error_max_retries: {last_error}"
        )
    
    def _extract_title(self, content: str, url: str) -> str:
        """Extract title from HTML content or use URL as fallback."""
        if BeautifulSoup:
            try:
                soup = BeautifulSoup(content, 'html.parser')
                title_tag = soup.find('title')
                if title_tag and title_tag.text.strip():
                    return title_tag.text.strip()
            except Exception as e:
                logger.warning(f"Error extracting title from {url}: {e}")
        
        # Fallback: try to extract from URL
        parsed_url = urlparse(url)
        path_parts = parsed_url.path.strip('/').split('/')
        if path_parts and path_parts[-1]:
            return path_parts[-1].replace('-', ' ').replace('_', ' ').title()
        
        return f"Blog from {parsed_url.netloc}"
    
    def _clean_content(self, content: str) -> str:
        """Clean HTML content to extract readable text."""
        if BeautifulSoup:
            try:
                soup = BeautifulSoup(content, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Get text and clean it up
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                return ' '.join(chunk for chunk in chunks if chunk)
                
            except Exception as e:
                logger.warning(f"Error cleaning content with BeautifulSoup: {e}")
        
        # Fallback: return raw content
        return content
    
    async def _parse_rss_feed(self, session: aiohttp.ClientSession, rss_url: str) -> List[str]:
        """
        Parse RSS feed and extract blog URLs.
        
        Args:
            session: aiohttp ClientSession
            rss_url: RSS feed URL
            
        Returns:
            List of blog URLs found in the RSS feed
        """
        try:
            logger.info(f"Parsing RSS feed: {rss_url}")
            async with session.get(rss_url, timeout=aiohttp.ClientTimeout(total=self.timeout)) as response:
                if response.status != 200:
                    logger.error(f"Failed to fetch RSS feed {rss_url}: HTTP {response.status}")
                    return []
                
                content = await response.text()
                root = ET.fromstring(content)
                
                urls = []
                
                # Try RSS 2.0 format
                for item in root.findall('.//item/link'):
                    if item.text:
                        urls.append(item.text.strip())
                
                # Try Atom format
                for link in root.findall('.//{http://www.w3.org/2005/Atom}link[@rel="alternate"]'):
                    href = link.get('href')
                    if href:
                        urls.append(href.strip())
                
                logger.info(f"Found {len(urls)} URLs in RSS feed {rss_url}")
                return urls
                
        except Exception as e:
            logger.error(f"Error parsing RSS feed {rss_url}: {e}")
            return []
    
    async def fetch_blogs(self, urls: Union[List[str], str]) -> List[BlogObject]:
        """
        Fetch multiple blogs asynchronously.
        
        Args:
            urls: List of blog URLs or a single RSS feed URL
            
        Returns:
            List of BlogObject instances with fetched content
        """
        if isinstance(urls, str):
            # Assume it's an RSS feed URL
            urls = [urls]
        
        connector = aiohttp.TCPConnector(limit=20, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            all_urls = []
            
            # Process each URL - could be individual blog URL or RSS feed
            for url in urls:
                if url.endswith('.xml') or 'rss' in url.lower() or 'feed' in url.lower():
                    # Treat as RSS feed
                    rss_urls = await self._parse_rss_feed(session, url)
                    all_urls.extend(rss_urls)
                else:
                    # Treat as individual blog URL
                    all_urls.append(url)
            
            if not all_urls:
                logger.warning("No valid URLs found to fetch")
                return []
            
            logger.info(f"Starting to fetch {len(all_urls)} blogs")
            start_time = time.time()
            
            # Create tasks for parallel execution
            tasks = [self.fetch_single_blog(session, url) for url in all_urls]
            
            # Execute all tasks concurrently
            blog_objects = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and log them
            results = []
            for i, result in enumerate(blog_objects):
                if isinstance(result, Exception):
                    logger.error(f"Task failed for URL {all_urls[i]}: {result}")
                    results.append(BlogObject(
                        title="Task Failed",
                        url=all_urls[i],
                        content="",
                        fetch_status=f"task_error: {str(result)}"
                    ))
                else:
                    results.append(result)
            
            end_time = time.time()
            logger.info(f"Completed fetching {len(results)} blogs in {end_time - start_time:.2f} seconds")
            
            # Log summary statistics
            success_count = sum(1 for blog in results if blog.fetch_status == "success")
            logger.info(f"Fetch summary: {success_count}/{len(results)} successful")
            
            return results


def slugify(text: str, max_length: int = 50) -> str:
    """
    Convert text to a filesystem-safe slug.
    
    Args:
        text: Text to slugify
        max_length: Maximum length of the slug
        
    Returns:
        Slugified text safe for use as directory name
    """
    # Convert to lowercase and replace spaces/special chars with dashes
    slug = re.sub(r'[^\w\s-]', '', text.lower())
    slug = re.sub(r'[-\s]+', '-', slug)
    slug = slug.strip('-')
    
    # Truncate to max length
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('-')
    
    # Ensure it's not empty
    if not slug:
        slug = "untitled"
    
    return slug


def validate_url(url: str) -> bool:
    """
    Validate if a string is a properly formatted URL.
    
    Args:
        url: URL string to validate
        
    Returns:
        True if URL is valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc]) and result.scheme in ['http', 'https']
    except Exception:
        return False


def load_urls_from_file(file_path: str) -> List[str]:
    """
    Load URLs from a text file, one URL per line.
    
    Args:
        file_path: Path to the text file containing URLs
        
    Returns:
        List of URLs found in the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        IOError: If there's an error reading the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            urls = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and not line.startswith('#'):  # Skip empty lines and comments
                    urls.append(line)
            return urls
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise IOError(f"Error reading file {file_path}: {e}")


def validate_and_clean_urls(urls: List[str], remove_duplicates: bool = True) -> tuple[List[str], List[str]]:
    """
    Validate URLs and optionally remove duplicates.
    
    Args:
        urls: List of URLs to validate
        remove_duplicates: Whether to remove duplicate URLs
        
    Returns:
        Tuple of (valid_urls, invalid_urls)
    """
    valid_urls = []
    invalid_urls = []
    seen_urls = set()
    
    for url in urls:
        url = url.strip()
        if not url:
            continue
            
        # Check for duplicates if removal is enabled
        if remove_duplicates and url in seen_urls:
            logger.info(f"Skipping duplicate URL: {url}")
            continue
            
        if validate_url(url):
            valid_urls.append(url)
            if remove_duplicates:
                seen_urls.add(url)
        else:
            invalid_urls.append(url)
            logger.warning(f"Invalid URL format: {url}")
    
    return valid_urls, invalid_urls


class BlogToPodcastPipeline:
    """
    Main pipeline for processing multiple blogs into podcasts.
    
    Handles the complete pipeline:
    1. Fetch blog content (using BlogListFetcher)
    2. Process each blog: summarization → script generation → TTS
    3. Save organized output files
    4. Maintain master index
    """
    
    def __init__(self, output_dir: str = "podcast_outputs", voice_id: str = "JBFqnCBsd6RMkjVDRZzb"):
        """
        Initialize the pipeline.
        
        Args:
            output_dir: Base directory for all podcast outputs
            voice_id: ElevenLabs voice ID to use
        """
        self.output_dir = Path(output_dir)
        self.voice_id = voice_id
        self.master_index_file = self.output_dir / "master_index.json"
        self.pipeline_status_file = self.output_dir / "pipeline_status.json"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
    
    def _create_blog_directory(self, blog: BlogObject) -> str:
        """Create a directory for a blog based on its title."""
        slug = slugify(blog.title)
        
        # Handle duplicate directory names
        base_dir = self.output_dir / slug
        counter = 1
        blog_dir = base_dir
        
        while blog_dir.exists():
            blog_dir = self.output_dir / f"{slug}-{counter}"
            counter += 1
        
        blog_dir.mkdir(exist_ok=True)
        return str(blog_dir)
    
    def _save_text_file(self, directory: str, filename: str, content: str) -> None:
        """Save text content to a file."""
        file_path = Path(directory) / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved {filename} to {file_path}")
    
    def _load_master_index(self) -> MasterIndex:
        """Load existing master index or create new one."""
        if self.master_index_file.exists():
            try:
                with open(self.master_index_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert dict back to dataclass instances
                    blogs = []
                    for blog_data in data.get('blogs', []):
                        blog_obj_data = blog_data['blog_object']
                        blog_obj = BlogObject(**blog_obj_data)
                        blog_data['blog_object'] = blog_obj
                        blogs.append(ProcessedBlog(**blog_data))
                    
                    return MasterIndex(
                        blogs=blogs,
                        created_at=data.get('created_at', datetime.now().isoformat()),
                        updated_at=datetime.now().isoformat()
                    )
            except Exception as e:
                logger.error(f"Error loading master index: {e}")
        
        # Create new master index
        return MasterIndex(
            blogs=[],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    
    def _save_master_index(self, master_index: MasterIndex) -> None:
        """Save master index to JSON file."""
        try:
            # Convert dataclasses to dict for JSON serialization
            data = asdict(master_index)
            data['updated_at'] = datetime.now().isoformat()
            
            with open(self.master_index_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved master index to {self.master_index_file}")
        except Exception as e:
            logger.error(f"Error saving master index: {e}")
    
    def _load_pipeline_status(self) -> Dict[str, PipelineStatus]:
        """Load existing pipeline status or create new empty dict."""
        if self.pipeline_status_file.exists():
            try:
                with open(self.pipeline_status_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    status_dict = {}
                    for url, status_data in data.items():
                        status_dict[url] = PipelineStatus(**status_data)
                    
                    return status_dict
            except Exception as e:
                logger.error(f"Error loading pipeline status: {e}")
        
        return {}
    
    def _save_pipeline_status(self, status_dict: Dict[str, PipelineStatus]) -> None:
        """Save pipeline status to JSON file."""
        try:
            # Convert dataclasses to dict for JSON serialization
            data = {}
            for url, status in status_dict.items():
                data[url] = asdict(status)
            
            with open(self.pipeline_status_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved pipeline status to {self.pipeline_status_file}")
        except Exception as e:
            logger.error(f"Error saving pipeline status: {e}")
    
    def _update_pipeline_status(self, status_dict: Dict[str, PipelineStatus], url: str, **updates) -> None:
        """Update pipeline status for a specific URL."""
        if url not in status_dict:
            status_dict[url] = PipelineStatus(url=url)
        
        # Update fields
        for field, value in updates.items():
            if hasattr(status_dict[url], field):
                setattr(status_dict[url], field, value)
        
        status_dict[url].update_timestamp()
        
        # Update overall status based on individual stages
        status = status_dict[url]
        if status.is_completed():
            status.overall_status = "completed"
        elif status.is_failed():
            status.overall_status = "failed"
        elif any([status.fetch_status == "success", 
                 status.summary_status in ["success", "processing"], 
                 status.tts_status in ["success", "processing"]]):
            status.overall_status = "processing"
    
    def _should_skip_url(self, url: str, status_dict: Dict[str, PipelineStatus], resume_mode: bool) -> bool:
        """Determine if a URL should be skipped during resume."""
        if not resume_mode:
            return False
        
        if url not in status_dict:
            return False
        
        status = status_dict[url]
        return status.is_completed()
    
    async def _process_single_blog(self, blog: BlogObject, agent: Agent, status_dict: Dict[str, PipelineStatus]) -> ProcessedBlog:
        """
        Process a single blog through the complete pipeline with detailed status tracking.
        
        Args:
            blog: Blog object to process
            agent: Configured Agno agent
            status_dict: Pipeline status dictionary for tracking
            
        Returns:
            ProcessedBlog with processing results
        """
        start_time = time.time()
        url = blog.url
        
        # Initialize pipeline status
        self._update_pipeline_status(status_dict, url, 
                                   processing_start=datetime.now().isoformat(),
                                   blog_title=blog.title,
                                   overall_status="processing")
        
        blog_dir = self._create_blog_directory(blog)
        self._update_pipeline_status(status_dict, url, directory_path=blog_dir)
        
        # Create pipeline status object for this blog
        pipeline_status = status_dict[url]
        
        processed_blog = ProcessedBlog(
            blog_object=blog,
            directory_path=blog_dir,
            processing_status="processing",
            pipeline_status=pipeline_status
        )
        
        try:
            logger.info(f"Processing blog: {blog.title}")
            
            # Step 0: Update fetch status based on blog fetch result
            if blog.fetch_status == "success":
                self._update_pipeline_status(status_dict, url, fetch_status="success")
                logger.info(f"✅ Fetch successful for: {blog.title}")
            else:
                self._update_pipeline_status(status_dict, url, 
                                           fetch_status="failed",
                                           fetch_error=f"Blog fetch failed: {blog.fetch_status}")
                raise Exception(f"Blog fetch failed: {blog.fetch_status}")
            
            # Step 1: Generate summary and script using the agent
            logger.info(f"🔄 Generating summary and script for: {blog.title}")
            
            try:
                prompt = f"""
                Please process this blog content:
                
                Title: {blog.title}
                URL: {blog.url}
                Content: {blog.content[:5000]}...  # Truncate if too long
                
                Create:
                1. A concise summary (maximum 300 words) that captures the main points
                2. A conversational podcast script (maximum 2000 characters) based on the summary
                
                Please format your response as:
                
                SUMMARY:
                [Your summary here]
                
                SCRIPT:
                [Your podcast script here]
                """
                
                # Set timeout for summary generation
                summary_start = time.time()
                response: RunResponse = agent.run(prompt)
                summary_duration = time.time() - summary_start
                
                # Check for timeout (adjust threshold as needed)
                if summary_duration > 120:  # 2 minutes timeout
                    self._update_pipeline_status(status_dict, url,
                                               summary_status="timeout",
                                               summary_error=f"Summary generation timed out after {summary_duration:.1f}s")
                    raise Exception(f"Summary generation timed out after {summary_duration:.1f}s")
                
                if not response.content:
                    raise Exception("No response content from agent")
                
                content = response.content
                
                # Parse summary and script from response
                summary_match = re.search(r'SUMMARY:\s*(.*?)(?=SCRIPT:|$)', content, re.DOTALL)
                script_match = re.search(r'SCRIPT:\s*(.*?)$', content, re.DOTALL)
                
                if not summary_match or not script_match:
                    raise Exception("Could not parse summary and script from agent response")
                
                summary = summary_match.group(1).strip()
                script = script_match.group(1).strip()
                
                # Save summary and script
                self._save_text_file(blog_dir, "summary.txt", summary)
                self._save_text_file(blog_dir, "script.txt", script)
                
                processed_blog.summary = summary
                processed_blog.script = script
                
                # Update summary status as successful
                self._update_pipeline_status(status_dict, url, summary_status="success")
                logger.info(f"✅ Summary generated successfully for: {blog.title}")
                
            except Exception as e:
                error_msg = str(e)
                if "timeout" in error_msg.lower():
                    self._update_pipeline_status(status_dict, url,
                                               summary_status="timeout",
                                               summary_error=error_msg)
                else:
                    self._update_pipeline_status(status_dict, url,
                                               summary_status="failed",
                                               summary_error=error_msg)
                raise Exception(f"Summary generation failed: {error_msg}")
            
            # Step 2: Generate audio from script
            logger.info(f"🔄 Generating audio for: {blog.title}")
            
            try:
                # Use the script for TTS generation
                tts_response: RunResponse = agent.run(f"Convert this text to audio: {script}")
                
                if tts_response.audio and len(tts_response.audio) > 0:
                    # Save audio file
                    audio_filename = Path(blog_dir) / "podcast.wav"
                    write_audio_to_file(
                        audio=tts_response.audio[0].base64_audio,
                        filename=str(audio_filename)
                    )
                    
                    processed_blog.audio_file = str(audio_filename)
                    
                    # Update TTS status as successful
                    self._update_pipeline_status(status_dict, url, tts_status="success")
                    logger.info(f"✅ Audio generated successfully for: {blog.title}")
                else:
                    raise Exception("No audio generated from TTS")
                    
            except Exception as e:
                error_msg = str(e)
                self._update_pipeline_status(status_dict, url,
                                           tts_status="failed",
                                           tts_error=error_msg)
                raise Exception(f"TTS generation failed: {error_msg}")
            
            # Mark as completed
            processing_time = time.time() - start_time
            processed_blog.processing_status = "completed"
            processed_blog.processing_time = processing_time
            
            # Update final pipeline status
            self._update_pipeline_status(status_dict, url,
                                       processing_end=datetime.now().isoformat(),
                                       processing_duration=processing_time,
                                       overall_status="completed")
            
            logger.info(f"🎉 Successfully processed {blog.title} in {processing_time:.2f}s")
            
        except Exception as e:
            processing_time = time.time() - start_time
            processed_blog.processing_status = "failed"
            processed_blog.processing_time = processing_time
            processed_blog.error_message = str(e)
            
            # Update final pipeline status
            self._update_pipeline_status(status_dict, url,
                                       processing_end=datetime.now().isoformat(),
                                       processing_duration=processing_time,
                                       overall_status="failed")
            
            logger.error(f"❌ Failed to process {blog.title}: {e}")
        
        # Save pipeline status after each blog
        self._save_pipeline_status(status_dict)
        
        return processed_blog
    
    async def process_blogs(self, blog_urls: Union[List[str], str], resume_mode: bool = False) -> MasterIndex:
        """
        Process multiple blogs through the complete pipeline with resume support.
        
        Args:
            blog_urls: List of blog URLs or RSS feed URL
            resume_mode: If True, skip already completed blogs
            
        Returns:
            Updated master index with processing results
        """
        logger.info("Starting batch blog processing pipeline")
        
        # Load existing pipeline status
        status_dict = self._load_pipeline_status()
        
        if resume_mode and status_dict:
            logger.info(f"Resume mode enabled. Found {len(status_dict)} existing status entries")
        
        # Step 1: Fetch all blog content
        fetcher = BlogListFetcher()
        blogs = await fetcher.fetch_blogs(blog_urls)
        
        if not blogs:
            logger.warning("No blogs fetched to process")
            return self._load_master_index()
        
        # Step 2: Filter blogs for resume mode
        blogs_to_process = []
        skipped_count = 0
        
        for blog in blogs:
            if self._should_skip_url(blog.url, status_dict, resume_mode):
                logger.info(f"⏭️ Skipping already completed: {blog.title}")
                skipped_count += 1
            else:
                blogs_to_process.append(blog)
        
        if resume_mode and skipped_count > 0:
            logger.info(f"📊 Resume summary: {skipped_count} blogs skipped, {len(blogs_to_process)} to process")
        
        if not blogs_to_process:
            logger.info("✅ All blogs already completed or no valid blogs to process")
            master_index = self._load_master_index()
            return master_index
        
        # Step 3: Create agent for processing
        agent = Agent(
            name="Blog to Podcast Agent",
            agent_id="blog_to_podcast_agent",
            model=OpenAIChat(id="gpt-4o"),
            tools=[
                ElevenLabsTools(
                    voice_id=self.voice_id,
                    model_id="eleven_multilingual_v2",
                    target_directory=str(self.output_dir),
                ),
            ],
            description="You are an AI agent that processes blog content into podcast scripts and audio.",
            instructions=[
                "When given blog content:",
                "1. Create a concise summary that captures the main points",
                "2. Generate a conversational podcast script based on the summary",
                "3. Keep the script under 2000 characters for TTS limits",
                "4. Make the content engaging and suitable for audio consumption",
            ],
            markdown=True,
            debug_mode=True,
        )
        
        # Step 4: Process blogs sequentially to avoid API rate limits
        processed_blogs = []
        for i, blog in enumerate(blogs_to_process):
            logger.info(f"Processing blog {i+1}/{len(blogs_to_process)}: {blog.title}")
            processed_blog = await self._process_single_blog(blog, agent, status_dict)
            processed_blogs.append(processed_blog)
            
            # Small delay between requests to avoid rate limits
            if i < len(blogs_to_process) - 1:
                await asyncio.sleep(2)
        
        # Step 5: Update master index
        master_index = self._load_master_index()
        master_index.blogs.extend(processed_blogs)
        master_index.updated_at = datetime.now().isoformat()
        
        # Recalculate stats
        master_index.__post_init__()
        
        # Save updated index
        self._save_master_index(master_index)
        
        # Final save of pipeline status
        self._save_pipeline_status(status_dict)
        
        logger.info(f"Batch processing completed. Processed {len(processed_blogs)} new blogs")
        logger.info(f"Pipeline status saved to: {self.pipeline_status_file}")
        logger.info(f"Overall success rate: {master_index.successful_blogs}/{master_index.total_blogs}")
        
        return master_index
    
    def generate_feed_index(self, format_type: str = "both") -> Dict[str, str]:
        """
        Generate index files for all successfully processed podcasts.
        
        Args:
            format_type: "json", "markdown", or "both"
            
        Returns:
            Dictionary with generated file paths
        """
        logger.info("Generating feed index for successfully processed podcasts")
        
        # Load existing data
        master_index = self._load_master_index()
        pipeline_status = self._load_pipeline_status()
        
        if not master_index.blogs:
            logger.warning("No processed blogs found in master index")
            return {}
        
        # Filter for successfully completed blogs
        successful_blogs = []
        for blog in master_index.blogs:
            url = blog.blog_object.url
            if url in pipeline_status and pipeline_status[url].is_completed():
                status = pipeline_status[url]
                
                # Extract one-line summary from the first sentence of full summary
                one_line_summary = ""
                if blog.summary:
                    # Get first sentence or first 150 characters
                    sentences = blog.summary.split('. ')
                    if sentences:
                        one_line_summary = sentences[0].strip()
                        if not one_line_summary.endswith('.'):
                            one_line_summary += '.'
                    # Fallback to truncated version
                    if len(one_line_summary) > 150:
                        one_line_summary = one_line_summary[:147] + "..."
                
                # Get relative path to audio file for web serving
                audio_path = None
                if blog.audio_file and Path(blog.audio_file).exists():
                    # Convert absolute path to relative from output directory
                    audio_path = Path(blog.audio_file).relative_to(self.output_dir)
                
                successful_blogs.append({
                    'title': blog.blog_object.title,
                    'url': blog.blog_object.url,
                    'audio_file': str(audio_path) if audio_path else None,
                    'audio_file_absolute': blog.audio_file,
                    'one_line_summary': one_line_summary,
                    'full_summary': blog.summary,
                    'directory_path': blog.directory_path,
                    'processing_duration': status.processing_duration,
                    'created_at': status.processing_end or status.last_updated,
                    'file_size_mb': self._get_file_size_mb(blog.audio_file) if blog.audio_file else 0
                })
        
        if not successful_blogs:
            logger.warning("No successfully completed blogs found")
            return {}
        
        # Sort by creation time (newest first)
        successful_blogs.sort(key=lambda x: x['created_at'], reverse=True)
        
        logger.info(f"Found {len(successful_blogs)} successfully processed podcasts")
        
        generated_files = {}
        
        # Generate JSON index
        if format_type in ["json", "both"]:
            json_file = self._generate_json_index(successful_blogs)
            generated_files["json"] = json_file
        
        # Generate Markdown index
        if format_type in ["markdown", "both"]:
            md_file = self._generate_markdown_index(successful_blogs)
            generated_files["markdown"] = md_file
        
        # Generate RSS feed
        if format_type in ["rss", "both"]:
            rss_file = self._generate_rss_feed(successful_blogs)
            generated_files["rss"] = rss_file
        
        logger.info(f"Generated index files: {list(generated_files.keys())}")
        return generated_files
    
    def _get_file_size_mb(self, file_path: str) -> float:
        """Get file size in MB."""
        try:
            return round(Path(file_path).stat().st_size / (1024 * 1024), 2)
        except:
            return 0.0
    
    def _generate_json_index(self, successful_blogs: List[Dict]) -> str:
        """Generate JSON index file."""
        json_file = self.output_dir / "index.json"
        
        index_data = {
            "metadata": {
                "title": "Blog to Podcast Feed",
                "description": "Generated podcasts from blog content",
                "generated_at": datetime.now().isoformat(),
                "total_podcasts": len(successful_blogs),
                "generator": "Blog to Podcast Agent"
            },
            "podcasts": successful_blogs
        }
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated JSON index: {json_file}")
        return str(json_file)
    
    def _generate_markdown_index(self, successful_blogs: List[Dict]) -> str:
        """Generate Markdown index file."""
        md_file = self.output_dir / "index.md"
        
        # Generate markdown content
        md_content = f"""# 🎙️ Blog to Podcast Feed

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
Total Podcasts: {len(successful_blogs)}

## 📋 Podcast Directory

"""
        
        for i, blog in enumerate(successful_blogs, 1):
            # Create markdown entry for each podcast
            audio_link = f"[🎧 Listen]({blog['audio_file']})" if blog['audio_file'] else "❌ Audio not available"
            file_size = f" ({blog['file_size_mb']} MB)" if blog['file_size_mb'] > 0 else ""
            
            md_content += f"""### {i}. {blog['title']}

**Original Post:** [{blog['url']}]({blog['url']})  
**Audio:** {audio_link}{file_size}  
**Summary:** {blog['one_line_summary']}  
**Generated:** {blog['created_at'][:10] if blog['created_at'] else 'Unknown'}  

---

"""
        
        # Add footer with statistics
        total_size = sum(blog['file_size_mb'] for blog in successful_blogs)
        avg_duration = sum(blog['processing_duration'] or 0 for blog in successful_blogs) / len(successful_blogs)
        
        md_content += f"""## 📊 Statistics

- **Total Audio Files:** {len(successful_blogs)}
- **Total Size:** {total_size:.2f} MB
- **Average Processing Time:** {avg_duration:.1f} seconds

---

*Generated by Blog to Podcast Agent*
"""
        
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(f"Generated Markdown index: {md_file}")
        return str(md_file)
    
    def _generate_rss_feed(self, successful_blogs: List[Dict]) -> str:
        """Generate RSS feed XML file."""
        rss_file = self.output_dir / "feed.xml"
        
        # Create RSS feed content
        rss_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
<channel>
    <title>Blog to Podcast Feed</title>
    <description>Automatically generated podcasts from blog content</description>
    <language>en-us</language>
    <lastBuildDate>{datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')}</lastBuildDate>
    <generator>Blog to Podcast Agent</generator>
    <itunes:category text="Technology"/>
    <itunes:explicit>false</itunes:explicit>
"""
        
        # Add each podcast as an RSS item
        for blog in successful_blogs:
            # Escape XML characters
            title = blog['title'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            summary = blog['one_line_summary'].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            
            # Convert created_at to RFC 2822 format for RSS
            try:
                created_dt = datetime.fromisoformat(blog['created_at'].replace('Z', '+00:00'))
                pub_date = created_dt.strftime('%a, %d %b %Y %H:%M:%S %z')
            except:
                pub_date = datetime.now().strftime('%a, %d %b %Y %H:%M:%S %z')
            
            # Create audio enclosure URL (you may need to adjust this for your setup)
            audio_url = f".//{blog['audio_file']}" if blog['audio_file'] else ""
            file_size_bytes = int(blog['file_size_mb'] * 1024 * 1024) if blog['file_size_mb'] > 0 else 0
            
            rss_content += f"""
    <item>
        <title>{title}</title>
        <description>{summary}</description>
        <link>{blog['url']}</link>
        <guid>{blog['url']}</guid>
        <pubDate>{pub_date}</pubDate>"""
            
            if audio_url and file_size_bytes > 0:
                rss_content += f"""
        <enclosure url="{audio_url}" type="audio/wav" length="{file_size_bytes}"/>"""
            
            rss_content += """
    </item>"""
        
        rss_content += """
</channel>
</rss>"""
        
        with open(rss_file, 'w', encoding='utf-8') as f:
            f.write(rss_content)
        
        logger.info(f"Generated RSS feed: {rss_file}")
        return str(rss_file)


# Streamlit Page Setup
st.set_page_config(page_title="📰 ➡️ 🎙️ Blog to Podcast Agent", page_icon="🎙️", layout="wide")
st.title("📰 ➡️ 🎙️ Blog to Podcast Agent")
st.subheader("Convert multiple blogs to podcasts in batch")

# Sidebar: API Keys
st.sidebar.header("🔑 API Keys")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
elevenlabs_api_key = st.sidebar.text_input("ElevenLabs API Key", type="password")

# Voice selection
voice_id = st.sidebar.selectbox(
    "ElevenLabs Voice",
    ["JBFqnCBsd6RMkjVDRZzb", "21m00Tcm4TlvDq8ikWAM", "AZnzlk1XvdvUeBnXmlld"],
    help="Select voice for podcast generation"
)

# Check if all keys are provided
keys_provided = all([openai_api_key, elevenlabs_api_key])

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Input URLs")
    
    # URL input methods
    input_method = st.radio(
        "Choose input method:",
        ["Individual URLs", "RSS Feed", "Bulk Text Input"]
    )
    
    urls = []
    
    if input_method == "Individual URLs":
        st.write("Enter blog URLs one by one:")
        
        # Dynamic URL inputs
        if 'url_count' not in st.session_state:
            st.session_state.url_count = 1
            
        for i in range(st.session_state.url_count):
            url = st.text_input(f"Blog URL {i+1}:", key=f"url_{i}")
            if url.strip():
                urls.append(url.strip())
        
        col_add, col_remove = st.columns(2)
        with col_add:
            if st.button("➕ Add URL"):
                st.session_state.url_count += 1
                st.rerun()
        
        with col_remove:
            if st.button("➖ Remove URL") and st.session_state.url_count > 1:
                st.session_state.url_count -= 1
                st.rerun()
    
    elif input_method == "RSS Feed":
        rss_url = st.text_input("RSS Feed URL:", placeholder="https://example.com/feed.xml")
        if rss_url.strip():
            urls = [rss_url.strip()]
    
    else:  # Bulk Text Input
        bulk_text = st.text_area(
            "Paste URLs (one per line):",
            height=150,
            placeholder="https://example1.com/blog/post1\nhttps://example2.com/blog/post2"
        )
        urls = [url.strip() for url in bulk_text.split('\n') if url.strip()]

with col2:
    st.header("📊 Processing Status")
    
    # Display current URLs
    if urls:
        st.write(f"**{len(urls)} URL(s) ready for processing:**")
        for i, url in enumerate(urls[:5], 1):  # Show first 5
            st.write(f"{i}. {url}")
        if len(urls) > 5:
            st.write(f"... and {len(urls) - 5} more")
    else:
        st.write("No URLs entered yet")

# Processing section
st.header("🚀 Process Blogs")

if not keys_provided:
    st.warning("⚠️ Please enter all required API keys in the sidebar to enable processing.")

# Process button
process_button = st.button(
    "🎙️ Process All Blogs", 
    disabled=not keys_provided or not urls,
    help="Process all blogs through the complete pipeline"
)

# Resume mode option
resume_mode = st.checkbox(
    "🔄 Resume Mode", 
    help="Skip already completed blogs and continue from where you left off"
)

if process_button:
    # Set API keys as environment variables
    os.environ["OPENAI_API_KEY"] = openai_api_key
    os.environ["ELEVEN_LABS_API_KEY"] = elevenlabs_api_key
    
    # Initialize pipeline
    pipeline = BlogToPodcastPipeline(voice_id=voice_id)
    
    # Create progress containers
    progress_container = st.container()
    results_container = st.container()
    
    with progress_container:
        st.write("🔄 **Processing Status:**")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process blogs asynchronously
        try:
            with st.spinner("Processing blogs through the complete pipeline..."):
                # Run the async pipeline
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    master_index = loop.run_until_complete(pipeline.process_blogs(urls, resume_mode=resume_mode))
                finally:
                    loop.close()
                
                progress_bar.progress(1.0)
                status_text.success("✅ Processing completed!")
            
            # Display results
            with results_container:
                st.header("📋 Processing Results")
                
                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Blogs", master_index.total_blogs)
                with col2:
                    st.metric("Successful", master_index.successful_blogs)
                with col3:
                    st.metric("Failed", master_index.failed_blogs)
                with col4:
                    success_rate = (master_index.successful_blogs / master_index.total_blogs * 100) if master_index.total_blogs > 0 else 0
                    st.metric("Success Rate", f"{success_rate:.1f}%")
                
                # Detailed results
                st.subheader("📁 Generated Content")
                
                for processed_blog in master_index.blogs[-len(urls):]:  # Show only current batch
                    with st.expander(f"🎙️ {processed_blog.blog_object.title} - {processed_blog.processing_status.title()}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**URL:** {processed_blog.blog_object.url}")
                            st.write(f"**Status:** {processed_blog.processing_status}")
                            if processed_blog.processing_time:
                                st.write(f"**Processing Time:** {processed_blog.processing_time:.2f}s")
                            if processed_blog.error_message:
                                st.error(f"**Error:** {processed_blog.error_message}")
                        
                        with col2:
                            st.write(f"**Directory:** {processed_blog.directory_path}")
                            
                            # Show generated files
                            if processed_blog.processing_status == "completed":
                                directory_path = Path(processed_blog.directory_path)
                                
                                # Audio file
                                if processed_blog.audio_file and Path(processed_blog.audio_file).exists():
                                    st.audio(processed_blog.audio_file)
                                    
                                    # Download button for audio
                                    with open(processed_blog.audio_file, 'rb') as f:
                                        st.download_button(
                                            "⬇️ Download Podcast",
                                            f.read(),
                                            file_name=f"{slugify(processed_blog.blog_object.title)}.wav",
                                            mime="audio/wav"
                                        )
                                
                                # Summary
                                if processed_blog.summary:
                                    with st.expander("📄 Summary"):
                                        st.write(processed_blog.summary)
                                
                                # Script
                                if processed_blog.script:
                                    with st.expander("📝 Script"):
                                        st.write(processed_blog.script)
                
                # Pipeline Status Details
                st.subheader("📊 Pipeline Status Details")
                
                # Load and display pipeline status
                pipeline_status = pipeline._load_pipeline_status()
                if pipeline_status:
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        fetch_success = len([s for s in pipeline_status.values() if s.fetch_status == "success"])
                        fetch_failed = len([s for s in pipeline_status.values() if s.fetch_status == "failed"])
                        st.metric("Fetch Stage", f"{fetch_success} / {fetch_success + fetch_failed}")
                    
                    with col2:
                        summary_success = len([s for s in pipeline_status.values() if s.summary_status == "success"])
                        summary_total = len([s for s in pipeline_status.values() if s.summary_status in ["success", "timeout", "failed"]])
                        st.metric("Summary Stage", f"{summary_success} / {summary_total}")
                    
                    with col3:
                        tts_success = len([s for s in pipeline_status.values() if s.tts_status == "success"])
                        tts_total = len([s for s in pipeline_status.values() if s.tts_status in ["success", "failed"]])
                        st.metric("TTS Stage", f"{tts_success} / {tts_total}")
                    
                    # Pipeline status download
                    pipeline_status_json = json.dumps({url: asdict(status) for url, status in pipeline_status.items()}, 
                                                    indent=2, ensure_ascii=False)
                    st.download_button(
                        "⬇️ Download Pipeline Status (JSON)",
                        pipeline_status_json,
                        file_name="pipeline_status.json",
                        mime="application/json"
                    )
                
                # Master index download
                st.subheader("📋 Master Index")
                master_index_json = json.dumps(asdict(master_index), indent=2, ensure_ascii=False)
                st.download_button(
                    "⬇️ Download Master Index (JSON)",
                    master_index_json,
                    file_name="master_index.json",
                    mime="application/json"
                )
                
                # Auto-generate index files
                if master_index.successful_blogs > 0:
                    st.subheader("📋 Feed Index Generation")
                    
                    # Show auto-generated indexes
                    try:
                        generated_files = pipeline.generate_feed_index("all")
                        if generated_files:
                            st.success("✅ Auto-generated feed index files:")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            # Download buttons for each format
                            for i, (file_type, file_path) in enumerate(generated_files.items()):
                                with [col1, col2, col3][i % 3]:
                                    if Path(file_path).exists():
                                        with open(file_path, 'r', encoding='utf-8') as f:
                                            content = f.read()
                                        
                                        file_ext = Path(file_path).suffix
                                        mime_type = {
                                            '.json': 'application/json',
                                            '.md': 'text/markdown',
                                            '.xml': 'application/xml'
                                        }.get(file_ext, 'text/plain')
                                        
                                        st.download_button(
                                            f"⬇️ Download {file_type.upper()}",
                                            content,
                                            file_name=Path(file_path).name,
                                            mime=mime_type
                                        )
                                        
                                        # Show preview for markdown
                                        if file_type == "markdown":
                                            with st.expander(f"📄 Preview {file_type.upper()}"):
                                                st.markdown(content)
                    except Exception as e:
                        st.warning(f"⚠️ Could not auto-generate indexes: {e}")
                
                st.success(f"🎉 Successfully processed {master_index.successful_blogs} out of {len(urls)} blogs!")
                
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            logger.error(f"Streamlit batch processing error: {e}")

# Generate Index Section
st.header("📋 Feed Index Generation")
st.write("Generate feed index files from existing processed blogs")

col1, col2 = st.columns(2)

with col1:
    index_format = st.selectbox(
        "Index Format",
        ["all", "json", "markdown", "rss"],
        help="Choose which index formats to generate"
    )

with col2:
    generate_index_button = st.button("🔄 Generate Feed Index")

if generate_index_button:
    try:
        pipeline = BlogToPodcastPipeline()
        generated_files = pipeline.generate_feed_index(index_format)
        
        if generated_files:
            st.success(f"✅ Successfully generated {len(generated_files)} index file(s):")
            
            # Create download buttons for each generated file
            cols = st.columns(len(generated_files))
            for i, (file_type, file_path) in enumerate(generated_files.items()):
                with cols[i]:
                    if Path(file_path).exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        file_ext = Path(file_path).suffix
                        mime_type = {
                            '.json': 'application/json',
                            '.md': 'text/markdown',
                            '.xml': 'application/xml'
                        }.get(file_ext, 'text/plain')
                        
                        st.download_button(
                            f"⬇️ {file_type.upper()}",
                            content,
                            file_name=Path(file_path).name,
                            mime=mime_type,
                            key=f"download_{file_type}"
                        )
            
            # Show preview of markdown if generated
            if "markdown" in generated_files and Path(generated_files["markdown"]).exists():
                with st.expander("📄 Markdown Preview"):
                    with open(generated_files["markdown"], 'r', encoding='utf-8') as f:
                        st.markdown(f.read())
        else:
            st.warning("⚠️ No successfully processed blogs found to generate indexes from")
            
    except Exception as e:
        st.error(f"❌ Failed to generate index: {e}")

# Display existing master index if available
if st.button("📊 View Existing Master Index"):
    try:
        pipeline = BlogToPodcastPipeline()
        master_index = pipeline._load_master_index()
        
        if master_index.blogs:
            st.header("📊 Existing Processed Blogs")
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processed", master_index.total_blogs)
            with col2:
                st.metric("Successful", master_index.successful_blogs)
            with col3:
                st.metric("Failed", master_index.failed_blogs)
            
            # List of blogs
            for blog in master_index.blogs:
                with st.expander(f"{blog.blog_object.title} - {blog.processing_status}"):
                    st.write(f"**URL:** {blog.blog_object.url}")
                    st.write(f"**Created:** {blog.created_at}")
                    st.write(f"**Directory:** {blog.directory_path}")
                    if blog.error_message:
                        st.error(f"**Error:** {blog.error_message}")
        else:
            st.info("No processed blogs found.")
            
    except Exception as e:
        st.error(f"Error loading master index: {e}")


# CLI Functions
def parse_cli_arguments() -> argparse.Namespace:
    """Parse command line arguments for CLI mode."""
    parser = argparse.ArgumentParser(
        description="Blog to Podcast Agent - Convert blogs to podcasts in batch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process URLs from a file
  python blog_to_podcast_agent.py --file blogs.txt
  
  # Process URLs from command line
  python blog_to_podcast_agent.py --urls "https://blog1.com/post1,https://blog2.com/post2"
  
  # Process with custom voice and output directory
  python blog_to_podcast_agent.py --file blogs.txt --voice 21m00Tcm4TlvDq8ikWAM --output-dir my_podcasts
  
  # Resume processing (skip already completed blogs)
  python blog_to_podcast_agent.py --file blogs.txt --resume
  
  # Generate feed index from existing processed blogs
  python blog_to_podcast_agent.py --generate-index --output-dir my_podcasts
  
  # Generate only JSON index
  python blog_to_podcast_agent.py --generate-index --index-format json
  
  # Generate all formats (JSON, Markdown, RSS)
  python blog_to_podcast_agent.py --generate-index --index-format all
  
  # Keep duplicates and show verbose output
  python blog_to_podcast_agent.py --urls "url1,url2" --keep-duplicates --verbose
  
  # Dry run with detailed status tracking
  python blog_to_podcast_agent.py --file blogs.txt --dry-run --verbose
        """
    )
    
    # Input source (mutually exclusive, but not required if generating index)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        '--file', '-f',
        type=str,
        help='Text file containing blog URLs (one per line)'
    )
    input_group.add_argument(
        '--urls', '-u',
        type=str,
        help='Comma-separated list of blog URLs'
    )
    
    # Processing options
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='podcast_outputs',
        help='Output directory for generated podcasts (default: podcast_outputs)'
    )
    parser.add_argument(
        '--voice',
        type=str,
        default='JBFqnCBsd6RMkjVDRZzb',
        choices=['JBFqnCBsd6RMkjVDRZzb', '21m00Tcm4TlvDq8ikWAM', 'AZnzlk1XvdvUeBnXmlld'],
        help='ElevenLabs voice ID to use (default: JBFqnCBsd6RMkjVDRZzb)'
    )
    parser.add_argument(
        '--keep-duplicates',
        action='store_true',
        help='Keep duplicate URLs instead of removing them'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate URLs without processing them'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume processing by skipping already completed blogs'
    )
    parser.add_argument(
        '--generate-index',
        action='store_true',
        help='Generate feed index files (JSON, Markdown, RSS) from existing processed blogs'
    )
    parser.add_argument(
        '--index-format',
        type=str,
        choices=['json', 'markdown', 'rss', 'both', 'all'],
        default='both',
        help='Format for generated index files (default: both)'
    )
    
    return parser.parse_args()


async def run_cli_mode() -> None:
    """Run the blog-to-podcast agent in CLI mode."""
    args = parse_cli_arguments()
    
    # Set up logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🎙️ Blog to Podcast Agent - CLI Mode")
    print("=" * 50)
    
    # Initialize pipeline first to check if we're just generating indexes
    pipeline = BlogToPodcastPipeline(output_dir=args.output_dir, voice_id=args.voice)
    
    # Handle index generation mode
    if args.generate_index:
        print("📋 Generating feed index from existing processed blogs...")
        try:
            format_type = "all" if args.index_format == "all" else args.index_format
            generated_files = pipeline.generate_feed_index(format_type)
            
            if generated_files:
                print(f"\n✅ Successfully generated index files:")
                for file_type, file_path in generated_files.items():
                    print(f"   📄 {file_type.upper()}: {file_path}")
                
                # Show preview of content
                master_index = pipeline._load_master_index()
                successful_count = len([b for b in master_index.blogs 
                                      if b.blog_object.url in pipeline._load_pipeline_status() 
                                      and pipeline._load_pipeline_status()[b.blog_object.url].is_completed()])
                print(f"\n📊 Index Summary:")
                print(f"   Total successful podcasts: {successful_count}")
                print(f"   Output directory: {args.output_dir}")
            else:
                print("⚠️ No successfully processed blogs found to index")
            
            return
        except Exception as e:
            print(f"❌ Failed to generate index: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    # For normal processing, check environment variables
    required_env_vars = ['OPENAI_API_KEY', 'ELEVEN_LABS_API_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Error: Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables before running the CLI.")
        print("Example:")
        print("  export OPENAI_API_KEY='your_openai_key'")
        print("  export ELEVEN_LABS_API_KEY='your_elevenlabs_key'")
        sys.exit(1)
    
    # Validate that URLs are provided for processing
    if not args.file and not args.urls:
        print("❌ Error: Either --file, --urls, or --generate-index must be provided")
        print("Use --help for usage examples")
        sys.exit(1)
    
    # Load URLs based on input method
    urls = []
    try:
        if args.file:
            print(f"📂 Loading URLs from file: {args.file}")
            urls = load_urls_from_file(args.file)
            print(f"   Found {len(urls)} URLs in file")
            
        elif args.urls:
            print("📝 Processing URLs from command line")
            urls = [url.strip() for url in args.urls.split(',')]
            print(f"   Found {len(urls)} URLs from command line")
    
    except (FileNotFoundError, IOError) as e:
        print(f"❌ Error loading URLs: {e}")
        sys.exit(1)
    
    if not urls:
        print("❌ No URLs provided")
        sys.exit(1)
    
    # Validate URLs
    print(f"🔍 Validating URLs...")
    remove_duplicates = not args.keep_duplicates
    valid_urls, invalid_urls = validate_and_clean_urls(urls, remove_duplicates)
    
    print(f"   ✅ Valid URLs: {len(valid_urls)}")
    if invalid_urls:
        print(f"   ❌ Invalid URLs: {len(invalid_urls)}")
        for invalid_url in invalid_urls:
            print(f"      - {invalid_url}")
    
    if remove_duplicates and len(valid_urls) < len(urls) - len(invalid_urls):
        duplicates_removed = len(urls) - len(invalid_urls) - len(valid_urls)
        print(f"   🗑️  Removed {duplicates_removed} duplicate(s)")
    
    if not valid_urls:
        print("❌ No valid URLs to process")
        sys.exit(1)
    
    # Show URLs to be processed
    print(f"\n📋 URLs to process:")
    for i, url in enumerate(valid_urls, 1):
        print(f"   {i}. {url}")
    
    # Dry run mode
    if args.dry_run:
        print(f"\n🧪 Dry run complete. {len(valid_urls)} URLs validated successfully.")
        print("Use without --dry-run to actually process the blogs.")
        return
    
    # Show pipeline configuration for normal processing
    print(f"\n🚀 Pipeline Configuration:")
    print(f"   Output directory: {args.output_dir}")
    print(f"   Voice ID: {args.voice}")
    print(f"   Resume mode: {'Enabled' if args.resume else 'Disabled'}")
    
    # Show existing status if resume mode
    if args.resume:
        existing_status = pipeline._load_pipeline_status()
        if existing_status:
            completed = len([s for s in existing_status.values() if s.is_completed()])
            failed = len([s for s in existing_status.values() if s.is_failed()])
            print(f"   Found existing status: {completed} completed, {failed} failed")
        else:
            print(f"   No existing status found")
    
    # Process blogs
    try:
        print(f"\n🔄 Starting blog processing...")
        start_time = time.time()
        
        master_index = await pipeline.process_blogs(valid_urls, resume_mode=args.resume)
        
        processing_time = time.time() - start_time
        
        # Display results
        print(f"\n📊 Processing Results:")
        print(f"   Total blogs: {master_index.total_blogs}")
        print(f"   Successful: {master_index.successful_blogs}")
        print(f"   Failed: {master_index.failed_blogs}")
        print(f"   Success rate: {(master_index.successful_blogs / master_index.total_blogs * 100):.1f}%")
        print(f"   Total time: {processing_time:.2f} seconds")
        
        # Show individual results
        print(f"\n📁 Generated Content:")
        for processed_blog in master_index.blogs[-len(valid_urls):]:  # Show only current batch
            status_icon = "✅" if processed_blog.processing_status == "completed" else "❌"
            print(f"   {status_icon} {processed_blog.blog_object.title}")
            print(f"      Directory: {processed_blog.directory_path}")
            if processed_blog.processing_status == "completed":
                print(f"      Files: podcast.wav, summary.txt, script.txt")
            elif processed_blog.error_message:
                print(f"      Error: {processed_blog.error_message}")
        
        # Show detailed pipeline status
        pipeline_status = pipeline._load_pipeline_status()
        if pipeline_status:
            print(f"\n📋 Detailed Pipeline Status:")
            
            # Group by status
            fetch_success = len([s for s in pipeline_status.values() if s.fetch_status == "success"])
            fetch_failed = len([s for s in pipeline_status.values() if s.fetch_status == "failed"])
            summary_success = len([s for s in pipeline_status.values() if s.summary_status == "success"])
            summary_timeout = len([s for s in pipeline_status.values() if s.summary_status == "timeout"])
            summary_failed = len([s for s in pipeline_status.values() if s.summary_status == "failed"])
            tts_success = len([s for s in pipeline_status.values() if s.tts_status == "success"])
            tts_failed = len([s for s in pipeline_status.values() if s.tts_status == "failed"])
            
            print(f"   Fetch stage:   {fetch_success} success, {fetch_failed} failed")
            print(f"   Summary stage: {summary_success} success, {summary_timeout} timeout, {summary_failed} failed")
            print(f"   TTS stage:     {tts_success} success, {tts_failed} failed")
        
        print(f"\n🎉 Processing completed!")
        print(f"📂 Master index saved to: {pipeline.master_index_file}")
        print(f"📊 Pipeline status saved to: {pipeline.pipeline_status_file}")
        
        # Auto-generate index files if any blogs were successfully processed
        if master_index.successful_blogs > 0:
            print(f"\n📋 Auto-generating feed index files...")
            try:
                generated_files = pipeline.generate_feed_index("all")
                if generated_files:
                    print(f"✅ Generated index files:")
                    for file_type, file_path in generated_files.items():
                        print(f"   📄 {file_type.upper()}: {file_path}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to generate index files: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def is_cli_mode() -> bool:
    """
    Determine if we're running in CLI mode or Streamlit mode.
    
    Returns:
        True if running in CLI mode, False if Streamlit mode
    """
    # Check if any CLI arguments are provided
    cli_args = ['--file', '-f', '--urls', '-u', '--help', '-h', '--generate-index']
    return any(arg in sys.argv for arg in cli_args)


# Main execution logic
if __name__ == "__main__" and is_cli_mode():
    # CLI Mode
    asyncio.run(run_cli_mode())
elif __name__ != "__main__":
    # Streamlit Mode (imported by streamlit run command)
    pass  # Streamlit code above will execute
else:
    # Default to showing help if no arguments provided
    print("🎙️ Blog to Podcast Agent")
    print("=" * 30)
    print("Usage:")
    print("  Streamlit UI: streamlit run blog_to_podcast_agent.py")
    print("  CLI mode:     python blog_to_podcast_agent.py --help")
    print("\nFor CLI help: python blog_to_podcast_agent.py --help")