#!/usr/bin/env python3
"""
Simple JobYaari Direct HTML Scraper
Scrapes job data directly from HTML structure without LLM processing.
Based on the specific CSS selectors from JobYaari.com
"""

import os
import time
import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# LangChain Ollama embeddings (keeping it simple)
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain.embeddings import OllamaEmbeddings

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('jobyaari_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleJobYaariDirectScraper:
    """Simple direct HTML scraper for JobYaari.com"""
    
    def __init__(self):
        """Initialize the scraper"""
        
        # Configuration
        self.base_url = "https://www.jobyaari.com"
        self.categories_url = "https://www.jobyaari.com/categories"
        self.scrape_delay = int(os.getenv('SCRAPE_DELAY', '2'))
        self.max_retries = int(os.getenv('MAX_RETRIES', '3'))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '30'))
        
        # Setup session
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': os.getenv('USER_AGENT', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        # Initialize embeddings
        try:
            self.embeddings = OllamaEmbeddings(
                base_url=os.getenv('OLLAMA_HOST', 'http://localhost:11434'),
                model=os.getenv('EMBEDDING_MODEL', 'nomic-embed-text:latest')
            )
            logger.info("Ollama embeddings initialized")
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise
        
        # Initialize Qdrant
        try:
            self.qdrant_client = QdrantClient(
                host=os.getenv('QDRANT_HOST', 'localhost'),
                port=int(os.getenv('QDRANT_PORT', '6333'))
            )
            self.collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'jobyaari_jobs')
            self._setup_qdrant_collection()
            logger.info("Qdrant initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {e}")
            raise
        
        # Statistics
        self.stats = {
            'categories_processed': 0,
            'jobs_found': 0,
            'jobs_stored': 0,
            'jobs_skipped': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
        
        logger.info("JobYaari Direct Scraper initialized successfully")
    
    def _setup_qdrant_collection(self):
        """Setup Qdrant collection"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_exists = any(col.name == self.collection_name for col in collections.collections)
            
            if not collection_exists:
                # Get embedding dimension
                test_embedding = self.embeddings.embed_query("test")
                embedding_dim = len(test_embedding)
                
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=embedding_dim,
                        distance=rest_models.Distance.COSINE
                    )
                )
                logger.info(f"Created Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Failed to setup Qdrant collection: {e}")
            raise
    
    def make_request(self, url: str) -> Optional[requests.Response]:
        """Make HTTP request with retry logic"""
        for attempt in range(self.max_retries):
            try:
                logger.debug(f"Requesting: {url} (attempt {attempt + 1})")
                response = self.session.get(url, timeout=self.request_timeout)
                
                if response.status_code == 200:
                    return response
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"Status {response.status_code} for {url}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Request error: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
        
        logger.error(f"Failed to fetch {url}")
        self.stats['errors'] += 1
        return None
    
    def get_job_categories(self) -> List[Dict[str, str]]:
        """Get job categories from /categories page"""
        logger.info("Fetching job categories...")
        
        response = self.make_request(self.categories_url)
        if not response:
            logger.error("Failed to fetch categories page")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        categories = []
        
        # Find the third container-fluid div, then look for .row under it
        container_divs = soup.find_all('div', class_='container-fluid')
        if len(container_divs) >= 3:
            third_container = container_divs[2]
            row_div = third_container.find('div', class_='row')
            
            if row_div:
                # Find all category items
                category_items = row_div.find_all('div', class_='hs__item')
                logger.info(f"Found {len(category_items)} category items")
                
                for item in category_items:
                    try:
                        link = item.find('a')
                        if link and link.get('href'):
                            title_span = item.find('span', class_='hs__item__title')
                            if title_span:
                                category_name = title_span.get_text(strip=True)
                                category_url = urljoin(self.base_url, link.get('href'))
                                categories.append({
                                    'name': category_name,
                                    'url': category_url
                                })
                                logger.debug(f"Found category: {category_name} -> {category_url}")
                    except Exception as e:
                        logger.error(f"Error processing category item: {e}")
                        continue
        
        logger.info(f"Retrieved {len(categories)} job categories")
        return categories
    
    def extract_jobs_from_category(self, category_url: str, category_name: str) -> List[Dict[str, Any]]:
        """Extract jobs from a category page using direct HTML parsing"""
        logger.info(f"Extracting jobs from category: {category_name}")
        
        response = self.make_request(category_url)
        if not response:
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        jobs = []
        
        # Find div with class='row' and id='find'
        job_container = soup.find('div', {'class': 'row', 'id': 'find'})
        if not job_container:
            logger.warning(f"No job container found for category: {category_name}")
            return []
        
        # Find all job cards: div with class='col-md-6 col-lg-4'
        job_cards = job_container.find_all('div', class_='col-md-6 col-lg-4')
        logger.info(f"Found {len(job_cards)} job cards in category: {category_name}")
        
        for i, card in enumerate(job_cards):
            try:
                job_data = self._extract_job_data_from_card(card, category_name, category_url)
                if job_data:
                    jobs.append(job_data)
                    logger.debug(f"Extracted job {i+1}: {job_data.get('job_title', 'Unknown')}")
            except Exception as e:
                logger.error(f"Error extracting job card {i+1}: {e}")
                continue
        
        self.stats['jobs_found'] += len(jobs)
        logger.info(f"Successfully extracted {len(jobs)} jobs from category: {category_name}")
        return jobs
    
    def _extract_job_data_from_card(self, card, category_name: str, category_url: str) -> Optional[Dict[str, Any]]:
        """Extract job data from individual job card"""
        try:
            job_data = {
                'job_title': '',
                'department_company': '',
                'salary_range': '',
                'experience': '',
                'education': '',
                'state': '',
                'last_date': '',
                'posted_on': '',
                'category': category_name,
                'age_limit': '',
                'job_openings': '',
                'official_notice_link': '',
                'official_website': '',
                'important_notice': '',
                'scraped_at': datetime.now().isoformat(),
                'source': 'jobyaari.com',
                'category_url': category_url,
                'job_detail_url': '',
                'tags': []
            }


            
            # --- Extract posted_on from the card (before opening detail page) ---
            try:
                posting_info = card.find('div', class_='posting-info') or card.find('div', class_='post-info') or card.find('span', class_='post-date')
                if posting_info:
                    posting_text = posting_info.get_text(" ", strip=True)
                    # look for explicit date like 21/09/2025 or 21-09-2025
                    m = re.search(r'Posted(?:\s*(?:on|:)?\s*)(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', posting_text, re.I)
                    if not m:
                        m = re.search(r'(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})', posting_text)
                    if m:
                        job_data['posted_on'] = m.group(1).strip()
                    else:
                        # fallback: capture any text following 'Posted'
                        m2 = re.search(r'Posted\s*[:\-]?\s*(.+)', posting_text, re.I)
                        if m2:
                            job_data['posted_on'] = m2.group(1).strip()
            except Exception:
                logger.warning("non-fatal: continue without posted_on")
                # non-fatal: continue without posted_on
                pass

            
            # First, extract job detail URL from the card
            job_detail_url = self._extract_job_detail_url(card)
            if job_detail_url:
                job_data['job_detail_url'] = job_detail_url
                # Get detailed information from job detail page
                detailed_data = self._extract_detailed_job_info(job_detail_url)
                if detailed_data:
                    job_data.update(detailed_data)
            
            # Fallback: Extract basic info from card if detailed extraction fails
            if not job_data.get('job_title'):
                # Extract job title from ribbon
                ribbon = card.find('span', class_='ribbon1-shape')
                if ribbon:
                    title_span = ribbon.find('span')
                    if title_span:
                        job_data['job_title'] = title_span.get_text(strip=True)
            
            if not job_data.get('department_company'):
                # Extract company/department
                profession_span = card.find('span', class_='drop__profession')
                if profession_span:
                    job_data['department_company'] = profession_span.get_text(strip=True)
            
            if not job_data.get('salary_range'):
                # Extract salary range
                salary_span = card.find('span', class_='salary-price')
                if salary_span:
                    salary_text_spans = salary_span.find_all('span')
                    for span in salary_text_spans:
                        text = span.get_text(strip=True)
                        if text and text not in ['₹', '']:
                            job_data['salary_range'] = text
                            break
            
            if not job_data.get('experience'):
                # Extract experience
                exp_span = card.find('span', class_='drop__exp')
                if exp_span:
                    exp_text_spans = exp_span.find_all('span')
                    for span in exp_text_spans:
                        text = span.get_text(strip=True)
                        if text and 'fa-' not in text:
                            job_data['experience'] = text
                            break
            
            if not job_data.get('education'):
                # Extract education
                education_div = card.find('div', class_='salary')
                if education_div:
                    education_text = education_div.get_text(strip=True)
                    education_clean = re.sub(r'^\s*[^\w]*', '', education_text)
                    if education_clean:
                        job_data['education'] = education_clean
            
            if not job_data.get('last_date'):
                # Extract dates from posting-info
                posting_info = card.find('div', class_='posting-info')
                if posting_info:
                    post_items = posting_info.find_all('div', class_='post-item')
                    for item in post_items:
                        text = item.get_text(strip=True)
                        if 'Last Date:' in text:
                            job_data['last_date'] = text.replace('Last Date:', '').strip()
                        elif 'Posted' in text and not job_data.get('posted_on'):
                            job_data['posted_on'] = text.replace('Posted', '').strip()
            
            # Validate that we have essential data
            if job_data['job_title'] or job_data['department_company']:
                return job_data
            else:
                logger.warning("Job card missing essential data, skipping")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting job data: {e}")
            return None
    
    def _extract_job_detail_url(self, card) -> Optional[str]:
        """Extract job detail URL from job card"""
        try:
            # Look for clickable elements that might contain the job detail URL
            clickable_div = card.find('div', class_='drop-name')
            if clickable_div and clickable_div.get('onclick'):
                onclick_content = clickable_div.get('onclick')
                # Extract URL from onclick="location.href='URL'"
                url_match = re.search(r'location\.href="([^"]+)"', onclick_content)
                if url_match:
                    return url_match.group(1)
            
            # Alternative: look for direct links
            links = card.find_all('a', href=True)
            for link in links:
                href = link.get('href')
                if 'jobdetails' in href:
                    return urljoin(self.base_url, href)
            
            return None
        except Exception as e:
            logger.error(f"Error extracting job detail URL: {e}")
            return None
    
    def _extract_detailed_job_info(self, job_detail_url: str) -> Optional[Dict[str, Any]]:
        """Extract detailed job information from job detail page"""
        try:
            logger.debug(f"Fetching job details from: {job_detail_url}")

            response = self.make_request(job_detail_url)
            if not response:
                return None

            soup = BeautifulSoup(response.content, 'html.parser')
            detailed_data = {}

            # Find the main job detail container
            job_detail_container = soup.find('div', class_='col-xl-8 col-lg-8 col-12')
            if not job_detail_container:
                logger.warning("Job detail container not found")
                return None

            # Extract job title
            post_name = job_detail_container.find('h5', class_='post-name')
            if post_name:
                detailed_data['job_title'] = post_name.get_text(strip=True)

            # Extract department/company
            profession_span = job_detail_container.find('span', class_='drop__profession')
            if profession_span:
                detailed_data['department_company'] = profession_span.get_text(strip=True)

            # Extract details from the details section
            details_sec = job_detail_container.find('div', class_='details-sec')
            if details_sec:
                detail_items = details_sec.find_all('li')
                for item in detail_items:
                    label = item.find('h5', class_='label-head')
                    if label:
                        label_text = label.get_text(strip=True).lower()
                        value_div = item.find('div', class_='job-post-info-text')
                        if value_div:
                            value = value_div.get_text(strip=True)
                            # Remove the label from the value
                            value = value.replace(label.get_text(strip=True), '').strip()

                            if 'experience' in label_text:
                                detailed_data['experience'] = value
                            elif 'salary' in label_text:
                                detailed_data['salary_range'] = value
                            elif 'qualification' in label_text:
                                detailed_data['education'] = value
                            elif 'last date' in label_text:
                                detailed_data['last_date'] = value.replace('Last Date:', '').strip()

            # ---------------------------
            # Robust Location extraction
            # ---------------------------
            # Try CSS selector first (handles multi-class match), then fallback to manual class-list check.
            location_div = soup.select_one('.location.job-post-det') or soup.find(
                lambda tag: tag.name == 'div' and tag.get('class') and
                            'location' in tag.get('class') and 'job-post-det' in tag.get('class')
            )
            if location_div:
                # prefer the specific p tag, but fall back to any <p> if markup varies
                location_p = location_div.find('p', class_='location-list') or location_div.find('p')
                if location_p:
                    location_text = location_p.get_text(separator=' ', strip=True)
                    # Remove leading icon characters and common prefixes like 'Location:'
                    location_text = re.sub(r'^[^\w\d]+', '', location_text).strip()
                    location_text = re.sub(r'^(location[:\s-]*)', '', location_text, flags=re.I).strip()
                    if location_text:
                        # normalize to lowercase (change if you prefer title-case)
                        detailed_data['state'] = location_text.title()

            # Extract notification details
            job_detail_detail = job_detail_container.find('div', class_='job-detail-detail')
            if job_detail_detail:
                detail_list_items = job_detail_detail.find_all('li')
                for item in detail_list_items:
                    text_div = item.find('div', class_='text')
                    if text_div:
                        text_content = text_div.get_text(strip=True).lower()
                        details_div = item.find('div', class_='details')

                        if 'age limit' in text_content:
                            age_location = item.find('div', class_='job-location')
                            if age_location:
                                detailed_data['age_limit'] = age_location.get_text(strip=True)
                        elif 'job openings' in text_content:
                            openings_div = None
                            if details_div:
                                # Prefer direct child divs (first is label, second is value)
                                child_divs = details_div.find_all('div', recursive=False)
                                if len(child_divs) >= 2:
                                    openings_div = child_divs[1]
                                else:
                                    # Fallback: find first child div without a class (likely the value)
                                    openings_div = details_div.find('div', class_=False)
                                    # Last-resort fallback: previous fragile behavior
                                    if not openings_div:
                                        openings_div = details_div.find_next('div')
                            if openings_div:
                                detailed_data['job_openings'] = openings_div.get_text(strip=True)
                        elif 'official notice' in text_content:
                            link = item.find('a', class_='updated-link')
                            if link and link.get('href'):
                                detailed_data['official_notice_link'] = urljoin(self.base_url, link.get('href'))
                        elif 'job website' in text_content or 'website' in text_content:
                            link = item.find('a', class_='updated-link')
                            if link and link.get('href'):
                                detailed_data['official_website'] = link.get('href')

            # Extract important notice/job details
            rich_text = job_detail_container.find('div', class_='rich-text w-richtext')
            if rich_text:
                # Get all text content and clean it
                notice_text = rich_text.get_text(separator='\n', strip=True)
                if notice_text:
                    detailed_data['important_notice'] = notice_text

            # ---------------------------
            # Extract specialization tags (from the right sidebar)
            # ---------------------------
            try:
                # Prefer searching within an aside with both classes 'side-bar' and 'right'
                aside = soup.select_one('aside.side-bar.right') or soup.find(
                    lambda tag: tag.name == 'aside' and tag.get('class') and
                                'side-bar' in tag.get('class') and 'right' in tag.get('class')
                )
                tags_list: List[str] = []
                if aside:
                    # Find the job-post-container inside the aside that contains job-post-category links
                    tag_container = aside.select_one('div.job-post-container') or aside.find('div', class_='job-post-container')
                    if tag_container:
                        anchors = tag_container.select('a.job-post-category') or tag_container.find_all('a', class_='job-post-category')
                        for a in anchors:
                            try:
                                t = a.get_text(strip=True)
                                if t:
                                    tags_list.append(t)
                            except Exception:
                                continue

                # final fallback: global search for job-post-category links (very safe)
                if not tags_list:
                    anchors = soup.select('a.job-post-category')
                    for a in anchors:
                        try:
                            t = a.get_text(strip=True)
                            if t:
                                tags_list.append(t)
                        except Exception:
                            continue

                if tags_list:
                    # normalize tags (strip and keep original casing) - change to .lower() if you prefer lowercase
                    detailed_data['tags'] = [t.strip() for t in tags_list]
            except Exception as e:
                logger.debug(f"Failed to extract specialization tags: {e}")

            logger.debug(f"Extracted detailed job info: {detailed_data.get('job_title', 'Unknown')}")
            return detailed_data

        except Exception as e:
            logger.error(f"Error extracting detailed job info from {job_detail_url}: {e}")
            return None

    
    def prepare_text_for_embedding(self, job_data: Dict[str, Any]) -> str:
        """Prepare job data as text for embedding"""
        text_parts = []
        
        if job_data.get('job_title'):
            text_parts.append(f"Job Title: {job_data['job_title']}")
        
        if job_data.get('department_company'):
            text_parts.append(f"Company: {job_data['department_company']}")
        
        if job_data.get('category'):
            text_parts.append(f"Category: {job_data['category']}")
        
        if job_data.get('state'):
            text_parts.append(f"Location: {job_data['state']}")
        
        if job_data.get('experience'):
            text_parts.append(f"Experience: {job_data['experience']}")
        
        if job_data.get('education'):
            text_parts.append(f"Education: {job_data['education']}")
        
        if job_data.get('salary_range'):
            text_parts.append(f"Salary: {job_data['salary_range']}")
        
        if job_data.get('age_limit'):
            text_parts.append(f"Age Limit: {job_data['age_limit']}")
        
        if job_data.get('job_openings'):
            text_parts.append(f"Openings: {job_data['job_openings']}")
        
        if job_data.get('important_notice'):
            # Include first 200 characters of important notice
            notice = job_data['important_notice'][:200]
            text_parts.append(f"Notice: {notice}")

        # show tags if present
        if job_data.get('tags'):
            # tags is a list; join into a small readable string for embedding
            tags_txt = ', '.join(job_data['tags'])
            text_parts.append(f"Tags: {tags_txt}")
        
        return ' | '.join(text_parts)
    
    def generate_job_id(self, job_data: Dict[str, Any]) -> str:
        """Generate unique ID for job"""
        unique_string = f"{job_data.get('job_title', '')}{job_data.get('department_company', '')}{job_data.get('category', '')}"
        return hashlib.md5(unique_string.encode()).hexdigest()
    
    def store_job_in_vector_db(self, job_data: Dict[str, Any]) -> bool:
        """Store job in Qdrant vector database"""
        try:
            # Generate job ID
            job_id = self.generate_job_id(job_data)
            
            # Check if job already exists
            try:
                existing = self.qdrant_client.retrieve(
                    collection_name=self.collection_name,
                    ids=[job_id]
                )
                if existing:
                    logger.debug(f"Job already exists, skipping: {job_data.get('job_title', 'Unknown')}")
                    self.stats['jobs_skipped'] += 1
                    return False
            except Exception:
                pass  # Job doesn't exist, continue

            # --- Prepare compact metadata for easier filtering/search and include it in payload ---
            metadata = {
                "id": job_id,
                "job_title": job_data.get("job_title"),
                "company": job_data.get("department_company"),
                "category": job_data.get("category"),
                "state": job_data.get("state"),
                "posted_on": job_data.get("posted_on"),
                "last_date": job_data.get("last_date"),
                "salary_range": job_data.get("salary_range"),
                "experience": job_data.get("experience"),
                "education": job_data.get("education"),
                "age_limit": job_data.get("age_limit"),
                "job_openings": job_data.get("job_openings"),
                "tags": job_data.get("tags", []),
                "job_detail_url": job_data.get("job_detail_url"),
                "scraped_at": job_data.get("scraped_at"),
                "source": job_data.get("source"),
            }

            # Build final payload: keep full job_data but add a compact metadata key
            payload = dict(job_data)  # shallow copy to avoid mutating original
            payload['metadata'] = metadata
            
            # Prepare text for embedding
            text = self.prepare_text_for_embedding(job_data)
            payload['page_content'] = text 

            if len(text.strip()) < 10:
                logger.warning("Job text too short for embedding")
                return False
            
            # Generate embedding
            embedding = self.embeddings.embed_query(text)
            if not embedding:
                logger.error("Failed to generate embedding")
                return False
            
            # Store in Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[rest_models.PointStruct(
                    id=job_id,
                    vector=embedding,
                    payload=payload
                )]
            )
            
            logger.debug(f"Stored job: {job_data.get('job_title', 'Unknown')}")
            self.stats['jobs_stored'] += 1


            # retrieved_points = self.qdrant_client.retrieve(
            #         collection_name=self.collection_name,
            #         ids=[job_id],
            #         with_payload=True,
            #         with_vectors=True  # Set to True if you want vectors too
            # )
            # logger.info("****Shaim stored Point data: ", retrieved_points)   # should include 'page_content' and other fields
            return True
            
        except Exception as e:
            logger.error(f"Failed to store job in vector DB: {e}")
            self.stats['errors'] += 1
            return False
    
    def scrape_category(self, category: Dict[str, str]) -> int:
        """Scrape all jobs from a category"""
        category_name = category['name']
        category_url = category['url']
        
        logger.info(f"Processing category: {category_name}")
        
        # Extract jobs from category
        jobs = self.extract_jobs_from_category(category_url, category_name)
        if not jobs:
            logger.warning(f"No jobs found in category: {category_name}")
            return 0
        
        # Process each job
        processed_count = 0
        for i, job_data in enumerate(jobs):
            logger.info(f"Processing job {i+1}/{len(jobs)}: {job_data.get('job_title', 'Unknown')}")
            
            # Print job data for debugging
            self._print_job_data(job_data)
            
            # Store in vector DB
            if self.store_job_in_vector_db(job_data):
                processed_count += 1
            
            # Rate limiting
            time.sleep(self.scrape_delay)
        
        self.stats['categories_processed'] += 1
        logger.info(f"Category '{category_name}' completed: {processed_count} jobs processed")
        return processed_count
    
    def _print_job_data(self, job_data: Dict[str, Any]):
        """Print formatted job data"""
        logger.info("=" * 60)
        logger.info(f"Job Title: {job_data.get('job_title', 'N/A')}")
        logger.info(f"Department/Company: {job_data.get('department_company', 'N/A')}")
        logger.info(f"Salary Range: {job_data.get('salary_range', 'N/A')}")
        logger.info(f"Experience: {job_data.get('experience', 'N/A')}")
        logger.info(f"Education: {job_data.get('education', 'N/A')}")
        logger.info(f"State: {job_data.get('state', 'N/A')}")
        logger.info(f"Last Date: {job_data.get('last_date', 'N/A')}")
        logger.info(f"Posted On: {job_data.get('posted_on', 'N/A')}")
        logger.info(f"Category: {job_data.get('category', 'N/A')}")
        logger.info(f"Age Limit: {job_data.get('age_limit', 'N/A')}")
        logger.info(f"Job Openings: {job_data.get('job_openings', 'N/A')}")
        logger.info(f"Tags: {', '.join(job_data.get('tags', [])) if job_data.get('tags') else 'N/A'}")
        logger.info(f"Official Notice: {job_data.get('official_notice_link', 'N/A')}")
        logger.info(f"Official Website: {job_data.get('official_website', 'N/A')}")
        if job_data.get('important_notice'):
            logger.info(f"Important Notice: {job_data.get('important_notice', 'N/A')[:200]}...")
        logger.info(f"Job Detail URL: {job_data.get('job_detail_url', 'N/A')}")
        logger.info("=" * 60)
    
    def run_scraping(self, max_categories: Optional[int] = None, test_mode: bool = False):
        """Main scraping function"""
        logger.info("=== Starting JobYaari Direct Scraping ===")
        
        try:
            # Get job categories
            categories = self.get_job_categories()
            if not categories:
                logger.error("No categories found, exiting")
                return
            
            # Limit categories for testing
            if test_mode:
                categories = categories[:1]
                logger.info("Test mode: processing only 1 category")
            elif max_categories:
                categories = categories[:max_categories]
                logger.info(f"Limited to {max_categories} categories")
            
            # Process each category
            for i, category in enumerate(categories):
                logger.info(f"\n--- Category {i+1}/{len(categories)}: {category['name']} ---")
                
                try:
                    self.scrape_category(category)
                    
                    # Wait between categories
                    if i < len(categories) - 1:
                        wait_time = self.scrape_delay * 3
                        logger.info(f"Waiting {wait_time}s before next category...")
                        time.sleep(wait_time)
                        
                except KeyboardInterrupt:
                    logger.info("Scraping interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error processing category '{category['name']}': {e}")
                    self.stats['errors'] += 1
                    continue
            
        except KeyboardInterrupt:
            logger.info("Scraping interrupted by user")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            self._print_final_stats()
    
    def _print_final_stats(self):
        """Print final statistics"""
        end_time = datetime.now()
        duration = end_time - self.stats['start_time']
        
        logger.info("\n=== SCRAPING COMPLETED ===")
        logger.info(f"Duration: {duration}")
        logger.info(f"Categories processed: {self.stats['categories_processed']}")
        logger.info(f"Jobs found: {self.stats['jobs_found']}")
        logger.info(f"Jobs stored: {self.stats['jobs_stored']}")
        logger.info(f"Jobs skipped: {self.stats['jobs_skipped']}")
        logger.info(f"Errors: {self.stats['errors']}")
        
        # Get collection info
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Total jobs in database: {collection_info.points_count}")
        except Exception:
            pass
        
        logger.info("========================")

def main():
    """Main function with CLI options"""
    import argparse
    
    parser = argparse.ArgumentParser(description='JobYaari Direct HTML Scraper')
    parser.add_argument('--test', action='store_true', help='Test mode (1 category only)')
    parser.add_argument('--max-categories', type=int, help='Max categories to process')
    parser.add_argument('--category', type=str, help='Specific category name to scrape')
    parser.add_argument('--core-categories', action='store_true', help='Limit scraping to Engineering, Science, Commerce, Education')
    args = parser.parse_args()
    
    try:
        scraper = SimpleJobYaariDirectScraper()
        
        if args.category:
            # Scrape specific category
            categories = scraper.get_job_categories()
            target_category = None
            for cat in categories:
                if args.category.lower() in cat['name'].lower():
                    target_category = cat
                    break
            
            if target_category:
                logger.info(f"Scraping specific category: {target_category['name']}")
                scraper.scrape_category(target_category)
            else:
                logger.error(f"Category '{args.category}' not found")
                logger.info(f"Available categories: {[cat['name'] for cat in categories]}")

        elif args.core_categories:
            # Limit run to the four core categories (by URL)
            allowed_urls = {
                "https://www.jobyaari.com/category/science",
                "https://www.jobyaari.com/category/engineering",
                "https://www.jobyaari.com/category/commerce",
                "https://www.jobyaari.com/category/education",
            }
            categories = scraper.get_job_categories()
            if not categories:
                logger.error("No categories found on categories page; exiting")
            else:
                # filter available categories to only allowed ones
                allowed_norm = {u.rstrip('/') for u in allowed_urls}
                selected = [
                    c for c in categories
                    if c.get('url') and c['url'].rstrip('/') in allowed_norm
                ]
                if not selected:
                    logger.error("None of the core categories were found on the categories page")
                    logger.info(f"Available categories: {[cat['name'] for cat in categories]}")
                else:
                    logger.info(f"Scraping core categories: {[c['name'] for c in selected]}")
                    for cat in selected:
                        scraper.scrape_category(cat)

        else:
            # Regular scraping
            scraper.run_scraping(
                max_categories=args.max_categories,
                test_mode=args.test
            )
            
    except KeyboardInterrupt:
        logger.info("Scraper stopped by user")
    except Exception as e:
        logger.error(f"Scraper failed: {e}")


def clear_qdrant_collection():
    """Clear all data from Qdrant collection before starting fresh scrape"""
    try:
        # Initialize Qdrant client
        qdrant_client = QdrantClient(
            host=os.getenv('QDRANT_HOST', 'localhost'),
            port=int(os.getenv('QDRANT_PORT', '6333'))
        )
        
        collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'jobyaari_jobs')
        
        # Check if collection exists
        collections = qdrant_client.get_collections()
        collection_exists = any(col.name == collection_name for col in collections.collections)
        
        if collection_exists:
            # Delete the collection
            qdrant_client.delete_collection(collection_name)
            logger.info(f"Cleared Qdrant collection: {collection_name}")
        else:
            logger.info(f"Collection {collection_name} doesn't exist, nothing to clear")
            
    except Exception as e:
        logger.error(f"Failed to clear Qdrant collection: {e}")
        raise


if __name__ == "__main__":
    clear_qdrant_collection()
    main()