#!/usr/bin/env python3
"""
MCP Corpus Stream Client - Unified Data Streaming for Language Expansion

High-level role (to disambiguate from sovereign_core.mcp.stream_client):
- This file streams **corpus / training text** for many languages
- It pulls from HuggingFace, Wikipedia, news, etc. using `mcp_connectors.yml`
- Output is used for **adapter/tokenizer training**, not user feedback

Usage:
    from mcp.stream_client import stream_language_data
    
    # Stream 1000 samples for Sinhala
    for sample in stream_language_data('sinhala', max_samples=1000):
        print(sample['text'])
"""

import os
import sys
import json
import yaml
import logging
from pathlib import Path
from typing import Generator, Dict, Any, Optional, List
from dataclasses import dataclass

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    """Configuration for a streaming source"""
    source_id: str
    source_type: str  # 'huggingface_stream', 'http_pages', 's3'
    language_code: str
    max_samples: int
    uri: Optional[str] = None
    base_url: Optional[str] = None
    bucket: Optional[str] = None


class MCPStreamClient:
    """
    Unified streaming client for language data
    
    Coordinates:
    - HuggingFace dataset streaming
    - Wikipedia scraping
    - News scraping
    - S3/cloud storage
    """
    
    def __init__(self, config_path: str = "config/mcp_connectors.yml"):
        """
        Initialize MCP stream client
        
        Args:
            config_path: Path to MCP connectors configuration
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.scrapers_available = self._check_scraper_availability()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load MCP connectors configuration"""
        try:
            if not self.config_path.exists():
                logger.warning(f"Config not found: {self.config_path}, using empty config")
                return {'sources': []}
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            logger.info(f"Loaded MCP config with {len(config.get('sources', []))} sources")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {'sources': []}
    
    def _check_scraper_availability(self) -> Dict[str, bool]:
        """Check which scrapers are available"""
        availability = {}
        
        # Check for HuggingFace datasets
        try:
            import datasets
            availability['huggingface'] = True
            logger.info("✅ HuggingFace datasets available")
        except ImportError:
            availability['huggingface'] = False
            logger.warning("❌ HuggingFace datasets not installed (pip install datasets)")
        
        # Check for web scraping tools
        try:
            import requests
            import bs4
            import feedparser
            availability['web_scraping'] = True
            logger.info("✅ Web scraping tools available")
        except ImportError:
            availability['web_scraping'] = False
            logger.warning("❌ Web scraping tools not installed (pip install requests beautifulsoup4 feedparser)")
        
        return availability
    
    def stream_huggingface(self, uri: str, language_code: str, max_samples: int) -> Generator[Dict[str, str], None, None]:
        """
        Stream data from HuggingFace dataset
        
        Args:
            uri: Dataset URI (format: "dataset_name:config")
            language_code: Language code
            max_samples: Maximum samples to yield
            
        Yields:
            Sample dictionaries with 'text', 'language', 'source'
        """
        if not self.scrapers_available.get('huggingface', False):
            logger.error("HuggingFace datasets not available, skipping")
            return
        
        try:
            from datasets import load_dataset
            
            # Parse URI
            parts = uri.split(':')
            dataset_name = parts[0]
            config_name = parts[1] if len(parts) > 1 else None
            split = 'train'
            
            logger.info(f"Streaming HuggingFace dataset: {dataset_name} (config: {config_name})")
            
            # Load dataset in streaming mode
            if config_name:
                dataset = load_dataset(dataset_name, config_name, split=split, streaming=True)
            else:
                dataset = load_dataset(dataset_name, split=split, streaming=True)
            
            count = 0
            for sample in dataset:
                if count >= max_samples:
                    break
                
                # Extract text (try common field names)
                text = None
                for field in ['text', 'content', 'sentence', 'passage']:
                    if field in sample:
                        text = sample[field]
                        break
                
                if not text or len(text) < 10:
                    continue
                
                yield {
                    'text': text.strip(),
                    'language': language_code,
                    'source': f'huggingface:{dataset_name}',
                    'metadata': {'dataset': dataset_name, 'config': config_name}
                }
                
                count += 1
                if count % 100 == 0:
                    logger.info(f"Streamed {count}/{max_samples} samples from HuggingFace")
            
            logger.info(f"✅ Completed streaming {count} samples from HuggingFace")
            
        except Exception as e:
            logger.error(f"Failed to stream from HuggingFace: {e}")
    
    def stream_wikipedia(self, language_code: str, max_pages: int) -> Generator[Dict[str, str], None, None]:
        """
        Stream data from Wikipedia scraper
        
        Args:
            language_code: Wikipedia language code
            max_pages: Maximum pages to scrape
            
        Yields:
            Sample dictionaries
        """
        if not self.scrapers_available.get('web_scraping', False):
            logger.error("Web scraping tools not available, skipping Wikipedia")
            return
        
        try:
            # Import scraper
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from scrapers.scrape_wiki import WikipediaScraper
            
            logger.info(f"Scraping Wikipedia for language: {language_code} (quality filter: ON)")
            
            scraper = WikipediaScraper(language_code=language_code, delay=1.0, enable_quality_filter=True)
            
            for page in scraper.scrape_pages(max_pages=max_pages):
                # Split text into sentences (simple split for now)
                text = page['text']
                
                # Yield full page
                yield {
                    'text': text,
                    'language': language_code,
                    'source': 'wikipedia',
                    'metadata': {'title': page['title'], 'url': page['url']}
                }
            
            logger.info(f"✅ Completed Wikipedia scraping")
            
        except Exception as e:
            logger.error(f"Failed to scrape Wikipedia: {e}")
    
    def stream_news(self, language: str, max_articles: int) -> Generator[Dict[str, str], None, None]:
        """
        Stream data from news scraper
        
        Args:
            language: Language name (e.g., 'vietnamese', 'thai')
            max_articles: Maximum articles to scrape
            
        Yields:
            Sample dictionaries
        """
        if not self.scrapers_available.get('web_scraping', False):
            logger.error("Web scraping tools not available, skipping news")
            return
        
        try:
            # Import scraper
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from scrapers.scrape_news import NewsScraper
            
            logger.info(f"Scraping news for language: {language}")
            
            scraper = NewsScraper(language=language, delay=2.0)
            
            for article in scraper.scrape_articles(max_articles=max_articles):
                yield {
                    'text': article['text'],
                    'language': article['language'],
                    'source': 'news',
                    'metadata': {'title': article['title'], 'url': article['url']}
                }
            
            logger.info(f"✅ Completed news scraping")
            
        except Exception as e:
            logger.error(f"Failed to scrape news: {e}")
    
    def stream_language_data(self, language: str, max_samples: int = 1000) -> Generator[Dict[str, str], None, None]:
        """
        Stream data for a language from all available sources
        
        Args:
            language: Language name or code
            max_samples: Maximum total samples to yield
            
        Yields:
            Sample dictionaries with 'text', 'language', 'source'
        """
        logger.info(f"Starting data streaming for language: {language} (target: {max_samples} samples)")
        
        total_samples = 0
        
        # Find matching sources in config
        for source_config in self.config.get('sources', []):
            if total_samples >= max_samples:
                break
            
            source_id = source_config.get('id', '')
            source_type = source_config.get('type', '')
            lang_code = source_config.get('language_code', '')
            
            # Check if source matches our language
            if language.lower() not in source_id.lower() and language.lower() != lang_code.lower():
                continue
            
            remaining = max_samples - total_samples
            source_max = min(source_config.get('max_samples', 1000), remaining)
            
            logger.info(f"Streaming from source: {source_id} (type: {source_type}, max: {source_max})")
            
            try:
                # Route to appropriate streamer
                if source_type == 'huggingface_stream':
                    uri = source_config.get('uri', '')
                    for sample in self.stream_huggingface(uri, lang_code, source_max):
                        yield sample
                        total_samples += 1
                        if total_samples >= max_samples:
                            break
                
                elif source_type == 'http_pages' and 'wiki' in source_id:
                    max_pages = source_config.get('max_pages', 100)
                    for sample in self.stream_wikipedia(lang_code, max_pages):
                        yield sample
                        total_samples += 1
                        if total_samples >= max_samples:
                            break
                
                elif source_type == 'http_pages' and 'news' in source_id:
                    for sample in self.stream_news(language, source_max):
                        yield sample
                        total_samples += 1
                        if total_samples >= max_samples:
                            break
                
                else:
                    logger.warning(f"Source type '{source_type}' not yet implemented for {source_id}")
            
            except Exception as e:
                logger.error(f"Error streaming from {source_id}: {e}")
                continue
        
        logger.info(f"✅ Streaming complete: {total_samples} total samples for {language}")


# Convenience function
def stream_language_data(language: str, max_samples: int = 1000, config_path: str = "config/mcp_connectors.yml") -> Generator[Dict[str, str], None, None]:
    """
    Convenience function to stream language data
    
    Args:
        language: Language name or code
        max_samples: Maximum samples to yield
        config_path: Path to MCP connectors config
        
    Yields:
        Sample dictionaries
    """
    client = MCPStreamClient(config_path=config_path)
    yield from client.stream_language_data(language, max_samples)


def main():
    """CLI for testing stream client"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Stream Client - Test data streaming")
    parser.add_argument('--language', '-l', required=True, help='Language to stream')
    parser.add_argument('--max_samples', '-n', type=int, default=10, help='Max samples (default: 10 for testing)')
    parser.add_argument('--config', '-c', default='mcp_connectors.yml', help='Config file path')
    parser.add_argument('--output', '-o', default=None, help='Output JSONL file (optional)')
    
    args = parser.parse_args()
    
    # Stream data
    count = 0
    samples = []
    
    for sample in stream_language_data(args.language, args.max_samples, args.config):
        count += 1
        samples.append(sample)
        print(f"\n{'='*60}")
        print(f"Sample {count}/{args.max_samples}")
        print(f"Language: {sample['language']}")
        print(f"Source: {sample['source']}")
        print(f"Text: {sample['text'][:200]}...")
        print(f"{'='*60}")
    
    # Save if output specified
    if args.output and samples:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"\n💾 Saved {len(samples)} samples to {output_path}")
    
    print(f"\n✅ Streamed {count} samples for {args.language}")


if __name__ == "__main__":
    main()

