#!/usr/bin/env python3
"""
Enhanced CLI Runner - Command-line interface for BHIV Core tasks with batch processing and multiple output formats.
"""

import argparse
import requests
import uuid
import os
import json
import csv
import mimetypes
import glob
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from tqdm import tqdm
from utils.logger import get_logger
from reinforcement.replay_buffer import replay_buffer
from reinforcement.reward_functions import get_reward_from_output
from time import sleep

logger = get_logger(__name__)

class OutputFormatter:
    """Handle different output formats for CLI results."""

    @staticmethod
    def format_json(results: List[Dict[str, Any]], pretty: bool = True) -> str:
        """Format results as JSON."""
        if pretty:
            return json.dumps(results, indent=2, default=str)
        return json.dumps(results, default=str)

    @staticmethod
    def format_text(results: List[Dict[str, Any]]) -> str:
        """Format results as human-readable text."""
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"=== Result {i} ===")

            if 'error' in result:
                output.append(f"Status: ERROR")
                output.append(f"Error: {result['error']}")
            else:
                agent_output = result.get('agent_output', {})
                output.append(f"Status: SUCCESS")
                output.append(f"Task ID: {result.get('task_id', 'N/A')}")
                output.append(f"Agent: {result.get('agent', 'N/A')}")
                output.append(f"Model: {agent_output.get('model', 'N/A')}")
                output.append(f"Input Type: {result.get('input_type', 'N/A')}")

                if 'result' in agent_output:
                    output.append(f"Summary: {agent_output['result']}")

                if 'confidence' in agent_output:
                    output.append(f"Confidence: {agent_output['confidence']:.2%}")

                if 'keywords' in agent_output and agent_output['keywords']:
                    output.append(f"Keywords: {', '.join(agent_output['keywords'])}")

                if 'processing_time' in result:
                    output.append(f"Processing Time: {result['processing_time']:.2f}s")

            output.append("")  # Empty line between results

        return "\n".join(output)

    @staticmethod
    def format_csv(results: List[Dict[str, Any]]) -> str:
        """Format results as CSV."""
        if not results:
            return ""

        # Prepare CSV data
        csv_data = []
        fieldnames = ['task_id', 'agent', 'model', 'input_type', 'status', 'summary', 'confidence', 'keywords', 'processing_time', 'error']

        for result in results:
            agent_output = result.get('agent_output', {})
            row = {
                'task_id': result.get('task_id', ''),
                'agent': result.get('agent', ''),
                'model': agent_output.get('model', ''),
                'input_type': result.get('input_type', ''),
                'status': 'ERROR' if 'error' in result else 'SUCCESS',
                'summary': agent_output.get('result', ''),
                'confidence': agent_output.get('confidence', ''),
                'keywords': ', '.join(agent_output.get('keywords', [])),
                'processing_time': result.get('processing_time', ''),
                'error': result.get('error', '')
            }
            csv_data.append(row)

        # Convert to CSV string
        import io
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
        return output.getvalue()

def discover_files(directory: str, extensions: List[str] = None) -> List[str]:
    """Discover files in directory with specified extensions."""
    if extensions is None:
        extensions = ['.pdf', '.txt', '.jpg', '.jpeg', '.png', '.wav', '.mp3', '.ogg', '.flac']

    files = []
    directory_path = Path(directory)

    if not directory_path.exists():
        logger.error(f"Directory not found: {directory}")
        return files

    for ext in extensions:
        pattern = f"**/*{ext}"
        found_files = list(directory_path.glob(pattern))
        files.extend([str(f) for f in found_files])

    return sorted(files)

def run_task(task: str, data: str, model: str, file_path: str = "", input_type: str = "text", retries: int = 3, delay: int = 2) -> Dict[str, Any]:
    """Run a task via the MCP bridge API with retries."""
    task_id = str(uuid.uuid4())
    start_time = datetime.now()

    for attempt in range(retries):
        try:
            # Determine if we need to upload a file
            if file_path and os.path.exists(file_path):
                logger.info(f"Processing file: {file_path} with type: {input_type}")

                try:
                    from requests_toolbelt.multipart.encoder import MultipartEncoder

                    mime_type = mimetypes.guess_type(file_path)[0] or 'application/octet-stream'

                    with open(file_path, 'rb') as f:
                        multipart_data = MultipartEncoder(
                            fields={
                                'agent': model,
                                'input': data,
                                'input_type': input_type,
                                'retries': str(retries),
                                'fallback_model': 'edumentor_agent',
                                'file': (os.path.basename(file_path), f, mime_type)
                            }
                        )

                        response = requests.post(
                            "http://localhost:8002/handle_task_with_file",
                            data=multipart_data,
                            headers={'Content-Type': multipart_data.content_type},
                            timeout=120
                        )
                except ImportError:
                    logger.warning("requests-toolbelt not installed. Using JSON API with file path.")
                    # Fallback: send file path directly
                    response = requests.post(
                        "http://localhost:8002/handle_task",
                        json={
                            "agent": model,
                            "input": data,
                            "pdf_path": file_path,  # This will be used as file path for all types
                            "input_type": input_type,
                            "retries": retries,
                            "fallback_model": "edumentor_agent"
                        },
                        timeout=120
                    )
            else:
                # Use regular JSON API
                response = requests.post(
                    "http://localhost:8002/handle_task",
                    json={
                        "agent": model,
                        "input": data,
                        "pdf_path": file_path,
                        "input_type": input_type,
                        "retries": retries,
                        "fallback_model": "edumentor_agent"
                    },
                    timeout=120
                )

            response.raise_for_status()
            result = response.json()

            # Add processing time and metadata
            processing_time = (datetime.now() - start_time).total_seconds()
            result['processing_time'] = processing_time
            result['file_path'] = file_path
            result['input_type'] = input_type
            result['agent'] = model

            logger.info(f"CLI task completed in {processing_time:.2f}s: {result.get('task_id', task_id)}")

            response_task_id = result.get('task_id', task_id)
            reward = get_reward_from_output(result.get('agent_output', {}), response_task_id)
            replay_buffer.add_run(response_task_id, data, result.get('agent_output', {}), model, model, reward)
            return result

        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1}/{retries} failed: {str(e)}")
            if attempt < retries - 1:
                sleep(delay)
            else:
                logger.error(f"Error running CLI task: {str(e)}")
                processing_time = (datetime.now() - start_time).total_seconds()
                output = {
                    "error": f"Task execution failed: {str(e)}",
                    "status": 500,
                    "processing_time": processing_time,
                    "file_path": file_path,
                    "input_type": input_type,
                    "agent": model,
                    "task_id": task_id
                }
                reward = get_reward_from_output(output, task_id)
                replay_buffer.add_run(task_id, data, output, model, model, reward)
                return output

def process_batch(directory: str, task: str, model: str, input_type: str = "auto", output_format: str = "json", output_file: str = None) -> List[Dict[str, Any]]:
    """Process all files in a directory."""
    files = discover_files(directory)

    if not files:
        logger.warning(f"No supported files found in directory: {directory}")
        return []

    logger.info(f"Found {len(files)} files to process in {directory}")
    results = []

    # Process files with progress bar
    with tqdm(files, desc="Processing files", unit="file") as pbar:
        for file_path in pbar:
            pbar.set_description(f"Processing {os.path.basename(file_path)}")

            # Auto-detect input type if needed
            current_input_type = input_type
            if input_type == "auto":
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.pdf']:
                    current_input_type = 'pdf'
                elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    current_input_type = 'image'
                elif ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
                    current_input_type = 'audio'
                else:
                    current_input_type = 'text'

            # Process file
            result = run_task(task, f"Process this {current_input_type} file", model, file_path, current_input_type)
            results.append(result)

            # Update progress bar with status
            status = "✓" if 'error' not in result else "✗"
            pbar.set_postfix(status=status, total=len(results))

    # Save results if output file specified
    if output_file:
        save_results(results, output_file, output_format)

    return results

def save_results(results: List[Dict[str, Any]], output_file: str, output_format: str):
    """Save results to file in specified format."""
    formatter = OutputFormatter()

    try:
        if output_format.lower() == 'json':
            content = formatter.format_json(results)
        elif output_format.lower() == 'text':
            content = formatter.format_text(results)
        elif output_format.lower() == 'csv':
            content = formatter.format_csv(results)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)

        logger.info(f"Results saved to {output_file} in {output_format.upper()} format")

    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

def print_rl_statistics():
    """Print RL performance statistics."""
    try:
        from reinforcement.model_selector import model_selector
        from reinforcement.agent_selector import agent_selector

        print("\n🧠 RL Performance Statistics:")
        print("=" * 40)

        # Model selector stats
        model_performance = model_selector.get_model_performance_summary()
        if model_performance:
            print("\n📊 Model Performance:")
            for model, stats in model_performance.items():
                print(f"  {model}:")
                print(f"    Avg Reward: {stats['avg_reward']:.3f}")
                print(f"    Tasks: {stats['count']}")
                print(f"    Confidence: {stats['confidence']:.1%}")

        # Agent selector stats
        print(f"\n🤖 Agent Statistics:")
        print(f"  Tracked Agents: {len(agent_selector.agent_scores)}")
        print(f"  Total Tasks: {agent_selector.total_tasks}")

        if agent_selector.agent_scores:
            best_agent = max(agent_selector.agent_scores.items(),
                           key=lambda x: x[1].get('avg_reward', 0))
            print(f"  Best Agent: {best_agent[0]} (reward: {best_agent[1].get('avg_reward', 0):.3f})")

        print()
    except Exception as e:
        logger.error(f"Error displaying RL statistics: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced BHIV Core CLI Runner with batch processing and multiple output formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file processing
  python cli_runner.py summarize "Analyze this document" edumentor_agent --file document.pdf

  # Batch processing
  python cli_runner.py summarize "Process documents" edumentor_agent --batch ./documents --output-format json

  # Save results to file
  python cli_runner.py summarize "Analyze files" edumentor_agent --batch ./input --output results.csv --output-format csv

  # Multi-input processing
  python cli_runner.py summarize "Multi-modal analysis" edumentor_agent --batch ./mixed_files --input-type multi
        """
    )

    # Positional arguments
    parser.add_argument("task", help="Task to perform (e.g., summarize, calculate)")
    parser.add_argument("data", help="Input data or description for processing")
    parser.add_argument("model", help="Model/agent to use (e.g., edumentor_agent, vedas_agent, wellness_agent)")

    # File processing options
    parser.add_argument("--file", help="Path to single file to process")
    parser.add_argument("--batch", help="Directory path for batch processing")
    parser.add_argument("--input-type", default="auto",
                       choices=["auto", "text", "pdf", "image", "audio", "multi"],
                       help="Input type (auto-detect by default)")

    # Output options
    parser.add_argument("--output-format", default="json",
                       choices=["json", "text", "csv"],
                       help="Output format (default: json)")
    parser.add_argument("--output", help="Output file path (prints to stdout if not specified)")

    # Processing options
    parser.add_argument("--retries", type=int, default=3, help="Number of retries for failed requests")
    parser.add_argument("--delay", type=int, default=2, help="Delay between retries in seconds")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    # RL options
    parser.add_argument("--use-rl", action="store_true", help="Enable RL-based model/agent selection")
    parser.add_argument("--no-rl", action="store_true", help="Disable RL-based model/agent selection")
    parser.add_argument("--exploration-rate", type=float, help="Override RL exploration rate (0.0-1.0)")
    parser.add_argument("--rl-stats", action="store_true", help="Show RL statistics after processing")

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    # Configure RL settings
    from config.settings import RL_CONFIG
    if args.use_rl:
        RL_CONFIG['use_rl'] = True
        logger.info("RL mode enabled via CLI flag")
    elif args.no_rl:
        RL_CONFIG['use_rl'] = False
        logger.info("RL mode disabled via CLI flag")

    if args.exploration_rate is not None:
        if 0.0 <= args.exploration_rate <= 1.0:
            RL_CONFIG['exploration_rate'] = args.exploration_rate
            logger.info(f"RL exploration rate set to {args.exploration_rate}")
        else:
            parser.error("Exploration rate must be between 0.0 and 1.0")

    logger.info(f"RL Configuration: use_rl={RL_CONFIG['use_rl']}, exploration_rate={RL_CONFIG['exploration_rate']}")

    # Validate arguments - support both text-only and file processing
    if not args.file and not args.batch and args.input_type == "auto":
        parser.error("For auto input type detection, either --file or --batch must be specified. For text processing, use --input-type text")
    
    if args.file and args.batch:
        parser.error("Cannot specify both --file and --batch")

    try:
        if args.batch:
            # Batch processing
            logger.info(f"Starting batch processing of directory: {args.batch}")
            results = process_batch(
                directory=args.batch,
                task=args.task,
                model=args.model,
                input_type=args.input_type,
                output_format=args.output_format,
                output_file=args.output
            )

            if not args.output:
                # Print results to stdout
                formatter = OutputFormatter()
                if args.output_format == 'json':
                    print(formatter.format_json(results))
                elif args.output_format == 'text':
                    print(formatter.format_text(results))
                elif args.output_format == 'csv':
                    print(formatter.format_csv(results))

            # Print summary
            successful = len([r for r in results if 'error' not in r])
            failed = len(results) - successful
            print(f"\nBatch processing completed: {successful} successful, {failed} failed out of {len(results)} total files")

            # Show RL statistics if requested
            if args.rl_stats:
                print_rl_statistics()

        else:
            # Single file processing
            input_type = args.input_type
            if args.file and input_type == "auto":
                ext = os.path.splitext(args.file)[1].lower()
                if ext in ['.pdf']:
                    input_type = 'pdf'
                elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                    input_type = 'image'
                elif ext in ['.mp3', '.wav', '.ogg', '.flac', '.m4a']:
                    input_type = 'audio'
                else:
                    input_type = 'text'
                logger.info(f"Auto-detected input type: {input_type} from file extension: {ext}")

            # Validate file exists
            if args.file and not os.path.exists(args.file):
                logger.error(f"File not found: {args.file}")
                return 1

            # Process single file
            result = run_task(args.task, args.data, args.model, args.file or "", input_type, args.retries, args.delay)
            results = [result]

            # Output results
            if args.output:
                save_results(results, args.output, args.output_format)
            else:
                formatter = OutputFormatter()
                if args.output_format == 'json':
                    print(formatter.format_json(results))
                elif args.output_format == 'text':
                    print(formatter.format_text(results))
                elif args.output_format == 'csv':
                    print(formatter.format_csv(results))

        return 0

    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    main()








