"""
OpenRouter vision model integration for highlight detection.
"""

import os
import base64
import json
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

import requests

from app.preprocessor.frame_preprocessor import FramePreprocessor


class OpenRouterVision:
    """
    OpenRouter vision model for highlight detection.
    
    Supports models like:
    - google/gemini-2.5-pro
    - qwen/qwen3-vl-8b-instruct
    - anthropic/claude-3.5-sonnet
    - openai/gpt-4-vision-preview
    """
    
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    TIME_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}(?:\.\d{3})?)")

    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: str = "google/gemini-2.5-pro",
        max_frames: int = 500
    ):
        """
        Initialize OpenRouter Vision.

        Args:
            api_key: OpenRouter API key (or set OPENROUTER_API_KEY env var)
            model: Model name (default: google/gemini-2.5-pro)
                   See https://openrouter.ai/docs for available models
            max_frames: Maximum number of frames to process per request
        """
        
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Set OPENROUTER_API_KEY environment variable or pass api_key parameter."
            )
        
        self.model = model
        self.max_frames = max_frames

    def _get_vision_analysis_schema(self) -> Dict[str, Any]:
        """Get JSON schema for structured vision analysis output."""
        return {
            "name": "highlight_pick",
            "strict": True,
            "schema": {
                "type": "array",
                "description": "Video clip time period matching the user description",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "int",
                            "description": "Video clip start at which second"
                        },
                        "end_time": {
                            "type": "int",
                            "description": "Video clip end at which second"
                        },
                    },
                    "required": ["start_time", "end_time"],
                    "additionalProperties": False
                }
            }
        }

    def _encode_image_to_data_url(self, image_path: str) -> str:
        """
        Encode image to base64 data URL.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded data URL
        """
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Detect image format from file extension
        ext = Path(image_path).suffix.lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')
        
        return f"data:{mime_type};base64,{b64}"

    def _parse_time_from_filename(self, filename: str) -> float:
        """
        Parse timestamp from filename like '00:00:03.500.jpg'.
        
        Args:
            filename: Frame filename with timestamp
            
        Returns:
            Time in seconds
        """
        m = self.TIME_RE.search(filename)
        if not m:
            return 0.0
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = float(m.group(3))
        return hh * 3600 + mm * 60 + ss

    def _seconds_to_hhmmss(self, sec: float) -> str:
        """
        Convert seconds to HH:MM:SS format.
        
        Args:
            sec: Time in seconds
            
        Returns:
            Formatted time string
        """
        hh = int(sec // 3600)
        mm = int((sec % 3600) // 60)
        ss = sec % 60
        return f"{hh:02d}:{mm:02d}:{ss:06.3f}"

    def _build_messages(
        self, 
        frame_paths: List[str],
        user_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Build OpenRouter-compatible messages.
        
        Args:
            frame_paths: List of frame file paths
            user_prompt: Optional user prompt for guided analysis
            
        Returns:
            List of messages for API call
        """
        # System prompt with detailed instructions for structured output
        system_prompt = """You are a video analysis expert. You need to find the video clips that best match the user's description based on the user's input, and output the time frames of these clips.

Your task:
1. Understand the user's input and locate the segment that best matches the user's description.
2. Output an interval to locate the start and end seconds of the segment you found. Your output should be in the format of: [(start time, end time)...]

Guidelines:
- The video clips you find should be of moderate length; select those that best match the user's description.
- The total number of video clips you find should not exceed 10; ideally, it should be between 3 and 8."""
        
        
        # Build user content with frames
        user_content = [
            {
                "type": "text",
                "text": user_prompt
            }
        ]
        
        # Add frames with timestamps
        for frame_path in frame_paths:
            if not os.path.exists(frame_path):
                continue
            
            # Parse timestamp from filename
            filename = Path(frame_path).name
            timestamp = self._parse_time_from_filename(filename)
            ts_str = self._seconds_to_hhmmss(timestamp)
            
            user_content.append({
                "type": "text",
                "text": f"Frame at {ts_str}"
            })
            
            user_content.append({
                "type": "image_url",
                "image_url": {
                    "url": self._encode_image_to_data_url(frame_path)
                }
            })
        
        # Build messages array with system and user roles
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ]
        
        return messages

    def analyze_frames(
        self, 
        preprocess_result, 
        user_prompt: Optional[str] = None
    ):
        """
        Analyze frames using OpenRouter API.

        Args:
            preprocess_result: PreprocessResult object from video preprocessing or List[str] of frame paths
            user_prompt: Optional user requirement for the short video generation
                        (e.g., "Create a funny recap", "Make a TikTok highlight reel")
                        This guides the vision analysis to focus on relevant aspects

        Returns:
            VisionResult with:
                - video_summary: Overall summary of the video
                - shots: List of VisionShot objects with timestamps and descriptions
                - metadata: Dict with model info and frames_analyzed count
        
        Raises:
            TypeError: If preprocess_result is not PreprocessResult or List[str]
            ValueError: If no valid frames are provided
            RuntimeError: If API call fails, response parsing fails, or unexpected errors occur
        """
        # Handle both PreprocessResult and List[str] for backward compatibility
        # Check if it's a PreprocessResult by checking for the 'frames' attribute
        if hasattr(preprocess_result, 'frames') and hasattr(preprocess_result, 'to_dict'):
            frame_paths = list(preprocess_result.frames.values())
        elif isinstance(preprocess_result, list):
            frame_paths = preprocess_result
        else:
            raise TypeError("preprocess_result must be PreprocessResult or List[str]")
        
        if not frame_paths:
            return [], "No frames provided for analysis"
        
        # Evenly sample frames if count exceeds max_frames
        if len(frame_paths) > self.max_frames:
            print(f"Warning: Evenly sampling {self.max_frames} frames from {len(frame_paths)} total frames")
            # Calculate step size for even sampling
            step = len(frame_paths) / self.max_frames
            frame_paths = [frame_paths[int(i * step)] for i in range(self.max_frames)]
        
        # Filter out non-existent frames
        valid_frames = [f for f in frame_paths if os.path.exists(f)]
        
        if not valid_frames:
            return [], "No valid frames found for analysis."
        
        # Build messages
        messages = self._build_messages(valid_frames, user_prompt)
        
        # Prepare API request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": self._get_vision_analysis_schema()
            }
        }
        
        print("Sending request to OpenRouter API...")
        print(f"Model: {self.model}")
        print(f"Using structured output: {'response_format' in payload}")

        try:
            # Call OpenRouter API
            response = requests.post(
                self.API_URL,
                headers=headers,
                json=payload,
                timeout=600  # 5 minute timeout for long requests
            )

            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            print(f"Response text length: {len(response.text)}")
            
            # Check for HTTP errors first
            response.raise_for_status()
            
            # Check if response is empty
            if not response.text or response.text.strip() == "":
                raise RuntimeError(
                    f"Empty response from OpenRouter API. "
                    f"Status: {response.status_code}, "
                    f"Headers: {dict(response.headers)}"
                )
            
            # Parse JSON response
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise RuntimeError(
                    f"Failed to parse JSON response: {str(e)}\n"
                    f"Response status: {response.status_code}\n"
                    f"Response headers: {dict(response.headers)}\n"
                    f"Response text (first 1000 chars): {response.text[:1000]}\n"
                    f"Response text (last 500 chars): {response.text[-500:]}"
                )
            
            # Check if response has expected structure
            if "choices" not in data or not data["choices"]:
                raise RuntimeError(
                    f"Unexpected API response structure. "
                    f"Missing 'choices' field or empty choices array. "
                    f"Response: {json.dumps(data)}"
                )
            
            raw_response = data["choices"][0]["message"]["content"]
            
            print(f"Raw response length: {len(raw_response)}")
            print(f"Raw response (first 200 chars): {raw_response[:200]}")
            
            # Strip markdown code blocks if present
            content = raw_response.strip()
            if content.startswith("```"):
                # Remove opening code fence (```json or ```javascript or just ```)
                content = re.sub(r'^```(?:json|javascript)?\s*\n?', '', content, flags=re.IGNORECASE)
                # Remove closing code fence
                content = re.sub(r'\n?```\s*$', '', content)
                content = content.strip()
                print("Stripped markdown code blocks")
            
            # Parse structured JSON response
            try:
                parsed_json = json.loads(content)
                
#                # Validate expected structure
#                if not isinstance(parsed_json, dict):
#                    raise ValueError("Response is not a dictionary")
                
#                if "video_summary" not in parsed_json or "shots" not in parsed_json:
#                    raise ValueError("Missing required fields: video_summary or shots")
#                
                # Convert shots to VisionShot objects
#                shots = [VisionShot(**shot) for shot in parsed_json["shots"]]
                
                # Create VisionResult with metadata
                metadata = {
                    "model": self.model,
                    "frames_analyzed": len(valid_frames),
                    "raw_response": raw_response
                }
                
                return raw_response
#                    video_summary=parsed_json["video_summary"],
#                    shots=shots,
                
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {str(e)}")
                print(f"Content (first 500): {content[:500]}")
                
                # Try to extract JSON from text
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        parsed_json = json.loads(json_match.group(0))
                        print("Successfully extracted JSON from text")
                    except:
                        raise RuntimeError(
                            f"Response content is not valid JSON: {str(e)}\n"
                            f"Raw response: {raw_response[:500]}"
                        )
                else:
                    raise RuntimeError(
                        f"Response content is not valid JSON: {str(e)}\n"
                        f"Raw response: {raw_response[:500]}"
                    )
            except ValueError as e:
                raise RuntimeError(
                    f"Invalid response structure: {str(e)}\n"
                    f"Response: {json.dumps(parsed_json)}"
                )
        
        except requests.exceptions.HTTPError as e:
            error_detail = ""
            if hasattr(e, 'response') and e.response is not None:
                error_detail = e.response.text[:500]
            raise RuntimeError(
                f"OpenRouter API HTTP error {e.response.status_code}: {error_detail}"
            )
        
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"OpenRouter API request failed: {str(e)}")
        
        except RuntimeError:
            # Re-raise RuntimeError as-is
            raise
        
        except Exception as e:
            # Catch any other unexpected exceptions and wrap them
            import traceback
            tb = traceback.format_exc()
            raise RuntimeError(
                f"Unexpected error during vision analysis: {str(e)}\n"
                f"Traceback:\n{tb}"
            )

def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


if __name__ == "__main__":
    
    
    video_path_list = ["RoripwjYFp8_60.0_210.0",
                       "_0EdHKxcRHM_60.0_210.0",
                       "_0EdHKxcRHM_210.0_360.0",
                       "_0EdHKxcRHM_510.0_660.0",                       
                       "_0EdHKxcRHM_660.0_810.0",
                       "_0ipsQzLdzA_60.0_210.0",
                       "_0ipsQzLdzA_210.0_360.0",
                       "_0ipsQzLdzA_360.0_510.0",
                       "_0ipsQzLdzA_510.0_660.0",
                       "_0ipsQzLdzA_660.0_810.0",
                       "_0u5I0OJP6U_60.0_210.0",
                       "_0u5I0OJP6U_210.0_360.0",
                       "_2mgEMfnYzw_60.0_210.0",
                       "_2mgEMfnYzw_210.0_360.0",
                       "_2mgEMfnYzw_360.0_510.0",
                       "_2mgEMfnYzw_510.0_660.0",
                       "_2mgEMfnYzw_660.0_810.0",
                       "_3t2jEtJX8g_210.0_360.0",
                       "_4LFOLSEYlU_60.0_210.0",
                       "NUsG9BgSes0_60.0_210.0",
                       "NUsG9BgSes0_360.0_510.0",
                      ]
    
    
    
    query_path = "example/queries3.jsonl"
    
    queries = load_jsonl(query_path)
     
    total_list = []
    
    for idx in range(19, len(video_path_list)):
        
        query_text_list = []
        
        video_path = "example/" + video_path_list[idx] + ".mp4"

        query_data = queries[idx]
        
        query_text_list = query_data["query"]

        
        a = FramePreprocessor(workspace_path="./temp_frames")
        
        b = OpenRouterVision()
        
        user_prompt = query_text_list
        
        print(user_prompt)
        
        
        preprocess_result = a.preprocess(video_path = video_path)
        
        output = b.analyze_frames(preprocess_result, user_prompt)
        
        
        
        total_info = {'query': query_data['query'], 
                    'video_path': video_path, 
                    'GT_windows': query_data['relevant_windows'],
                    'Predicted_windows': output,
                     }
                     
        total_list.append(total_info)

    with open("data.json", "w") as file:
    
        json.dump(total_list, file)

        print("data.json has been saved...")

  