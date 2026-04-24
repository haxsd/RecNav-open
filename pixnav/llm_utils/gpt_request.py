import base64
import os
from mimetypes import guess_type

import cv2
import numpy as np
from openai import AzureOpenAI, OpenAI

_API_VERSION = "2024-02-15-preview"
_GPT4_DEPLOYMENT = os.environ.get("PIXNAV_GPT4_DEPLOYMENT", "gpt-4-1106-preview")
_GPT4V_DEPLOYMENT = os.environ.get("PIXNAV_GPT4V_DEPLOYMENT", "gpt-4-vision-preview")
_TEXT_MODEL = os.environ.get("PIXNAV_TEXT_MODEL", os.environ.get("DEEPSEEK_MODEL", os.environ.get("OPENAI_MODEL", "deepseek-chat")))
_VISION_MODEL = os.environ.get("PIXNAV_VISION_MODEL", os.environ.get("DEEPSEEK_VISION_MODEL", _TEXT_MODEL))
_DASHSCOPE_TEXT_MODEL = os.environ.get("DASHSCOPE_TEXT_MODEL", "qwen-plus")
_DASHSCOPE_VISION_MODEL = os.environ.get("DASHSCOPE_VISION_MODEL", "qwen-vl-max")

_gpt4_client = None
_gpt4_model = None
_gpt4v_client = None
_gpt4v_model = None
_gpt4_client_is_azure = False
_gpt4v_client_is_azure = False


def _build_client(deployment_name: str):
    azure_base = os.environ.get("OPENAI_API_ENDPOINT", "").strip()
    azure_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if azure_base and azure_key:
        return (
            AzureOpenAI(
                api_key=azure_key,
                api_version=_API_VERSION,
                base_url=f"{azure_base}/openai/deployments/{deployment_name}",
            ),
            True,
            deployment_name,
        )

    dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if dashscope_key:
        base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
        return OpenAI(api_key=dashscope_key, base_url=base_url), False, _DASHSCOPE_TEXT_MODEL

    api_key = os.environ.get("OPENAI_API_KEY", "").strip() or os.environ.get("DEEPSEEK_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or os.environ.get("DEEPSEEK_BASE_URL", "").strip()
    if not api_key:
        return None, False, _TEXT_MODEL
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs), False, _TEXT_MODEL


def _build_vision_client():
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    base_url = os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1").strip()
    if api_key:
        return OpenAI(api_key=api_key, base_url=base_url), False, _DASHSCOPE_VISION_MODEL
    client, is_azure, _ = _build_client(_GPT4V_DEPLOYMENT)
    model = _GPT4V_DEPLOYMENT if is_azure else _VISION_MODEL
    return client, is_azure, model

# Function to encode a local image into data URL 
def local_image_to_data_url(image):
    if isinstance(image,str):
        mime_type, _ = guess_type(image)
        with open(image, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{base64_encoded_data}"
    elif isinstance(image,np.ndarray):
        base64_encoded_data = base64.b64encode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY, 60])[1]).decode('utf-8')
        return f"data:image/jpeg;base64,{base64_encoded_data}"

def gptv_response(text_prompt,image_prompt,system_prompt=""):
    global _gpt4v_client, _gpt4v_client_is_azure, _gpt4v_model
    if _gpt4v_client is None:
        _gpt4v_client, _gpt4v_client_is_azure, _gpt4v_model = _build_vision_client()
    if _gpt4v_client is None:
        raise RuntimeError("No LLM credentials found. Set DASHSCOPE_API_KEY, DEEPSEEK_API_KEY or OPENAI_API_KEY.")

    prompt = [{'role':'system','content':system_prompt},
             {'role':'user','content':[{'type':'text','text':text_prompt},
                                       {'type':'image_url','image_url':{'url':local_image_to_data_url(image_prompt)}}]}]
    response = _gpt4v_client.chat.completions.create(model=_gpt4v_model,
                                                    messages=prompt,
                                                    max_tokens=300,
                                                    temperature=0.0)
    return response.choices[0].message.content

def gpt_response(text_prompt,system_prompt=""):
    global _gpt4_client, _gpt4_client_is_azure, _gpt4_model
    if _gpt4_client is None:
        _gpt4_client, _gpt4_client_is_azure, _gpt4_model = _build_client(_GPT4_DEPLOYMENT)
    if _gpt4_client is None:
        raise RuntimeError("No LLM credentials found. Set DASHSCOPE_API_KEY or OPENAI_API_KEY.")

    prompt = [{'role':'system','content':system_prompt},
              {'role':'user','content':[{'type':'text','text':text_prompt}]}]
    response = _gpt4_client.chat.completions.create(model=_gpt4_model,
                                              messages=prompt,
                                              max_tokens=1000,
                                              temperature=0.0)
    return response.choices[0].message.content
