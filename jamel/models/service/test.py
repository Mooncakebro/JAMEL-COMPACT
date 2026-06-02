#!/usr/bin/env python3
"""
测试补全功能的简单脚本
"""

import asyncio
import base64
import io
import traceback
from PIL import Image
import requests
import json

def image_to_base64(image_path):
    """将图片转换为 base64"""
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

def test_chat_completion(base_url="http://localhost:3210"):
    """测试 OpenAI 兼容的聊天补全接口"""
    print("Testing chat completion...")

    url = f"{base_url}/v1/chat/completions"
    user_content = "Who are you?"
    data = {
        "model": "gpt-3.5-turbo",  # 使用标准模型名
        "messages": [
            # {
            #     "role": "system",
            #     "content": "You are a helpful assistant."
            # },
            {
                "role": "user",
                "content": user_content
            }
        ],
        "max_tokens": 50,
        "temperature": 0.1,
        "stream": False
    }

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Chat completion successful:")
            print(f"   User: {user_content}")
            print(f"   Assistant: {result['choices'][0]['message']['content']}")
            if 'usage' in result:
                print(f"   Usage: {result['usage']}")
            return True
        else:
            print(f"❌ Chat completion failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Error testing chat completion: {e}")
        return False

def test_text_completion(base_url="http://localhost:3210"):
    """测试文本补全"""
    print("Testing text completion...")

    url = f"{base_url}/v1/completions"
    data = {
        "prompt": "\"Who are you?\"",
        "max_tokens": 50,
        "temperature": 0.7,
        "top_p": 0.9
    }

    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Text completion successful:")
            print(f"   Prompt: {data['prompt']}")
            print(f"   Completion: {result['choices'][0]['text']}")
            if 'usage' in result:
                print(f"   Usage: {result['usage']}")
            return True
        else:
            print(f"❌ Text completion failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing text completion: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 Starting completion API tests...\n")

    # 可以通过命令行参数指定服务器地址
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3210"

    tests = [
        test_chat_completion,
        # test_text_completion
    ]

    results = []
    for test in tests:
        try:
            result = test(base_url)
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    print(f"\n📊 Test Results:")
    print(f"   Passed: {sum(results)}/{len(results)}")
    print(f"   Failed: {len(results) - sum(results)}/{len(results)}")

    if all(results):
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed.")

if __name__ == "__main__":
    main()