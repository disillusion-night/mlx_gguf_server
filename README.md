# mlx_gguf_server
## Load MLX and GGUF format LLM Models on Apple Silicon Macs.

## Table of Contents
- [Abstract](#abstract)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [API Requests and Responses](#api-requests-and-responses)
  - [Loading and Accessing Multiple Models](#loading-and-accessing-multiple-models)
  - [Process Management](#process-management)
- [Unsupported/Future improvements](#unsupported/future-improvements)


# Abstract
This is my practice program for learning LLMs and Python. 

Serve multiple Large Language Models simultaneously on Apple Silicon Macs.

Supports both MLX format and llama.cpp(gguf) format models. The MLX format models are loaded using the `mlx` and `mlx_lm` libraries, while the llama.cpp(gguf) format models are loaded using the `llama-cpp-python` library. 

By leveraging multiprocessing, load, unload, and switch between multiple LLM models

This program is using MLX framework, so run only on Apple Silicon Macs.


# Installation
1. Install the required dependencies by running `pip install -r requirements.txt`.

2. Place your MLX format(folder) or gguf format of model files into `models` directory.

3. Start the server by running `python main.py`. By default, `http://127.0.0.1:4000` will be used for service. You can change listen address and port by using arguments.

Use the provided FastAPI endpoints to interact with the LLM models. the following are examples of API execution and its output.


# Usage

## Major APIs
`/v1/internal/model/list`

Get a list of available models.

```console
$ curl -X GET http://localhost:4000/v1/internal/model/list

{"model_names":["Mistral-7B-Instruct-v0.2","Mixtral-8x7B-Instruct-v0.1_Q4","gemma-2b","gemma-2b.gguf"]}
```
In this case, there are four models are stored in the models directory. "Mistral-7B-Instruct-v0.2", "Mixtral-8x7B-Instruct-v0.1_Q4" and "gemma-2b" are directories that be stored MLX format LLM models. "gemma-2b.gguf" is a single file of GGUF format.
```
├── main.py
└── models
    ├── Mistral-7B-Instruct-v0.2
    ├── Mixtral-8x7B-Instruct-v0.1_Q4
    ├── gemma-2b
    └── gemma-2b.gguf
```

`/v1/internal/model/load`

Load a specific model. If the load is successful, {"load": "success"} is return.

```console
$ curl -X POST -H "Content-Type: application/json" -d '{"llm_model_name": "gemma-2b"}' http://localhost:4000/v1/internal/model/load

{"load":"success"}
```

For gguf load, ``chat_format`` parameter is supported. This parameter is optional.
```console
$ curl -X POST -H "Content-Type: application/json" -d '{"llm_model_name": "gemma-2b.gguf","chat_format":"gemma" }' http://localhost:4000/v1/internal/model/load

{"load":"success"}
```

`/v1/completions`

Generate completions using loaded LLM model. Supported parameters are listed in ``llm_process/llm_model.py``

```console
curl -s -X POST -H "Content-Type: application/json" -d '{"prompt": "Your prompt here", "max_tokens": 50}' http://localhost:4000/v1/completions | jq
{
  "id": "2182c466-12f0-41da-83fe-c868c85bbdcb",
  "object": "text_completion",
  "created": 1713714528,
  "model": "gemma-2b-it_Q8_0",
  "choices": [
    {
      "text": " is a bit too vague. To improve the clarity, please specify the following:\n\n* What do you want the user to be able to do with the generated abstract?\n* What type of information do you want the abstract to include?\n*"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 50,
    "total_tokens": 54
  }
}
```

`/v1/chat/completions`

Generate chat completions using loaded LLM model. Supporting parameters are almost same as ``v1/completions`` but the user sends a ``messages`` List. This server automatically tries to apply chat-templates based on model's information. 

```console
$ curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Your prompt here", "max_tokens": 50}' http://localhost:4000/v1/completions

{"id":"d0ae05ea-c507-4397-ad83-14d26e208a69","object":"text_completion","created":1712068570,"model":"gemma-2b","choices":[{"text":" has led me to a number of things that I never suspected before. For example:\n\nMy understanding is that the <code>.git/HEAD</code> file, among other things, contains a SHA. However, that SHA is also the <code>commit"}],"usage":{"prompt_tokens":4,"completion_tokens":50,"total_tokens":54}}
```

```console
curl -s -X POST -H "Content-Type: application/json" -H "X-Model-Id: 0" -d '{"messages": [{"role": "user", "content": "hello"}]}' http://localhost:4000/v1/chat/completions | jq 
{
  "id": "f84da751-9a03-466e-aa4d-b40eaf5f7613",
  "object": "chat.completion",
  "created": 1713716076,
  "model": "gemma-2b-it_Q8_0",
  "choices": [
    {
      "message": {
        "content": "Hello! 👋  It's great to hear from you. How can I assist you today? 😊"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 11,
    "completion_tokens": 21,
    "total_tokens": 32
  }
}
```

`/v1/internal/token-count`

Get the token count for a given prompt.

```console 
$ curl -X POST -H "Content-Type: application/json" -d '{"prompt": "Your prompt here"}' http://localhost:4000/v1/internal/token-count

{"length":3}
```
          
`/v1/internal/model/unload`

Unload model.

```console
$ curl -X POST http://localhost:4000/v1/internal/model/unload

{"unload":"success"}%
```

## Loading and Accessing Multiple Models
You can load and access multiple models simultaneously by using `X-Model-Id"` header.

```console
$ curl -X POST -H "Content-Type: application/json" -H "X-Model-Id: 0" -d '{"llm_model_name": "gemma-2b"}' http://localhost:4000/v1/internal/model/load
{"load":"success"}

$ curl -X POST -H "Content-Type: application/json" -H "X-Model-Id: 1" -d '{"llm_model_name": "gemma-2b.gguf"}' http://localhost:4000/v1/internal/model/load
{"load":"success"}

$ curl -X POST -H "Content-Type: application/json" -H "X-Model-Id: 2" -d '{"llm_model_name": "Mixtral-8x7B-Instruct-v0.1_Q4"}' http://localhost:4000/v1/internal/model/load
{"load":"success"}
```

The above commands load "gemma-2b (load by MLX)" to Model ID 0, "gemma-2b.gguf (load by llama.cpp)" to Model ID 1 and "Mixtral-8x7B-Instruct-v0.1_Q4 (load by MLX)" to Model ID 2. 
If the HTTP request does not contain an "X-Model-Id" header, the request targets model_id 0 (same as -H "X-Model-Id: 0").

## processes management
If you run several models, each models are run as dedicated process. Each task (load, completion, token-count) are passed through FIFO queue and executed one by one.
You can check the each processes status and current queue through following API.

`/management/processes`

This API does not require an X-Model-Id.　All on loading processes information is returned.
In this example, two models are loaded, neither of which currently has any particular task.

```console
$ curl -s -X GET -H "Content-Type: application/json"  http://localhost:4000/management/processes | jq
{
  "processes": [
    {
      "model_id": "0",
      "model_name": "gemma-1.1-2b-it_Q8_0",
      "model_path": "models/gemma-1.1-2b-it_Q8_0",
      "model_type": "mlx",
      "context_length": 8192,
      "process_id": 29214,
      "cpu_usage": 0.0,
      "memory_usage": 3827122176,
      "current_queue": {
        "request_queue_size": 0,
        "response_queue_size": 0,
        "queues": {}
      }
    },
    {
      "model_id": "1",
      "model_name": "gemma-1.1-2b-it-GGUF_Q8_0.gguf",
      "model_path": "models/gemma-1.1-2b-it-GGUF_Q8_0.gguf",
      "model_type": "llama-cpp",
      "context_length": 8192,
      "process_id": 29219,
      "cpu_usage": 0.0,
      "memory_usage": 448593920,
      "current_queue": {
        "request_queue_size": 0,
        "response_queue_size": 0,
        "queues": {}
      }
    }
  ]
}
```

In this second example, Model Id 1 has two chat-completion tasks in queue.
```console
curl -s -X GET -H "Content-Type: application/json"  http://localhost:4000/management/processes | jq
{
  "processes": [
    {
      "model_id": "0",
      "model_name": "gemma-1.1-2b-it_Q8_0",
      "model_path": "models/gemma-1.1-2b-it_Q8_0",
      "model_type": "mlx",
      "context_length": 8192,
      "process_id": 29278,
      "cpu_usage": 0.0,
      "memory_usage": 3817078784,
      "current_queue": {
        "request_queue_size": 0,
        "response_queue_size": 1,
        "queues": {}
      }
    },
    {
      "model_id": "1",
      "model_name": "gemma-1.1-2b-it-GGUF_Q8_0.gguf",
      "model_path": "models/gemma-1.1-2b-it-GGUF_Q8_0.gguf",
      "model_type": "llama-cpp",
      "context_length": 8192,
      "process_id": 29284,
      "cpu_usage": 0.0,
      "memory_usage": 506052608,
      "current_queue": {
        "request_queue_size": 0,
        "response_queue_size": 53,
        "queues": {
          "b966a45c-6560-46d9-b70d-445cff6faf46": {
            "completions_stream": {
              "model": "dummy",
              "prompt": "",
              "messages": [
                {
                  "role": "user",
                  "content": "Hello!"
                }
              ],
              "max_tokens": 50,
              "temperature": 0.0,
              "seed": null,
              "stream": true,
              "apply_chat_template": true,
              "complete_text": false,
              "top_p": 1.0,
              "stop": [],
              "repetition_penalty": null,
              "repetition_context_size": 20,
              "top_k": 40,
              "min_p": 0.05,
              "typical_p": 1.0,
              "frequency_penalty": 0.0,
              "presence_penalty": 0.0,
              "repet_penalty": 1.1,
              "mirostat_mode": 0,
              "mirostat_tau": 5.0,
              "mirostat_eta": 0.1
            },
            "start_time": **********.*******
          },
          "3d6dfd64-5fcb-4edd-b4c3-62dda062c24f": {
            "completions_stream": {
              "model": "dummy",
              "prompt": "",
              "messages": [
                {
                  "role": "user",
                  "content": "Hello!"
                }
              ],
              "max_tokens": 50,
              "temperature": 0.0,
              "seed": null,
              "stream": true,
              "apply_chat_template": true,
              "complete_text": true,
              "top_p": 1.0,
              "stop": [],
              "repetition_penalty": null,
              "repetition_context_size": 20,
              "top_k": 40,
              "min_p": 0.05,
              "typical_p": 1.0,
              "frequency_penalty": 0.0,
              "presence_penalty": 0.0,
              "repet_penalty": 1.1,
              "mirostat_mode": 0,
              "mirostat_tau": 5.0,
              "mirostat_eta": 0.1
            },
            "start_time": **********.*******
          }
        }
      }
    }
  ]
}
```

If you disconnect the client-server connection during stream output, tasks will continue to remain in the Queue. To forcefully empty the Queue, use the following API.

`/management/process/clean-up`

This API specifies the parameter "timeout". Tasks that are older than the time specified by that value will be deleted from the Queue.

```console
curl -X POST -H "Content-Type: application/json" -H "X-Model-Id: 1" -d '{"timeout": 1}' http://localhost:4000/management/process/clean-up

{"process_clean_up":"success"}
```


# Unsupported/Future improvements

* Load LoRA (adapter) is currently unsupported yet.
