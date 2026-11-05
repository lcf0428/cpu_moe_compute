# Running AISBench with a Custom Configuration File
The standard command invocation method for AISBench specifies the model task via `--models`, the dataset task via `--datasets`, and the result presentation task via `--summarizer` to run an evaluation task. Additionally, AISBench supports specifying a **custom configuration file** that combines the configuration information of these three types of tasks, enabling the execution of custom task combinations.


## Usage Instructions
```bash
ais_bench ais_bench/configs/{model_type}_examples/{task_config_filename}
# Example:
ais_bench ais_bench/configs/api_examples/infer_vllm_api_general.py
```


## Example of Using a Custom Configuration File for Accuracy Evaluation
### Editing the Example Content
The following example demonstrates how to evaluate the performance of two service interfaces ([`v1/chat/completions`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) and [`v1/completions`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)) on the [GSM8K](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/gsm8k/README_en.md) and [MATH datasets](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/math/README_en.md). Refer to the sample file: [demo_infer_vllm_api.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/demo_infer_vllm_api.py):

```python
from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask
from ais_bench.benchmark.models import VLLMCustomAPIChat

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general

# Use only a subset of samples for demo testing
gsm8k_0_shot_cot_str[0]['abbr'] = 'demo_' + gsm8k_0_shot_cot_str[0]['abbr']
gsm8k_0_shot_cot_str[0]['reader_cfg']['test_range'] = '[0:8]'

math500_gen_0_shot_cot_chat[0]['abbr'] = 'demo_' + math500_gen_0_shot_cot_chat[0]['abbr']
math500_gen_0_shot_cot_chat[0]['reader_cfg']['test_range'] = '[0:8]'

# Specify the dataset list; add different dataset configurations by concatenation
datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat
# Specify the model configuration list
models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='demo-vllm-api-general-chat',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",  # Specify the IP address of the inference service
        host_port = 8080,       # Specify the port of the inference service
        max_out_len = 512,
        batch_size=1,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        )
    )
]

work_dir = 'outputs/demo_api-vllm-general-chat/'
```


### Executing the Custom Task Combination
After modifying the configuration file, run the following command to start the accuracy evaluation:
```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_general_chat.py
```

If you need to execute multiple tasks in parallel, you can add the [`--max-num-workers`](../base_tutorials/all_params/cli_args.md#common-parameters) parameter to the command line to specify the maximum number of parallel tasks. Example:
```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_general_chat.py --max-num-workers 4
```


### Output Results
```bash
dataset                 version  metric   mode  demo-vllm-api-general-chat demo-vllm-api-general
----------------------- -------- -------- ----- -------------------------- ---------------------
demo_gsm8k              401e4c   accuracy gen                     62.50                62.50
demo_math_prm800k_500   c4b6f0   accuracy gen                     50.00                62.50
```

## Example of Using a Custom Configuration File for Performance Evaluation
### Editing the Example Content
The following example demonstrates how to evaluate the performance of two service interfaces ([`v1/chat/completions`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) and [`v1/completions`](https://github.com/AISBench/benchmark/tree/master/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)) using synthetic datasets for performance evaluation. Refer to the sample file: [demo_infer_vllm_api_perf.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py):

```python
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.synthetic.synthetic_gen_string import (
        synthetic_datasets,
    )
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_stream import (
        models as vllm_api_general_stream,
    )
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import (
        models as vllm_api_stream_chat,
    )

datasets = synthetic_datasets  # Specify the dataset list

vllm_api_general_stream[0]["abbr"] = "demo-" + vllm_api_general_stream[0]["abbr"]
vllm_api_stream_chat[0]["abbr"] = "demo-" + vllm_api_stream_chat[0]["abbr"]

models = vllm_api_general_stream + vllm_api_stream_chat # Specify the model list

work_dir = "outputs/demo_api-vllm-stream-perf/"
```

### Executing the Custom Task Combination
After modifying the configuration file, run the following command to start the performance evaluation:
```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py -m perf
```

If you need to execute multiple tasks in parallel, you can add the [`--max-num-workers`](../base_tutorials/all_params/cli_args.md#common-parameters) parameter to the command line to specify the maximum number of parallel tasks. Example:
```bash
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_perf.py -m perf --max-num-workers 2
```

### Output Results
```bash
[2025-12-05 12:10:44,147] [ais_bench] [INFO] Performance Results of task [demo-vllm-api-general-stream/syntheticdataset]: 
╒══════════════════════════╤═════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤═════╕
│ Performance Parameters   │ Stage   │ Average         │ Min             │ Max             │ Median          │ P75             │ P90             │ P99             │  N  │
╞══════════════════════════╪═════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪═════╡
│ E2EL                     │ total   │ 1734.3 ms       │ 544.8 ms        │ 3692.3 ms       │ 1664.0 ms       │ 2081.5 ms       │ 2748.4 ms       │ 3597.9 ms       │ 10  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼─────┤
│ TTFT                     │ total   │ 103.5 ms        │ 102.4 ms        │ 107.0 ms        │ 103.1 ms        │ 103.3 ms        │ 104.2 ms        │ 106.8 ms        │ 10  │
...
[2025-12-05 12:10:44,149] [ais_bench] [INFO] Performance Result files located in outputs/demo_api-vllm-general-stream-chat-perf/20251205_121020/performances/demo-vllm-api-general-stream-chat.
[2025-12-05 12:10:44,149] [ais_bench] [INFO] Performance Results of task [demo-vllm-api-stream-chat/syntheticdataset]: 
╒══════════════════════════╤═════════╤═════════════════╤═════════════════╤═════════════════╤═════════════════╤════════════════╤═════════════════╤═════════════════╤═════╕
│ Performance Parameters   │ Stage   │ Average         │ Min             │ Max             │ Median          │ P75            │ P90             │ P99             │  N  │
╞══════════════════════════╪═════════╪═════════════════╪═════════════════╪═════════════════╪═════════════════╪════════════════╪═════════════════╪═════════════════╪═════╡
│ E2EL                     │ total   │ 3406.7 ms       │ 372.4 ms        │ 5772.4 ms       │ 3589.8 ms       │ 4476.6 ms      │ 4921.1 ms       │ 5647.1 ms       │ 10  │
├──────────────────────────┼─────────┼─────────────────┼─────────────────┼─────────────────┼─────────────────┼────────────────┼─────────────────┼─────────────────┼─────┤
│ TTFT                     │ total   │ 103.2 ms        │ 102.0 ms        │ 107.5 ms        │ 102.9 ms        │ 103.4 ms       │ 104.3 ms        │ 107.2 ms        │ 10  │
```

## Custom Model and Dataset Combinations
By default, model and dataset combinations in custom configuration files are automatically generated as a Cartesian product based on the `models` list in the model configuration file and the `datasets` list in the dataset configuration file. The number of combinations equals the product of the lengths of the `models` list and the `datasets` list. Users can customize model-dataset combinations by configuring `model_dataset_combinations` in the configuration file.

```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general_chat import models as vllm_api_general_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_stream_chat import models as vllm_api_stream_chat

models = vllm_api_general + vllm_api_general_chat + vllm_api_stream_chat    
datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat
model_dataset_combinations = [
    dict(models=[models[0]], datasets=[datasets[0]]), # Combination 1: Use model 0 (vllm_api_general) with dataset 0 (gsm8k_0_shot_cot_str)
    dict(models=[models[1]], datasets=[datasets[1]]), # Combination 2: Use model 1 (vllm_api_general_chat) with dataset 1 (math500_gen_0_shot_cot_chat)
    dict(models=[models[2]], datasets=[datasets[0], datasets[1]]), # Combination 3: Use model 2 (vllm_api_stream_chat) with dataset 0 (gsm8k_0_shot_cot_str) and dataset 1 (math500_gen_0_shot_cot_chat)
    ...
]
```

> ⚠️ **Note**: The `abbr` parameter must be used to specify a unique identifier for models and datasets. In the same configuration file, models and datasets with the same `abbr` can only be combined once. In the following example, `vllm_api_general_copy` and `vllm_api_general` have the same `abbr`, so combination 2 will be considered the same task as combination 1 and will be skipped, even if the internal parameters differ:

```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat

vllm_api_general_copy = vllm_api_general.copy()
vllm_api_general_copy[0]['port'] = 8081
models = vllm_api_general_copy + vllm_api_general
datasets = math500_gen_0_shot_cot_chat
model_dataset_combinations = [
    dict(models=[models[1]], datasets=datasets), # Combination 1: Use model 1 (vllm_api_general) with dataset (math500_gen_0_shot_cot_chat)
    dict(models=[models[0]], datasets=datasets), # Combination 2: Use model 0 (vllm_api_general_copy) with dataset 0 (math500_gen_0_shot_cot_chat). Since vllm_api_general_copy and vllm_api_general have the same abbr, this will be considered the same task as combination 1 and will be skipped, even if the internal parameters differ
]
```

Correct approach: When reusing model or dataset configurations, modify the `abbr` parameter to make it different from the original model or dataset. For example:

```python
vllm_api_general_copy = vllm_api_general.copy()
vllm_api_general_copy[0]['abbr'] = vllm_api_general[0]['abbr'] + '-copy' # Modify abbr to identify the model
```

In this way, `vllm_api_general_copy[0]` and `vllm_api_general[0]` have different `abbr` values, so combination 2 and combination 1 are different tasks and will be executed normally.

## List of Preset Custom Configuration File Samples

| Filename | Description |
| --- | --- |
| [infer_vllm_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general.py) | Evaluates the `v1/completions` sub-service using vLLM API (version 0.6+) on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |
| [infer_mindie_stream_api_general.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_mindie_stream_api_general.py) | Evaluates the `infer` sub-service using MindIE Stream API on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |
| [infer_vllm_api_old.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_old.py) | Evaluates the `generate` sub-service using vLLM API (version 0.2.6) on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |
| [infer_vllm_api_general_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_general_chat.py) | Evaluates the `v1/chat/completions` sub-service using vLLM API (version 0.6+) on the GSM8K dataset. The prompt format is a conversation format, and the dataset path is customized. |
| [infer_vllm_api_stream_chat.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/infer_vllm_api_stream_chat.py) | Evaluates the `v1/chat/completions` sub-service with streaming inference using vLLM API (version 0.6+) on the GSM8K dataset. The prompt format is a conversation format, and the dataset path is customized. |
| [infer_hf_base_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_base_model.py) | Evaluates using the inference interface of a Hugging Face base model on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |
| [infer_hf_chat_model.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/hf_example/infer_hf_chat_model.py) | Evaluates using the inference interface of a Hugging Face chat model on the GSM8K dataset. The prompt format is a string, and the dataset path is customized. |

**Note**: To evaluate other datasets using the above custom configuration files, import additional datasets from [ais_bench/configs/api_examples/all_dataset_configs.py](https://github.com/AISBench/benchmark/tree/master/ais_bench/configs/api_examples/all_dataset_configs.py).
