from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.postprocess.model_postprocessors import extract_non_reasoning_content

# /home is bind-mounted into the container, so outputs are visible on the host.
# ais_bench automatically appends a timestamp subdirectory.

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr="longbench_8153",
        path = "/mnt/data/models/DeepSeek-V4-Flash",
        model="deepseek_v4",
        stream=False,
        request_rate=0,
        use_timestamp=False,
        retry=2,
        host_ip="127.0.0.1",
        host_port=8153,
        max_out_len=4096,
        batch_size=1,
        trust_remote_code=False,
        generation_kwargs=dict(
            temperature=1,
            top_p = 0.95,
            chat_template_kwargs={"enable_thinking": True}
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
]
