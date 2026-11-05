from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import GPQADataset, GPQAEvaluator
from ais_bench.benchmark.utils.postprocess.text_postprocessors import first_option_postprocess

# 临时续跑配置：仅含 20260909_170443 run 中失败的 136 题（62..197），
# 行序经同余对齐保持每题选项排列/gold 与原 run 一致。跑完后删除本文件与
# gpqa_diamond_resume.csv 即还原。
gpqa_reader_cfg = dict(
    input_columns=['question', 'A', 'B', 'C', 'D'],
    output_column='answer')

gpqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='What is the correct answer to this question: {question}\nChoices:\n'
                                          '(A){A}\n'
                                          '(B){B}\n'
                                          '(C){C}\n'
                                          '(D){D}\n'
                                          'Format your response as follows: "The correct answer is (insert answer here)"'),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer))

gpqa_eval_cfg = dict(evaluator=dict(type=GPQAEvaluator),
                     pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'))

gpqa_datasets = [
    dict(
        abbr='GPQA_diamond',
        type=GPQADataset,
        path='ais_bench/datasets/gpqa/',
        name='gpqa_diamond.csv',
        reader_cfg=gpqa_reader_cfg,
        infer_cfg=gpqa_infer_cfg,
        eval_cfg=gpqa_eval_cfg)
]
