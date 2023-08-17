# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import distutils.util
import os
import paddle
#import fastdeploy as fd
import time

from paddlenlp.transformers import (
    ChatGLMv2Config,
    ChatGLMv2ForConditionalGeneration,
    ChatGLMv2Tokenizer,
)

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"函数 {func.__name__}的执行时间: {execution_time} seconds")
        return result
    return wrapper
def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="THUDM/chatglm2-6b", help="The directory of model.")
    parser.add_argument("--model_path", default="/workspace/PaddleNLP/examples/language_model/chatglm_v2/inference", help="The directory of model.")
    parser.add_argument("--model_prefix", type=str, default="chatglm", help="The model and params file prefix.")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Type of inference device, support 'cpu' or 'gpu'.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="paddle",
        choices=["onnx_runtime", "paddle", "openvino", "tensorrt", "paddle_tensorrt"],
        help="The inference runtime backend.",
    )
    # parser.add_argument(
    #     "--device",
    #     type=str,
    #     default="gpu",
    #     choices=["gpu", "cpu"],
    #     help="Type of inference device, support 'cpu' or 'gpu'.",
    # )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size of data.")
    parser.add_argument("--src_length", type=int, default=128, help="The batch size of data.")
    parser.add_argument("--tgt_length", type=int, default=128, help="The batch size of data.")
    parser.add_argument("--device_id", type=int, default=0, help="Select which gpu device to train model.")
    
    return parser.parse_args()


def batchfy_text(texts, batch_size):
    batch_texts = []
    batch_start = 0
    while batch_start < len(texts):
        batch_texts += [texts[batch_start : min(batch_start + batch_size, len(texts))]]
        batch_start += batch_size
    return batch_texts



class Predictor(object):
    def __init__(self, args):
        self.tokenizer = ChatGLMv2Tokenizer.from_pretrained(args.model_name_or_path)
        self.batch_size = args.batch_size
        self.src_length = args.src_length
        self.tgt_length = args.tgt_length

        model_path = os.path.join(args.model_path, args.model_prefix + ".pdmodel")
        params_path = os.path.join(args.model_path, args.model_prefix + ".pdiparams")
        config = paddle.inference.Config(model_path, params_path)

        if args.device == "gpu":
            # set GPU configs accordingly
            config.enable_use_gpu(100, 0)
        elif args.device == "cpu":
            # set CPU configs accordingly,
            # such as enable_mkldnn, set_cpu_math_library_num_threads
            config.disable_gpu()
        config.disable_glog_info()
        config.switch_ir_optim(False)
        self.predictor = paddle.inference.create_predictor(config)
    

    def preprocess(self, input_text):
        inputs = self.tokenizer(
            input_text,
            return_tensors="np",
            padding=True,
            max_length=self.src_length,
            truncation=True,
            truncation_side="left",
        )
        return inputs
    @measure_time
    def infer(self, input_map):
        input_handles = {}
        for name in self.predictor.get_input_names():
            input_handles[name] = self.predictor.get_input_handle(name)
            input_handles[name].copy_from_cpu(input_map[name])

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        output_handle = self.predictor.get_output_handle(output_names[0])
        results = output_handle.copy_to_cpu()
        return results
        # results = self.predictor(dict(input_map))
        # return results

    def postprocess(self, infer_data):
        result = []
        for x in infer_data.tolist():
            res = self.tokenizer.decode(x, skip_special_tokens=True)
            res = res.strip("\n")
            result.append(res)
        out_dict = {"result": result}
        return out_dict

    def predict(self, texts):
        input_map = self.preprocess(texts)
        infer_result = self.infer(input_map)
        output = self.postprocess(infer_result)
        return output


if __name__ == "__main__":
    args = parse_arguments()
    predictor = Predictor(args)
    all_texts = [
        "你好",
        "[Round 0]\n问：你好\n答：你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。\n[Round 1]\n问：晚上睡不着应该怎么办\n答：",
    ]
    batch_texts = batchfy_text(all_texts, args.batch_size)
    for bs, texts in enumerate(batch_texts):
        outputs = predictor.predict(texts)
        for text, result in zip(texts, outputs["result"]):
            print("{} \n {}".format(text, result))
