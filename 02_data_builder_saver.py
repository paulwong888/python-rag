import os, time, re, json
import logging, backoff
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI
from dotenv import load_dotenv

class DatasetBuilderAndSaver():

    def __init__(self):
        # 设置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        load_dotenv()
        # 以下是本地模型的调用,非在线模型服务调用
        # model_name = ""
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.openai = OpenAI()

    def read_file(self, file_name: str):
        with open(file_name, "r", encoding="utf-8") as file:
            return file.read()
        
    def gen_dataset(self, files_dir: str, entries_per_file: int = 2):
        result_list = []
        file_list = os.listdir(files_dir)
        for file_name in file_list:
            if file_name.endswith(".txt"):
                file_path = os.path.join(files_dir, file_name)
                file_content = self.read_file(file_path)
                for j in range(entries_per_file):
                    self.logger.info(f"  生成第 {j + 1}/{entries_per_file} 个条目")
                    output = self.gen_single_entry(file_content)
                    if output and all(key in output for key in ["instruction", "input", "output", "text"]):
                        result_list.append(output)
                        self.logger.info(f"  成功生成 1 个完整条目")
                    else:
                        self.logger.warning(f"  跳过不完整的条目")
                    time.sleep(2) # 在请求之间增加延迟到2秒
        
        return result_list

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def gen_single_entry(self, input: str):
        prompt = f"""
            基于以下文本，生成1个用于指令数据集的高质量条目。条目应该直接关联到给定的文本内容，提出相关的问题或任务。
            请确保生成多样化的指令类型，例如：
            - 分析类："分析..."
            - 比较类："比较..."
            - 解释类："解释..."
            - 评价类："评价..."
            - 问答类："为什么..."

            文本内容：
            {input}

            请以下面的格式生成条目，确保所有字段都有适当的内容：
            {{
                "instruction": "使用上述多样化的指令类型之一，提出一个具体的、与文本相关的问题或任务",
                "input": "如果需要额外的上下文信息，请在这里提供，否则留空",
                "output": "对instruction的详细回答或任务的完成结果"
            }}
            确保所有生成的内容都与给定的文本直接相关，生成的是有效的JSON格式，并且内容高质量、准确、详细。
            """
        # self.logger.info(prompt)
        try:
            response = self.openai.chat.completions.create(
                model="gpt-4o-mini", 
                messages = [{"role": "user", "content": prompt}],
                temperature=0.7, max_tokens=4098
            )
            response_content = response.choices[0].message.content
            self.logger.info(f"API 响应: {response_content}")

            json_match = re.search("\{.*\}", response_content, re.DOTALL)
            if json_match:
                entry = json.loads(json_match.group())
            required_keys = ['instruction', 'input', 'output']
            if isinstance(entry, dict) and all(key in entry for key in required_keys):
                # 根据 input 是否为空来设置 text 字段
                if entry['input'].strip():
                    entry[
                        'text'] = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.### Instruction: {entry['instruction']}\n### Input: {entry['input']}\n### Response: {entry['output']}"
                else:
                    entry[
                        'text'] = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.### Instruction: {entry['instruction']}\n### Input: {entry['input']}\n### Response: {entry['output']}"

                self.logger.info("成功生成完整条目")
                return entry
            else:
                self.logger.warning("JSON 解析成功，但缺少必要字段")
                return []

        except Exception as e:
            self.logger.error("无法从API响应中提取有效的JSON")
            raise

    def save_as_pq(self, content_list: list[dict], save_dir: str):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        output_file = os.path.join(save_dir, "instruction_dataset.parquet")
        t1 = pa.string()
        # schema = pa.schema([
        #     pa.field("instruction", t1),
        #     pa.field("input", t1),
        #     pa.field("output", t1),
        #     pa.field("text", t1),
        # ])
        schema = ["instruction", "input", "output", "text"]
        arrays = [
            pa.array([content["instruction"] for content in content_list]),
            pa.array([content["input"] for content in content_list]),
            pa.array([content["output"] for content in content_list]),
            pa.array([content["text"] for content in content_list]),
        ]
        table = pa.Table.from_arrays(arrays, schema)
        pq.write_table(table, output_file)

def build_data():
    dataset_builder_saver = DatasetBuilderAndSaver()
    logger = dataset_builder_saver.logger
    input_dir = "data/hong_lou_meng/output"
    save_dir = "data/hong_lou_meng/datasets"
    logger.info("开始生成数据集")
    data_item_list = dataset_builder_saver.gen_dataset(input_dir)
    dataset_builder_saver.save_as_pq(data_item_list, save_dir)
    logger.info(f"数据集已生成并保存到 {save_dir}")
    logger.info(f"共生成 {len(data_item_list)} 个有效条目")

def test_pyarrow():
    dataset_builder_saver = DatasetBuilderAndSaver()
    content_list = [{
        "instruction": [111],
        "input": [222],
        "output": [333],
        "text": [555],
    }]
    save_dir = "data/hong_lou_meng/datasets"
    dataset_builder_saver.save_as_pq(content_list, save_dir)

def test_1():
    import pyarrow as pa
    import pyarrow.dataset as pt
    # Create a PyArrow table
    data = [
        pa.array([1, 2, 3]),
        pa.array(['a', 'b', 'c']),
        pa.array([1.1, 2.2, 3.3])
    ]
    table = pa.Table.from_arrays(data, names=['col1', 'col2', 'col3'])
    # Display the table
    print(table)

if __name__ == "__main__":
    build_data()
    # test_pyarrow()
    # test_1()