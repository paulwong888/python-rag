import re
import torch
import os
from transformers import BertTokenizer, BertModel
from scipy.spatial.distance import cosine

class TxtSplitterAndSaver():

    def __init__(self, bert_model_name: str = "google-bert/bert-base-chinese"):
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.bert_tokenizesr: BertTokenizer = BertTokenizer.from_pretrained(bert_model_name)

    def read_text(self, file_name: str):
        with open(file_name, "r", encoding="utf-8") as file:
            return file.read()

    def split_text_by_semantic(self, long_text: str, max_length, similarity_threshold, sentence_size: int = -1):
        
        """
        被分解为一行文字,一行单一标点, 需要合并
        '第1章 甄士隐梦幻识通灵 贾雨村风尘怀闺秀\n\u3000\u3000此开卷第一回也'
        '。'
        '\n\u3000\u3000作者自云：因曾历过一番梦幻之后，故将真事隐去，而借通灵说撰此《石头记》一书也，故曰“甄士隐”云云'
        '。'
        """
        spllited_text_list = re.split(r"(。|！|？|；)", long_text)
        cleaned_spllited_text_list = [s + p for s, p in zip(spllited_text_list[::2], spllited_text_list[1::2]) if s]

        chunks = []
        current_chunk = cleaned_spllited_text_list[0]
        current_embedding = self.get_embeddings(current_chunk)

        for sentence in cleaned_spllited_text_list[1:sentence_size]:
            sentence_embedding = self.get_embeddings(sentence)
            similarity = 1 - cosine(current_embedding, sentence_embedding)
            if similarity > similarity_threshold and len(self.bert_tokenizesr.tokenize(current_chunk + sentence)) <= max_length:
                current_chunk += sentence
                current_embedding = (current_embedding + sentence_embedding) / 2
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
                current_embedding = sentence_embedding

        #添加最后一个chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def get_embeddings(self, text:str):
        input = self.bert_tokenizesr(text, return_tensors="pt", max_length=512, padding="max_length", truncation=True)
        with torch.no_grad():
            output = self.bert_model(**input)
        # print(type(output.pooler_output))
        # print(output.pooler_output.shape)
        # print(type(output.last_hidden_state))
        # print(output.last_hidden_state.shape)
        return output.pooler_output.squeeze().numpy()
        # return output.last_hidden_state.mean(dim=1).squeeze().numpy()
 
    def save_chunks_to_file(self, text_chunk_list: list[str], save_dir: str):
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        
        for i, text_chunk in enumerate(text_chunk_list):
            file_path = os.path.join(save_dir, f"chunk_{i+1}.txt")
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_chunk)
            print(f"已保存第{i+1} 文本块到 {file_path}")

def test_full_text():
    txt_splitter_and_saver = TxtSplitterAndSaver()

    input_file = "data/input/红楼梦.txt"
    output_dir = "data/output"

    long_text = txt_splitter_and_saver.read_text(input_file)
    spillted_text_list = txt_splitter_and_saver.split_text_by_semantic(long_text, 512, 0.5, 50).sw;
    txt_splitter_and_saver.save_chunks_to_file(spillted_text_list, output_dir)

def test_get_embedding():
    txt_splitter_and_saver = TxtSplitterAndSaver()
    text = "却说那女娲氏炼石补天之时，于大荒山无稽崖炼成高十二丈、见方二十四丈大的顽石三万六千五百零一块。"
    print(txt_splitter_and_saver.get_embeddings(text))

if __name__ == "__main__":
    # test_get_embedding()
    test_full_text()