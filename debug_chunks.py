"""调试向量库分块"""
from rag.vector_store import VectorStoreService
import yaml

config = yaml.safe_load(open('config/config.yaml', encoding='utf-8'))
vs = VectorStoreService(config)
vs.load_or_create_vectorstore()

payload = vs.store.get(include=['documents', 'metadatas'])
print(f'总共有 {len(payload["documents"])} 个分块')
for i, (doc, meta) in enumerate(zip(payload['documents'], payload['metadatas'])):
    print(f'--- 分块 {i+1} ---')
    print(f'来源: {meta.get("source", "未知")}')
    print(f'内容: {doc[:150]}...')
    print()