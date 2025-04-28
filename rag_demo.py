from lazyllm import Document, Retriever, Reranker, SentenceSplitter
import lazyllm


# 1. 初始化llm,embed,rerank
llm = lazyllm.OnlineChatModule(
    source="glm",
)
embed_glm = lazyllm.OnlineEmbeddingModule(
    source="glm",
)
rerank_glm = lazyllm.OnlineEmbeddingModule(
    type="rerank",
    source="glm",
)

# 2. 初始化documents
chroma_dir = "D:/projects/rag/index/"
chroma_store_conf = {
    "type": "chroma",
    "kwargs": {
        "dir": chroma_dir,
    },
    # "indices": { #windows不支持milvus_lite,先不使用
    #     # 使用 milvus 索引后端
    #     "smart_embedding_index": {
    #         "backend": "milvus",
    #         "kwargs": {
    #             "uri": chroma_dir+"milvus.db",
    #             "index_kwargs": {
    #                 "index_type": "HNSW",
    #                 "metric_type": "COSINE",
    #             },
    #         },
    #     },
    # },
}

documents = Document(
    dataset_path="D:/projects/rag/resources",
    embed=embed_glm,
    manager=False,
    store_conf=chroma_store_conf,
)
documents.create_node_group(
    name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100
)

# 3. 初始化retriever, reranker
retriever1 = Retriever(
    documents,
    group_name="CoarseChunk",
    similarity="bm25_chinese",
    similarity_cut_off=0.1,
    topk=2,
)
retriever2 = Retriever(
    doc=documents,
    group_name="CoarseChunk",
    similarity="cosine",
    similarity_cut_off=0.1,
    topk=2,
)
retriever3 = Retriever(
    doc=documents,
    group_name="sentences",
    similarity="cosine",
    similarity_cut_off=0.1,
    topk=5,
)
reranker = Reranker(name="ModuleReranker", model=rerank_glm, topk=3)

# 4. 调用retriever组件，传入query
query = "deepseek V3主要改进"
retriever_result1 = retriever1(query=query)
retriever_result2 = retriever2(query=query)
retriever_result3 = retriever3(query=query)
for node in retriever_result1:
    print("retriever_result1:", node.get_content())
for node in retriever_result2:
    print("retriever_result2:", node.get_content())
for node in retriever_result3:
    print("retriever_result3:", node.get_content())

# 5. 去重: 合并结果并基于内容去重
combined_results = retriever_result1 + retriever_result2 + retriever_result3
unique_nodes = []
unique_contents = set()

for node in combined_results:
    content = node.get_content()
    if content not in unique_contents:
        unique_contents.add(content)
        unique_nodes.append(node)

print(f"原始结果数: {len(combined_results)}, 去重后结果数: {len(unique_nodes)}")

# 6. 调用reranker组件，传入去重后的结果
retriever_result = reranker(
    nodes=unique_nodes, query=query
)
for node in retriever_result:
    print("reranked_result:", node.get_content())

# 7. 调用llm组件，传入query和context_str
prompt = (
    "你将扮演一个人工智能问答助手的角色，完成一项对话任务。"
    "在这个任务中，你需要根据给定的上下文以及问题，给出你的回答。"
)
llm.prompt(lazyllm.ChatPrompter(
    instruction=prompt, extro_keys=["context_str"]
))
res = llm(
    {
        "query": query,
        "context_str": "".join(
            [node.get_content() for node in retriever_result]
        ),
    }
)

print(f"answer: {res}")
