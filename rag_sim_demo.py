from lazyllm import Document, Retriever, Reranker, SentenceSplitter
import lazyllm
from typing import List, Tuple
import sys
import heapq
import jieba
import jieba.analyse
from lazyllm.tools.rag import DocNode


# 自定义词汇权重
jieba.add_word("deepseek-v3", freq=4000)
jieba.add_word("deepseek-r1", freq=4000)


@lazyllm.tools.rag.register_similarity(mode="text", batch=True)
def jieba_weight_similarity(
    query: str, nodes: List[DocNode], **kwargs
) -> List[Tuple[DocNode, float]]:
    # 使用jieba分词
    def tokenize(text):
        return list(jieba.cut(text))

    # 获取query的关键词和权重
    query_keywords = jieba.analyse.extract_tags(query, withWeight=True)
    print(f"Query keywords: {query_keywords}")  # 打印关键词和权重
    query_keywords_dict = {kw: weight for kw, weight in query_keywords}
    query_keywords_set = set(query_keywords_dict.keys())

    # 计算query中各关键词的出现次数
    query_tokens = tokenize(query)
    query_term_freq = {}
    for token in query_tokens:
        if token in query_keywords_set:
            query_term_freq[token] = query_term_freq.get(token, 0) + 1

    # 对语料库文档进行分词
    corpus_texts = [node.get_text() for node in nodes]
    corpus_tokens = [tokenize(text) for text in corpus_texts]

    # 计算相似度 - 考虑词频因素
    scores = []
    for doc_tokens in corpus_tokens:
        # 计算文档中各关键词的出现次数
        doc_term_freq = {}
        for token in doc_tokens:
            if token in query_keywords_set:
                doc_term_freq[token] = doc_term_freq.get(token, 0) + 1

        # 计算得分：关键词词频乘以权重
        if not query_keywords_set:
            scores.append(0.0)
        else:
            score = 0.0
            total_possible_score = 0.0

            for kw in query_keywords_set:
                # 获取词频和权重
                doc_freq = doc_term_freq.get(kw, 0)
                query_freq = query_term_freq.get(kw, 0)
                weight = query_keywords_dict[kw]

                # 计算有效词频（不超过查询中词频）
                effective_freq = min(doc_freq, query_freq) if query_freq > 0 else 0

                # 累加得分：词频 * 权重
                score += effective_freq * weight
                total_possible_score += query_freq * weight

            # 归一化得分
            scores.append(score / total_possible_score if total_possible_score > 0 else 0.0)

    # 获取topk结果
    topk = min(len(nodes), kwargs.get("topk", sys.maxsize))
    indexes = heapq.nlargest(topk, range(len(scores)), scores.__getitem__)

    # 构建结果元组列表，包含节点和分数
    results = []
    print("Top-k nodes with scores:")
    for idx in indexes:
        node = nodes[idx]
        score = scores[idx]
        # 将分数保存为节点的score属性
        node.score = score  # 使用直接赋值而不是setattr
        print(f"  Node {idx}: score={score}")
        results.append((node, score))   
    return results


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

# 添加日志来显示模块是否正确初始化
print("初始化完成:")
print(f"  LLM: {type(llm).__name__}")
print(f"  Embed: {type(embed_glm).__name__}")
print(f"  Rerank: {type(rerank_glm).__name__}")

# 2. 初始化documents
chroma_dir = "D:/projects/rag/index/"
chroma_store_conf = {
    "type": "chroma",
    "kwargs": {
        "dir": chroma_dir,
    },
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
    similarity="jieba_weight_similarity",
    similarity_cut_off=0.1,
    topk=3,
)
retriever2 = Retriever(
    doc=documents,
    group_name="CoarseChunk",
    similarity="cosine",
    similarity_cut_off=0.1,
    topk=3,
)

reranker = Reranker(name="ModuleReranker", model=rerank_glm, topk=3)


# 4. 调用retriever组件，传入query
query = "deepseek-v3主要改进"
retriever_result1 = retriever1(query=query)
retriever_result2 = retriever2(query=query)

# 打印结果和分数
for i, node in enumerate(retriever_result1):
    print(f"\n---retriever_result1 #{i+1}:\n", node.get_content())
    print(f"---retriever_result1 #{i+1}:Score: {getattr(node, 'score', 'N/A')}")

for i, node in enumerate(retriever_result2):
    print(f"\n---retriever_result2 #{i+1}:\n", node.get_content())
    print(f"---retriever_result2 #{i+1}:Score: {getattr(node, 'score', 'N/A')}")

# 5. 去重: 合并结果并基于内容去重
combined_results = retriever_result1 + retriever_result2
unique_nodes = []
unique_contents = set()

for node in combined_results:
    content = node.get_content()
    if content not in unique_contents:
        unique_contents.add(content)
        unique_nodes.append(node)

print(f"原始结果数: {len(combined_results)}, 去重后结果数: {len(unique_nodes)}")

# 6. 调用reranker组件，传入去重后的结果
print("\n执行reranker...")
reranker_result = reranker(
    nodes=unique_nodes, query=query
)


# 打印重排序后的结果和分数
for i, node in enumerate(reranker_result):
    print(f"\n---reranked_result #{i+1}:\n", node.get_content())
    print(f"Rerank #{i+1}:Score: {getattr(node, 'score', 'N/A')}")

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
            [node.get_content() for node in reranker_result]
        ),
    }
)

print(f"answer: {res}")
