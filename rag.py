from llm_config import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

def build_rag_chain(vectorstore, system_prompt):
    # 1. Dynamic Model Setup
    llm = get_llm(temperature=0.1)

    # 2. Dynamic Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt + "\n\n" + 
         "LANGUAGE INSTRUCTION:\n{language_instruction}\n\n" +
         "Relevant Context from Knowledge Base:\n{context}"),
        ("human", "{input}")
    ])

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 3. Chain Construction
    rag_chain = (
        {
            "context": itemgetter("input") | retriever | format_docs,
            "input": itemgetter("input"),
            "language_instruction": itemgetter("language_instruction")
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever