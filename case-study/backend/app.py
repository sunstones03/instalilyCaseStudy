from flask import Flask, jsonify, request, session
from flask_cors import CORS

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_deepseek import ChatDeepSeek
from langchain.prompts import (
    SystemMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate
)

import numpy as np

system_prompt = """
You are an expert support agent for partselect.com.
Your role is to answer user questions related to refrigerator and dishwasher parts strictly and exclusively using the information found on the company website partselect.com.

Respond only and exclusively using the information contained in the provided website.
Do not introduce any information that is not present in the website.
If the provided website does not contain sufficient information to answer the question, or if the website does not directly address the user’s query, redirect the user to customer support then prompt them to ask a question within your expertise in a happy manner.

Do not explain your answer or provide any additional commentary.
Your responses should be concise and focused on addressing the user's query using only the provided information.

Adhere to the context and limitations at all times.
If any part of the question cannot be answered with the provided website, you must refrain from speculation or the use of external knowledge.

If asked about order status or confirmation, ask for email address and order number, then redirect them to https://www.partselect.com/user/self-service/.
Ask follow up questions if necessary.
Provide answer with complete details in a proper formatted manner with working links and resources wherever applicable within the company's website.
Never provide wrong links.
"""

app = Flask(__name__)
CORS(app)  # Allow CORS for all routes
app.config["SECRET_KEY"] = "secret"
app.config["thread_id"] = np.random.randint(1000000)

@app.route('/api/chat', methods=["POST","GET"])
def chat():
    '''
    description: produces the AI message given the human query
    '''
    if request.method == "POST":

        data = request.get_json()
        query = data.get('query', '')

        ## create input for llm agent
        config = {"configurable" : {"thread_id" : app.config['thread_id']}}
        msg = {"messages": [HumanMessage(content=query)]}
        
        print(app.config['thread_id'])
        ## invoke the model to produce the answer
        answer = agent_executor.invoke(msg, config)["messages"][-1].content
        return jsonify({"answer": answer})

@app.route('/api/init_thread', methods=["POST","GET"])
def init_thread():
    '''
    description: initializes the conversation thread, generates new thread_id, and sends the initial system prompt
    '''
    app.config["thread_id"] = np.random.randint(1000000)
    print(app.config['thread_id'])
    config = {"configurable" : {"thread_id" : app.config['thread_id']}}
    msg = {"messages": [SystemMessage(content=system_prompt)]}
    agent_executor.invoke(msg, config)

    return jsonify(dict())

if __name__ == '__main__':

    ## create memory
    memory = MemorySaver()

    ## call deepseek-chat model
    model = ChatDeepSeek(
        model="deepseek-chat",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    ## create search tool
    search = TavilySearchResults(max_results=4)
    tools = [search]
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    ## set up flask server
    app.run(debug=True)
