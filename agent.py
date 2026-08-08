from dotenv import load_dotenv
from typing import Annotated, List
from flask import Flask
from flask_socketio import SocketIO
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from socket_manager import socketio
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
import os
from web_opreation import serp_search,reddit_search_api,reddit_post_retrieval
from prompt import (
get_bing_analysis_messages,
get_reddit_analysis_messages,
get_reddit_url_analysis_messages,
get_google_analysis_messages,
get_synthesis_messages
)


load_dotenv()



llm = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    base_url="https://openrouter.ai/api/v1",
    api_key= os.getenv("OPENAI_API_KEY"),
    temperature=0
)



class State(TypedDict):
    messages:Annotated[list, add_messages]
    user_question: str| None
    google_reasult: str| None
    bing_reasult: str| None
    reddit_reasult: str| None
    selected_reddit_url: list[str]|None
    reddit_post_data: list|None
    google_analysis: str | None
    bing_analysis:str|None
    reddit_analysis : str|None  
    final_answer: str |None
    
class RedditURLAnalysis(BaseModel):
    selected_urls: list[str] = Field(
        description="List of Reddit URLs that contain valuable information for answering the user's question"
    )

def emit_status(step, status, data=None):
    socketio.emit(
        "agent_status",
        {
            "step": step,
            "status": status,
            "data": data
        }
    )


def google_search(state: State):
    user_question = state.get("user_question",'')
    print(f"Searching for Google for {user_question}")
    emit_status("Google Search", "running")
    google_reasult=serp_search(user_question,engine="google")
    print(google_reasult)
    return {"google_reasult":google_reasult}


def bing_search(state: State):
    user_question = state.get("user_question",'')
    emit_status("Bing Search", "running")
    print(f"Searching for Bing for {user_question}")
    bing_reasult=serp_search(user_question,engine="bing")
    print(bing_reasult)
    return {"bing_reasult":bing_reasult}
   
    

def reddit_search(state: State):
    user_question = state.get("user_question",'')
    emit_status("Reddit Search", "running")
    print(f"Searching for Reddit for {user_question}")
    reddit_reasult=reddit_search_api(user_question)
    return {"reddit_reasult":reddit_reasult}


def analyis_reddit_post(state: State):
    user_question = state.get("user_question","")
    reddit_reasult =state.get("reddit_reasult","")
    if not reddit_reasult:
        return {"selected_reddit_url":[]}
    structured_llm = llm.with_structured_output(RedditURLAnalysis)
    emit_status("Selecting Reddit Posts", "running")
    messages = get_reddit_url_analysis_messages(user_question, reddit_reasult)

    try:
        analysis = structured_llm.invoke(messages)
        print(analysis)
        print(type(analysis))
        selected_urls = analysis.selected_urls
        print(selected_urls)
        for i, url in enumerate(selected_urls, 1):
          print(f"{i}. {url}")
    except Exception as e:
        print(e)
        selected_urls = []

    return {"selected_reddit_url": selected_urls}



def retrive_redit_posts(state: State):
    print("Getting Reddit post comments")

    selected_urls = state.get("selected_reddit_url", [])

    if not selected_urls:
        return {"reddit_post_data": []}

    print(f"Processing {len(selected_urls)} Reddit URLs")
    print(selected_urls)
    print(type(selected_urls))

    reddit_post_data = reddit_post_retrieval(selected_urls)
    emit_status("Downloading Reddit Posts", "running")

    if reddit_post_data:
        print(f"Successfully got {len(reddit_post_data)} posts")
    else:
        print("Failed to get post data")
        reddit_post_data = []

    print(reddit_post_data)
    emit_status(
    "Downloading Reddit Posts",
    "completed",
    {"posts": len(reddit_post_data)}
)

    return {"reddit_post_data": reddit_post_data}



def analys_google_reasult(state: State):
    print('analysing google search')
    user_question =  state.get("user_question","")
    google_reasult = state.get("google_reasult","")
    emit_status("Analyzing Google", "running")
    message = get_google_analysis_messages(user_question,google_reasult)
    reply = llm.invoke(message)
    emit_status("Analyzing Google", "completed")
    return {"google_analysis":reply.content}



def analyis_bing_reasult(state: State):
    print('analysing bing search')
    user_question =  state.get("user_question","")
    bing_reasult = state.get("bing_reasult","")
    emit_status("Analyzing Bing", "running")
    message = get_bing_analysis_messages(user_question,bing_reasult)
    reply = llm.invoke(message)
    emit_status("Analyzing Bing", "completed")
    return {"bing_analysis":reply.content}




def analyis_reddit_reasult(state: State):
    print('analysing reddit search')
    user_question =  state.get("user_question","")
    reddit_reasult = state.get("reddit_reasult","")
    reddit_post_Data =state.get('reddit_post_data')
    emit_status("Analyzing Reddit", "running")
    message = get_reddit_analysis_messages(user_question,reddit_reasult,reddit_post_Data)
    reply = llm.invoke(message)
    emit_status("Analyzing Reddit", "completed")
    return {"reddit_analysis":reply.content}
    



def syntheesize_analyses(state: State):
    print('combine all reasult together')
    user_question = state.get('user_question',"")
    google_analysis = state.get('google_analysis',"")
    bing_analysis = state.get('bing_analysis',"")
    reddit_analysis = state.get('reddit_analysis',"")
    emit_status("Generating Final Answer", "running")
    messages = get_synthesis_messages(
        user_question,google_analysis,bing_analysis,reddit_analysis
    )
    reply = llm.invoke(messages)
    final_answer = reply.content
    emit_status("Generating Final Answer", "completed")
    return {"final_answer":final_answer,"messages":[{"role":"assistant","content":final_answer}]}


graph_builder = StateGraph(State)

graph_builder.add_node("google_search", google_search)
graph_builder.add_node("bing_search", bing_search)
graph_builder.add_node("reddit_search", reddit_search)
graph_builder.add_node("analyis_reddit_post", analyis_reddit_post)
graph_builder.add_node("retrive_redit_posts", retrive_redit_posts)
graph_builder.add_node("analys_google_reasult", analys_google_reasult)
graph_builder.add_node("analyis_bing_reasult", analyis_bing_reasult)
graph_builder.add_node("analyis_reddit_reasult", analyis_reddit_reasult)
graph_builder.add_node("syntheesize_analyses", syntheesize_analyses)



graph_builder.add_edge(START, "google_search")
graph_builder.add_edge(START, "bing_search")
graph_builder.add_edge(START, "reddit_search")


graph_builder.add_edge('google_search','analyis_reddit_post')
graph_builder.add_edge('bing_search','analyis_reddit_post')
graph_builder.add_edge('reddit_search','analyis_reddit_post')
graph_builder.add_edge('analyis_reddit_post','retrive_redit_posts')


graph_builder.add_edge('retrive_redit_posts','analys_google_reasult')
graph_builder.add_edge('retrive_redit_posts','analyis_bing_reasult')
graph_builder.add_edge('retrive_redit_posts','analyis_reddit_reasult')


graph_builder.add_edge('analys_google_reasult','syntheesize_analyses')
graph_builder.add_edge('analyis_bing_reasult','syntheesize_analyses')
graph_builder.add_edge('analyis_reddit_reasult','syntheesize_analyses')

graph_builder.add_edge('syntheesize_analyses',END)


graph =graph_builder.compile()


def run_chatbot(query: str):
    state = {
        "messages": [{"role": "user", "content": query}],
        "user_question": query,
        "google_reasult": None,
        "bing_reasult": None,
        "reddit_reasult": None,
        "selected_reddit_url": None,
        "reddit_post_data": None,
        "google_analysis": None,
        "bing_analysis": None,
        "reddit_analysis": None,
        "final_answer": None
    }

    print("\nStarting parallel process...")
    final_state = graph.invoke(state)

    return final_state.get("final_answer")

if __name__ == "__main__":
    run_chatbot(query)