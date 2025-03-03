from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_google_genai import ChatGoogleGenerativeAI

import tkinter
import tkinter.scrolledtext

window = tkinter.Tk()
window.title("AI Deep Research")
window.geometry("960x500")
window.resizable(0,0)

tkinter.Label(window, text="Enter Your Research Query").pack(pady=10)
query = tkinter.StringVar()
querybox = tkinter.Entry(window, textvariable=query, width= 110)
querybox.focus()
querybox.pack()

frame = tkinter.Frame(window)
frame.pack(pady=10)

tool = TavilySearchResults(max_results=2)
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm_with_tools = llm.bind_tools([tool])

class State(TypedDict):
    messages: Annotated[list, add_messages]

def research_agent(state: State):
    last_message = state["messages"][-1]
    query = last_message.content if hasattr(last_message, "content") else str(last_message)
    
    search_results = tool.invoke(query)

    formatted_result = "Search Results: "
    if search_results:
        for i in search_results:
            formatted_result = formatted_result + i['content']
    else:
        formatted_result = "No relevant search results found."

    return {"messages": state["messages"] + [formatted_result]}

def answer_agent(state: State):
    response = llm_with_tools.invoke(state["messages"])

    if hasattr(response, "content"):
        response_text = response.content

    return {"messages": state["messages"] + [response_text]}

graph = StateGraph(State)
graph.add_node("research", research_agent)
graph.add_node("answer", answer_agent)
graph.add_edge("research", "answer")
graph.add_edge(START, "research")

graph_compiled = graph.compile()

state = {"messages": []}

def restart():
    if query.get()!="":
        global state
        state = {"messages": []}
        search()

def search():
    if query.get()!="":
        text_area.delete('1.0', tkinter.END)
        global state
        state["messages"].append(query.get())
        state = graph_compiled.invoke(state)
        output = state["messages"][-1].content
        text_area.insert(tkinter.INSERT, output)
        querybox.delete(0, tkinter.END)

tkinter.Button(frame, text="Search (Refine)", command=search).pack(side="left", padx=10)
tkinter.Button(frame, text="Restart Search", command=restart).pack(side="right", padx=10)

text_area=tkinter.scrolledtext.ScrolledText(window, height = 20, width =85)
text_area.pack()
window.mainloop()