import streamlit as st
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler


# tools
api_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200)
arxiv=ArxivQueryRun(api_wrapper=api_wrapper)

api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200)
wiki=WikipediaQueryRun(api_wrapper=api_wrapper)

search=DuckDuckGoSearchRun(name='Search')


st.set_page_config(page_title='Search Engine')
st.title("🔎 Search Engine with Tools and Agents")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

# sidebar for settings
st.sidebar.title('Settings')
api_key=st.sidebar.text_input(label='Enter your GrOQ API Key:',type='password')

if 'messages' not in st.session_state:
    st.session_state['messages']=[
        {
            'role':'assistant',
            'content':'Hi, I am a chatbot who can search the web.How Can i help you?'
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg.get('role')).write(msg.get('content'))

prompt=st.chat_input(placeholder='what is machine learning?')    

if prompt:
    st.session_state.messages.append({'role':'user','content':prompt})
    st.chat_message('user').write(prompt)
    
    llm=ChatGroq(model_name='Llama3-8b-8192',groq_api_key=api_key,streaming=True)
    tools=[search,wiki,arxiv]
    search_agent=initialize_agent(tools=tools,llm=llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)
    
    with st.chat_message(name='assistant'):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
        response=search_agent.run(st.session_state.messages,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant','content':response})
        st.write(response)
