#
# Fivetran Snowflake Cortex Streamlit Lab
# Build a California Wine Country Travel Assistant Chatbot
#

import streamlit as st
from snowflake.snowpark.context import get_active_session
import pandas as pd
import time

# Change this list as needed to add/remove model capabilities.
MODELS = [
    "llama3.2-3b",
    "claude-3-5-sonnet",
    "mistral-large2",
    "llama3.1-8b",
    "llama3.1-405b",
    "llama3.1-70b",
    "mistral-7b",
    "jamba-1.5-large",
    "mixtral-8x7b",
    "reka-flash",
    "gemma-7b"
]

# Change this value to control the number of tokens
CHUNK_NUMBER = [4,6,8,10,12,14,16]

# Preset prompts
WINERY_OVERVIEW_PROMPT = """Create a well-formatted comprehensive overview of the following winery:

**Winery Name:** {winery}

The overview should include:

1. **Basic Information**:
   - Location, including region and any notable geographical features.
   - A brief history of the winery (e.g., founding date, founders, or mission).
   - Specialties in wine production (e.g., types of wine, signature varietals).

2. **Unique Characteristics and Offerings**:
   - At least 5 unique features, activities, or experiences the winery offers.
   - Any seasonal or recurring events.

3. **Tasting Room and Tours**:
   - Details about the tasting room ambiance.
   - Information about guided tours, tastings, or other experiences.

4. **Hospitality**:
   - Unique amenities, such as picnic areas, accommodations, or partnerships.
   - Pet and children policies and details.

5. **Notable Recognition**:
   - Awards, certifications, or mentions in media.

6. **Pricing**:
   - Cost range for tastings, tours, or special events.

7. **Insider Tips**:
   - Recommendations for visitors to make the most of their experience."""

TRIP_PLAN_PROMPT = """Create a well-formatted, detailed and engaging travel guide for a 2-day wine country getaway, complete with a catchy itinerary name. The trip should include visits to {winery}.

Please include:
1. A unique name for this winery trip
2. A logical day-by-day itinerary that visits the winery
3. At least 5 unique characteristics or activities at the winery
4. Recommended nearby restaurants for lunch and dinner
5. Hotel recommendations for the overnight stay
6. Other complementary activities
7. Information about any pet-friendly features
8. Tips for making the most of the visits
9. Advice on what to wear for this time of year
10. Estimated cost of the trip"""

def get_winery_list(session):
    """Fetch the list of wineries from Snowflake."""
    winery_cmd = """
    SELECT DISTINCT winery_or_vineyard 
    FROM vineyard_data_vectors 
    ORDER BY winery_or_vineyard
    """
    winery_df = session.sql(winery_cmd).to_pandas()
    return winery_df['WINERY_OR_VINEYARD'].tolist()

def on_preset_prompt_change():
    """Handle preset prompt changes"""
    if st.session_state.preset_prompt == "None":
        # Clear both the full and display questions when None is selected
        st.session_state.current_question = None
        st.session_state.display_question = None
    elif st.session_state.selected_winery:
        if st.session_state.preset_prompt == "Winery Overview":
            # Store the full prompt for processing
            st.session_state.current_question = WINERY_OVERVIEW_PROMPT.format(winery=st.session_state.selected_winery)
            # Store the simplified display version
            st.session_state.display_question = f"Create a well-formatted comprehensive overview of {st.session_state.selected_winery}"
        else:
            # Store the full prompt for processing
            st.session_state.current_question = TRIP_PLAN_PROMPT.format(winery=st.session_state.selected_winery)
            # Store the simplified display version
            st.session_state.display_question = f"Create a well-formatted, detailed and engaging travel guide for a 2-day wine country getaway, complete with a catchy itinerary name. The trip should include visits to {st.session_state.selected_winery}."
    
def build_layout():
    # Initialize session state
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = []
    if 'current_question' not in st.session_state:
        st.session_state.current_question = None
    if 'reset_key' not in st.session_state:
        st.session_state.reset_key = 0

    # Set page config - must be called first
    st.set_page_config(layout="wide")

    # Build the layout
    st.title(":wine_glass: California Wine Country Visit Assistant :wine_glass:")
    st.write("""I'm an interactive California Wine Country Visit Assistant. A bit about me...I'm a RAG-based, Gen AI app **built 
      with and powered by Fivetran, Snowflake, Streamlit, and Cortex** and I use a custom, structured dataset!""")
    st.caption("""Let me help plan your trip to California wine country. Using the dataset you just moved into the Snowflake Data Cloud with Fivetran, I'll assist you with winery and vineyard information and provide visit recommendations from numerous models available in Snowflake Cortex (including Claude 3.5 Sonnet). You can even pick the model you want to use or try out all the models. The dataset includes over 700 wineries and vineyards across all CA wine-producing regions including the North Coast, Central Coast, Central Valley, South Coast and various AVAs sub-AVAs. Let's get started!""")

    # Sidebar components
    st.sidebar.selectbox("Select a Snowflake Cortex model:", MODELS, key="model_name")
    st.sidebar.checkbox('Use your Fivetran dataset as context?', key="dataset_context")

    # Winery search feature
    wineries = get_winery_list(session)
    st.sidebar.selectbox(
        "Search for a specific winery:",
        options=[""] + wineries,  # Add empty option as first choice
        key="selected_winery",
        help="Type to search for a specific winery"
    )

    # Preset prompt selection
    if st.session_state.get('selected_winery') and st.session_state.get('selected_winery') != "":
        st.sidebar.radio(
            "Select a preset prompt:",
            options=["None", "Winery Overview", "Trip Plan"],
            key="preset_prompt",
            on_change=on_preset_prompt_change,
            help="Select a preset prompt to generate information about the selected winery"
        )

    # Reset conversation button
    if st.button('Reset conversation'):
        st.session_state.conversation_state = []
        st.session_state.current_question = None
        st.session_state.display_question = None
        st.session_state.reset_key += 1  # Add this
        st.rerun()

    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        st.selectbox("Select number of context chunks:", CHUNK_NUMBER, key="num_retrieved_chunks")

    # Sidebar caption and logo
    st.sidebar.caption("""I use **Snowflake Cortex** which provides instant access to industry-leading large language models including Claude, Llama, and Snowflake Arctic that have been trained by researchers at companies like Anthropic, Meta, Mistral, Google, Reka, and Snowflake.""")
    for _ in range(6):
        st.sidebar.write("")
    url = 'https://i.imgur.com/9lS8Y34.png'
    col1, col2, col3 = st.sidebar.columns([1,2,1])
    with col2:
        st.image(url, width=150)
    caption_col1, caption_col2, caption_col3 = st.sidebar.columns([0.22,2,0.005])
    with caption_col2:
        st.caption("Fivetran, Snowflake, Streamlit, & Cortex")

    # Main content area
    processing_placeholder = st.empty()
    text_input = st.text_input(
        "",
        placeholder="Message your personal CA Wine Country Visit Assistant...",
        key=f"text_input_{st.session_state.reset_key}",  # Modify this
        label_visibility="collapsed"
    )

    # Dataset context caption
    if st.session_state.dataset_context:
        st.caption("""Please note that :green[**_I am_**] using your Fivetran dataset as context. All models are very 
          creative and can make mistakes. Consider checking important information before heading out to wine country.""")
    else:
        st.caption("""Please note that :red[**_I am NOT_**] using your Fivetran dataset as context. All models are very 
          creative and can make mistakes. Consider checking important information before heading out to wine country.""")

    # Return the text input if no preset prompt is selected, otherwise return the preset prompt
    return text_input if st.session_state.get('preset_prompt') == "None" or not st.session_state.get('current_question') else st.session_state.current_question

def build_prompt(question):
    # Build the RAG prompt if the user chooses
    chunks_used = []
    if st.session_state.dataset_context:
        context_cmd = f"""
          with context_cte as
          (select winery_or_vineyard, winery_information as winery_chunk, vector_cosine_similarity(winery_embedding,
                snowflake.cortex.embed_text_1024('snowflake-arctic-embed-l-v2.0', ?)) as v_sim
          from vineyard_data_vectors
          having v_sim > 0
          order by v_sim desc
          limit ?)
          select winery_or_vineyard, winery_chunk from context_cte 
          """
        chunk_limit = st.session_state.num_retrieved_chunks
        context_df = session.sql(context_cmd, params=[question, chunk_limit]).to_pandas()
        context_len = len(context_df) -1
        chunks_used = context_df['WINERY_OR_VINEYARD'].tolist()
        
        rag_context = ""
        for i in range(0, context_len):
            rag_context += context_df.loc[i, 'WINERY_CHUNK']
        rag_context = rag_context.replace("'", "''")
        
        new_prompt = f"""
          Act as a California winery visit expert for visitors to California wine country who want an incredible visit and 
          tasting experience. You are a personal visit assistant named Snowflake CA Wine Country Visit Assistant. Provide 
          the most accurate information on California wineries based only on the context provided. Only provide information 
          if there is an exact match below. Do not go outside the context provided.  
          Context: {rag_context}
          Question: {question} 
          Answer: 
          """
    else:
        new_prompt = f"""
          Act as a California winery visit expert for visitors to California wine country who want an incredible visit and 
          tasting experience. You are a personal visit assistant named Snowflake CA Wine Country Visit Assistant. Provide 
          the most accurate information on California wineries.
          Question: {question} 
          Answer: 
          """

    return new_prompt, chunks_used

def get_model_token_count(prompt_or_response) -> int:
    token_count = 0
    try:
        token_cmd = f"""select SNOWFLAKE.CORTEX.COUNT_TOKENS(?, ?) as token_count;"""
        tc_data = session.sql(token_cmd, params=[st.session_state.model_name, prompt_or_response]).collect()
        token_count = tc_data[0][0]
    except Exception:
        token_count = -9999

    return token_count

def calc_times(start_time, first_token_time, end_time, token_count):
    time_to_first_token = first_token_time - start_time
    total_duration = end_time - start_time
    time_for_remaining_tokens = total_duration - time_to_first_token
    
    tokens_per_second = token_count / total_duration if total_duration > 0 else 1
    
    if time_to_first_token < 0.01:
        time_to_first_token = total_duration / 2

    return time_to_first_token, time_for_remaining_tokens, tokens_per_second

def run_prompt(question):
    formatted_prompt, chunks_used = build_prompt(question)
    token_count = get_model_token_count(formatted_prompt)
    start_time = time.time()
    
    cortex_cmd = f"""
             select SNOWFLAKE.CORTEX.COMPLETE(?,?) as response
           """    
    sql_resp = session.sql(cortex_cmd, params=[st.session_state.model_name, formatted_prompt])
    first_token_time = time.time()
    answer_df = sql_resp.collect()
    end_time = time.time()
    
    time_to_first_token, time_for_remaining_tokens, tokens_per_second = calc_times(
        start_time, first_token_time, end_time, token_count)

    return answer_df, time_to_first_token, time_for_remaining_tokens, tokens_per_second, int(token_count), chunks_used

def main():
    question = build_layout()
    
    if question:
        with st.spinner("Thinking..."):
            try:
                # Run the prompt using the full question
                data, time_to_first_token, time_for_remaining_tokens, tokens_per_second, token_count, chunks_used = run_prompt(question)
                response = data[0][0]
                
                if response:
                    token_count += get_model_token_count(response)
                    rag_delim = ", "
                    
                    # Use the display question if it exists, otherwise use the original question
                    display_question = st.session_state.get('display_question') if st.session_state.get('display_question') else question
                    
                    # Append conversation state
                    st.session_state.conversation_state.append(
                        (f":information_source: RAG Chunks/Records Used:",
                         f"""<span style='color:#808080;'> {(rag_delim.join([str(ele) for ele in chunks_used])) if chunks_used else 'none'} 
                         </span><br/><br/>""")
                    )
                    st.session_state.conversation_state.append(
                        (f":1234: Token Count for '{st.session_state.model_name}':", 
                         f"""<span style='color:#808080;'>{token_count} tokens :small_blue_diamond: {tokens_per_second:.2f} tokens/s :small_blue_diamond: 
                         {time_to_first_token:.2f}s to first token + {time_for_remaining_tokens:.2f}s.</span>""")
                    )
                    st.session_state.conversation_state.append(
                        (f"CA Wine Country Visit Assistant ({st.session_state.model_name}):", response)
                    )
                    st.session_state.conversation_state.append(("You:", display_question))
                    
                    # Clear the current questions after processing
                    st.session_state.current_question = None
                    st.session_state.display_question = None
                    
            except Exception as e:
                st.warning(f"An error occurred while processing your question: {e}")
                
        # Display conversation history
        if st.session_state.conversation_state:
            for i in reversed(range(len(st.session_state.conversation_state))):
                label, message = st.session_state.conversation_state[i]
                if 'Token Count' in label or 'RAG Chunks' in label:
                    st.markdown(f"**{label}** {message}", unsafe_allow_html=True)
                elif i % 2 == 0:
                    st.write(f":wine_glass:**{label}** {message}")
                else:
                    st.write(f":question:**{label}** {message}")

if __name__ == "__main__":
    session = get_active_session()
    main()