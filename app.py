import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import traceback

# Set the page config as the first Streamlit command
st.set_page_config(page_title="IFFA Australia", layout="centered")

# Handling the session state for user login
if 'username' not in st.session_state:
    st.session_state.username = None
    st.session_state.login_attempts = 0

# Add debug session state display
if st.sidebar.checkbox("Show Debug Info", False, key="debug_info_sidebar"):
    st.sidebar.write("Session State:", st.session_state)

# Custom CSS styling
st.markdown(
    """
    <style>
    /* Base styling for the app */
    body {
        background-color: #141414;
        color: #ffffff;
        font-family: 'Arial', sans-serif;
    }
    
    /* Netflix-inspired styling */
    .netflix-row {
        overflow-x: auto;
        padding: 20px 0;
        white-space: nowrap;
    }
    
    .movie-card {
        display: inline-block;
        position: relative;
        width: 220px;
        height: 330px;
        margin-right: 10px;
        border-radius: 5px;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        vertical-align: top;
    }
    
    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.8);
        z-index: 10;
    }
    
    .movie-poster {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: filter 0.3s ease;
    }
    
    .movie-card:hover .movie-poster {
        filter: brightness(60%);
    }
    
    .movie-info {
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        padding: 15px;
        background: linear-gradient(to top, rgba(0, 0, 0, 0.9), transparent);
        transform: translateY(100%);
        transition: transform 0.3s ease;
        text-align: left;
    }
    
    .movie-card:hover .movie-info {
        transform: translateY(0);
    }
    
    .movie-title {
        color: white;
        font-weight: bold;
        margin-bottom: 5px;
        font-size: 16px;
        white-space: normal;
    }
    
    .movie-genre {
        color: #aaa;
        font-size: 12px;
        white-space: normal;
    }
    
    /* Animate posters with subtle movement */
    @keyframes subtleZoom {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    .movie-poster {
        animation: subtleZoom 8s infinite ease-in-out;
    }
    
    /* Streamlit component overrides */
    .stButton>button {
        background-color: #E50914;
        color: white;
        padding: 8px 16px;
        border: none;
        border-radius: 4px;
        font-size: 16px;
        transition: background-color 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #F40612;
    }
    
    /* Netflix-style title header */
    .netflix-header {
        font-size: 26px;
        font-weight: bold;
        margin-bottom: 20px;
        color: white;
    }
    
    /* Streamlit main area */
    .main {
        background-color: #141414;
    }
    
    /* Custom container for featured content */
    .featured-container {
        position: relative;
        width: 100%;
        height: 400px;
        margin-bottom: 30px;
        overflow: hidden;
        border-radius: 8px;
    }
    
    .featured-image {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .featured-info {
        position: absolute;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 20px;
        background: linear-gradient(to top, rgba(0,0,0,0.9) 0%, transparent 100%);
    }
    
    .featured-title {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .featured-description {
        font-size: 16px;
        margin-bottom: 15px;
        max-width: 600px;
    }
    
    /* For rating stars */
    .rating-stars {
        color: #E50914;
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def load_data():
    try:
        # Check if file exists
        if not os.path.exists("Movie_list.csv"):
            st.error("Error: Movie_list.csv file not found. Please ensure the file exists in the current directory.")
            return None
            
        # Load data
        df = pd.read_csv("Movie_list.csv")
        
        # Clean data
        columns_to_drop = ['Unnamed: 0', 'Tracking Number', 'Rating', 'Recommendations']
        df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
        
        # Define text columns to use for recommendations
        text_cols = ['Project Title', 'Synopsis', 'Genres', 'Language', 'Country',
                   'Country of Origin', 'Country of Filming', 'Directors',
                   'Writers', 'Producers', 'Key Cast', 'Submission Categories']
        
        # Make sure all text columns exist
        for col in text_cols:
            if col not in df.columns:
                df[col] = ""
            else:
                df[col] = df[col].fillna('')
        
        df['combined_features'] = (
            df['Genres'] + ' ' +
            df['Synopsis'] + ' ' +
            df['Language'] + ' ' +
            df.get('Country of Origin', '') + ' ' +
            df['Key Cast'] + ' ' +
            df.get('Submission Categories', '')
        )
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.code(traceback.format_exc())
        return None

def create_recommendation_engine(df):
    try:
        # TF-IDF and similarity
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['combined_features'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        indices = pd.Series(df.index, index=df['Project Title']).drop_duplicates()
        return cosine_sim, indices
    except Exception as e:
        st.error(f"Error creating recommendation engine: {str(e)}")
        st.code(traceback.format_exc())
        return None, None

def get_recommendations(title, df, cosine_sim, indices):
    try:
        idx = indices.get(title)
        if idx is None:
            return pd.DataFrame()
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:6]  # Exclude itself
        movie_indices = [i[0] for i in sim_scores]
        
        # Reset index to make sure we don't display index values instead of actual titles
        return df.iloc[movie_indices].reset_index(drop=True)
    except Exception as e:
        st.error(f"Error getting recommendations: {str(e)}")
        st.code(traceback.format_exc())
        return pd.DataFrame()

def login_page():
    # Custom Netflix-style hero banner
    st.markdown("""
    <div style="width: 100%; height: 200px; background: linear-gradient(90deg, #000 0%, transparent 60%), 
                linear-gradient(0deg, #000 10%, transparent 50%), url('https://assets.nflxext.com/ffe/siteui/vlv3/a73c4363-1dcd-4719-b3b1-3725418fd91d/fe1147dd-78be-44aa-a0e5-2d2994305a13/US-en-20210102-popsignuptwoweeks-perspective_alpha_website_small.jpg');
                background-size: cover; background-position: center; padding: 20px; margin-bottom: 30px;">
        <h1 style="color: #E50914; font-size: 44px; margin-top: 100px;">IFFA</h1>
        <h2 style="color: white; font-size: 24px;">Short Film Platform</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Login form with Netflix styling
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="background-color: rgba(0,0,0,0.8); padding: 30px; border-radius: 5px; color: white;">
            <h2 style="color: white; text-align: center; margin-bottom: 20px;">Sign In</h2>
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Email or username", key="username_input")
        password = st.text_input("Password", type="password", key="password_input")
        
        # Debug information to help troubleshoot (hidden in production)
        if st.sidebar.checkbox("Show Debug Info", False, key="debug_info_login"):
            st.write(f"Debug: Username entered: '{username}'")
        
        col_login, col_remember = st.columns([2, 1])
        
        with col_login:
            login_button = st.button("Sign In", key="login_button", 
                                    help="Click to sign in with your credentials")
            if login_button:
                if username and password:
                    # Show a Netflix-style loading animation
                    with st.spinner("Signing in..."):
                        import time
                        time.sleep(1)  # Simulate loading
                    
                    st.session_state.username = username
                    st.success(f"Welcome to IFFA, {username}!")
                    # Use the newer rerun method with fallback to experimental_rerun
                    try:
                        st.rerun()
                    except AttributeError:
                        st.experimental_rerun()
                else:
                    st.warning("Please enter both email/username and password.")
        
        with col_remember:
            st.checkbox("Remember me", value=True, key="remember_me")
        
        # Need help section
        st.markdown("""
        <div style="margin-top: 20px; text-align: center;">
            <span style="color: #737373;">New to IFFA? </span>
            <a href="#" style="color: white; text-decoration: none;">Sign up now</a>.
        </div>
        """, unsafe_allow_html=True)
        
        # Special test login button for debugging only
        if st.sidebar.checkbox("Show Test Login", False, key="show_test_login"):
            if st.button("Test Login (Bypass Authentication)", key="test_login"):
                st.session_state.username = "test_user"
                st.success("Test login successful!")
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()

def main_page(df):
    try:
        # Create recommendation engine
        cosine_sim, indices = create_recommendation_engine(df)
        
        # Logo (check if file exists)
        logo_path = "iffa_logo.png"
        if os.path.exists(logo_path):
            st.image(logo_path, width=200)
        else:
            st.warning("Logo file 'iffa_logo.png' not found")
            
        st.title(f"Welcome to IFFA, {st.session_state.username}")
        
        if st.button("Logout", key="logout_button"):
            st.session_state.username = None
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()
        
        # Search bar
        search_query = st.text_input("Search for a short film:", key="search_query").lower()
        
        if not df.empty:
            filtered_titles = df['Project Title'][df['Project Title'].str.lower().str.contains(search_query)]
            
            # Dropdown from search results
            if not filtered_titles.empty:
                selected_title = st.selectbox("Choose a short film:", sorted(filtered_titles.unique()), key="film_select")
            else:
                if search_query:
                    st.warning("No films match your search. Showing default list.")
                selected_title = st.selectbox("Choose a short film:", sorted(df['Project Title'].unique()), key="film_select_default")
            
            # Show selected film info
            film = df[df['Project Title'] == selected_title].iloc[0]
            st.subheader(f"📽️ {film['Project Title']}")
            
            # Display poster if exists
            if 'Poster' in df.columns and pd.notna(film['Poster']) and str(film['Poster']).startswith("http"):
                st.image(film['Poster'], width=300)
            
            st.markdown(f"**Genre:** {film['Genres']}")
            st.markdown(f"**Language:** {film['Language']}")
            
            if 'Country of Origin' in df.columns:
                st.markdown(f"**Country of Origin:** {film['Country of Origin']}")
                
            st.markdown(f"**Synopsis:** {film['Synopsis']}")
            
            # Ratings (User Feedback)
            st.markdown("**Rate this film:**")
            rating = st.slider("Give a rating (1-5 stars)", 1, 5, 3, key="rating_slider")
            
            # Save rating
            if st.button("Submit Rating", key="submit_rating"):
                rating_file = "ratings.csv"
                new_row = pd.DataFrame([[st.session_state.username, selected_title, rating]], 
                                      columns=["Username", "Film Title", "Rating"])
                
                try:
                    if os.path.exists(rating_file):
                        existing_ratings = pd.read_csv(rating_file)
                        existing_ratings = existing_ratings[~((existing_ratings['Username'] == st.session_state.username) & 
                                                            (existing_ratings['Film Title'] == selected_title))]
                        updated_ratings = pd.concat([existing_ratings, new_row], ignore_index=True)
                    else:
                        updated_ratings = new_row
                    
                    updated_ratings.to_csv(rating_file, index=False)
                    st.success(f"{st.session_state.username}, you rated '{selected_title}' {rating} ⭐")
                except Exception as e:
                    st.error(f"Error saving rating: {str(e)}")
            
            # Recommendations
            if cosine_sim is not None and indices is not None:
                recs = get_recommendations(selected_title, df, cosine_sim, indices)
                st.markdown("---")
                st.subheader("🎞️ Recommended for You:")
                
                if recs.empty:
                    st.info("No recommendations found.")
                else:
                    cols = st.columns(min(5, len(recs)))
                    for idx, row in enumerate(recs.itertuples()):
                        with cols[idx % len(cols)]:
                            if 'Poster' in df.columns and pd.notna(getattr(row, 'Poster', None)) and str(getattr(row, 'Poster', '')).startswith("http"):
                                st.image(row.Poster, width=150)
                            
                            # Directly access the Project Title from the DataFrame to ensure we get the title
                            movie_title = recs.iloc[idx]['Project Title']
                            st.markdown(f"**{movie_title}**")
                            
                            # Get genres safely
                            movie_genres = recs.iloc[idx].get('Genres', 'Unknown')
                            st.markdown(f"*{movie_genres}*")
    except Exception as e:
        st.error(f"Error in main page: {str(e)}")
        st.code(traceback.format_exc())

# Main app flow
def main():
    # Load data first
    df = load_data()
    
    # Debug button to clear session state
    if st.sidebar.button("Reset Session State", key="reset_session"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("Session state has been reset!")
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
    
    # Render appropriate page based on login status
    if st.session_state.get('username') is None:
        login_page()
    else:
        if df is not None:
            main_page(df)
        else:
            st.error("Cannot load main page due to data loading errors.")
            if st.button("Logout", key="error_logout"):
                st.session_state.username = None
                try:
                    st.rerun()
                except AttributeError:
                    st.experimental_rerun()

if __name__ == "__main__":
    main()