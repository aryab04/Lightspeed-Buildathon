import os
import streamlit as st
from keybert import KeyBERT
import shutil

# --- Page Configuration ---
st.set_page_config(page_title="Synthetic Image Studio", page_icon="🧠", layout="wide")

# --- Directories ---
DATASET_DIR = "datasets"  # Folder containing your dataset folders
ZIP_DIR = "zips"          # Folder to store temporary ZIP files
os.makedirs(ZIP_DIR, exist_ok=True)

# --- Load Keyword Model ---
@st.cache_resource
def load_keyword_model():
    return KeyBERT()

kw_model = load_keyword_model()

# --- Keyword Extraction ---
def extract_keyword(query):
    keywords = kw_model.extract_keywords(
        query,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=1
    )
    return keywords[0][0] if keywords else query.strip()

# --- Dataset Matching ---
def find_matching_datasets(keyword):
    matches = []
    for folder in os.listdir(DATASET_DIR):
        if keyword.lower() in folder.lower():
            matches.append(folder)
    return matches

# --- Simulated Navigation ---
pages = {
    "🏠 Home": "home",
    "🧠 Convert MRI to CT": "mri_ct",
    "🩻 Convert CT to MRI": "ct_mri",
    "📊 Image Segmentation": "analysis",
    "📤 Upload Datasets": "upload"
}
selected_page = st.sidebar.radio("## KEY FUNCTIONALITIES", list(pages.keys()))

# --- Top Navigation Bar ---
st.markdown("""
    <style>
    .top-bar {
        background-color: #e3edf7;
        padding: 1rem 2rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'Segoe UI', sans-serif;
        max-width: 80rem;
        margin: auto;
    }
    .logo {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1c2b39;
    }
    .nav-left {
        display: flex;
        gap: 1.5rem;
        align-items: center;
    }
    .nav-center {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1c2b39;
        margin: 0 auto;
    }
    .nav-right {
        margin-left: auto;
    }
    .dropdown {
        background: none;
        border: none;
        font-size: 1rem;
        color: #1c2b39;
        font-weight: bold;
        cursor: pointer;
    }
    .dropdown:hover {
        text-decoration: underline;
    }
    </style>
    <div class="top-bar">
        <div class="nav-left">
            <select class="dropdown">
                <option>Partners</option>
                <option>AIIMS, New Delhi</option>
                <option>Mayo Clinic</option>
                <option>CMC, Vellore</option>
                <option>John Hopkins</option>
                <option>Stanford Health</option>
            </select>
            <select class="dropdown">
                <option>About</option>
                <option>Overview</option>
                <option>Team</option>
                <option>Contact</option>
            </select>
        </div>
        <div class="nav-center">SimuData</div>
        <div class="nav-right">
            <select class="dropdown">
                <option>Sign In</option>
                <option>Create Account</option>
            </select>
        </div>
    </div>
""", unsafe_allow_html=True)


# --- Page Content ---
if pages[selected_page] == "home":
    st.markdown("""
    <h1 style='text-align: center; margin-top: 2rem; color: #ADD8E6; font-size: 3.5rem;'>
        SimuData : A one stop platform for synthetic data processing
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
        <h3 style='text-align: center; font-size: 2rem; margin-top: 1rem; color: white;'>
            What datasets do you need?
        </h3>
    """, unsafe_allow_html=True)


    query = st.text_input("", placeholder="Search datasets...", label_visibility="collapsed")

    if query:
        keyword = extract_keyword(query)
        # st.markdown(f"📌 **Detected keyword**: `{keyword}`")

        matching_folders = find_matching_datasets(keyword)

        if matching_folders:
            st.success(f"✅ Found {len(matching_folders)} matching dataset folder(s):")

            for folder in matching_folders:
                st.subheader(f"📁 {folder}")
                folder_path = os.path.join(DATASET_DIR, folder)
                zip_path = os.path.join(ZIP_DIR, f"{folder}.zip")

                if not os.path.exists(zip_path):
                    shutil.make_archive(
                        base_name=os.path.join(ZIP_DIR, folder),
                        format='zip',
                        root_dir=folder_path
                    )

                with open(zip_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download folder as ZIP",
                        data=f,
                        file_name=f"{folder}.zip",
                        mime="application/zip"
                    )
        else:
            st.warning("😕 No matching datasets found.")

    st.markdown("""
        <div style='display: flex; justify-content: center; gap: 3rem; margin-top: 4rem;'>
            <div style='text-align: center; background-color: #fff; padding: 1rem 1.5rem; border-radius: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); width: 160px;'>
                <h4 style='margin-bottom: 0.5rem; color: #1c2b39;'>Number of Datasets</h4>
                <h2 style='color: #1c2b39;'>3,000+</h2>
            </div>
            <div style='text-align: center; background-color: #fff; padding: 1rem 1.5rem; border-radius: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); width: 160px;'>
                <h4 style='margin-bottom: 0.5rem; color: #1c2b39;'>Active Users</h4>
                <h2 style='color: #1c2b39;'>10,000+</h2>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif pages[selected_page] == "mri_ct":
    st.title("🧠 Convert MRI to CT")
    st.info("Simulated MRI to CT conversion interface")
    uploaded_file = st.file_uploader("Upload an MRI image", type=["png", "jpg", "jpeg"])

    st.markdown("Converting...")
    st.image("mrict.jpeg", use_column_width=True)

elif pages[selected_page] == "ct_mri":
    st.title("🩻 Convert CT to MRI")
    st.info("Simulated CT to MRI conversion interface")
    uploaded_file = st.file_uploader("Upload a CT image", type=["png", "jpg", "jpeg"])

    st.markdown("Converting...")
    st.image("ctmri.jpeg", use_column_width=True)

elif pages[selected_page] == "analysis":
    st.title("📊 Image Segmentation")
    st.info("Image Segmentation dashboard")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    st.markdown("Analysing...")
    st.image("seg.jpeg", caption="Segmentation Output", use_column_width=True)

elif pages[selected_page] == "upload":
    st.title("📤 Upload Datasets")
    uploaded = st.file_uploader("Choose a dataset to upload")
    if uploaded:
        dest = os.path.join(DATASET_DIR, uploaded.name)
        with open(dest, "wb") as f:
            f.write(uploaded.read())
        st.success(f"Uploaded {uploaded.name} successfully!")

# --- Footer ---
st.markdown("""<div style='text-align: center; margin-top: 3rem; color: gray;'>© 2025 GenAI Hackathon Project • </div>""", unsafe_allow_html=True)
