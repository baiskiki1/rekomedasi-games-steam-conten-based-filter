from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uvicorn
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- PROSES ANALISIS DATASET ---
print("--- MEMULAI ANALISIS DATASET ---")
try:
    # Memuat dataset dan reset indeks agar koordinat data akurat
    df = pd.read_csv('games.csv', low_memory=False).fillna('')
    df = df.reset_index(drop=True) 
    
    # Deteksi Kolom Nama Game
    name_col = 'Name'
    for col in df.columns:
        sample = str(df[col].iloc[0])
        if not re.search(r'\d{4}', sample) and not sample.isdigit() and len(sample) > 3:
            name_col = col
            break

    # Deteksi Kolom AppID
    appid_col = 'AppID'
    for col in df.columns:
        if str(df[col].iloc[0]).isdigit() and col != 'Required age':
            appid_col = col
            break

    print(f"SISTEM: Terdeteksi Nama: '{name_col}', AppID: '{appid_col}'")
    
    # Gabungkan fitur teks untuk AI
    df['features'] = (
        df['Genres'].astype(str) + " " + 
        df['Categories'].astype(str) + " " + 
        df['Tags'].astype(str)
    ).str.lower().fillna('')
    
    # Inisialisasi TF-IDF (Basis referensi 25.000 game)
    tfidf = TfidfVectorizer(stop_words='english')
    limit_ref = min(25000, len(df))
    tfidf_matrix = tfidf.fit_transform(df['features'].head(limit_ref))
    
    print("--- BACKEND SERVER SIAP DI PORT 8001 ---")
except Exception as e:
    print(f"ERROR STARTUP: {e}")

def safe_int(val):
    try:
        clean_val = re.sub(r'\D', '', str(val))
        return int(clean_val) if clean_val else 0
    except: return 0

@app.get("/recommend/{name}")
def get_recommendations(name: str):
    try:
        query = name.strip().lower()
        matched = df[df[name_col].astype(str).str.lower().str.contains(query, na=False)]
        
        if matched.empty:
            return {"error": f"Game '{name}' tidak ditemukan."}
        
        actual_idx = matched.index[0]
        game_features = str(df.loc[actual_idx, 'features'])
        
        # Hitung kemiripan dinamis
        query_vec = tfidf.transform([game_features])
        sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        # Ambil indeks dengan skor tertinggi
        related_indices = sim_scores.argsort()[-7:][::-1]
        
        results = []
        for i in related_indices:
            if i == actual_idx: continue
            results.append({
                "Name": str(df.loc[i, name_col]),
                "Genres": str(df.loc[i, 'Genres']),
                "Header image": str(df.loc[i, 'Header image']),
                "AppID": safe_int(df.loc[i, appid_col]),
                "Score": float(sim_scores[i]) # Mengirim skor akurasi
            })
        return results[:6]
    except Exception as e:
        return {"error": f"Gagal menghitung AI: {str(e)}"}

@app.get("/popular")
def get_popular():
    try:
        popular = df.head(12)
        res = []
        for i in popular.index:
            res.append({
                "Name": str(df.loc[i, name_col]), 
                "Genres": str(df.loc[i, 'Genres']), 
                "Header image": str(df.loc[i, 'Header image']), 
                "AppID": safe_int(df.loc[i, appid_col]),
                "Score": 1.0 # Default 100% untuk populer
            })
        return res
    except: return {"error": "Gagal memuat data."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)