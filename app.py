from flask import Flask, render_template, request
from pinecone import Pinecone  # Ensure this works
from openai import OpenAI
import webbrowser
from PyPDF2 import PdfReader
import os
from pdf2image import convert_from_path
import re

app = Flask(__name__, static_folder='static')

# Initialize Pinecone with environment variable
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "novel-data"
pinecone_index = pc.Index(index_name)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# PDF loader with chunking and page-specific image extraction
def load_pdf(pdf_path, prefix=""):
    reader = PdfReader(pdf_path)
    pages_text = [page.extract_text() or "" for page in reader.pages]
    all_text = " ".join(pages_text)
    words = all_text.split()
    chunk_size = 500
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    # Extract images
    images = convert_from_path(pdf_path)
    image_dir = os.path.join(os.path.dirname(__file__), "static", "images")
    os.makedirs(image_dir, exist_ok=True)
    image_paths = []
    for i, img in enumerate(images):
        img_filename = f"{prefix}page_{i}.png"
        img_path = os.path.join(image_dir, img_filename)
        img.save(img_path, "PNG")
        relative_path = f"/static/images/{img_filename}"
        image_paths.append(relative_path)

    # Map chunks to images by keyword relevance
    vectors = []
    for i, chunk in enumerate(chunks):
        if chunk:
            keywords = r"(key fob|remote key|keyless|smart key)"
            chunk_lower = chunk.lower()
            best_page = None
            if re.search(keywords, chunk_lower):
                for page_num, page_text in enumerate(pages_text):
                    if re.search(keywords, page_text.lower()):
                        best_page = page_num
                        break
            if best_page is None:
                words_per_page = len(words) / len(pages_text) if pages_text else 1
                chunk_start_word = i * chunk_size
                best_page = min(int(chunk_start_word / words_per_page), len(image_paths) - 1) if words_per_page else 0
            
            img_path = image_paths[best_page] if best_page < len(image_paths) else None
            vector = {
                "id": f"{prefix}id_{i}",
                "values": client.embeddings.create(input=chunk, model="text-embedding-ada-002").data[0].embedding,
                "metadata": {"text": chunk, "image": img_path if img_path else ""}
            }
            vectors.append(vector)
    pinecone_index.upsert(vectors=vectors)
    print(f"Uploaded {len(vectors)} vectors from {pdf_path} with {len(image_paths)} images")

# Load Mercedes manual—run locally once, not on deploy
# load_pdf("C:/Mercedes/23MercedesManual.pdf", prefix="merc_")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        query = request.form["query"]
        query_embedding = client.embeddings.create(input=query, model="text-embedding-ada-002").data[0].embedding
        results = pinecone_index.query(vector=query_embedding, top_k=1, include_metadata=True)
        if results["matches"]:
            match = results["matches"][0]
            retrieved_text = match.metadata["text"]
            image_path = match.metadata.get("image", "")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant summarizing a Mercedes-Benz user manual."},
                    {"role": "user", "content": f"Based on this: {retrieved_text}, answer: {query}"}
                ]
            )
            answer = response.choices[0].message.content
            return render_template("result.html", query=query, text=retrieved_text, answer=answer, image=image_path)
        else:
            return render_template("result.html", query=query, text="No match found", answer="Sorry, I could not find an answer.", image="")
    return render_template("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = "0.0.0.0" if os.environ.get("FLASK_ENV") == "production" else "127.0.0.1"
    debug = os.environ.get("FLASK_ENV") != "production"
    
    if debug:
        try:
            with open("app_running.txt", "r") as f:
                pass
        except FileNotFoundError:
            with open("app_running.txt", "w") as f:
                f.write("running")
            webbrowser.open(f"http://{host}:{port}")
        app.run(debug=debug, host=host, port=port, use_reloader=False)
        os.remove("app_running.txt")
    else:
        app.run(debug=debug, host=host, port=port)