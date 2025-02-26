from flask import Flask, request, render_template
import os
from openai import OpenAI

app = Flask(__name__)

# Environment variables
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form['query']
        # Your OpenAI logic here
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": query}]
        )
        answer = response.choices[0].message.content
        return render_template('index.html', answer=answer, query=query)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)