import subprocess
subprocess.run(["pip", "install", "openai==1.12.0", "--force-reinstall", "--no-cache-dir"], check=True)
from app import app

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)