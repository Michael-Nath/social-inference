import requests
import io

# Hardcoded values
repo_id = "facebook/opt-350m"  # Example model
filename = "model.safetensors"
start_offset = 1000  # Starting byte
length = 5000  # Number of bytes to download

# Construct URL
url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

# Set up range request header
headers = {"Range": f"bytes={start_offset}-{start_offset+length-1}"}

# Make the request
response = requests.get(url, headers=headers, stream=True)

# Check if we got partial content
if response.status_code != 206:  # 206 is Partial Content
    print(f"Failed with status code: {response.status_code}")
else:
    # Stream to buffer
    buffer = io.BytesIO()
    for chunk in response.iter_content(chunk_size=8192):
        buffer.write(chunk)
    buffer.seek(0)
    
    # Print first few bytes to verify
    print(f"Downloaded {len(buffer.getvalue())} bytes")
    print(f"First 20 bytes: {buffer.getvalue()[:20]}")