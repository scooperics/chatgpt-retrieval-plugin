import os
import subprocess
from dotenv import load_dotenv
from pathlib import Path

def load_env_file(env_file_path):
    # Read the content of the original .env file
    with open(env_file_path, 'r') as file:
        content = file.read()

    # Remove 'export' keyword and write the modified content to a temporary file
    temp_env_path = Path("temp_env_file")
    with open(temp_env_path, 'w') as temp_file:
        temp_file.write(content.replace("export ", ""))

    # Load environment variables from the temporary file
    load_dotenv(dotenv_path=temp_env_path)

    # Remove the temporary file
    os.remove(temp_env_path)

def get_docker_container_ip(container_name):
    result = subprocess.run(["docker", "inspect", "-f", "{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}", container_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode == 0:
        return result.stdout.decode('utf-8').strip()
    else:
        raise Exception(f"Error getting container IP: {result.stderr.decode('utf-8')}")