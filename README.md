# YoloCuento

# Requisites
Python: https://www.python.org/downloads/

# Set up

python3 -m venv venv # Recreate a new, empty venv

venv/scripts/activate.bat # Activate it

pip3 install -r requirements.txt # Install the dependencies

pip3 uninstall torch torchvision torchaudio

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124



# Run

python yoloCount.py
