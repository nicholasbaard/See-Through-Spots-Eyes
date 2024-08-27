# See-Through-Spots-Eyes
A Computer Vision Project to demonstrate various augmentations and overlays of streamed video

## Installation

### Python version

This project requires Python 3.11 or higher. Make sure you have the correct version installed before proceeding with the installation steps.

To check your Python version, run:


python --version


If you need to upgrade or install Python 3.11+, visit the official Python website: https://www.python.org/downloads/

### Clone the repository

```bash
git clone https://github.com/nicholasbaard/See-Through-Spots-Eyes.git
cd See-Through-Spots-Eyes
```

### Create a virtual environment

It's recommended to use a virtual environment to avoid conflicts with other projects or system-wide packages.


# Create a virtual environment

```bash
python -m venv venv
```

# Activate the virtual environment
# On Windows:

```bash
venv\Scripts\activate
```

# On macOS and Linux:

```bash
source venv/bin/activate
```

After activating the virtual environment, your command prompt should show the name of the virtual environment, indicating that it's active.

### Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

To run the various solutions, run the following commands:

## Note:

stream_url is currently set to None in each of the scipts. This will cause the streamlit app to use your webcam on your laptop. To stream from spot, you will need to change the stream_url to the spot url.

```python
stream_url = None 
```

### cartoonify

#### cartoon

```bash
cd cartoon
streamlit run cartoonise_simple.py
```

#### pencilize

```bash
cd cartoon
streamlit run pencilize_simple.py
```

#### ppedetector
Note: currently not working
```bash
cd cartoon
streamlit run ppe_detector.py
```

### segmentation

```bash
cd segmentation
streamlit run st_pipeline_segmentation.py
```

### pose

```bash
cd pose
streamlit run st_pipeline_multi_pose.py
```