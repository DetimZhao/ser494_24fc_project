# Introduction
This repository contains files for the individual course project in SER494: Data Science for Software Engineers (fall 2024) created by Detim Zhao for partial fulfillment of the course requirements.

It was cleared by course staff (R. Acuna) for public release on 1/12/2025.

# Paper

To check out the paper refer to: [Paper.pdf](/Paper.pdf) in the content root. 

# Steps to download and run project:
## Setup
1. Clone the repository:
    ```sh
    git clone https://github.com/DetimZhao/ser494_24fc_project.git
    ```
2. Navigate to the project directory:
    ```sh
    cd ser494_24fc_project
    ```
3. Create a virtual environment:
    ```sh
    python3 -m venv venv
    ```
4. Activate the virtual environment:
    - On macOS/Linux:
      ```sh
      source venv/bin/activate
      ```
    - On Windows:
      ```sh
      .\venv\Scripts\activate
      ```
5. If you're on macOS, uncomment the two requirements in the `requirements.txt` file.
6. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
## Running workflow
1. Locate the `wf_core.py` file in the project directory.

2. Run the script using Python:
    ```sh
    python wf_core.py
    ```

3. Alternatively, you can run the script using something like Visual Studio Code:
    - Open Visual Studio Code.
    - Open the project directory in Visual Studio Code.
    - Locate the `wf_core.py` file in the file explorer.
    - Right-click on the file and select "Run Python File in Terminal".
