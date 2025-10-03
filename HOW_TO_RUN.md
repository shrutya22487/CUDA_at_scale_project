## How to Run the Code

1. Make sure you have Python installed.

2. Set Up a Virtual Environment in VS Code:

* Open VS Code.
* Open the terminal.
* Navigate to your project directory:

* *  cd path\to\your\project
* Create a virtual environment:

* * python -m venv venv
* Activate the virtual environment:
* * Windows:
* * * .\venv\Scripts\activate

* * Mac/Linux:
* * * source venv/bin/activate

3.  Install Required Python Packages:

* In the terminal with the virtual environment activated, install the necessary packages:

* * pip install numpy opencv-python-headless

4. Configure VS Code Python Interpreter:

* Open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P).
* Type Python: Select Interpreter and select the interpreter that matches your virtual environment. It should show up as something like Python 3.x.x (venv).

5. Run the Script:

* In the terminal, ensure your virtual environment is activated.
Run the script:

* * python main.py

* Delete previous outputs to see the changes.

* Following these steps should help you successfully run the image processing code using only NumPy and OpenCV in VS Code. If you encounter any issues, please let me know!

## Proof of Execution

* Input and Output Folders before and after running the code. (Shown with 3 inputs for example)

![alt text](image.png)
* Running the code.

![alt text](image-1.png)
* Input and Output Folders after running the code.

![alt text](image-2.png)

* Works with multiple inputs as well. 