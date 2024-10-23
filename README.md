## Bug Reproducibility for Deep Learning Systems (Accepted in EMSE, 2024)

Paper Link: https://arxiv.org/abs/2401.03069
This repository is the replication package for the project "Enhancing the Reproducibility of Deep Learning Bugs: An Empirical Study". This study is conducted under the supervision and guidance of Dr. Masud Rahman and Dr. Foutse Khomh.

### Abstract
**Context:** Deep learning has achieved remarkable progress in various domains. However, like any software system, deep learning systems contain bugs, some of which can have severe impacts, as evidenced by crashes involving autonomous vehicles. Despite substantial advancements in deep learning techniques, little research has focused on reproducing deep learning bugs, which hinders resolving them. Existing literature suggests that only 3\% of machine/deep learning bugs are reproducible, underscoring the need for further research.

**Objective:** This paper examines the reproducibility of deep learning bugs. We identify edit actions and useful information that could improve deep learning bug reproducibility.

**Method:** First, we construct a dataset of 668 deep learning bugs from Stack Overflow and Defects4ML across three frameworks and 22 architectures. Second, out of the 668 bugs, we select 165 bugs using stratified sampling and try to determine their reproducibility. While reproducing these bugs, we identify edit actions and useful information necessary for their reproduction. Third, we used the Apriori algorithm to identify useful information and edit actions required to reproduce specific bug types. Finally, we conduct a user study with 22 developers to assess the effectiveness of our findings in real-life settings. 

**Results:** We successfully reproduced 148 out of 165 bugs attempted. We identified ten edit actions and five useful information categories that can help us reproduce the deep learning bugs. Our findings improved bug reproducibility by 22.92\% and reduced reproduction time by 24.35\% based on a user study with 22 developers.

**Conclusions:** Our research addresses the critical issue of deep learning bug reproducibility. Practitioners and researchers can leverage our findings to improve deep learning bug reproducibility.

### Materials Included
* Analysis Folder: This folder contains Jupyter notebooks focused on dataset analysis. The notebooks include code for implementing the Apriori algorithm, which is used to identify critical edit actions. These actions are essential for gathering the necessary information required to reproduce bugs.
* Dataset Folder: Within this directory, you'll find various datasets, including those for PyTorch, TensorFlow (TF), and Keras Posts. The folder also contains queries used to filter the data and retrieve specific posts. Additionally, reproducibility results are included, along with corresponding edit actions and vital bug report details. Finally, this folder also contains the results for the user study conducted as a part of the third research question.
* Bugs Folder In this folder, a collection of bugs is organized, alongside their original code snippets sourced from Stack Overflow. Completed code snippets associated with each bug are also provided. Each specific bug folder contains the following elements:
  - Finalized code snippet after applying the edit actions
  - Original code snippet from Stack Overflow/Github.
  - `requirements.txt` file is generated to facilitate the installation of necessary dependencies for each specific bug.
* User-Study-RQ3: Contains the forms and hint formulation technique + results for the RQ3.
* Statistical-Tests-RQ3: GLM Dataset and Code for Statistical Analysis of the impact of hints on bug reproducibility.
* LLAMA3-Experiments: Contains the experimental results for the experiment conducted in Appendix B.
* Cohen-Kappa: Contains the data for the bug reproduction and agreement analysis between first author and independent collaborator.

### System Requirements
- **Operating System:** Windows 10 or higher
- **Python Version:** 3.10
- **Disk Space:** ~3.5 GB
- **Development Environment:** Visual Studio Code (VS Code)
- **RAM:** 16GB
- **GPU:** N/A

### Installation Instructions

To replicate the work, follow these steps:

#### Step 1: Setting Up the Virtual Environment
1. Create a virtual environment using the following command:
    ```shell
    python -m venv venv
    ```

##### Step 2: Installing Dependencies
1. After creating the virtual environment, activate it by following the instructions [here](https://docs.python.org/3/library/venv.html).
2. Install the necessary dependencies for the required bug:
    ```shell
    cd Bugs/<BugID>
    pip install -r requirements.txt
    ```
###  Bug Reproduction
After installing the dependencies, run the following command to reproduce the bug
```shell
python main.py &> output.txt
```
This ensures that the code to reproduce the bug is run, and the results are stored in the output file. To check the original bug report, go to the `Dataset_Manual_Reproduction.csv` and find the Stack Overflow Post of the corresponding Bug ID. This will help you verify the output of the reproduced bug and the original error message.


### Bug Reproduction - All Bugs
To reproduce all the bugs, download the data and dependencies for the bugs, and run the following command.
```shell
cd Bugs
python script.py
```

### Analysis
To analyze the results and run the code for Apriori implementation, go to the respective Jupyter notebook in the Analysis folder and run the cells in the notebook sequentially.

### Licensing Information
This project is licensed under the MIT License, a permissive open-source license that allows others to use, modify, and distribute the project's code with very few restrictions. This license can benefit research by promoting collaboration and encouraging the sharing of ideas and knowledge. With this license, researchers can build on existing code to create new tools, experiments, or projects, and easily adapt and customize the code to suit their specific research needs without worrying about legal implications. The open-source nature of the MIT License can help foster a collaborative research community, leading to faster innovation and progress in their respective fields. Additionally, the license can help increase the visibility and adoption of the project, attracting more researchers to use and contribute to it.
