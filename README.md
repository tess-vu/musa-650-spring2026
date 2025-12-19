# Geospatial Machine Learning in Remote Sensing (MUSA 650)

[Overview](#overview) | [Objectives](#objectives) | [Format](#format) | [Tips](#tips-for-success) | [Grading](#grading) | [Software](#software) | [Schedule](#schedule) | [Academic Integrity](#academic-integrity-and-ai-use)

[![Join Slack](https://img.shields.io/badge/Join-Slack-blue?logo=slack)](https://join.slack.com/t/musa650spring2025/shared_invite/zt-2xlntepg4-mhmTgu10OlqMA~vvznUChA)

## Overview

Satellite remote sensing is the science of converting raw aerial imagery into actionable intelligence about the built and natural environment. In the context of city planning, this is applied to use cases such as building footprint extraction, multi-class object detection (e.g., cars), and land cover/land use classification. This course will provide you the foundation necessary for application of machine learning algorithms on satellite imagery. We will cover the core concepts of machine learning, including supervised and unsupervised learning; model selection; feature engineering; and performance evaluation. The course will cover traditional methods and algorithms as well as recent deep learning methods using convolutional neural networks and their application to semantic image segmentation.

### Instructors

| Name | Email | Office Hours |
|------|-------|-------------|
| [Guray Erus](https://www.med.upenn.edu/cbica/aibil/gerus.html), PhD | [guray.erus@pennmedicine.upenn.edu](mailto:guray.erus@pennmedicine.upenn.edu) | Friday 3-5 PM (or by appointment in case of conflict) |
| [Nissim Lebovits](https://nlebovits.github.io/), MCP | [nissimlebovits@proton.me](mailto:nissimlebovits@proton.me) | By appointment |


## Objectives

The main learning goal of this class is to provide you with the foundational context and skills necessary for use in machine learning applied to remote sensing and enable you to independently pursue further study and work. We will focus specifically on remote sensing use cases in city planning, with a special emphasis on deep learning applications (e.g., CNNs). You will learn how to define a problem, select appropriate algorithms and tools, design and implement their machine learning models, and apply and validate their models on new datasets.

Given the ready availability of code documentation and AI-generated code, we will emphasize a strong conceptual understanding of remote sensing and machine learning fundamentals. This is a hands-on class involving multiple examples that will be explained and run real-time during the lectures. The course will primarily use Python-based implementations of remote sensing, such as `pystac` and `geemap`, although you will be encouraged to explore other relevant tools on your own.

## Format

The course will be divided into two halves. In the first half of the semester, we will focus on building a strong foundational understanding of remote sensing and machine learning. The second half of the semester will delve into more advanced and specific use cases, such as deep learning and data pipelines.

Most weeks, we will divide class time evenly between lectures and labs. Lectures will focus on a conceptual overview of the material, while labs will offer hands-on time to work on pairs or groups on assignments designed to build remote sensing skills applied to planning-specific use cases, including in-class exercises, homeworks, and the final project.

## Tips for Success

Based on our experiences teaching (and taking!) this class in previous years, here are a couple pieces of advice for how to approach the class:

1. Be patient. Remote sensing and machine learning are big domains, and it will take you a little while to wrap your head around the jargon and the core concepts. We will deliberately return to key topics repeatedly throughout the semester in order to give you a ample opportunity to absorb all of the most important information.

2. Give yourself enough time to complete assignments. Programming always involves a lot of debugging, and this class is no exception. We recommend budgeting at least twice as much time as you _think_ you'll need to complete an assigingment. Trust us--this will save you a lot of stress.

3. Come ready to explore! Remote sensing and machine learning are massive domains--more than we can possibly cover in one semester. This class is meant to be a launching pad; we hope you'll leave it with a sense of excitement and curiosity about where else remote sensing and machine learning can take you.

## Grading

There are five assignments over the course of the semester: [three homeworks](assignments/), each worth 10% of the overall grade (for a total of 30%), one [final project proposal](assignments/FINAL_PROJECT_PROPOSAL.md) worth 10%, and one [final project](assignments/FINAL_PROJECT.md) worth 40% of the overall grade. A further 20% of the grade is based on participation. For more information, please see the rubrics included with each assignment.

Homework can be started at any time but is due by the end of class by the date indicated on the syllabus. Unexcused late homework will not be accepted.

## Submission Guidelines

**All assignments must be completed in groups of 2-3 students.** One member of the group should submit on behalf of everyone, making sure to include all group members' names at the top of the notebook.

**Notebook Requirements:**

- All Jupyter notebooks must include the assignment number in the filename
- All notebooks must contain the complete assignment instructions, followed by the relevant code chunks
- All notebooks must include group members' names and submission date
- All code must be well-formatted with appropriate code chunks (no overly long code blocks)
- All code must be linted and formatted using [`ruff`](https://docs.astral.sh/ruff/) before submission

**Submission Structure:**
Assignments should be submitted via a pull request to the main branch of this repository with the following structure:

```
assignments/
  submissions/
    [ASSIGNMENT_NAME]/
      [ASSIGNMENT_NAME].ipynb
      [other required files]
```

**Visualization Requirements:**
For assignments that include interactive visualizations (e.g., `geemap`), please include a .gif of you clicking through each layer in your map. Embed this .gif in your notebook and include it in your submission folder. Do not include the interactive widget itself, as this often doesn't render well on GitHub.

## Software

This course relies on the use of Python and various related packages. All software is open-source and freely available. We will use common tools to facilitate collaboration and ease of use. You are expected to use VSCode, Git, and `uv`during the course in order to make sure that we're all using consistent environments and minimize dependency issues and other kinds of software problems.

### Installation

1. Install [VSCode](https://code.visualstudio.com/download).
2. Install [Git Bash](https://git-scm.com/install/).
3. Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
4. Create a fork of this repository.
5. Clone your fork to your local machine in the location of your choice.
6. In VSCode, open the cloned repository. Open a Git Bash terminal and run `uv sync` to install the dependencies.
7. That's it! You're now set up.

## Schedule

MUSA 650 meets in person on Thursday from 8:30am-11:29am in Meyerson B13. In person participation is required. Online participation is an option only in case of a valid excuse with approval.

| Week # | Date | Topic | Assignment | Lab |
|--------|------|-------|------------|-------|
| 1 | 1/15 | Course intro, overview of remote sensing and applications in city planning | None | | 
| 2 | 1/22 | Data sources and IO (Landsat, Sentinel, STAC API, `pystac-client`, etc.) | None | |
| 3 | 1/29 | Non-ML approaches to remote sensing (indices, thresholds, change detection, time series) | None | [Valencia Flooding](week03/README.md) | 
| 4 | 2/5 | Introduction to machine learning (unsupervised and supervised) | None | |
| 5 | 2/12 | Supervised learning for land cover classification, Part 1: training data, cross-validation, method and model selection | None | |
| 6 | 2/19 | Supervised learning for land cover classification, Part 2: validation, evaluation, multi-class classification, parameter optimization | None | |
| 7 | 2/26 | | Midterm Exam | Google Earth Engine lab (LINK TO DO) |
| Spring Break | | | |
| 8 | 3/12 | Fundamentals of deep learning | None | | 
| 9 | 3/19 | Recent advances in deep learning: literature review and paper discussion | Final Project Proposal Due (LINK TO DO) | | 
| 10 | 3/26 | Convolutional neural networks for image classification; UNet architecture for semantic segmentation | None | |
| 11 | 4/2 | Case study: deep learning applications (Guest: Tristan) | EuroSat Assignment Due (LINK TO DO) |
| 12 | 4/9 | Case study: predictive modeling using deep learning (Guest: TBD) | None | |
| 13 | 4/16 | Big data and machine learning: techniques, tools, challenges, future directions | Final Exam (LINK TO DO) | None |
| 14 | 4/23 | Final project presentations | Final Project Due (LINK TO DO) | None |

## Academic Integrity and AI Use

All students are expected to comply with with Penn's [Code of Academic Integrity](http://www.upenn.edu/academicintegrity/ai_codeofacademicintegrity.html), and obvious copying is prohibited. That said, this course relies extensively on [pair programming](https://www.codecademy.com/resources/blog/what-is-pair-programming/) and aims to prepare you for real-world work. So, you are encouraged to learn how to collaborate effectively and use the internet and other sources to support your work, so long as you attribute it clearly when adapting large chunks of code or other material from other sources. In cases where this is unclear, please make sure to attribute your source.

Relatedly, we have no issue with using AI tools to help with coding (note: you have all have [free GitHub Copilot access as long as you are students](https://education.github.com/pack)). That said, when using any of these tools, you are subject to the standard citation guidelines. Also, be aware that AI tools are subject to mistakes, especially as you get deeper into specialized technical tools. If you are too reliant on these tools, you run the risk of wasting time debugging nonsense code, or not understanding the underlying strucure of what you're writing. Ultimately, it's in your best interest to learn to use AI tools as _tools_--not as replacements for critical thinking--and to learn how to use them _intelligently and effectively_.

