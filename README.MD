# Multiprocessing Optimization and Visualization of Genomic Data(Big Data in Genomic)
Main technology stack: Python, Dash, CSS;

## Introduction

For tens of terabytes of genomic data, a single thread/single core run would be time consuming. The goal of this project is to optimise an existing serial solution with multiple processes, and to develop an web applicaiton for data visualization, which should enable researchers to analyze and visualize the data as well as the results.

It consists of 2 tasks:
  1. Multiprocessing
  2. Data visualization
  
Multiprocessing task: To optimise an existing serial solution with multiple processes, running simultaneously on multiple cores, to reduce code runtime and improve efficiency.
responsible for optimizing the existing genomic coding serial scheme, adopting the producer-consumer mode, constructing three types of workers (Reader, Encoder, Writer) and two task queues respectively, using locks to avoid process conflicts, significantly optimizing the program running time.

Data visualization task: To develop Web applications based on Dash and DashBio framework to build the data visualization tool to show the details of FASTQ data via alignment chart, filterable data table. Using CSS to design the layout of the components and make them more aesthetically pleasing and easy for users, realizing the visualization of gene data and real-time interaction functions to provide better intuitive feedback to the researchers.



## Visualization: Web application Layout
![Web application Layout](https://github.com/ChenErdi/Multiprocessing/blob/master/imgs/Layout.jpg)

---

![Drop-down](https://github.com/ChenErdi/Multiprocessing/blob/master/imgs/drop-down_heatmap.jpg)

---

![Sliders](https://github.com/ChenErdi/Multiprocessing/blob/master/imgs/Sliders.jpg)

---

## Multiprocessing
In this multiprocessing scenario, a total of **N** processes are used, with one process running the reader, **N-2** processes running the encoder and **1** process running the writer.
In practice, **N** is set to 10 and 16 (the number of processes is set according to the actual hardware).

![Cores number: **N=10**](https://github.com/ChenErdi/Multiprocessing/blob/master/core.png)

---
![Multiprocessing start: **N=10**](https://github.com/ChenErdi/Multiprocessing/blob/master/process_start.png)
