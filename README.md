# Shadow Flicker Assessment

**If you encounter any issues, please don't hesitate
to [report
them](https://github.com/leotiger/shadow-flicker-assessment/issues).**

![Screenshot](Screenshot.png)

The **Shadow Flicker Assessment** allows you to visualize shadow flicker caused by wind turbines.
The tool implements the [German protocol for Shadow Flicker assessment](https://www.lai-immissionsschutz.de/documents/wka_schattenwurfhinweise_stand_23_1588595757.01.pdf) which is a de facto standard applied in many countries,
including Spain, Catalonia, Germany, Chile and many, many others.

The tool is not meant for oficial documentation, this tool helps you to check shadow flicker studies provided by promotors for their projects within the process to obtain environmental approval for their proposed projects.

Nevertheless, the **Shadow Flicker Assessment** obtains reliable results comparable to professional assessment tools used by the industry like WindPro.

## Installation

Shadow Flicker Assessment is a command line script for python.
Your python enviroment needs to be setup to support all necessary libraries.
Please use conda or pip to install all missing libraries if the script complains on execution.


## Requirements

python with support for:

- matplotlib
- rasterio
- numpy
- shapely
- numba

The rest of imported libraries should be available within your standard python environment.

## Configuration

The Shadow Flicker Assessment tool reads project configurations via command line args using a .yaml file to 
obtain essential project data, e.g. wind turbine(s) data, receptors data, elevation map data (DEM)...

You can use the --fast (-f) flag to run quick tests which reduces the calculation amount considerably. For final assessments you should
run the script without the fast flag, but be advised that the script may run for hours as the computational workload is impressive.

To run only one of the two scenarios available, WORST and REALISTIC, please provide the scenario using the --scene (-s) flag.

To run assessments for specific projects you'll have to gather a lot of data first. Once you have the data available you can edit the assessment configuration .yaml for the wind park you want to investigate. You provide your .yaml configuration via the --config (-c) flag.

### Important

Despite having been reviewed and found error free, be advised that this script is a helper tool. You may want (or need) to calculate some data points manually to prove correctness of the assumptions and results provided by this script.

### ToDo

- [x] Clean code base
- [x] Export of all data for traceability
- [x] Accelerate processing by applying precalc and cache of solar posistions
- [x] Improve yaml documentation and comments

