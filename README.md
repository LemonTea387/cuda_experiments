# Project Description
- Mainly to investigate improved performance in matrix operations using GPU Computing
- Implemented addition, multiplcation and inversion (Gaussian-Jordan algorithm)
- CUDA C/C++ is the main language used with CUDA implementation
- CUDA 11.7 version is used
- OpenMP 5.0 is used for cpu parallelism

# Build
## Required Installation to run the application
- Microsoft Windows (Windows specific library used)
- CUDA Toolkit 11.7.x version
- OpenMP 5.0 compatible c compiler (MinGW gcc, clang)
## Compilation Commands
`nvcc` and `MinGW gcc` is used.

```bash
nvcc kernel.cu -o kernel.exe
gcc -fopenmp cpu_powered.c -o cpu.exe
```

# Result Graph Plotting
## Required Python Packages
- Matplotlib 3.5.3
## Steps to replicate graph
1) Run kernel program and cpu program
2) Copy and paste the results of both into a text file (order of whichever program's result does not matter) named `result.txt`.
3) Run `python3 plot.py`

# Presentation Video
https://drive.google.com/file/d/1DWWVazghMX0E5ABddE85y1z5EibXjj6o/view?usp=sharing