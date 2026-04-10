\# Uniform Feasibility for Smoothed Backup Control Barrier Functions



MATLAB code accompanying the paper:



> A. Alan and B. De Schutter, "Uniform Feasibility for Smoothed Backup Control Barrier Functions," \*European Control Conference (ECC)\*, 2026.



The script reproduces the double-integrator example of Section V: it computes the system data $(M, r, d)$, the smoothing threshold $\\theta^\*$, and the class-$\\mathcal{K}$ gain $\\gamma^\*$ on a grid, and simulates the classical and smoothed backup CBF safety filters for several values of $\\gamma$.



\## Requirements



\- MATLAB R2023b or later

\- Optimization Toolbox (`quadprog`)

\- Control System Toolbox (`lyap`)



\## Usage



Clone the repository and run the main script from MATLAB:



```matlab

>> main

```



The script prints the computed bounds, thresholds, and timing information to the console, and produces the figures shown in the paper.



\## Citation



```bibtex

@inproceedings{alan2026uniform,

&#x20; title     = {Uniform Feasibility for Smoothed Backup Control Barrier Functions},

&#x20; author    = {Alan, Anil and De Schutter, Bart},

&#x20; booktitle = {European Control Conference (ECC)},

&#x20; year      = {2026}

}

```

