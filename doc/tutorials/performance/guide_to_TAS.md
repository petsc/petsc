# Time, Accuracy, Speed Analysis (TAS)

```{highlight} none
```

Below is the guide to running TAS using ex13, which is a Poisson Problem in 2D and 3D with Finite Elements:

1. Compile `ex13.c`

> 1. This example source file, and the corresponding `makefile` are located in `PETSC_DIR/src/snes/tutorials/`
>
> 2. Compile with the command:
>
>    > ```console
>    > $ make ex13
>    > ```

2. Run ex13 with the following command:

   ```
   mpiexec -n 2 ./ex13 -log_view :ex_13_test.csv:ascii_csv \
     -dm_distribute \
     -dm_plex_box_faces 8,8 \
     -potential_petscspace_degree 1 \
     -snes_convergence_estimate \
     -convest_num_refine 5
   ```

3. A log file called `ex_13_test.csv` should now be present in the current directory.

4. Analyze the CSV log with `petsc_tas_analysis.py`. The Python environment must
   have NumPy, pandas, and Matplotlib installed:

   ```
   python3 "$PETSC_DIR/lib/petsc/bin/petsc_tas_analysis.py" -f ex_13_test.csv
   ```

5. You should see something similar to the following in your terminal window:

   > ```{literalinclude} exampleTASOutPut.txt
   > ```

6. The graphs are written to the current directory unless a different output
   directory has been set through `defaultGraphs` in `configureTAS.py`.

For more detailed help in using TAS:
: 1. See detailed user's guide
  2. On the command line use `python3 "$PETSC_DIR/lib/petsc/bin/petsc_tas_analysis.py" -h`
