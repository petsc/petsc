c
c  Used for original Julianne solver (uniprocessor only)
c  Not used for PETSc solvers.
c
       common /gs/ res(5,d_ni,d_nj,d_nk),vec(5,d_ni,d_nj,d_nk)
       double precision res, vec
