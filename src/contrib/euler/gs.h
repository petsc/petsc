!
!  Used for original Julianne solver (uniprocessor only)
!  Not used for PETSc solvers.
!
       common /gs/ res(5,d_ni,d_nj,d_nk),vec(5,d_ni,d_nj,d_nk)
       double precision res, vec
