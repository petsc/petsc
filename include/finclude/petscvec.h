C
C      Include file for Fortran use of the Vec package in PETSc
C
#define Vec        integer
#define VecScatter integer 
C
      integer INSERT_VALUES, ADD_VALUES
      integer SCATTER_REVERSE,SCATTER_DOWN,SCATTER_UP,SCATTER_ALL,
     *        SCATTER_ALL_REVERSE
      integer PIPELINE_DOWN,PIPELINE_UP

      parameter (INSERT_VALUES = 1, ADD_VALUES = 2)
      parameter (SCATTER_REVERSE = 1,SCATTER_DOWN = 2,SCATTER_UP = 4,
     *           SCATTER_ALL = 8, SCATTER_ALL_REVERSE = 9)
      parameter (PIPELINE_DOWN = 0,PIPELINE_UP = 1)
C
C      End of Fortran include file for the Vec package in PETSc

