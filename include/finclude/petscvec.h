C
C      Include file for Fortran use of the Vec package in PETSc
C
#define Vec        integer
#define VecScatter integer 
#define NormType   integer
C
      integer INSERT_VALUES, ADD_VALUES
      integer SCATTER_REVERSE,SCATTER_DOWN,SCATTER_UP,SCATTER_ALL,
     *        SCATTER_ALL_REVERSE
      integer PIPELINE_DOWN,PIPELINE_UP

      parameter (INSERT_VALUES = 1, ADD_VALUES = 2)
      parameter (SCATTER_REVERSE = 1,SCATTER_DOWN = 2,SCATTER_UP = 4,
     *           SCATTER_ALL = 8, SCATTER_ALL_REVERSE = 9)
      parameter (PIPELINE_DOWN = 0,PIPELINE_UP = 1)

      integer NORM_1,NORM_2,NORM_FROBENIUS,NORM_INFINITY

      parameter( NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4)
C
C      End of Fortran include file for the Vec package in PETSc

