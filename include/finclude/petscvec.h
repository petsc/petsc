C
C  $Id: vec.h,v 1.14 1996/11/27 22:58:23 bsmith Exp balay $;
C
C  Include file for Fortran use of the Vec package in PETSc
C
#define Vec          integer
#define VecScatter   integer 
#define NormType     integer
#define InsertMode   integer
#define ScatterMode  integer 
#define VecType      integer
C
C  VecType
C     
      integer VECSAME, VECSEQ, VECMPI
      parameter (VECSAME=-1, VECSEQ=0, VECMPI=1)
C
C  Types of vector and matrix norms
C
      integer NORM_1,NORM_2,NORM_FROBENIUS,NORM_INFINITY,NORM_MAX,
     *        NORM_1_AND_2
      parameter(NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4,
     *          NORM_MAX=4,NORM_1_AND_2=5)
C
C  Flags for VecSetValues() and MatSetValues()
C
      integer NOT_SET_VALUES,INSERT_VALUES, ADD_VALUES
      parameter (NOT_SET_VALUES=0,INSERT_VALUES=1, ADD_VALUES=2)
C
C  Types of vector scatters
C
      integer SCATTER_FORWARD,SCATTER_REVERSE,SCATTER_FORWARD_LOCAL,
              SCATTER_REVERSE_LOCAL,SCATTER_LOCAL
      parameter (SCATTER_FORWARD=0,SCATTER_REVERSE=1,
     *           SCATTER_FORWARD_LOCAL=2,SCATTER_REVERSE_LOCAL=3,
     *           SCATTER_LOCAL=2)
C
C  End of Fortran include file for the Vec package in PETSc

