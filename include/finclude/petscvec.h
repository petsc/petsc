C
C  $Id: vec.h,v 1.12 1996/04/16 00:40:41 balay Exp bsmith $;
C
C  Include file for Fortran use of the Vec package in PETSc
C
#define Vec          integer
#define VecScatter   integer 
#define NormType     integer
#define InsertMode   integer
#define ScatterMode  integer 

C
C  VecType
C     
      integer VECSAME, VECSEQ, VECMPI
      parameter (VECSAME=-1, VECSEQ=0, VECMPI=1)

C
C  Types of vector and matrix norms
C
      integer NORM_1,NORM_2,NORM_FROBENIUS,NORM_INFINITY,NORM_MAX

      parameter(NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4,
     *          NORM_MAX=4)

C
C  Flags for VecSetValues() and MatSetValues()
C
      integer NOT_SET_VALUES,INSERT_VALUES, ADD_VALUES

      parameter (NOT_SET_VALUES=0,INSERT_VALUES=1, ADD_VALUES=2)
C
C  Types of vector scatters
C
      integer SCATTER_REVERSE,SCATTER_ALL

      parameter (SCATTER_REVERSE = 1,SCATTER_ALL = 8)

C
C  End of Fortran include file for the Vec package in PETSc

