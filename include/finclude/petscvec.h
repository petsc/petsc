!
!  $Id: vec.h,v 1.27 1999/03/24 18:11:56 balay Exp balay $;
!
!  Include file for Fortran use of the Vec package in PETSc
!
#if !defined (__VEC_H)
#define __VEC_H

#define Vec               PetscFortranAddr
#define VecScatter        PetscFortranAddr
#define Map               PetscFortranAddr
#define NormType          integer
#define InsertMode        integer
#define ScatterMode       integer 
#define VecOption         integer
#define VecType           character*(80)
#define PipelineDirection integer
#define PipelineType      integer
#define VecPipeline       integer
#define VecOperation      integer

#define VEC_SEQ    'seq'
#define VEC_MPI    'mpi'
#define VEC_SHARED 'shared'

#endif
!
!
!  Types of vector and matrix norms
!
      integer NORM_1,NORM_2,NORM_FROBENIUS,NORM_INFINITY
      integer NORM_MAX,NORM_1_AND_2

      parameter (NORM_1=1,NORM_2=2,NORM_FROBENIUS=3,NORM_INFINITY=4)
      parameter (NORM_MAX=4,NORM_1_AND_2=5)
!
!  Flags for VecSetValues() and MatSetValues()
!
      integer NOT_SET_VALUES,INSERT_VALUES, ADD_VALUES, MAX_VALUES

      parameter (NOT_SET_VALUES=0,INSERT_VALUES=1, ADD_VALUES=2)
      parameter (MAX_VALUES=3)
!
!  Types of vector scatters
!
      integer SCATTER_FORWARD,SCATTER_REVERSE,SCATTER_FORWARD_LOCAL
      integer SCATTER_REVERSE_LOCAL,SCATTER_LOCAL

      parameter (SCATTER_FORWARD=0,SCATTER_REVERSE=1)
      parameter (SCATTER_FORWARD_LOCAL=2,SCATTER_REVERSE_LOCAL=3)
      parameter (SCATTER_LOCAL=2)
!
!  PipelineDirection
!
      integer PIPELINE_DOWN,PIPELINE_UP

      parameter (PIPELINE_DOWN=0,PIPELINE_UP=1)
!
!  PipelineType
!
      integer PIPELINE_NONE,PIPELINE_SEQUENTIAL,PIPELINE_REDBLACK
      integer PIPELINE_MULTICOLOUR

      parameter (PIPELINE_NONE=1,PIPELINE_SEQUENTIAL=2)
      parameter (PIPELINE_REDBLACK=3,PIPELINE_MULTICOLOUR=4)
!
!  VecOption
!
      integer VEC_IGNORE_OFF_PROC_ENTRIES

      parameter (VEC_IGNORE_OFF_PROC_ENTRIES=0)
!
!  VecOperation
!
      integer VEC_VIEW,VECOP_LOADINTOVECTOR

      parameter (VEC_VIEW=32,VECOP_LOADINTOVECTOR=40)
!
!  End of Fortran include file for the Vec package in PETSc


