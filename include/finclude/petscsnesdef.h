!
!  Include file for Fortran use of the SNES package in PETSc
!
#if !defined (__PETSCSNESDEF_H)
#define __PETSCSNESDEF_H

#include "finclude/petsckspdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define SNES PetscFortranAddr
#endif
#define SNESType character*(80)
#define SNESConvergedReason PetscEnum
#define SNESLineSearchType  PetscEnum
#define MatMFFD PetscFortranAddr
#define MatMFFDType PetscFortranAddr
!
!  SNESType
!
#define SNESLS          'ls'
#define SNESTR          'tr'
#define SNESPYTHON      'python'
#define SNESTEST        'test'
#define SNESNRICHARDSON 'nrichardson'
#define SNESKSPONLY     'ksponly'
#define SNESVIRS        'virs'
#define SNESVISS        'viss'
#define SNESNGMRES      'ngmres'
#define SNESQN          'qn'
#define SNESSHELL       'shell'
#define SNESNCG         'ncg'
#define SNESSORQN       'sorqn'
#define SNESFAS         'fas'

!
! MatSNESMF
! 
#define MATMFFD_DEFAULT 'ds'
#define MATMFFD_WP 'wp'

#endif
