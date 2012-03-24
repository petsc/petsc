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
#define SNESMSType character*(80)
#define SNESConvergedReason PetscEnum
#define SNESLineSearchType  PetscEnum
#define MatMFFD PetscFortranAddr
#define MatMFFDType PetscFortranAddr
#define SNESLineSearch PetscFortranAddr
#define SNESLineSearchOrder PetscEnum
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
#define SNESMS          'ms'

!
! SNESLineSearchType
!

#define SNES_LINESEARCH_BASIC 'basic'
#define SNES_LINESEARCH_BT    'bt'
#define SNES_LINESEARCH_L2    'l2'
#define SNES_LINESEARCH_CP    'cp'
#define SNES_LINESEARCH_SHELL 'shell'

!
!  SNESMSType
!
#define SNESMSEULER     'euler'
#define SNESMSM62       'm62'
#define SNESMSJAMESON83 'jameson83'
#define SNESMSVLTP21    'vltp21'
#define SNESMSVLTP31    'vltp31'
#define SNESMSVLTP41    'vltp41'
#define SNESMSVLTP51    'vltp51'
#define SNESMSVLTP61    'vltp61'

!
! MatSNESMF
! 
#define MATMFFD_DEFAULT 'ds'
#define MATMFFD_WP 'wp'

#endif
