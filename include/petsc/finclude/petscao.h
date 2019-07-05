!
!
!  Include file for Fortran use of the AO (application ordering) package in PETSc
!
#if !defined (PETSCAODEF_H)
#define PETSCAODEF_H

#include "petsc/finclude/petscis.h"

#define AO PetscFortranAddr
#define AOType character*(80)

#define AOBASIC           'basic'
#define AOADVANCED        'advanced'
#define AOMAPPING         'mapping'
#define AOMEMORYSCALABLE  'memoryscalable'

#endif
