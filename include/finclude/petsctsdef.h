!
!  Include file for Fortran use of the TS (timestepping) package in PETSc
!
#if !defined (__PETSCTSDEF_H)
#define __PETSCTSDEF_H

#include "finclude/petscsnesdef.h"

#if !defined(PETSC_USE_FORTRAN_TYPES)
#define TS PetscFortranAddr
#endif
#define TSType character*(80)
#define TSSundialsType PetscEnum
#define TSProblemType PetscEnum 
#define TSSundialsGramSchmidtType PetscEnum
#define TSSundialsLmmType PetscEnum

#define TS_EULER 'euler'
#define TS_BEULER 'beuler'
#define TS_PSEUDO 'pseudo'
#define TS_SUNDIALS 'sundials'
#define TS_CRANK_NICHOLSON 'crank-nicholson'
#define TS_RUNGE_KUTTA 'runge-kutta'
#define TS_PYTHON 'python'
#endif
