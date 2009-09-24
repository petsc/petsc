!
!  Include file for Fortran use of the TS (timestepping) package in PETSc
!
#if !defined (__PETSCTSDEF_H)
#define __PETSCTSDEF_H

#include "finclude/petscsnesdef.h"

#if !defined(PETSC_USE_FORTRAN_DATATYPES)
#define TS PetscFortranAddr
#endif
#define TSType character*(80)
#define TSSundialsType PetscEnum
#define TSProblemType PetscEnum 
#define TSSundialsGramSchmidtType PetscEnum
#define TSSundialsLmmType PetscEnum

#define TSEULER 'euler'
#define TSBEULER 'beuler'
#define TSPSEUDO 'pseudo'
#define TSCRANK_NICHOLSON 'crank-nicholson'
#define TSSUNDIALS 'sundials'
#define TSRUNGE_KUTTA 'runge-kutta'
#define TSPYTHON 'python'
#define TSTHETA 'theta'
#define TSGL 'gl'
#endif
