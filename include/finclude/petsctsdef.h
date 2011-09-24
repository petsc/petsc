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
#define TSConvergedReason PetscEnum
#define TSSundialsType PetscEnum
#define TSProblemType PetscEnum 
#define TSSundialsGramSchmidtType PetscEnum
#define TSSundialsLmmType PetscEnum

#define TSEULER           'euler'
#define TSBEULER          'beuler'
#define TSPSEUDO          'pseudo'
#define TSCN              'cn'
#define TSSUNDIALS        'sundials'
#define TSRK              'rk'
#define TSPYTHON          'python'
#define TSTHETA           'theta'
#define TSALPHA           'alpha'
#define TSGL              'gl'
#define TSSSP             'ssp'
#define TSARKIMEX         'arkimex'
#if defined(PETSC_HAVE_ROSW)
#define TSROSW            'rosw'
#endif

#define TSSSPType character*(80)
#define TSSSPRKS2  'rks2'
#define TSSSPRKS3  'rks3'
#define TSSSPRK104 'rk104'

#define TSGLAdaptType character*(80)
#define TSGLADAPT_NONE 'none'
#define TSGLADAPT_SIZE 'size'
#define TSGLADAPT_BOTH 'both'

#define TSARKIMEXType character*(80)
#define TSARKIMEX2D '2d'
#define TSARKIMEX2E '2e'
#define TSARKIMEX3  '3'
#define TSARKIMEX4  '4'
#define TSARKIMEX5  '5'

#if defined(PETSC_HAVE_ROSW)
#define TSROSW2M      '2m'
#define TSROSW2P      '2p'
#define TSROSWRA3PW   'ra3pw'
#define TSROSWRA34PW2 'ra34pw2'

#endif

#endif
