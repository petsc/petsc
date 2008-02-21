#include "private/fortranimpl.h"
#include "petscts.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tsdefaultcomputejacobian_            TSDEFAULTCOMPUTEJACOBIAN
#define tsdefaultcomputejacobiancolor_       TSDEFAULTCOMPUTEJACOBIANCOLOR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tsdefaultcomputejacobian_            tsdefaultcomputejacobian
#define tsdefaultcomputejacobiancolor_       tsdefaultcomputejacobiancolor
#endif

EXTERN_C_BEGIN
/* function */
void tsdefaultcomputejacobian_(TS *ts,PetscReal *t,Vec *xx1,Mat *J,Mat *B,MatStructure *flag,void *ctx,PetscErrorCode *ierr)
{
  *ierr = TSDefaultComputeJacobian(*ts,*t,*xx1,J,B,flag,ctx);
}

/* function */
void tsdefaultcomputejacobiancolor_(TS *ts,PetscReal *t,Vec *xx1,Mat *J,Mat *B,MatStructure *flag,void *ctx,PetscErrorCode *ierr)
{
  *ierr = TSDefaultComputeJacobianColor(*ts,*t,*xx1,J,B,flag,*(MatFDColoring*)ctx);
}
EXTERN_C_END
