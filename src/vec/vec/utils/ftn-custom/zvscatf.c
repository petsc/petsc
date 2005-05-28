#include "zpetsc.h"
#include "petscvec.h"
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecscattercreate_         VECSCATTERCREATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecscattercreate_         vecscattercreate
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL vecscattercreate_(Vec *xin,IS *ix,Vec *yin,IS *iy,VecScatter *newctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(ix);
  CHKFORTRANNULLOBJECTDEREFERENCE(iy);
  *ierr = VecScatterCreate(*xin,*ix,*yin,*iy,newctx);
}

EXTERN_C_END
