#include "zpetsc.h"
#include "petscvec.h"
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecscattercreatetoall_   VECSCATTERCREATETOALL
#define vecscattercreatetozero_  VECSCATTERCREATETOZERO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecscattercreatetoall_   vecscattercreatetoall
#define vecscattercreatetozero_  vecscattercreatetozero
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL   vecscattercreatetoall_(Vec *vin,VecScatter *ctx,Vec *vout, int *ierr ){
  CHKFORTRANNULLOBJECT(vout);
  *ierr = VecScatterCreateToAll(*vin,ctx,vout);
}
void PETSC_STDCALL   vecscattercreatetozero_(Vec *vin,VecScatter *ctx,Vec *vout, int *ierr ){
  CHKFORTRANNULLOBJECT(vout);
  *ierr = VecScatterCreateToZero(*vin,ctx,vout);
}

EXTERN_C_END
