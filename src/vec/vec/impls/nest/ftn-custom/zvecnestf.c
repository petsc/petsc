#include <petsc-private/fortranimpl.h>
#include <petscvec.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecnestgetsubvecs_            VECNESTGETSUBVECS
#define vecnestsetsubvecs_            VECNESTSETSUBVECS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecnestgetsubvecs_            vecnestgetsubvecs
#define vecnestsetsubvecs_            vecnestsetsubvecs
#endif

PETSC_EXTERN_C void PETSC_STDCALL vecnestgetsubvecs_(Vec *X,PetscInt *N,Vec *sx,PetscErrorCode *ierr)
{
  Vec *tsx;
  PetscInt  i,n;
  CHKFORTRANNULLINTEGER(N);
  *ierr = VecNestGetSubVecs(*X,&n,&tsx); if (*ierr) return;
  if (N) *N = n;
  CHKFORTRANNULLOBJECT(sx);
  if (sx) {
    for (i=0; i<n; i++) {
      sx[i] = tsx[i];
    }
  }
}

PETSC_EXTERN_C void PETSC_STDCALL vecnestsetsubvecs_(Vec *X,PetscInt *N,PetscInt *idxm,Vec *sx,PetscErrorCode *ierr)
{
  *ierr = VecNestSetSubVecs(*X,*N,idxm,sx);
}
