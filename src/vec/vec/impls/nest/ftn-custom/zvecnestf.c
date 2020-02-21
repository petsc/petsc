#include <petsc/private/fortranimpl.h>
#include <petscvec.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define vecnestgetsubvecs_            VECNESTGETSUBVECS
#define vecnestsetsubvecs_            VECNESTSETSUBVECS
#define veccreatenest_                VECCREATENEST
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecnestgetsubvecs_            vecnestgetsubvecs
#define vecnestsetsubvecs_            vecnestsetsubvecs
#define veccreatenest_                veccreatenest
#endif

PETSC_EXTERN void vecnestgetsubvecs_(Vec *X,PetscInt *N,Vec *sx,PetscErrorCode *ierr)
{
  Vec      *tsx;
  PetscInt i,n;
  CHKFORTRANNULLINTEGER(N);
  *ierr = VecNestGetSubVecs(*X,&n,&tsx); if (*ierr) return;
  if (N) *N = n;
  CHKFORTRANNULLOBJECT(sx);
  if (sx) {
    for (i=0; i<n; i++) sx[i] = tsx[i];
  }
}

PETSC_EXTERN void vecnestsetsubvecs_(Vec *X,PetscInt *N,PetscInt *idxm,Vec *sx,PetscErrorCode *ierr)
{
  *ierr = VecNestSetSubVecs(*X,*N,idxm,sx);
}

PETSC_EXTERN void veccreatenest_(MPI_Fint *comm,PetscInt *nb,IS is[],Vec x[],Vec *Y,int *ierr)
{
  CHKFORTRANNULLOBJECT(is);
  CHKFORTRANNULLOBJECT(x);
  *ierr = VecCreateNest(MPI_Comm_f2c(*comm),*nb,is,x,Y);
}
