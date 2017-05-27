#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspmonitorlgresidualnormcreate_        KSPMONITORLGRESIDUALNORMCREATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspmonitorlgresidualnormcreate_        kspmonitorlgresidualnormcreate
#endif

/*
   Possible bleeds memory but cannot be helped.
*/
PETSC_EXTERN void PETSC_STDCALL kspmonitorlgresidualnormcreate_(
                    MPI_Fint *comm,char* host PETSC_MIXED_LEN(len1),
                    char* label PETSC_MIXED_LEN(len2),int *x,int *y,int *m,int *n,PetscDrawLG *lgctx,
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *t1,*t2;

  FIXCHAR(host,len1,t1);
  FIXCHAR(label,len2,t2);
  *ierr = KSPMonitorLGResidualNormCreate(MPI_Comm_f2c(*comm),t1,t2,*x,*y,*m,*n,lgctx);
}

