#include <petsc/private/fortranimpl.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define kspmonitorlgcreate_        KSPMONITORLGCREATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspmonitorlgcreate_        kspmonitorlgcreate
#endif

/*
   Possible bleeds memory but cannot be helped.
*/
PETSC_EXTERN void kspmonitorlgcreate_(
                    MPI_Fint *comm,char* host,
                    char* label,char* metric,int l,const char **names,int *x,int *y,int *m,int *n,PetscDrawLG *lgctx,
                    PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len1,PETSC_FORTRAN_CHARLEN_T len2,PETSC_FORTRAN_CHARLEN_T len3)
{
  char *t1,*t2,*t3;

  FIXCHAR(host,len1,t1);
  FIXCHAR(label,len2,t2);
  FIXCHAR(metric,len3,t3);
  *ierr = KSPMonitorLGCreate(MPI_Comm_f2c(*comm),t1,t2,t3,l,names,*x,*y,*m,*n,lgctx);
}
