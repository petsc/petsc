/*$Id: zf90da.c,v 1.12 2000/09/22 18:53:58 balay Exp $*/

#include "petscksp.h"
#include "petscf90.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define kspgetresidualhistoryf90_     KSPGETRESIDUALHISTORYF90
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define kspgetresidualhistoryf90_     kspgetresidualhistoryf90
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL kspgetresidualhistoryf90_(KSP *ksp,F90Array1d *indices,int *n,int *ierr)
{
  PetscReal *hist;
  *ierr = KSPGetResidualHistory(*ksp,&hist,n); if (*ierr) return;
  *ierr = F90Array1dCreate(hist,PETSC_DOUBLE,1,*n,indices);
}
EXTERN_C_END

char *PetscErrorStrings[] = {
    "Out of memory",
    "No support for this operation for this object type",
    "Signal received",
    "Floating point exception",
    "Corrupted Petsc object",
    "Error in external library",
    "Petsc has generated inconsistent data",
    "Memory corruption",
    "Nonconforming object sizes",
    "Argument aliasing not permitted",
    "Invalid argument",
    "Null or corrupt argument",
    "Argument out of range",
    "Invalid pointer",
    "Arguments must have same type",
    "Object is in wrong state",
    "Arguments are incompatible",
    "Unable to open file",
    "Read from file failed",
    "Write to file failed",
    "Unexpected data in file",
    "Detected breakdown in Krylov method",
    "Detected zero pivot in LU factorization",
    "Detected zero pivot in Cholesky factorization"};





