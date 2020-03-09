#include <petsc/private/f90impl.h>
#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerbinarywriteint_ PETSCVIEWERBINARYWRITEINT
#define petscviewerbinarywritescalar_ PETSCVIEWERBINARYWRITESCALAR
#define petscviewerbinarywritereal_ PETSCVIEWERBINARYWRITEREAL
#define petscviewerbinaryreadint_ PETSCVIEWERBINARYREADINT
#define petscviewerbinaryreadscalar_ PETSCVIEWERBINARYREADSCALAR
#define petscviewerbinaryreadreal_ PETSCVIEWERBINARYREADREAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerbinarywriteint_ petscviewerbinarywriteint
#define petscviewerbinarywritescalar_ petscviewerbinarywritescalar
#define petscviewerbinarywritereal_ petscviewerbinarywritereal
#define petscviewerbinaryreadint_ petscviewerbinaryreadint
#define petscviewerbinaryreadscalar_ petscviewerbinaryreadscalar
#define petscviewerbinaryreadreal_ petscviewerbinaryreadreal
#endif

PETSC_EXTERN void petscviewerbinarywriteint_(PetscViewer *viewer,PetscInt *a,PetscInt *len,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinaryWrite(v,a,*len,PETSC_INT);
}

PETSC_EXTERN void petscviewerbinarywritescalar_(PetscViewer *viewer,PetscScalar *a,PetscInt *len,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinaryWrite(v,a,*len,PETSC_SCALAR);
}

PETSC_EXTERN void petscviewerbinarywritereal_(PetscViewer *viewer,PetscReal *a,PetscInt *len,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinaryWrite(v,a,*len,PETSC_REAL);
}

PETSC_EXTERN void petscviewerbinaryreadint_(PetscViewer *viewer,PetscInt *a,PetscInt *len,PetscInt *count,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscViewerBinaryRead(v,a,*len,count,PETSC_INT);
}

PETSC_EXTERN void petscviewerbinaryreadscalar_(PetscViewer *viewer,PetscScalar *a,PetscInt *len,PetscInt *count,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscViewerBinaryRead(v,a,*len,count,PETSC_SCALAR);
}

PETSC_EXTERN void petscviewerbinaryreadreal_(PetscViewer *viewer,PetscReal *a,PetscInt *len,PetscInt *count,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  CHKFORTRANNULLINTEGER(count);
  *ierr = PetscViewerBinaryRead(v,a,*len,count,PETSC_REAL);
}

