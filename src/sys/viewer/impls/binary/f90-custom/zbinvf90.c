#include "../src/sys/f90-src/f90impl.h"
#include "private/fortranimpl.h"
#include "petscviewer.h"

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

EXTERN_C_BEGIN


void PETSC_STDCALL petscviewerbinarywriteint_(PetscViewer *viewer,PetscInt *a,PetscInt *len,PetscTruth *tmp,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinaryWrite(v,a,*len,PETSC_INT,*tmp);
}

void PETSC_STDCALL petscviewerbinarywritescalar_(PetscViewer *viewer,PetscScalar *a,PetscInt *len,PetscTruth *tmp,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinaryWrite(v,a,*len,PETSC_SCALAR,*tmp);
}

void PETSC_STDCALL petscviewerbinarywritereal_(PetscViewer *viewer,PetscReal *a,PetscInt *len,PetscTruth *tmp,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinaryWrite(v,a,*len,PETSC_REAL,*tmp);
}

void PETSC_STDCALL petscviewerbinaryreadint_(PetscViewer *viewer,PetscInt *a,PetscInt *len,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinaryRead(v,a,*len,PETSC_INT);
}

void PETSC_STDCALL petscviewerbinaryreadscalar_(PetscViewer *viewer,PetscScalar *a,PetscInt *len,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinaryRead(v,a,*len,PETSC_SCALAR);
}

void PETSC_STDCALL petscviewerbinaryreadreal_(PetscViewer *viewer,PetscReal *a,PetscInt *len,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerBinaryRead(v,a,*len,PETSC_REAL);
}

EXTERN_C_END
