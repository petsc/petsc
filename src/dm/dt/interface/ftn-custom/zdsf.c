#include <petsc/private/fortranimpl.h>
#include <petscds.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdsviewfromoptions_   PETSCDSVIEWFROMOPTIONS
#define petscdsview_  PETSCDSVIEW
#define petscdssetcontext_  PETSCDSSETCONTEXT
#define petscdssetriemannsolver_ PETSCDSSETRIEMANNSOLVER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdsviewfromoptions_   petscdsviewfromoptions
#define petscdsview_  petscdsview
#define petscdssetcontext_ petscdssetcontext
#define petscdssetriemannsolver_ petscdssetriemannsolver
#endif

static PetscFortranCallbackId riemannsolver;

static PetscErrorCode ourriemannsolver(PetscInt dim,PetscInt Nf,PetscReal x[],PetscReal n[],PetscScalar uL[],PetscScalar uR[],PetscInt numConstants,PetscScalar constants[],PetscScalar flux[],void *ctx)
{
  PetscObjectUseFortranCallback((PetscDS)ctx,riemannsolver,(PetscInt*,PetscInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*),(&dim,&Nf,x,n,uL,uR,&numConstants,constants,flux,_ctx,&ierr));
}

PETSC_EXTERN void petscdsviewfromoptions_(PetscDS *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscDSViewFromOptions(*ao,obj,t);if (*ierr) return;
  FREECHAR(type,t);
}

PETSC_EXTERN void petscdsview_(PetscDS *prob,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscDSView(*prob,v);if (*ierr) return;
}

PETSC_EXTERN void petscdssetcontext_(PetscDS *prob,PetscInt *f,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscDSSetContext(*prob,*f,*prob);if (*ierr) return;
}

PETSC_EXTERN void petscdssetriemannsolver_(PetscDS *prob,PetscInt *f,void (*rs)(PetscInt*,PetscInt*,PetscReal*,PetscReal*,PetscScalar*,PetscScalar*,PetscInt*,PetscScalar*,PetscScalar*,void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*prob,PETSC_FORTRAN_CALLBACK_CLASS,&riemannsolver,(PetscVoidFunction)rs,NULL);if (*ierr) return;
  *ierr = PetscDSSetRiemannSolver(*prob,*f,(void*)ourriemannsolver);if (*ierr) return;
}