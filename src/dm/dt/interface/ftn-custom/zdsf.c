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

// We can't use PetscObjectUseFortranCallback() because this function returns void
static void ourriemannsolver(PetscInt dim,PetscInt Nf,const PetscReal x[],const PetscReal n[],const PetscScalar uL[],const PetscScalar uR[],PetscInt numConstants,const PetscScalar constants[],PetscScalar flux[],void *ctx)
{
  void (*func)(PetscInt *dim,PetscInt *Nf,const PetscReal x[],const PetscReal n[],const PetscScalar uL[],const PetscScalar uR[],const PetscInt *numConstants,const PetscScalar constants[],PetscScalar flux[],void *ctx);
  void *_ctx;
  PetscCallAbort(PETSC_COMM_SELF,PetscObjectGetFortranCallback((PetscObject)ctx,PETSC_FORTRAN_CALLBACK_CLASS,riemannsolver,(PetscVoidFunction*)&func,&_ctx));
  if (func) {
    (*func)(&dim,&Nf,x,n,uL,uR,&numConstants,constants,flux,_ctx);
  }
}

PETSC_EXTERN void petscdsviewfromoptions_(PetscDS *ao,PetscObject obj,char* type,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type,len,t);
  CHKFORTRANNULLOBJECT(obj);
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
  *ierr = PetscDSSetRiemannSolver(*prob,*f,ourriemannsolver);if (*ierr) return;
}
