#include "zpetsc.h"
#include "petscpc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define pcshellsetapply_           PCSHELLSETAPPLY
#define pcshellsetapplyrichardson_ PCSHELLSETAPPLYRICHARDSON
#define pcshellsetapplytranspose_  PCSHELLSETAPPLYTRANSPOSE
#define pcshellsetsetup_           PCSHELLSETSETUP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcshellsetapply_           pcshellsetapply
#define pcshellsetapplyrichardson_ pcshellsetapplyrichardson
#define pcshellsetapplytranspose_  pcshellsetapplytranspose
#define pcshellsetsetup_           pcshellsetsetup
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f1)(void*,Vec*,Vec*,PetscErrorCode*);
static void (PETSC_STDCALL *f2)(void*,Vec*,Vec*,Vec*,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscErrorCode*);
static void (PETSC_STDCALL *f3)(void*,Vec*,Vec*,PetscErrorCode*);
static void (PETSC_STDCALL *f9)(void*,PetscErrorCode*);
EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourshellapply(void *ctx,Vec x,Vec y)
{
  PetscErrorCode ierr = 0;
  (*f1)(ctx,&x,&y,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourapplyrichardson(void *ctx,Vec x,Vec y,Vec w,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt m)
{
  PetscErrorCode ierr = 0;

  (*f2)(ctx,&x,&y,&w,&rtol,&abstol,&dtol,&m,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourshellapplytranspose(void *ctx,Vec x,Vec y)
{
  PetscErrorCode ierr = 0;
  (*f3)(ctx,&x,&y,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourshellsetup(void *ctx)
{
  PetscErrorCode ierr = 0;

  (*f9)(ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL pcshellsetapply_(PC *pc,void (PETSC_STDCALL *apply)(void*,Vec *,Vec *,PetscErrorCode*),
                                    PetscErrorCode *ierr)
{
  f1 = apply;
  *ierr = PCShellSetApply(*pc,ourshellapply);
}

void PETSC_STDCALL pcshellsetapplyrichardson_(PC *pc,
         void (PETSC_STDCALL *apply)(void*,Vec *,Vec *,Vec *,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscErrorCode*),
         PetscErrorCode *ierr)
{
  f2 = apply;
  *ierr = PCShellSetApplyRichardson(*pc,ourapplyrichardson);
}

void PETSC_STDCALL pcshellsetapplytranspose_(PC *pc,void (PETSC_STDCALL *applytranspose)(void*,Vec *,Vec *,PetscErrorCode*),
                                             PetscErrorCode *ierr)
{
  f3 = applytranspose;
  *ierr = PCShellSetApplyTranspose(*pc,ourshellapplytranspose);
}

void PETSC_STDCALL pcshellsetsetup_(PC *pc,void (PETSC_STDCALL *setup)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  f9 = setup;
  *ierr = PCShellSetSetUp(*pc,ourshellsetup);
}

/* -----------------------------------------------------------------*/

EXTERN_C_END
