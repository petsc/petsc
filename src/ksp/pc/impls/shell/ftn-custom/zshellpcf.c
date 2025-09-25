#include <petsc/private/ftnimpl.h>
#include <petscpc.h>
#include <petscksp.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define pcshellsetapply_               PCSHELLSETAPPLY
  #define pcshellsetapplysymmetricleft_  PCSHELLSETAPPLYSYMMETRICLEFT
  #define pcshellsetapplysymmetricright_ PCSHELLSETAPPLYSYMMETRICRIGHT
  #define pcshellsetapplyba_             PCSHELLSETAPPLYBA
  #define pcshellsetapplyrichardson_     PCSHELLSETAPPLYRICHARDSON
  #define pcshellsetapplytranspose_      PCSHELLSETAPPLYTRANSPOSE
  #define pcshellsetsetup_               PCSHELLSETSETUP
  #define pcshellsetdestroy_             PCSHELLSETDESTROY
  #define pcshellsetpresolve_            PCSHELLSETPRESOLVE
  #define pcshellsetpostsolve_           PCSHELLSETPOSTSOLVE
  #define pcshellsetview_                PCSHELLSETVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define pcshellsetapply_               pcshellsetapply
  #define pcshellsetapplysymmetricleft_  pcshellsetapplysymmetricleft
  #define pcshellsetapplysymmetricright_ pcshellsetapplysymmetricright
  #define pcshellsetapplyba_             pcshellsetapplyba
  #define pcshellsetapplyrichardson_     pcshellsetapplyrichardson
  #define pcshellsetapplytranspose_      pcshellsetapplytranspose
  #define pcshellsetsetup_               pcshellsetsetup
  #define pcshellsetdestroy_             pcshellsetdestroy
  #define pcshellsetpresolve_            pcshellsetpresolve
  #define pcshellsetpostsolve_           pcshellsetpostsolve
  #define pcshellsetview_                pcshellsetview
#endif

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourshellapply(PC pc, Vec x, Vec y)
{
  PetscCallFortranVoidFunction((*(void (*)(PC *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[0]))(&pc, &x, &y, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshellapplysymmetricleft(PC pc, Vec x, Vec y)
{
  PetscCallFortranVoidFunction((*(void (*)(PC *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[9]))(&pc, &x, &y, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshellapplysymmetricright(PC pc, Vec x, Vec y)
{
  PetscCallFortranVoidFunction((*(void (*)(PC *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[10]))(&pc, &x, &y, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshellapplyctx(PC pc, Vec x, Vec y)
{
  void *ctx;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCallFortranVoidFunction((*(void (*)(PC *, void *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[0]))(&pc, ctx, &x, &y, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshellapplyba(PC pc, PCSide side, Vec x, Vec y, Vec work)
{
  PetscCallFortranVoidFunction((*(void (*)(PC *, PCSide *, Vec *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[1]))(&pc, &side, &x, &y, &work, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourapplyrichardson(PC pc, Vec x, Vec y, Vec w, PetscReal rtol, PetscReal abstol, PetscReal dtol, PetscInt m, PetscBool guesszero, PetscInt *outits, PCRichardsonConvergedReason *reason)
{
  PetscCallFortranVoidFunction((*(void (*)(PC *, Vec *, Vec *, Vec *, PetscReal *, PetscReal *, PetscReal *, PetscInt *, PetscBool *, PetscInt *, PCRichardsonConvergedReason *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[2]))(&pc, &x, &y, &w, &rtol, &abstol, &dtol, &m, &guesszero, outits, reason, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshellapplytranspose(PC pc, Vec x, Vec y)
{
  PetscCallFortranVoidFunction((*(void (*)(void *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[3]))(&pc, &x, &y, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshellsetup(PC pc)
{
  PetscCallFortranVoidFunction((*(void (*)(PC *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[4]))(&pc, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshellsetupctx(PC pc)
{
  void *ctx;
  PetscCall(PCShellGetContext(pc, &ctx));
  PetscCallFortranVoidFunction((*(void (*)(PC *, void *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[4]))(&pc, ctx, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshelldestroy(PC pc)
{
  PetscCallFortranVoidFunction((*(void (*)(void *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[5]))(&pc, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshellpresolve(PC pc, KSP ksp, Vec x, Vec y)
{
  PetscCallFortranVoidFunction((*(void (*)(PC *, KSP *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[6]))(&pc, &ksp, &x, &y, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshellpostsolve(PC pc, KSP ksp, Vec x, Vec y)
{
  PetscCallFortranVoidFunction((*(void (*)(PC *, KSP *, Vec *, Vec *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[7]))(&pc, &ksp, &x, &y, &ierr));
  return PETSC_SUCCESS;
}

static PetscErrorCode ourshellview(PC pc, PetscViewer view)
{
  PetscCallFortranVoidFunction((*(void (*)(PC *, PetscViewer *, PetscErrorCode *))(((PetscObject)pc)->fortran_func_pointers[8]))(&pc, &view, &ierr));
  return PETSC_SUCCESS;
}

PETSC_EXTERN void pcshellsetapply_(PC *pc, void (*apply)(void *, Vec *, Vec *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[0] = (PetscFortranCallbackFn *)apply;

  *ierr = PCShellSetApply(*pc, ourshellapply);
}

PETSC_EXTERN void pcshellsetapplysymmetricleft_(PC *pc, void (*apply)(void *, Vec *, Vec *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[9] = (PetscFortranCallbackFn *)apply;

  *ierr = PCShellSetApplySymmetricLeft(*pc, ourshellapplysymmetricleft);
}

PETSC_EXTERN void pcshellsetapplysymmetricright_(PC *pc, void (*apply)(void *, Vec *, Vec *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[10] = (PetscFortranCallbackFn *)apply;

  *ierr = PCShellSetApplySymmetricRight(*pc, ourshellapplysymmetricright);
}

PETSC_EXTERN void pcshellsetapplyctx_(PC *pc, void (*apply)(void *, void *, Vec *, Vec *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[0] = (PetscFortranCallbackFn *)apply;

  *ierr = PCShellSetApply(*pc, ourshellapplyctx);
}

PETSC_EXTERN void pcshellsetapplyba_(PC *pc, void (*apply)(void *, PCSide *, Vec *, Vec *, Vec *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[1] = (PetscFortranCallbackFn *)apply;

  *ierr = PCShellSetApplyBA(*pc, ourshellapplyba);
}

PETSC_EXTERN void pcshellsetapplyrichardson_(PC *pc, void (*apply)(void *, Vec *, Vec *, Vec *, PetscReal *, PetscReal *, PetscReal *, PetscInt *, PetscBool *, PetscInt *, PCRichardsonConvergedReason *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[2] = (PetscFortranCallbackFn *)apply;
  *ierr                                        = PCShellSetApplyRichardson(*pc, ourapplyrichardson);
}

PETSC_EXTERN void pcshellsetapplytranspose_(PC *pc, void (*applytranspose)(void *, Vec *, Vec *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[3] = (PetscFortranCallbackFn *)applytranspose;

  *ierr = PCShellSetApplyTranspose(*pc, ourshellapplytranspose);
}

PETSC_EXTERN void pcshellsetsetupctx_(PC *pc, void (*setup)(void *, void *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[4] = (PetscFortranCallbackFn *)setup;

  *ierr = PCShellSetSetUp(*pc, ourshellsetupctx);
}

PETSC_EXTERN void pcshellsetsetup_(PC *pc, void (*setup)(void *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[4] = (PetscFortranCallbackFn *)setup;

  *ierr = PCShellSetSetUp(*pc, ourshellsetup);
}

PETSC_EXTERN void pcshellsetdestroy_(PC *pc, void (*setup)(void *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[5] = (PetscFortranCallbackFn *)setup;

  *ierr = PCShellSetDestroy(*pc, ourshelldestroy);
}

PETSC_EXTERN void pcshellsetpresolve_(PC *pc, void (*presolve)(void *, void *, Vec *, Vec *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[6] = (PetscFortranCallbackFn *)presolve;

  *ierr = PCShellSetPreSolve(*pc, ourshellpresolve);
}

PETSC_EXTERN void pcshellsetpostsolve_(PC *pc, void (*postsolve)(void *, void *, Vec *, Vec *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[7] = (PetscFortranCallbackFn *)postsolve;

  *ierr = PCShellSetPostSolve(*pc, ourshellpostsolve);
}

PETSC_EXTERN void pcshellsetview_(PC *pc, void (*view)(void *, PetscViewer *, PetscErrorCode *), PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*pc, 11);
  ((PetscObject)*pc)->fortran_func_pointers[8] = (PetscFortranCallbackFn *)view;

  *ierr = PCShellSetView(*pc, ourshellview);
}
