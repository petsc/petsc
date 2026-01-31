#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/ftnimpl.h"
#include <petscsys.h>
#include <petscoptions.h>
#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscobjectaddoptionshandler_ PETSCOBJECTADDOPTIONSHANDLER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscobjectaddoptionshandler_ petscobjectaddoptionshandler
#endif

static struct {
  PetscFortranCallbackId handler;
  PetscFortranCallbackId destroy;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  PetscFortranCallbackId handler_pgiptr;
  PetscFortranCallbackId destroy_pgiptr;
#endif
} _cb;

static PetscErrorCode ourhandler(PetscObject obj, PetscOptionItems items, PetscCtx ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)obj, PETSC_FORTRAN_CALLBACK_CLASS, _cb.handler_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(obj, _cb.handler, (PetscObject *, PetscOptionItems *, PetscCtx, PetscErrorCode *PETSC_F90_2PTR_PROTO_NOVAR), (&obj, &items, _ctx, &ierr PETSC_F90_2PTR_PARAM(ptr)));
}

static PetscErrorCode ourdestroy(PetscObject obj, PetscCtx ctx)
{
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  void *ptr;
  PetscCall(PetscObjectGetFortranCallback((PetscObject)obj, PETSC_FORTRAN_CALLBACK_CLASS, _cb.destroy_pgiptr, NULL, &ptr));
#endif
  PetscObjectUseFortranCallback(obj, _cb.destroy, (PetscObject *, PetscCtx, PetscErrorCode *PETSC_F90_2PTR_PROTO_NOVAR), (&obj, _ctx, &ierr PETSC_F90_2PTR_PARAM(ptr)));
}

PETSC_EXTERN void petscobjectaddoptionshandler_(PetscObject *obj, void (*handle)(PetscObject *, PetscOptionItems *, PetscCtx, PetscErrorCode), void (*destroy)(PetscObject *, PetscCtx, PetscErrorCode *), PetscCtx ctx, PetscErrorCode *ierr PETSC_F90_2PTR_PROTO(ptr1) PETSC_F90_2PTR_PROTO(ptr2))
{
  *ierr = PetscObjectSetFortranCallback((PetscObject)*obj, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.handler, (PetscFortranCallbackFn *)handle, ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*obj, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.handler_pgiptr, NULL, ptr1);
  if (*ierr) return;
#endif
  *ierr = PetscObjectSetFortranCallback((PetscObject)*obj, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.destroy, (PetscFortranCallbackFn *)destroy, ctx);
  if (*ierr) return;
#if defined(PETSC_HAVE_F90_2PTR_ARG)
  *ierr = PetscObjectSetFortranCallback((PetscObject)*obj, PETSC_FORTRAN_CALLBACK_CLASS, &_cb.destroy_pgiptr, NULL, ptr2);
  if (*ierr) return;
#endif
  *ierr = PetscObjectAddOptionsHandler(*obj, ourhandler, ourdestroy, NULL);
}
