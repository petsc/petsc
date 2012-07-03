#include <petsc-private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define sneslinesearchsettype_          SNESLINESEARCHSETTYPE
#define sneslinesearchsetprecheck_      SNESLINESEARCHSETPRECHECK
#define sneslinesearchgetprecheck_      SNESLINESEARCHGETPRECHECK
#define sneslinesearchsetpostcheck_     SNESLINESEARCHSETPOSTCHECK
#define sneslinesearchgetpostcheck_     SNESLINESEARCHGETPOSTCHECK
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sneslinesearchsettype_          sneslinesearchsettype
#define sneslinesearchsetprecheck_      sneslinesearchsetprecheck
#define sneslinesearchgetprecheck_      sneslinesearchgetprecheck
#define sneslinesearchsetpostcheck_     sneslinesearchsetpostcheck
#define sneslinesearchgetpostcheck_     sneslinesearchgetpostcheck

#endif

/* fortranpointers go: shell, precheck, postcheck */

static PetscErrorCode oursneslinesearchprecheck(SNESLineSearch linesearch, Vec X, Vec Y, PetscBool * changed, void * ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(SNESLineSearch*,Vec*, Vec*, PetscBool*,void*,PetscErrorCode*))(((PetscObject)linesearch)->fortran_func_pointers[1]))(&linesearch,&X,&Y,changed,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode oursneslinesearchpostcheck(SNESLineSearch linesearch, Vec X, Vec Y, Vec W, PetscBool * changed_Y, PetscBool * changed_W, void * ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(SNESLineSearch*,Vec*,Vec*,Vec*,PetscBool*,PetscBool*,void*,PetscErrorCode*))
   (((PetscObject)linesearch)->fortran_func_pointers[2]))(&linesearch,&X,&Y,&W,changed_Y,changed_W,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL sneslinesearchsettype_(SNESLineSearch *linesearch,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = SNESLineSearchSetType(*linesearch,t);
  FREECHAR(type,t);
}


void PETSC_STDCALL sneslinesearchsetprecheck_(SNESLineSearch *linesearch,
                                              void (PETSC_STDCALL *func)(SNESLineSearch*,Vec*,Vec*,PetscBool*,PetscErrorCode*),
                                              void *ctx,
                                              PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*linesearch,3);
  ((PetscObject)*linesearch)->fortran_func_pointers[1] = (PetscVoidFunction)func;
  *ierr = SNESLineSearchSetPreCheck(*linesearch,oursneslinesearchprecheck,ctx);
}

void PETSC_STDCALL sneslinesearchsetpostcheck_(SNESLineSearch *linesearch,
                                                void (PETSC_STDCALL *func)(SNESLineSearch*,Vec*,Vec*,Vec*,PetscBool*,PetscBool*,PetscErrorCode*,void*),
                                               void *ctx,
                                               PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*linesearch,3);
  ((PetscObject)*linesearch)->fortran_func_pointers[2] = (PetscVoidFunction)func;
  *ierr = SNESLineSearchSetPostCheck(*linesearch,oursneslinesearchpostcheck,ctx);
}


EXTERN_C_END
