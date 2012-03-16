#include <private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsclinesearchsettype_          PETSCLINESEARCHSETTYPE
#define petsclinesearchsetprecheck_      PETSCLINESEARCHSETPRECHECK
#define petsclinesearchgetprecheck_      PETSCLINESEARCHGETPRECHECK
#define petsclinesearchsetpostcheck_     PETSCLINESEARCHSETPOSTCHECK
#define petsclinesearchgetpostcheck_     PETSCLINESEARCHGETPOSTCHECK
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petsclinesearchsettype_          petsclinesearchsettype
#define petsclinesearchsetprecheck_      petsclinesearchsetprecheck
#define petsclinesearchgetprecheck_      petsclinesearchgetprecheck
#define petsclinesearchsetpostcheck_     petsclinesearchsetpostcheck
#define petsclinesearchgetpostcheck_     petsclinesearchgetpostcheck

#endif

/* fortranpointers go: shell, precheck, postcheck */

static PetscErrorCode ourpetsclinesearchprecheck(PetscLineSearch linesearch, Vec X, Vec Y, PetscBool * changed, void * ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(PetscLineSearch*,Vec*, Vec*, PetscBool*,void*,PetscErrorCode*))(((PetscObject)linesearch)->fortran_func_pointers[1]))(&linesearch,&X,&Y,changed,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourpetsclinesearchpostcheck(PetscLineSearch linesearch, Vec X, Vec Y, Vec W, PetscBool * changed_Y, PetscBool * changed_W, void * ctx)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(PetscLineSearch*,Vec*,Vec*,Vec*,PetscBool*,PetscBool*,void*,PetscErrorCode*))
   (((PetscObject)linesearch)->fortran_func_pointers[2]))(&linesearch,&X,&Y,&W,changed_Y,changed_W,ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN

void PETSC_STDCALL petsclinesearchsettype_(PetscLineSearch *linesearch,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PetscLineSearchSetType(*linesearch,t);
  FREECHAR(type,t);
}


void PETSC_STDCALL petsclinesearchsetprecheck_(PetscLineSearch *linesearch,
                                               void (PETSC_STDCALL *func)(PetscLineSearch*,Vec*,Vec*,PetscBool*,PetscErrorCode*),
                                               void *ctx,
                                               PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*linesearch,3);
  ((PetscObject)*linesearch)->fortran_func_pointers[1] = (PetscVoidFunction)func;
  *ierr = PetscLineSearchSetPreCheck(*linesearch,ourpetsclinesearchprecheck,ctx);
}

void PETSC_STDCALL petsclinesearchsetpostcheck_(PetscLineSearch *linesearch,
                                                void (PETSC_STDCALL *func)(PetscLineSearch*,Vec*,Vec*,Vec*,PetscBool*,PetscBool*,PetscErrorCode*,void*),
                                               void *ctx,
                                               PetscErrorCode *ierr)
{
  PetscObjectAllocateFortranPointers(*linesearch,3);
  ((PetscObject)*linesearch)->fortran_func_pointers[2] = (PetscVoidFunction)func;
  *ierr = PetscLineSearchSetPostCheck(*linesearch,ourpetsclinesearchpostcheck,ctx);
}


EXTERN_C_END
