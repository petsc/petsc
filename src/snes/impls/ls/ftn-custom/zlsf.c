#include "zpetsc.h"
#include "petscsnes.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define sneslinesearchsetpostcheck_      SNESLINESEARCHSETPOSTCHECK
#define sneslinesearchset_               SNESLINESEARCHSET
#define sneslinesearchcubic_             SNESLINESEARCHCUBIC
#define sneslinesearchquadratic_         SNESLINESEARCHQUADRATIC
#define sneslinesearchno_                SNESLINESEARCHNO
#define sneslinesearchnonorms_           SNESLINESEARCHNONORMS
#define sneslinesearchsetprecheck_       SNESLINESEARCHSETPRECHECK
#define snessetupdate_                   SNESSETUPDATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sneslinesearchsetpostcheck_      sneslinesearchsetpostcheck
#define sneslinesearchset_               sneslinesearchset
#define sneslinesearchcubic_             sneslinesearchcubic     
#define sneslinesearchquadratic_         sneslinesearchquadratic    
#define sneslinesearchno_                sneslinesearchno    
#define sneslinesearchnonorms_           sneslinesearchnonorms    
#define sneslinesearchsetprecheck_       sneslinesearchsetprecheck
#define snessetupdate_                   snessetupdate
#endif
EXTERN_C_BEGIN
static void (PETSC_STDCALL *f74)(SNES*,Vec*,Vec*,Vec*,void*,PetscTruth*,PetscTruth*,PetscErrorCode*);
static void (PETSC_STDCALL *f73)(SNES*,void *,Vec*,Vec*,Vec*,Vec*,Vec*,PetscReal*,PetscReal*,PetscReal*,PetscTruth*,PetscErrorCode*);
static void (PETSC_STDCALL *f75)(SNES*,Vec*,Vec*,void*,PetscTruth*,PetscErrorCode*);
static void (PETSC_STDCALL *f76)(SNES*,PetscInt*,PetscErrorCode*);
EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
PetscErrorCode OurSNESLineSearch(SNES snes,void *ctx,Vec x,Vec f,Vec g,Vec y,Vec w,PetscReal fnorm,PetscReal*ynorm,PetscReal*gnorm,PetscTruth *flag)
{
  PetscErrorCode ierr = 0;
  (*f73)(&snes,(void*)&ctx,&x,&f,&g,&y,&w,&fnorm,ynorm,gnorm,flag,&ierr);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode OurSNESLineSearchPostCheck(SNES snes,Vec x,Vec y,Vec z,void *checkCtx,PetscTruth *flag1,PetscTruth *flag2)
{
  PetscErrorCode ierr = 0;
  (*f74)(&snes,&x,&y,&z,(void*)&checkCtx,flag1,flag2,&ierr);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode OurSNESLineSearchPreCheck(SNES snes,Vec x,Vec y,void *checkCtx,PetscTruth *flag1)
{
  PetscErrorCode ierr = 0;
  (*f75)(&snes,&x,&y,(void*)&checkCtx,flag1,&ierr);CHKERRQ(ierr);
  return 0;
}
PetscErrorCode OurSNESSetUpdate(SNES snes,PetscInt b)
{
  PetscErrorCode ierr = 0;
  (*f76)(&snes,&b,&ierr);CHKERRQ(ierr);
  return 0;
}

EXTERN_C_BEGIN
void PETSC_STDCALL sneslinesearchsetpostcheck_(SNES *snes,void (PETSC_STDCALL *f)(SNES*,Vec*,Vec *,Vec *,void *,PetscTruth*,PetscTruth*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  f74 = f;
  *ierr = SNESLineSearchSetPostCheck(*snes,OurSNESLineSearchPostCheck,ctx);
}  

void PETSC_STDCALL sneslinesearchsetprecheck_(SNES *snes,void (PETSC_STDCALL *f)(SNES*,Vec*,Vec *,void *,PetscTruth*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  f75 = f;
  *ierr = SNESLineSearchSetPreCheck(*snes,OurSNESLineSearchPreCheck,ctx);
}  

void PETSC_STDCALL snessetupdate_(SNES *snes,void (PETSC_STDCALL *f)(SNES*,PetscInt*,PetscErrorCode*),PetscErrorCode *ierr)
{
  f76 = f;
  *ierr = SNESSetUpdate(*snes,OurSNESSetUpdate);
}  
/* -----------------------------------------------------------------------------------------------------*/
void sneslinesearchcubic_(SNES *snes,void *lsctx,Vec *x,Vec *f,Vec *g,Vec *y,Vec *w,PetscReal*fnorm,
                                        PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag,PetscErrorCode *ierr)
{
  *ierr = SNESLineSearchCubic(*snes,lsctx,*x,*f,*g,*y,*w,*fnorm,ynorm,gnorm,flag);
}
void sneslinesearchquadratic_(SNES *snes,void *lsctx,Vec *x,Vec *f,Vec *g,Vec *y,Vec *w,PetscReal*fnorm,
                                        PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag,PetscErrorCode *ierr)
{
  *ierr = SNESLineSearchQuadratic(*snes,lsctx,*x,*f,*g,*y,*w,*fnorm,ynorm,gnorm,flag);
}
void sneslinesearchno_(SNES *snes,void *lsctx,Vec *x,Vec *f,Vec *g,Vec *y,Vec *w,PetscReal*fnorm,
                                        PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag,PetscErrorCode *ierr)
{
  *ierr = SNESLineSearchNo(*snes,lsctx,*x,*f,*g,*y,*w,*fnorm,ynorm,gnorm,flag);
}
void sneslinesearchnonorms_(SNES *snes,void *lsctx,Vec *x,Vec *f,Vec *g,Vec *y,Vec *w,PetscReal*fnorm,
                                        PetscReal *ynorm,PetscReal *gnorm,PetscTruth *flag,PetscErrorCode *ierr)
{
  *ierr = SNESLineSearchNoNorms(*snes,lsctx,*x,*f,*g,*y,*w,*fnorm,ynorm,gnorm,flag);
}

void PETSC_STDCALL sneslinesearchset_(SNES *snes,void (PETSC_STDCALL *f)(SNES*,void *,Vec*,Vec*,Vec*,Vec*,Vec*,PetscReal*,PetscReal*,PetscReal*,PetscTruth*,PetscErrorCode*),void *ctx,PetscErrorCode *ierr)
{
  if ((FCNVOID)f == (FCNVOID)sneslinesearchcubic_) {
    *ierr = SNESLineSearchSet(*snes,SNESLineSearchCubic,ctx);
  } else if ((FCNVOID)f == (FCNVOID)sneslinesearchquadratic_) {
    *ierr = SNESLineSearchSet(*snes,SNESLineSearchQuadratic,ctx);
  } else if ((FCNVOID)f == (FCNVOID)sneslinesearchno_) {
    *ierr = SNESLineSearchSet(*snes,SNESLineSearchNo,ctx);
  } else if ((FCNVOID)f == (FCNVOID)sneslinesearchnonorms_) {
    *ierr = SNESLineSearchSet(*snes,SNESLineSearchNoNorms,ctx);
  } else {
    f73 = f;
    *ierr = SNESLineSearchSet(*snes,OurSNESLineSearch,ctx);
  }
}

/* -----------------------------------------------------------------------------------------------------*/
EXTERN_C_END
