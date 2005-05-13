
#include "zpetsc.h"
#include "petscksp.h"
#include "petscda.h"
#include "petscdmmg.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define dmmgcreate_              DMMGCREATE
#define dmmgdestroy_             DMMGDESTROY
#define dmmgsetup_               DMMGSETUP
#define dmmgsetdm_               DMMGSETDM
#define dmmgview_                DMMGVIEW
#define dmmgsolve_               DMMGSOLVE
#define dmmggetda_               DMMGGETDA
#define dmmgsetksp_              DMMGSETKSP
#define dmmggetx_                DMMGGETX
#define dmmggetj_                DMMGGETJ
#define dmmggetb_                DMMGGETB
#define dmmggetrhs_              DMMGGETRHS
#define dmmggetksp_              DMMGGETKSP
#define dmmggetlevels_           DMMGGETLEVELS
#define dmmgsetinitialguess_     DMMGSETINITIALGUESS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmmggetrhs_              dmmggetrhs
#define dmmggetx_                dmmggetx
#define dmmggetj_                dmmggetj
#define dmmggetb_                dmmggetb
#define dmmggetksp_              dmmggetksp
#define dmmggetda_               dmmggetda
#define dmmggetlevels_           dmmggetlevels
#define dmmgsetksp_              dmmgsetksp
#define dmmgdestroy_             dmmgdestroy
#define dmmgcreate_              dmmgcreate
#define dmmgsetup_               dmmgsetup
#define dmmgsetdm_               dmmgsetdm
#define dmmgview_                dmmgview
#define dmmgsolve_               dmmgsolve
#define dmmgsetinitialguess_     dmmgsetinitialguess
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *theirmat)(DMMG*,Mat*,PetscErrorCode*);
EXTERN_C_END

static PetscErrorCode ourrhs(DMMG dmmg,Vec vec)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMMG*,Vec*,PetscErrorCode*))(((PetscObject)dmmg->dm)->fortran_func_pointers[0]))(&dmmg,&vec,&ierr);
  return ierr;
}

static PetscErrorCode ourinitialguess(DMMG dmmg,Vec vec)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMMG*,Vec*,PetscErrorCode*))(((PetscObject)dmmg->dm)->fortran_func_pointers[1]))(&dmmg,&vec,&ierr);
  return ierr;
}

/*
   Since DMMGSetKSP() immediately calls the matrix functions for each level we do not need to store
  the mat() function inside the DMMG object
*/
static PetscErrorCode ourmat(DMMG dmmg,Mat mat)
{
  PetscErrorCode ierr = 0;
  (*theirmat)(&dmmg,&mat,&ierr);
  return ierr;
}

EXTERN_C_BEGIN

void PETSC_STDCALL dmmggetx_(DMMG **dmmg,Vec *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetx(*dmmg);
}

void PETSC_STDCALL dmmggetj_(DMMG **dmmg,Mat *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetJ(*dmmg);
}

void PETSC_STDCALL dmmggetb_(DMMG **dmmg,Mat *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetB(*dmmg);
}

void PETSC_STDCALL dmmggetrhs_(DMMG **dmmg,Vec *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetRHS(*dmmg);
}

void PETSC_STDCALL dmmggetksp_(DMMG **dmmg,KSP *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetKSP(*dmmg);
}

void PETSC_STDCALL dmmggetlevels_(DMMG **dmmg,PetscInt *x,PetscErrorCode *ierr)
{
  *ierr = 0;
  *x    = DMMGGetLevels(*dmmg);
}

/* ----------------------------------------------------------------------------------------------------------*/
void PETSC_STDCALL dmmgsetinitialguess_(DMMG **dmmg,void (PETSC_STDCALL *initialguess)(DMMG*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt i;

  *ierr = DMMGSetInitialGuess(*dmmg,ourinitialguess);
  /*
    Save the fortran initial guess function in the DM on each level; ourinitialguess() pulls it out when needed
  */
  for (i=0; i<(**dmmg)->nlevels; i++) {
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[1] = (FCNVOID)initialguess;
  }
}

void PETSC_STDCALL dmmgsetksp_(DMMG **dmmg,void (PETSC_STDCALL *rhs)(DMMG*,Vec*,PetscErrorCode*),void (PETSC_STDCALL *mat)(DMMG*,Mat*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt i;
  theirmat = mat;
  *ierr = DMMGSetKSP(*dmmg,ourrhs,ourmat);
  /*
    Save the fortran rhs function in the DM on each level; ourrhs() pulls it out when needed
  */
  for (i=0; i<(**dmmg)->nlevels; i++) {
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[0] = (FCNVOID)rhs;
  }
}

/* ----------------------------------------------------------------------------------------------------------*/

void PETSC_STDCALL dmmggetda_(DMMG *dmmg,DA *da,PetscErrorCode *ierr)
{
  *da   = (DA)(*dmmg)->dm;
  *ierr = 0;
}

void PETSC_STDCALL dmmgsetdm_(DMMG **dmmg,DM *dm,PetscErrorCode *ierr)
{
  PetscInt i;
  *ierr = DMMGSetDM(*dmmg,*dm);if (*ierr) return;
  /* loop over the levels added a place to hang the function pointers in the DM for each level*/
  for (i=0; i<(**dmmg)->nlevels; i++) {
    *ierr = PetscMalloc(4*sizeof(FCNVOID),&((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers);if (*ierr) return;
  }
}

void PETSC_STDCALL dmmgview_(DMMG **dmmg,PetscViewer *viewer,PetscErrorCode *ierr)
{
  *ierr = DMMGView(*dmmg,*viewer);
}

void PETSC_STDCALL dmmgsolve_(DMMG **dmmg,PetscErrorCode *ierr)
{
  *ierr = DMMGSolve(*dmmg);
}

void PETSC_STDCALL dmmgcreate_(MPI_Comm *comm,PetscInt *nlevels,void *user,DMMG **dmmg,PetscErrorCode *ierr)
{
  *ierr = DMMGCreate((MPI_Comm)PetscToPointerComm(*comm),*nlevels,user,dmmg);
}

void PETSC_STDCALL dmmgdestroy_(DMMG **dmmg,PetscErrorCode *ierr)
{
  *ierr = DMMGDestroy(*dmmg);
}

void PETSC_STDCALL dmmgsetup_(DMMG **dmmg,PetscErrorCode *ierr)
{
  *ierr = DMMGSetUp(*dmmg);
}

EXTERN_C_END


