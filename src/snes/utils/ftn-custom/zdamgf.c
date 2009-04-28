#include "private/fortranimpl.h"
#include "petscdmmg.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmmgarraygetdmmg_        DMMGARRAYGETDMMG
#define dmmgsetksp_              DMMGSETKSP
#define dmmgsetinitialguess_     DMMGSETINITIALGUESS
#define dmmgsetdm_               DMMGSETDM
#define dmmgview_                DMMGVIEW
#define dmmgsolve_               DMMGSOLVE
#define dmmgcreate_              DMMGCREATE
#define dmmgdestroy_             DMMGDESTROY
#define dmmgsetup_               DMMGSETUP
#define dmmgsetuser_             DMMGSETUSER
#define dmmggetuser_             DMMGGETUSER
#define dmmggetdm_               DMMGGETDM
#define dmmgsetnullspace_        DMMGSETNULLSPACE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmmgarraygetdmmg_        dmmgarraygetdmmg_
#define dmmgsetksp_              dmmgsetksp
#define dmmgsetinitialguess_     dmmgsetinitialguess
#define dmmgsetdm_               dmmgsetdm
#define dmmgview_                dmmgview
#define dmmgsolve_               dmmgsolve
#define dmmgcreate_              dmmgcreate
#define dmmgdestroy_             dmmgdestroy
#define dmmgsetup_               dmmgsetup
#define dmmgsetuser_             dmmgsetuser
#define dmmggetuser_             dmmggetuser
#define dmmggetdm_               dmmggetdm
#define dmmgsetnullspace_        dmmgsetnullspace
#endif

static PetscErrorCode ournullspace(DMMG dmmg,Vec vec[])
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMMG*,Vec*,PetscErrorCode*))(((PetscObject)dmmg->dm)->fortran_func_pointers[3]))(&dmmg,vec,&ierr);
  return ierr;
}

EXTERN_C_BEGIN

void PETSC_STDCALL dmmgsetnullspace_(DMMG **dmmg,PetscTruth *has_cnst,PetscInt *n,void (PETSC_STDCALL *nullspace)(DMMG*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt i;

  CHKFORTRANNULLFUNCTION(nullspace);
  if (!nullspace) {
    *ierr = DMMGSetNullSpace(*dmmg,*has_cnst,*n,PETSC_NULL);
  } else {
    *ierr = DMMGSetNullSpace(*dmmg,*has_cnst,*n,ournullspace);if (*ierr) return;
    /*
    Save the fortran nullspace function in the DM on each level; ournullspace() pulls it out when needed
    */
    for (i=0; i<(**dmmg)->nlevels; i++) {
      ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[3] = (PetscVoidFunction)nullspace;
    }
  }
}

EXTERN_C_END

/* ----------------------------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
static void (PETSC_STDCALL *theirmat)(DMMG*,Mat*,Mat*,PetscErrorCode*);
EXTERN_C_END

static PetscErrorCode ourrhs(DMMG dmmg,Vec vec)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMMG*,Vec*,PetscErrorCode*))(((PetscObject)dmmg->dm)->fortran_func_pointers[0]))(&dmmg,&vec,&ierr);
  return ierr;
}

/*
   Since DMMGSetKSP() immediately calls the matrix functions for each level we do not need to store
  the mat() function inside the DMMG object
*/
static PetscErrorCode ourmat(DMMG dmmg,Mat mat,Mat mat2)
{
  PetscErrorCode ierr = 0;
  (*theirmat)(&dmmg,&mat,&mat2,&ierr);
  return ierr;
}

static PetscErrorCode ourinitialguess(DMMG dmmg,Vec vec)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(DMMG*,Vec*,PetscErrorCode*))(((PetscObject)dmmg->dm)->fortran_func_pointers[2]))(&dmmg,&vec,&ierr);
  return ierr;
}

EXTERN_C_BEGIN

void PETSC_STDCALL dmmgsetksp_(DMMG **dmmg,void (PETSC_STDCALL *rhs)(DMMG*,Vec*,PetscErrorCode*),void (PETSC_STDCALL *mat)(DMMG*,Mat*,Mat *,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt i;
  theirmat = mat;
  *ierr = DMMGSetKSP(*dmmg,ourrhs,ourmat);
  /*
    Save the fortran rhs function in the DM on each level; ourrhs() pulls it out when needed
  */
  for (i=0; i<(**dmmg)->nlevels; i++) {
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[0] = (PetscVoidFunction)rhs;
  }
}

/* ----------------------------------------------------------------------------------------------------------*/

void PETSC_STDCALL dmmgsetinitialguess_(DMMG **dmmg,void (PETSC_STDCALL *initialguess)(DMMG*,Vec*,PetscErrorCode*),PetscErrorCode *ierr)
{
  PetscInt i;

  if (FORTRANNULLFUNCTION(initialguess)) {
    *ierr = DMMGSetInitialGuess(*dmmg,PETSC_NULL);
  } else {
    *ierr = DMMGSetInitialGuess(*dmmg,ourinitialguess);
    /*
      Save the fortran initial guess function in the DM on each level; ourinitialguess() pulls it out when needed
    */
    for (i=0; i<(**dmmg)->nlevels; i++) {
      ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[2] = (PetscVoidFunction)initialguess;
    }
  }
}

void PETSC_STDCALL dmmgsetdm_(DMMG **dmmg,DM *dm,PetscErrorCode *ierr)
{
  PetscInt i;
  *ierr = DMMGSetDM(*dmmg,*dm);if (*ierr) return;
  /* loop over the levels added a place to hang the function pointers in the DM for each level*/
  for (i=0; i<(**dmmg)->nlevels; i++) {
        PetscObjectAllocateFortranPointers((*dmmg)[i]->dm,4);
  }
}

void PETSC_STDCALL dmmgview_(DMMG **dmmg,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = DMMGView(*dmmg,v);
}

void PETSC_STDCALL dmmgsolve_(DMMG **dmmg,PetscErrorCode *ierr)
{
  *ierr = DMMGSolve(*dmmg);
}

void PETSC_STDCALL dmmgcreate_(MPI_Comm *comm,PetscInt *nlevels,void *user,DMMG **dmmg,PetscErrorCode *ierr)
{
  *ierr = DMMGCreate(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*nlevels,user,dmmg);
}

void PETSC_STDCALL dmmgdestroy_(DMMG **dmmg,PetscErrorCode *ierr)
{
  *ierr = DMMGDestroy(*dmmg);
}

void PETSC_STDCALL dmmgsetup_(DMMG **dmmg,PetscErrorCode *ierr)
{
  *ierr = DMMGSetUp(*dmmg);
}

void PETSC_STDCALL dmmgsetuser_(DMMG **dmmg,PetscInt *level,void *ptr,PetscErrorCode *ierr)
{
  *ierr = DMMGSetUser(*dmmg,*level,ptr);
}

void PETSC_STDCALL dmmggetuser_(DMMG *dmmg,void **ptr,PetscErrorCode *ierr)
{
  *ierr = 0;
  *ptr  = (*dmmg)->user;
}

void PETSC_STDCALL dmmggetdm_(DMMG *dmmg,DM *dm,PetscErrorCode *ierr)
{
  *ierr = 0;
  *dm = (*dmmg)->dm;
}

void PETSC_STDCALL dmmgarraygetdmmg_(DMMG **dmmg,DMMG *dmmglevel,PetscErrorCode *ierr)
{
  *ierr = 0;
  *dmmglevel = (*dmmg)[(**dmmg)->nlevels-1];
}

EXTERN_C_END

