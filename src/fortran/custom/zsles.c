/*$Id: zsles.c,v 1.37 2001/09/11 16:34:57 bsmith Exp $*/

#include "src/fortran/custom/zpetsc.h"
#include "petscksp.h"
#include "petscda.h"

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
#define dmmggetksp_              DMMGGETKSP
#define dmmggetlevels_           DMMGGETLEVELS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
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
#endif

EXTERN_C_BEGIN
static int (PETSC_STDCALL *theirmat)(DMMG*,Mat*,int*);
EXTERN_C_END

static int ourrhs(DMMG dmmg,Vec vec)
{
  int              ierr = 0;
  (*(int (PETSC_STDCALL *)(DMMG*,Vec*,int*))(((PetscObject)dmmg->dm)->fortran_func_pointers[0]))(&dmmg,&vec,&ierr);
  return ierr;
}

/*
   Since DMMGSetKSP() immediately calls the matrix functions for each level we do not need to store
  the mat() function inside the DMMG object
*/
static int ourmat(DMMG dmmg,Mat mat)
{
  int              ierr = 0;
  (*theirmat)(&dmmg,&mat,&ierr);
  return ierr;
}

EXTERN_C_BEGIN

void PETSC_STDCALL dmmggetx_(DMMG **dmmg,Vec *x,int *ierr)
{
  *ierr = 0;
  *x    = DMMGGetx(*dmmg);
}

void PETSC_STDCALL dmmggetj_(DMMG **dmmg,Mat *x,int *ierr)
{
  *ierr = 0;
  *x    = DMMGGetJ(*dmmg);
}

void PETSC_STDCALL dmmggetB_(DMMG **dmmg,Mat *x,int *ierr)
{
  *ierr = 0;
  *x    = DMMGGetB(*dmmg);
}

void PETSC_STDCALL dmmggetksp_(DMMG **dmmg,KSP *x,int *ierr)
{
  *ierr = 0;
  *x    = DMMGGetKSP(*dmmg);
}

void PETSC_STDCALL dmmggetlevels_(DMMG **dmmg,int *x,int *ierr)
{
  *ierr = 0;
  *x    = DMMGGetLevels(*dmmg);
}

/* ----------------------------------------------------------------------------------------------------------*/

void PETSC_STDCALL dmmgsetksp_(DMMG **dmmg,int (PETSC_STDCALL *rhs)(DMMG*,Vec*,int*),int (PETSC_STDCALL *mat)(DMMG*,Mat*,int*),int *ierr)
{
  int i;
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

void PETSC_STDCALL dmmggetda_(DMMG *dmmg,DA *da,int *ierr)
{
  *da   = (DA)(*dmmg)->dm;
  *ierr = 0;
}

void PETSC_STDCALL dmmgsetdm_(DMMG **dmmg,DM *dm,int *ierr)
{
  int i;
  *ierr = DMMGSetDM(*dmmg,*dm);if (*ierr) return;
  /* loop over the levels added a place to hang the function pointers in the DM for each level*/
  for (i=0; i<(**dmmg)->nlevels; i++) {
    *ierr = PetscMalloc(3*sizeof(FCNVOID),&((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers);if (*ierr) return;
  }
}

void PETSC_STDCALL dmmgview_(DMMG **dmmg,PetscViewer *viewer,int *ierr)
{
  *ierr = DMMGView(*dmmg,*viewer);
}

void PETSC_STDCALL dmmgsolve_(DMMG **dmmg,int *ierr)
{
  *ierr = DMMGSolve(*dmmg);
}

void PETSC_STDCALL dmmgcreate_(MPI_Comm *comm,int *nlevels,void *user,DMMG **dmmg,int *ierr)
{
  *ierr = DMMGCreate((MPI_Comm)PetscToPointerComm(*comm),*nlevels,user,dmmg);
}

void PETSC_STDCALL dmmgdestroy_(DMMG **dmmg,int *ierr)
{
  *ierr = DMMGDestroy(*dmmg);
}

void PETSC_STDCALL dmmgsetup_(DMMG **dmmg,int *ierr)
{
  *ierr = DMMGSetUp(*dmmg);
}

EXTERN_C_END


