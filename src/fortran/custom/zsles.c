/*$Id: zsles.c,v 1.37 2001/09/11 16:34:57 bsmith Exp $*/

#include "src/fortran/custom/zpetsc.h"
#include "petscsles.h"
#include "petscda.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define slesdestroy_             SLESDESTROY
#define slescreate_              SLESCREATE
#define slessetoptionsprefix_    SLESSETOPTIONSPREFIX
#define slesappendoptionsprefix_ SLESAPPENDOPTIONSPREFIX
#define slesgetksp_              SLESGETKSP
#define slesgetoptionsprefix_    SLESGETOPTIONSPREFIX
#define slesview_                SLESVIEW
#define dmmgcreate_              DMMGCREATE
#define dmmgdestroy_             DMMGDESTROY
#define dmmgsetup_               DMMGSETUP
#define dmmgsetdm_               DMMGSETDM
#define dmmgview_                DMMGVIEW
#define dmmgsolve_               DMMGSOLVE
#define dmmggetda_               DMMGGETDA
#define dmmgsetsles_             DMMGSETSLES
#define dmmggetx_                DMMGGETX
#define dmmggetj_                DMMGGETJ
#define dmmggetb_                DMMGGETB
#define dmmggetsles_             DMMGGETSLES
#define dmmggetlevels_           DMMGGETLEVELS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmmggetx_                dmmggetx
#define dmmggetj_                dmmggetj
#define dmmggetb_                dmmggetb
#define dmmggetsles_             dmmggetsles
#define dmmggetda_               dmmggetda
#define dmmggetlevels_           dmmggetlevels
#define dmmgsetsles_             dmmgsetsles
#define dmmgdestroy_             dmmgdestroy
#define dmmgcreate_              dmmgcreate
#define dmmgsetup_               dmmgsetup
#define slessetoptionsprefix_    slessetoptionsprefix
#define slesappendoptionsprefix_ slesappendoptionsprefix
#define slesdestroy_             slesdestroy
#define slescreate_              slescreate
#define slesgetksp_              slesgetksp
#define slesgetoptionsprefix_    slesgetoptionsprefix
#define slesview_                slesview
#define dmmgsetdm_               dmmgsetdm
#define dmmgview_                dmmgview
#define dmmgsolve_               dmmgsolve
#endif

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

void PETSC_STDCALL dmmggetsles_(DMMG **dmmg,SLES *x,int *ierr)
{
  *ierr = 0;
  *x    = DMMGGetSLES(*dmmg);
}

void PETSC_STDCALL dmmggetlevels_(DMMG **dmmg,int *x,int *ierr)
{
  *ierr = 0;
  *x    = DMMGGetLevels(*dmmg);
}

/* ----------------------------------------------------------------------------------------------------------*/
static int ourrhs(DMMG dmmg,Vec vec)
{
  int              ierr = 0;
  (*(int (PETSC_STDCALL *)(DMMG*,Vec*,int*))(((PetscObject)dmmg->dm)->fortran_func_pointers[0]))(&dmmg,&vec,&ierr);
  return ierr;
}

/*
   Since DMMGSetSLES() immediately calls the matrix functions for each level we do not need to store
  the mat() function inside the DMMG object
*/
static int (PETSC_STDCALL *theirmat)(DMMG*,Mat*,int*);
static int ourmat(DMMG dmmg,Mat mat)
{
  int              ierr = 0;
  (*theirmat)(&dmmg,&mat,&ierr);
  return ierr;
}

void PETSC_STDCALL dmmgsetsles_(DMMG **dmmg,int (PETSC_STDCALL *rhs)(DMMG*,Vec*,int*),int (PETSC_STDCALL *mat)(DMMG*,Mat*,int*),int *ierr)
{
  int i;
  theirmat = mat;
  *ierr = DMMGSetSLES(*dmmg,ourrhs,ourmat);
  /*
    Save the fortran rhs function in the DM on each level; ourrhs() pulls it out when needed
  */
  for (i=0; i<(**dmmg)->nlevels; i++) {
    ((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers[0] = (void (*)(void))rhs;
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
    *ierr = PetscMalloc(3*sizeof(void (*)(void)),&((PetscObject)(*dmmg)[i]->dm)->fortran_func_pointers);if (*ierr) return;
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

void PETSC_STDCALL slesview_(SLES *sles,PetscViewer *viewer, int *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = SLESView(*sles,v);
}

void PETSC_STDCALL slessetoptionsprefix_(SLES *sles,CHAR prefix PETSC_MIXED_LEN(len),
                                         int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = SLESSetOptionsPrefix(*sles,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL slesappendoptionsprefix_(SLES *sles,CHAR prefix PETSC_MIXED_LEN(len),
                                            int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = SLESAppendOptionsPrefix(*sles,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL slesgetksp_(SLES *sles,KSP *ksp,int *ierr)
{
  *ierr = SLESGetKSP(*sles,ksp);
}

void PETSC_STDCALL slesdestroy_(SLES *sles,int *ierr)
{
  *ierr = SLESDestroy(*sles);
}

void PETSC_STDCALL slescreate_(MPI_Comm *comm,SLES *outsles,int *ierr)
{
  *ierr = SLESCreate((MPI_Comm)PetscToPointerComm(*comm),outsles);

}

void PETSC_STDCALL slesgetoptionsprefix_(SLES *sles,CHAR prefix PETSC_MIXED_LEN(len),
                                         int *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = SLESGetOptionsPrefix(*sles,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *ierr = PetscStrncpy(prefix,tname,len);
#endif
}

EXTERN_C_END


