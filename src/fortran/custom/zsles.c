/*$Id: zsles.c,v 1.30 2001/03/28 04:09:01 bsmith Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "petscsles.h"
#include "petscda.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define slesdestroy_             SLESDESTROY
#define slescreate_              SLESCREATE
#define slesgetpc_               SLESGETPC
#define slessetoptionsprefix_    SLESSETOPTIONSPREFIX
#define slesappendoptionsprefix_ SLESAPPENDOPTIONSPREFIX
#define slesgetksp_              SLESGETKSP
#define slesgetoptionsprefix_    SLESGETOPTIONSPREFIX
#define slesview_                SLESVIEW
#define dmmgcreate_              DMMGCREATE
#define dmmgdestroy_             DMMGDESTROY
#define dmmgsetup_               DMMGSETUP
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmmgdestroy_             dmmgdestroy
#define dmmgcreate_              dmmgcreate
#define dmmgsetup_               dmmgsetup
#define slessetoptionsprefix_    slessetoptionsprefix
#define slesappendoptionsprefix_ slesappendoptionsprefix
#define slesdestroy_             slesdestroy
#define slescreate_              slescreate
#define slesgetpc_               slesgetpc
#define slesgetksp_              slesgetksp
#define slesgetoptionsprefix_    slesgetoptionsprefix
#define slesview_                slesview
#endif

EXTERN_C_BEGIN

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

void PETSC_STDCALL slesgetpc_(SLES *sles,PC *pc,int *ierr)
{
  *ierr = SLESGetPC(*sles,pc);
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


