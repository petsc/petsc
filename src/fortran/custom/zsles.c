/*$Id: zsles.c,v 1.20 1999/10/04 22:51:03 balay Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "sles.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define slesdestroy_             SLESDESTROY
#define slescreate_              SLESCREATE
#define slesgetpc_               SLESGETPC
#define slessetoptionsprefix_    SLESSETOPTIONSPREFIX
#define slesappendoptionsprefix_ SLESAPPENDOPTIONSPREFIX
#define slesgetksp_              SLESGETKSP
#define slesgetoptionsprefix_    SLESGETOPTIONSPREFIX
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define slessetoptionsprefix_    slessetoptionsprefix
#define slesappendoptionsprefix_ slesappendoptionsprefix
#define slesdestroy_             slesdestroy
#define slescreate_              slescreate
#define slesgetpc_               slesgetpc
#define slesgetksp_              slesgetksp
#define slesgetoptionsprefix_    slesgetoptionsprefix
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL slessetoptionsprefix_(SLES *sles,CHAR prefix PETSC_MIXED_LEN(len),
                                         int *__ierr PETSC_END_LEN(len) )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SLESSetOptionsPrefix(*sles,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL slesappendoptionsprefix_(SLES *sles,CHAR prefix PETSC_MIXED_LEN(len),
                                            int *__ierr PETSC_END_LEN(len) )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SLESAppendOptionsPrefix(*sles,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL slesgetksp_(SLES *sles,KSP *ksp, int *__ierr )
{
  *__ierr = SLESGetKSP(*sles,ksp);
}

void PETSC_STDCALL slesgetpc_(SLES *sles,PC *pc, int *__ierr )
{
  *__ierr = SLESGetPC(*sles,pc);
}

void PETSC_STDCALL slesdestroy_(SLES *sles, int *__ierr )
{
  *__ierr = SLESDestroy(*sles);
}

void PETSC_STDCALL slescreate_(MPI_Comm *comm,SLES *outsles, int *__ierr )
{
  *__ierr = SLESCreate((MPI_Comm)PetscToPointerComm( *comm ),outsles);

}

void PETSC_STDCALL slesgetoptionsprefix_(SLES *sles, CHAR prefix PETSC_MIXED_LEN(len),
                                         int *__ierr PETSC_END_LEN(len) )
{
  char *tname;

  *__ierr = SLESGetOptionsPrefix(*sles,&tname);
#if defined(USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *__ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *__ierr = PetscStrncpy(prefix,tname,len);
#endif
}

EXTERN_C_END


