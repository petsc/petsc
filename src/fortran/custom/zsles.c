#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zsles.c,v 1.16 1999/04/05 18:20:15 balay Exp balay $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "sles.h"

#ifdef HAVE_FORTRAN_CAPS
#define slesdestroy_             SLESDESTROY
#define slescreate_              SLESCREATE
#define slesgetpc_               SLESGETPC
#define slessetoptionsprefix_    SLESSETOPTIONSPREFIX
#define slesappendoptionsprefix_ SLESAPPENDOPTIONSPREFIX
#define slesgetksp_              SLESGETKSP
#define slesgetoptionsprefix_    SLESGETOPTIONSPREFIX
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define slessetoptionsprefix_    slessetoptionsprefix
#define slesappendoptionsprefix_ slesappendoptionsprefix
#define slesdestroy_             slesdestroy
#define slescreate_              slescreate
#define slesgetpc_               slesgetpc
#define slesgetksp_              slesgetksp
#define slesgetoptionsprefix_    slesgetoptionsprefix
#endif

EXTERN_C_BEGIN

void slessetoptionsprefix_(SLES *sles,CHAR prefix, int *__ierr,int len )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SLESSetOptionsPrefix(*sles,t);
  FREECHAR(prefix,t);
}

void slesappendoptionsprefix_(SLES *sles,CHAR prefix, int *__ierr,int len )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SLESAppendOptionsPrefix(*sles,t);
  FREECHAR(prefix,t);
}

void slesgetksp_(SLES *sles,KSP *ksp, int *__ierr )
{
  *__ierr = SLESGetKSP(*sles,ksp);
}

void slesgetpc_(SLES *sles,PC *pc, int *__ierr )
{
  *__ierr = SLESGetPC(*sles,pc);
}

void slesdestroy_(SLES *sles, int *__ierr )
{
  *__ierr = SLESDestroy(*sles);
}

void slescreate_(MPI_Comm *comm,SLES *outsles, int *__ierr )
{
  *__ierr = SLESCreate((MPI_Comm)PetscToPointerComm( *comm ),outsles);

}

void slesgetoptionsprefix_(SLES *sles, CHAR prefix,int *__ierr,int len)
{
  char *tname;

  *__ierr = SLESGetOptionsPrefix(*sles,&tname);
#if defined(USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    PetscStrncpy(t,tname,len1);
  }
#else
  PetscStrncpy(prefix,tname,len);
#endif
}

EXTERN_C_END


