#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zsles.c,v 1.13 1998/03/30 22:22:21 balay Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "sles.h"
#include "pinclude/petscfix.h"

#ifdef HAVE_FORTRAN_CAPS
#define slesdestroy_             SLESDESTROY
#define slescreate_              SLESCREATE
#define slesgetpc_               SLESGETPC
#define slessetoptionsprefix_    SLESSETOPTIONSPREFIX
#define slesappendoptionsprefix_ SLESAPPENDOPTIONSPREFIX
#define slesgetksp_              SLESGETKSP
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define slessetoptionsprefix_    slessetoptionsprefix
#define slesappendoptionsprefix_ slesappendoptionsprefix
#define slesdestroy_             slesdestroy
#define slescreate_              slescreate
#define slesgetpc_               slesgetpc
#define slesgetksp_              slesgetksp
#endif

#if defined(__cplusplus)
extern "C" {
#endif

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

#if defined(__cplusplus)
}
#endif


