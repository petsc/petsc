#ifndef lint
static char vcid[] = "$Id: zsles.c,v 1.10 1997/04/04 19:10:23 bsmith Exp bsmith $";
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

void slessetoptionsprefix_(SLES sles,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SLESSetOptionsPrefix((SLES)PetscToPointer( *(int*)(sles) ),t);
  FREECHAR(prefix,t);
}

void slesappendoptionsprefix_(SLES sles,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = SLESAppendOptionsPrefix((SLES)PetscToPointer( *(int*)(sles) ),t);
  FREECHAR(prefix,t);
}

void slesgetksp_(SLES sles,KSP *ksp, int *__ierr ){
  KSP joe;
  *__ierr = SLESGetKSP((SLES)PetscToPointer( *(int*)(sles) ),&joe);
  *(int*) ksp = PetscFromPointer(joe);
}

void slesgetpc_(SLES sles,PC *pc, int *__ierr ){
  PC joe;
  *__ierr = SLESGetPC((SLES)PetscToPointer( *(int*)(sles) ),&joe);
  *(int*) pc = PetscFromPointer(joe);
}

void slesdestroy_(SLES sles, int *__ierr )
{
  *__ierr = SLESDestroy((SLES)PetscToPointer( *(int*)(sles) ));
  PetscRmPointer( *(int*)(sles) );
}

void slescreate_(MPI_Comm *comm,SLES *outsles, int *__ierr )
{
  SLES sles;
  *__ierr = SLESCreate((MPI_Comm)PetscToPointerComm( *comm ),&sles);
  *(int*) outsles = PetscFromPointer(sles);

}

#if defined(__cplusplus)
}
#endif
