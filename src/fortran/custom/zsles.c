#ifndef lint
static char vcid[] = "$Id: zsles.c,v 1.6 1996/01/15 22:37:53 balay Exp bsmith $";
#endif

#include "zpetsc.h"
#include "sles.h"
#include "draw.h"
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

void slessetoptionsprefix_(SLES sles,char *prefix, int *__ierr,int len ){
  char *t=0;
  if (prefix[len] != 0) {
    t = (char *) PetscMalloc( (len+1)*sizeof(char) ); 
    PetscStrncpy(t,prefix,len);
    t[len] = 0;
  }
  else t = prefix;
  *__ierr = SLESSetOptionsPrefix((SLES)MPIR_ToPointer( *(int*)(sles) ),t);
  if( t != prefix) PetscFree(t);
}
void slesappendoptionsprefix_(SLES sles,char *prefix, int *__ierr,int len ){
  char *t=0;
  if (prefix[len] != 0) {
    t = (char *) PetscMalloc( (len+1)*sizeof(char) ); 
    PetscStrncpy(t,prefix,len);
    t[len] = 0;
  }
  else t = prefix;
  *__ierr = SLESAppendOptionsPrefix((SLES)MPIR_ToPointer( *(int*)(sles) ),t);
  if( t != prefix) PetscFree(t);
}
void slesgetksp_(SLES sles,KSP *ksp, int *__ierr ){
  KSP joe;
  *__ierr = SLESGetKSP((SLES)MPIR_ToPointer( *(int*)(sles) ),&joe);
  *(int*) ksp = MPIR_FromPointer(joe);
}
void slesgetpc_(SLES sles,PC *pc, int *__ierr ){
  PC joe;
  *__ierr = SLESGetPC((SLES)MPIR_ToPointer( *(int*)(sles) ),&joe);
  *(int*) pc = MPIR_FromPointer(joe);
}

void slesdestroy_(SLES sles, int *__ierr )
{
  *__ierr = SLESDestroy((SLES)MPIR_ToPointer( *(int*)(sles) ));
  MPIR_RmPointer( *(int*)(sles) );
}

void slescreate_(MPI_Comm comm,SLES *outsles, int *__ierr )
{
  SLES sles;
  *__ierr = SLESCreate((MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),&sles);
  *(int*) outsles = MPIR_FromPointer(sles);

}

#if defined(__cplusplus)
}
#endif
