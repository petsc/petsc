#ifndef lint
static char vcid[] = "$Id: zsles.c,v 1.4 1995/11/23 04:15:38 bsmith Exp balay $";
#endif

#include "zpetsc.h"
#include "sles.h"
#include "draw.h"
#include "pinclude/petscfix.h"

#ifdef FORTRANCAPS
#define slesdestroy_             SLESDESTROY
#define slescreate_              SLESCREATE
#define slesgetpc_               SLESGETPC
#define slessetoptionsprefix_    SLESSETOPTIONSPREFIX
#define slesappendoptionsprefix_ SLESAPPENDOPTIONSPREFIX
#define slesgetksp_              SLESGETKSP
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define slessetoptionsprefix_    slessetoptionsprefix
#define slesappendoptionsprefix_ slesappendoptionsprefix
#define slesdestroy_             slesdestroy
#define slescreate_              slescreate
#define slesgetpc_               slesgetpc
#define slesgetksp_              slesgetksp
#endif

void slessetoptionsprefix_(SLES sles,char *prefix, int *flg, int *__ierr,int len ){
  char *t=0;
  if (prefix[len] != 0) {
    t = (char *) PetscMalloc( (len+1)*sizeof(char) ); 
    PetscStrncpy(t,prefix,len);
    t[len] = 0;
  }
  else t = prefix;
  *__ierr = SLESSetOptionsPrefix((SLES)MPIR_ToPointer( *(int*)(sles) ),t, flg);
  if( t != prefix) PetscFree(t);
}
void slesappendoptionsprefix_(SLES sles,char *prefix, int *flg, int *__ierr,int len ){
  char *t=0;
  if (prefix[len] != 0) {
    t = (char *) PetscMalloc( (len+1)*sizeof(char) ); 
    PetscStrncpy(t,prefix,len);
    t[len] = 0;
  }
  else t = prefix;
  *__ierr = SLESAppendOptionsPrefix((SLES)MPIR_ToPointer( *(int*)(sles) ),t, flg);
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
