#ifndef lint
static char vcid[] = "$Id: zoptions.c,v 1.1 1995/08/21 19:56:20 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "sles.h"
#include "draw.h"
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#include "pinclude/petscfix.h"

#ifdef FORTRANCAPS
#define slesdestroy_          SLESDESTROY
#define slescreate_           SLESCREATE
#define slesgetpc_            SLESGETPC
#define slessetoptionsprefix_ SLESSETOPTIONSPREFIX
#define slesgetksp_           SLESGETKSP
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define slessetoptionsprefix_ slessetoptionsprefix
#define slesdestroy_          slesdestroy
#define slescreate_           slescreate
#define slesgetpc_            slesgetpc
#define slesgetksp_           slesgetksp
#endif

/*
   Can bleed memory 
*/
void slessetoptionsprefix_(SLES sles,char *prefix, int *__ierr,int len ){
  char *t;
  if (prefix[len] != 0) {
    t = (char *) PETSCMALLOC( (len+1)*sizeof(char) ); 
    strncpy(t,prefix,len);
    t[len] = 0;
  }
  else t = prefix;
  *__ierr = SLESSetOptionsPrefix((SLES)MPIR_ToPointer(*(int*)(sles)),t);
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
