#ifndef lint
static char vcid[] = "$Id: zoptions.c,v 1.1 1995/08/21 19:56:20 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "petsc.h"
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#include "pinclude/petscfix.h"

#ifdef FORTRANCAPS
#define petscsetdebugger_     PETSCSETDEBUGGER
#define petscattachdebugger_  PETSCATTACHDEBUGGER
#define petscpoperrorhandler_ PETSCPOPERRORHANDLER
#define plogallbegin_         PLOGALLBEGIN
#define plogdestroy_          PLOGDESTROY
#define plogbegin_            PLOGBEGIN
#define petscobjectsetname_   PETSCOBJECTSETNAME
#define petscobjectdestroy_   PETSCOBJECTDESTROY
#define petscobjectgetcomm_   PETSCOBJECTGETCOMM
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscsetdebugger_     petscsetdebugger
#define petscattachdebugger_  petscattachdebugger
#define petscpoperrorhandler_ petscpoperrorhandler
#define plogallbegin_         plogallbegin
#define plogdestroy_          plogdestroy
#define plogbegin_            plogbegin
#define petscobjectsetname_   petscobjectsetname
#define petscobjectdestroy_   petscobjectdestroy
#define petscobjectgetcomm_   petscobjectgetcomm
#endif



void petscobjectdestroy_(PetscObject obj, int *__ierr ){
  *__ierr = PetscObjectDestroy((PetscObject)MPIR_ToPointer(*(int*)(obj)));
  MPIR_RmPointer(*(int*)(obj));
}
void petscobjectgetcomm_(PetscObject obj,MPI_Comm *comm, int *__ierr ){
  MPI_Comm c;
  *__ierr = PetscObjectGetComm((PetscObject)MPIR_ToPointer(*(int*)(obj)),&c);
  *(int*)comm = MPIR_FromPointer(c);
}

void petscsetdebugger_(char *debugger,int *xterm,char *display, int *__ierr ){
  *__ierr = PetscSetDebugger(debugger,*xterm,display);
}

void petscattachdebugger_(int *__ierr){
  *__ierr = PetscAttachDebugger();
}

void petscpoperrorhandler_(int *__ierr){
  *__ierr = PetscPopErrorHandler();
}

void plogallbegin_(int *__ierr){
  *__ierr = PLogAllBegin();
}

void plogdestroy_(int *__ierr){
  *__ierr = PLogDestroy();
}

void plogbegin_(int *__ierr){
  *__ierr = PLogBegin();
}

/*
      This bleeds memory, but no easy way to get around it
*/
void petscobjectsetname_(PetscObject obj,char *name,int *__ierr,int len)
{
  char *t1;
  if (name[len] != 0) {
    t1 = (char *) PETSCMALLOC( (len+1)*sizeof(char) ); 
    if (!t1) { *__ierr = 1; return;}
    strncpy(t1,name,len);
    t1[len] = 0;
  }
  else t1 = name;
  *__ierr = PetscObjectSetName((PetscObject)MPIR_ToPointer(*(int*)(obj)),t1);
}
