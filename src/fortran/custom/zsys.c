#ifndef lint
static char vcid[] = "$Id: zsys.c,v 1.2 1995/09/04 17:18:58 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "petsc.h"
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
#define petscobjectgetname_   PETSCOBJECTGETNAME
#define plogdump_             PLOGDUMP
#define plogeventregister_    PLOGEVENTREGISTER
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
#define petscobjectgetname_   petscobjectgetname
#define plogeventregister_    plogeventregister
#define plogdump_             plogdump
#endif


void plogdump_(char* name, int *__ierr,int len ){
  char *t1;
  if (name[len] != 0) {
    t1 = (char *) PETSCMALLOC( (len+1)*sizeof(char) ); 
    if (!t1) { *__ierr = 1; return;}
    PetscStrncpy(t1,name,len);
    t1[len] = 0;
  }
  else t1 = name;
  *__ierr = PLogDump(t1);
  if (t1 != name) PETSCFREE(t1);
}
void plogeventregister_(int *e,char *string, int *__ierr,int len ){
  char *t1;
  if (string[len] != 0) {
    t1 = (char *) PETSCMALLOC( (len+1)*sizeof(char) ); 
    if (!t1) { *__ierr = 1; return;}
    PetscStrncpy(t1,string,len);
    t1[len] = 0;
  }
  else t1 = string;
  *__ierr = PLogEventRegister(*e,t1);
}

void petscobjectgetname(PetscObject obj, char *name, int *__ierr, int len)
{
  char *tmp;
  *__ierr = PetscObjectGetName((PetscObject)MPIR_ToPointer(*(int*)(obj)),
                               &tmp);
  PetscStrncpy(name,tmp,len);
}

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
    PetscStrncpy(t1,name,len);
    t1[len] = 0;
  }
  else t1 = name;
  *__ierr = PetscObjectSetName((PetscObject)MPIR_ToPointer(*(int*)(obj)),t1);
}
