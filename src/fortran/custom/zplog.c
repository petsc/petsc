#ifndef lint
static char vcid[] = "$Id: zplog.c,v 1.9 1996/12/16 16:17:38 balay Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "sys.h"
#include "pinclude/petscfix.h"

#ifdef HAVE_FORTRAN_CAPS
#define plogeventbegin_       PLOGEVENTBEGIN
#define plogeventend_         PLOGEVENTEND
#define plogflops_            PLOGFLOPS
#define plogallbegin_         PLOGALLBEGIN
#define plogdestroy_          PLOGDESTROY
#define plogbegin_            PLOGBEGIN
#define plogdump_             PLOGDUMP
#define plogeventregister_    PLOGEVENTREGISTER
#define plogstagepop_         PLOGSTAGEPOP
#define plogstageregister_    PLOGSTAGEREGISTER
#define plogstagepush_        PLOGSTAGEPUSH
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define plogeventbegin_       plogeventbegin
#define plogeventend_         plogeventend
#define plogflops_            plogflops
#define plogallbegin_         plogallbegin
#define plogdestroy_          plogdestroy
#define plogbegin_            plogbegin
#define plogeventregister_    plogeventregister
#define plogdump_             plogdump
#define plogstagepop_         plogstagepop  
#define plogstageregister_    plogstageregister
#define plogstagepush_        plogstagepush
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void plogdump_(CHAR name, int *__ierr,int len ){
#if defined(PETSC_LOG)
  char *t1;
  FIXCHAR(name,len,t1);
  *__ierr = PLogDump(t1);
  FREECHAR(name,t1);
#endif
}
void plogeventregister_(int *e,CHAR string,CHAR color,int *__ierr,int len1,
                        int len2){
#if defined(PETSC_LOG)
  char *t1,*t2,*n1,*n2;
  FIXCHAR(string,len1,t1);
  FIXCHAR(color,len2,t2);

  /* we have to copy the strings because some Fortran compilers */
  /* do not keep them as static */
  n1 = (char *) PetscMalloc( (len1+1)*sizeof(char) ); if (!n1) {*__ierr = 1; return;}
  PetscStrncpy(n1,t1,len1); n1[len1] = 0; FREECHAR(string,t1);
  n2 = (char *) PetscMalloc( (len2+1)*sizeof(char) ); if (!n2) {*__ierr = 1; return;}
  PetscStrncpy(n2,t2,len2); n2[len2] = 0; FREECHAR(color,t2);
  *__ierr = PLogEventRegister(e,n1,n2);
#endif
}

void plogallbegin_(int *__ierr){
#if defined(PETSC_LOG)
  *__ierr = PLogAllBegin();
#endif
}

void plogdestroy_(int *__ierr){
#if defined(PETSC_LOG)
  *__ierr = PLogDestroy();
#endif
}

void plogbegin_(int *__ierr){
#if defined(PETSC_LOG)
  *__ierr = PLogBegin();
#endif
}

void plogeventbegin_(int *e,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4){
#if defined(PETSC_LOG)
  PetscObject t1,t2,t3,t4;
  if (o1) t1 = (PetscObject) PetscToPointer(*(int*)(o1)); else t1 = 0;
  if (o2) t2 = (PetscObject) PetscToPointer(*(int*)(o2)); else t2 = 0;
  if (o3) t3 = (PetscObject) PetscToPointer(*(int*)(o3)); else t3 = 0;
  if (o4) t4 = (PetscObject) PetscToPointer(*(int*)(o4)); else t4 = 0;

  if (_PLogPLB) (*_PLogPLB)(*e,1,t1,t2,t3,t4);
#if defined(HAVE_MPE)
  if (UseMPE && PLogEventMPEFlags[*e]) MPE_Log_event(MPEBEGIN+2*(*e),0,"");
#endif
#endif
}

void plogeventend_(int *e,PetscObject o1,PetscObject o2,PetscObject o3,PetscObject o4){
#if defined(PETSC_LOG)
  PetscObject t1,t2,t3,t4;
  if (o1) t1 = (PetscObject) PetscToPointer(*(int*)(o1)); else t1 = 0;
  if (o2) t2 = (PetscObject) PetscToPointer(*(int*)(o2)); else t2 = 0;
  if (o3) t3 = (PetscObject) PetscToPointer(*(int*)(o3)); else t3 = 0;
  if (o4) t4 = (PetscObject) PetscToPointer(*(int*)(o4)); else t4 = 0;
  if (_PLogPLE) (*_PLogPLE)(*e,1,t1,t2,t3,t4);
#if defined(HAVE_MPE)
  if (UseMPE && PLogEventMPEFlags[*e]) MPE_Log_event(MPEBEGIN+2*(*e)+1,0,"");
#endif
#endif
}

void plogflops_(int *f) {
  PLogFlops(*f);
}

void plogstagepop_(int *__ierr )
{
#if defined(PETSC_LOG)
  *__ierr = PLogStagePop();
#endif
}

void plogstageregister_(int *stage,CHAR sname, int *__ierr,int len){
#if defined(PETSC_LOG)
  char *t;
  FIXCHAR(sname,len,t);
  *__ierr = PLogStageRegister(*stage,t);
#endif
}

void plogstagepush_(int *stage, int *__ierr ){
#if defined(PETSC_LOG)
  *__ierr = PLogStagePush(*stage);
#endif
}

#if defined(__cplusplus)
}
#endif
