/*$Id: zPetscLog.c,v 1.25 2000/09/06 22:50:45 balay Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "petscsys.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define PetscLogeventbegin_       PetscLogEVENTBEGIN
#define PetscLogeventend_         PetscLogEVENTEND
#define PetscLogflops_            PetscLogFLOPS
#define PetscLogallbegin_         PetscLogALLBEGIN
#define PetscLogdestroy_          PetscLogDESTROY
#define PetscLogbegin_            PetscLogBEGIN
#define PetscLogdump_             PetscLogDUMP
#define PetscLogeventregister_    PetscLogEVENTREGISTER
#define PetscLogstagepop_         PetscLogSTAGEPOP
#define PetscLogstageregister_    PetscLogSTAGEREGISTER
#define PetscLogstagepush_        PetscLogSTAGEPUSH
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define PetscLogeventbegin_       PetscLogeventbegin
#define PetscLogeventend_         PetscLogeventend
#define PetscLogflops_            PetscLogflops
#define PetscLogallbegin_         PetscLogallbegin
#define PetscLogdestroy_          PetscLogdestroy
#define PetscLogbegin_            PetscLogbegin
#define PetscLogeventregister_    PetscLogeventregister
#define PetscLogdump_             PetscLogdump
#define PetscLogstagepop_         PetscLogstagepop  
#define PetscLogstageregister_    PetscLogstageregister
#define PetscLogstagepush_        PetscLogstagepush
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL PetscLogdump_(CHAR name PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(name,len,t1);
  *ierr = PetscLogDump(t1);
  FREECHAR(name,t1);
#endif
}
void PETSC_STDCALL PetscLogeventregister_(int *e,CHAR string PETSC_MIXED_LEN(len1),
               CHAR color PETSC_MIXED_LEN(len2),int *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
#if defined(PETSC_USE_LOG)
  char *t1,*t2;
  FIXCHAR(string,len1,t1);
  FIXCHAR(color,len2,t2);

  *ierr = PetscLogEventRegister(e,t1,t2);
  FREECHAR(string,t1);
  FREECHAR(color,t2);
#endif
}

void PETSC_STDCALL PetscLogallbegin_(int *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogAllBegin();
#endif
}

void PETSC_STDCALL PetscLogdestroy_(int *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogDestroy();
#endif
}

void PETSC_STDCALL PetscLogbegin_(int *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogBegin();
#endif
}

void PETSC_STDCALL PetscLogeventbegin_(int *e,PetscObject *o1,PetscObject *o2,PetscObject *o3,PetscObject *o4, int *_ierr){
  *_ierr = PetscLogEventBegin(*e,*o1,*o2,*o3,*o4);
}

void PETSC_STDCALL PetscLogeventend_(int *e,PetscObject *o1,PetscObject *o2,PetscObject *o3,PetscObject *o4, int *_ierr){
  *_ierr = PetscLogEventEnd(*e,*o1,*o2,*o3,*o4);
}

void PETSC_STDCALL PetscLogflops_(int *f,int *_ierr) {
  *_ierr = PetscLogFlops(*f);
}

void PETSC_STDCALL PetscLogstagepop_(int *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogStagePop();
#endif
}

void PETSC_STDCALL PetscLogstageregister_(int *stage,CHAR sname PETSC_MIXED_LEN(len),
                                      int *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t;
  FIXCHAR(sname,len,t);
  *ierr = PetscLogStageRegister(*stage,t);
#endif
}

void PETSC_STDCALL PetscLogstagepush_(int *stage,int *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PetscLogStagePush(*stage);
#endif
}

EXTERN_C_END
