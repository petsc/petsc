/*$Id: zplog.c,v 1.24 2000/09/06 21:03:24 balay Exp balay $*/

#include "src/fortran/custom/zpetsc.h"
#include "petscsys.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
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
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
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

EXTERN_C_BEGIN

void PETSC_STDCALL plogdump_(CHAR name PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(name,len,t1);
  *ierr = PLogDump(t1);
  FREECHAR(name,t1);
#endif
}
void PETSC_STDCALL plogeventregister_(int *e,CHAR string PETSC_MIXED_LEN(len1),
               CHAR color PETSC_MIXED_LEN(len2),int *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
#if defined(PETSC_USE_LOG)
  char *t1,*t2;
  FIXCHAR(string,len1,t1);
  FIXCHAR(color,len2,t2);

  *ierr = PLogEventRegister(e,t1,t2);
  FREECHAR(string,t1);
  FREECHAR(color,t2);
#endif
}

void PETSC_STDCALL plogallbegin_(int *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PLogAllBegin();
#endif
}

void PETSC_STDCALL plogdestroy_(int *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PLogDestroy();
#endif
}

void PETSC_STDCALL plogbegin_(int *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PLogBegin();
#endif
}

void PETSC_STDCALL plogeventbegin_(int *e,PetscObject *o1,PetscObject *o2,PetscObject *o3,PetscObject *o4, int *_ierr){
  *_ierr = PLogEventBegin(*e,*o1,*o2,*o3,*o4);
}

void PETSC_STDCALL plogeventend_(int *e,PetscObject *o1,PetscObject *o2,PetscObject *o3,PetscObject *o4, int *_ierr){
  *_ierr = PLogEventEnd(*e,*o1,*o2,*o3,*o4);
}

void PETSC_STDCALL plogflops_(int *f,int *_ierr) {
  *_ierr = PLogFlops(*f);
}

void PETSC_STDCALL plogstagepop_(int *ierr)
{
#if defined(PETSC_USE_LOG)
  *ierr = PLogStagePop();
#endif
}

void PETSC_STDCALL plogstageregister_(int *stage,CHAR sname PETSC_MIXED_LEN(len),
                                      int *ierr PETSC_END_LEN(len))
{
#if defined(PETSC_USE_LOG)
  char *t;
  FIXCHAR(sname,len,t);
  *ierr = PLogStageRegister(*stage,t);
#endif
}

void PETSC_STDCALL plogstagepush_(int *stage,int *ierr){
#if defined(PETSC_USE_LOG)
  *ierr = PLogStagePush(*stage);
#endif
}

EXTERN_C_END
