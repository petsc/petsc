#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zplog.c,v 1.17 1999/04/06 18:13:58 balay Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "sys.h"

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

void plogdump_(CHAR name, int *__ierr,int len ){
#if defined(PETSC_USE_LOG)
  char *t1;
  FIXCHAR(name,len,t1);
  *__ierr = PLogDump(t1);
  FREECHAR(name,t1);
#endif
}
void plogeventregister_(int *e,CHAR string,CHAR color,int *__ierr,int len1,
                        int len2){
#if defined(PETSC_USE_LOG)
  char *t1,*t2;
  FIXCHAR(string,len1,t1);
  FIXCHAR(color,len2,t2);

  *__ierr = PLogEventRegister(e,t1,t2);
  FREECHAR(string,t1);
  FREECHAR(color,t2);
#endif
}

void plogallbegin_(int *__ierr){
#if defined(PETSC_USE_LOG)
  *__ierr = PLogAllBegin();
#endif
}

void plogdestroy_(int *__ierr){
#if defined(PETSC_USE_LOG)
  *__ierr = PLogDestroy();
#endif
}

void plogbegin_(int *__ierr){
#if defined(PETSC_USE_LOG)
  *__ierr = PLogBegin();
#endif
}

void plogeventbegin_(int *e,PetscObject *o1,PetscObject *o2,PetscObject *o3,PetscObject *o4){
  PLogEventBegin(*e,*o1,*o2,*o3,*o4);
}

void plogeventend_(int *e,PetscObject *o1,PetscObject *o2,PetscObject *o3,PetscObject *o4){
  PLogEventEnd(*e,*o1,*o2,*o3,*o4);
}

void plogflops_(int *f) {
  PLogFlops(*f);
}

void plogstagepop_(int *__ierr )
{
#if defined(PETSC_USE_LOG)
  *__ierr = PLogStagePop();
#endif
}

void plogstageregister_(int *stage,CHAR sname, int *__ierr,int len){
#if defined(PETSC_USE_LOG)
  char *t;
  FIXCHAR(sname,len,t);
  *__ierr = PLogStageRegister(*stage,t);
#endif
}

void plogstagepush_(int *stage, int *__ierr ){
#if defined(PETSC_USE_LOG)
  *__ierr = PLogStagePush(*stage);
#endif
}

EXTERN_C_END
