#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zpc.c,v 1.24 1999/04/05 18:22:51 balay Exp balay $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "sles.h"
#include "mg.h"

#ifdef HAVE_FORTRAN_CAPS
#define pcregisterdestroy_         PCREGISTERDESTROY
#define pcdestroy_                 PCDESTROY
#define pccreate_                  PCCREATE
#define pcgetoperators_            PCGETOPERATORS
#define pcgetfactoredmatrix_       PCGETFACTOREDMATRIX
#define pcsetoptionsprefix_        PCSETOPTIONSPREFIX
#define pcappendoptionsprefix_     PCAPPENDOPTIONSPREFIX
#define pcbjacobigetsubsles_       PCBJACOBIGETSUBSLES
#define pcasmgetsubsles_           PCASMGETSUBSLES
#define mggetcoarsesolve_          MGGETCOARSESOLVE
#define mggetsmoother_             MGGETSMOOTHER
#define mggetsmootherup_           MGGETSMOOTHERUP
#define mggetsmootherdown_         MGGETSMOOTHERDOWN
#define pcshellsetapply_           PCSHELLSETAPPLY
#define pcshellsetapplyrichardson_ PCSHELLSETAPPLYRICHARDSON
#define pcgettype_                 PCGETTYPE
#define pcsettype_                 PCSETTYPE
#define pcgetoptionsprefix_        PCGETOPTIONSPREFIX
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define pcregisterdestroy_         pcregisterdestroy
#define pcdestroy_                 pcdestroy
#define pccreate_                  pccreate
#define pcgetoperators_            pcgetoperators
#define pcgetfactoredmatrix_       pcgetfactoredmatrix
#define pcsetoptionsprefix_        pcsetoptionsprefix
#define pcappendoptionsprefix_     pcappendoptionsprefix
#define pcbjacobigetsubsles_       pcbjacobigetsubsles
#define pcasmgetsubsles_           pcasmgetsubsles
#define mggetcoarsesolve_          mggetcoarsesolve
#define mggetsmoother_             mggetsmoother
#define mggetsmootherup_           mggetsmootherup
#define mggetsmootherdown_         mggetsmootherdown
#define pcshellsetapplyrichardson_ pcshellsetapplyrichardson
#define pcshellsetapply_           pcshellsetapply
#define pcgettype_                 pcgettype
#define pcsettype_                 pcsettype
#define pcgetoptionsprefix_        pcgetoptionsprefix
#endif

EXTERN_C_BEGIN

void pcsettype_(PC *pc,CHAR itmethod, int *__ierr,int len )
{
  char *t;

  FIXCHAR(itmethod,len,t);
  *__ierr = PCSetType(*pc,t);
  FREECHAR(itmethod,t);
}


static void (*f1)(void *,Vec*,Vec*,int*);
static int ourshellapply(void *ctx,Vec x,Vec y)
{
  int              ierr = 0;
  (*f1)(ctx,&x,&y,&ierr); CHKERRQ(ierr);
  return 0;
}

void pcshellsetapply_(PC *pc,void (*apply)(void*,Vec *,Vec *,int*),void *ptr,
                      int *__ierr )
{
  f1 = apply;
  *__ierr = PCShellSetApply(*pc,ourshellapply,ptr);
}

static void (*f9)(void *,int*);
static int ourshellsetup(void *ctx)
{
  int              ierr = 0;

  (*f9)(ctx,&ierr); CHKERRQ(ierr);
  return 0;
}

void pcshellsetsetup_(PC *pc,void (*setup)(void*,int*), int *__ierr )
{
  f9 = setup;
  *__ierr = PCShellSetSetUp(*pc,ourshellsetup);
}

/* -----------------------------------------------------------------*/
static void (*f2)(void*,Vec*,Vec*,Vec*,int*,int*);
static int ourapplyrichardson(void *ctx,Vec x,Vec y,Vec w,int m)
{
  int              ierr = 0;

  (*f2)(ctx,&x,&y,&w,&m,&ierr); CHKERRQ(ierr);
  return 0;
}

void pcshellsetapplyrichardson_(PC *pc,
         void (*apply)(void*,Vec *,Vec *,Vec *,int*,int*),
         void *ptr, int *__ierr )
{
  f2 = apply;
  *__ierr = PCShellSetApplyRichardson(*pc,ourapplyrichardson,ptr);
}

void mggetcoarsesolve_(PC *pc,SLES *sles, int *__ierr )
{
  *__ierr = MGGetCoarseSolve(*pc,sles);
}

void mggetsmoother_(PC *pc,int *l,SLES *sles, int *__ierr )
{
  *__ierr = MGGetSmoother(*pc,*l,sles);
}

void mggetsmootherup_(PC *pc,int *l,SLES *sles, int *__ierr )
{
  *__ierr = MGGetSmootherUp(*pc,*l,sles);
}

void mggetsmootherdown_(PC *pc,int *l,SLES *sles, int *__ierr )
{
  *__ierr = MGGetSmootherDown(*pc,*l,sles);
}

void pcbjacobigetsubsles_(PC *pc,int *n_local,int *first_local,SLES *sles,int *__ierr )
{
  SLES *tsles;
  int  i;
  *__ierr = PCBJacobiGetSubSLES(*pc,n_local,first_local,&tsles);
  for ( i=0; i<*n_local; i++ ){
    sles[i] = tsles[i];
  }
}

void pcasmgetsubsles_(PC *pc,int *n_local,int *first_local,SLES *sles,int *__ierr )
{
  SLES *tsles;
  int  i;
  *__ierr = PCASMGetSubSLES(*pc,n_local,first_local,&tsles);
  for ( i=0; i<*n_local; i++ ){
    sles[i] = tsles[i];
  }
}

void pcgetoperators_(PC *pc,Mat *mat,Mat *pmat,MatStructure *flag, int *__ierr)
{
  if (FORTRANNULLINTEGER(flag)) flag = PETSC_NULL;
  *__ierr = PCGetOperators(*pc,mat,pmat,flag);
}

void pcgetfactoredmatrix_(PC *pc,Mat *mat, int *__ierr )
{
  *__ierr = PCGetFactoredMatrix(*pc,mat);
}
 
void pcsetoptionsprefix_(PC *pc,CHAR prefix, int *__ierr,int len )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = PCSetOptionsPrefix(*pc,t);
  FREECHAR(prefix,t);
}

void pcappendoptionsprefix_(PC *pc,CHAR prefix, int *__ierr,int len )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = PCAppendOptionsPrefix(*pc,t);
  FREECHAR(prefix,t);
}

void pcdestroy_(PC *pc, int *__ierr )
{
  *__ierr = PCDestroy(*pc);
}

void pccreate_(MPI_Comm comm,PC *newpc, int *__ierr )
{
  *__ierr = PCCreate((MPI_Comm)PetscToPointerComm( *(int*)(comm) ),newpc);
}

void pcregisterdestroy_(int *__ierr)
{
  *__ierr = PCRegisterDestroy();
}

void pcgettype_(PC *pc,CHAR name,int *__ierr,int len)
{
  char *tname;

  *__ierr = PCGetType(*pc,&tname);
#if defined(USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  PetscStrncpy(t,tname,len1);
  }
#else
  PetscStrncpy(name,tname,len);
#endif
}

void pcgetoptionsprefix_(PC *pc, CHAR prefix,int *__ierr,int len)
{
  char *tname;

  *__ierr = PCGetOptionsPrefix(*pc,&tname);
#if defined(USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    PetscStrncpy(t,tname,len1);
  }
#else
  PetscStrncpy(prefix,tname,len);
#endif
}



EXTERN_C_END

