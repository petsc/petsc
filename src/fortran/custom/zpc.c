#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zpc.c,v 1.19 1998/04/21 18:23:47 balay Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "sles.h"
#include "mg.h"
#include "pinclude/petscfix.h"

#ifdef HAVE_FORTRAN_CAPS
#define pcregisterdestroy_         PCREGISTERDESTROY
#define pcdestroy_                 PCDESTROY
#define pccreate_                  PCCREATE
#define pcgetoperators_            PCGETOPERATORS
#define pcgetfactoredmatrix_       PCGETFACTOREDMATRIX
#define pcsetoptionsprefix_        PCSETOPTIONSPREFIX
#define pcappendoptionsprefix_     PCAPPENDOPTIONSPREFIX
#define pcbjacobigetsubsles_       PCBJACOBIGETSUBSLES
#define mggetcoarsesolve_          MGGETCOARSESOLVE
#define mggetsmoother_             MGGETSMOOTHER
#define mggetsmootherup_           MGGETSMOOTHERUP
#define mggetsmootherdown_         MGGETSMOOTHERDOWN
#define pcshellsetapply_           PCSHELLSETAPPLY
#define pcshellsetapplyrichardson_ PCSHELLSETAPPLYRICHARDSON
#define pcgettype_                 PCGETTYPE
#define pcsettype_                 PCSETTYPE
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define pcregisterdestroy_         pcregisterdestroy
#define pcdestroy_                 pcdestroy
#define pccreate_                  pccreate
#define pcgetoperators_            pcgetoperators
#define pcgetfactoredmatrix_       pcgetfactoredmatrix
#define pcsetoptionsprefix_        pcsetoptionsprefix
#define pcappendoptionsprefix_     pcappendoptionsprefix
#define pcbjacobigetsubsles_       pcbjacobigetsubsles
#define mggetcoarsesolve_          mggetcoarsesolve
#define mggetsmoother_             mggetsmoother
#define mggetsmootherup_           mggetsmootherup
#define mggetsmootherdown_         mggetsmootherdown
#define pcshellsetapplyrichardson_ pcshellsetapplyrichardson
#define pcshellsetapply_           pcshellsetapply
#define pcgettype_                 pcgettype
#define pcsettype_                 pcsettype
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void pcsettype_(PC pc,CHAR itmethod, int *__ierr,int len )
{
  char *t;

  FIXCHAR(itmethod,len,t);
  *__ierr = PCSetType((PC)PetscToPointer(pc),t);
  FREECHAR(itmethod,t);
}


static void (*f1)(void *,PetscFortranAddr*,PetscFortranAddr*,int*);
static int ourshellapply(void *ctx,Vec x,Vec y)
{
  int              ierr = 0;
  PetscFortranAddr s1,s2;

  s1 = PetscFromPointer(x);
  s2 = PetscFromPointer(y);
  (*f1)(ctx,&s1,&s2,&ierr); CHKERRQ(ierr);
  PetscRmPointer(&s1);
  PetscRmPointer(&s2); 
  return 0;
}
void pcshellsetapply_(PC pc,void (*apply)(void*,PetscFortranAddr*,PetscFortranAddr*,int*),void *ptr,
                      int *__ierr ){
  f1 = apply;
  *__ierr = PCShellSetApply(
	(PC)PetscToPointer(pc),ourshellapply,ptr);
}

static void (*f9)(void *,int*);
static int ourshellsetup(void *ctx)
{
  int              ierr = 0;

  (*f9)(ctx,&ierr); CHKERRQ(ierr);
  return 0;
}
void pcshellsetsetup_(PC pc,void (*setup)(void*,int*), int *__ierr ){
  f9 = setup;
  *__ierr = PCShellSetSetUp((PC)PetscToPointer(pc),ourshellsetup);
}

/* -----------------------------------------------------------------*/
static void (*f2)(void*,PetscFortranAddr*,PetscFortranAddr*,PetscFortranAddr*,int*,int*);
static int ourapplyrichardson(void *ctx,Vec x,Vec y,Vec w,int m)
{
  int              ierr = 0;
  PetscFortranAddr s1,s2,s3;

  s1 = PetscFromPointer(x);
  s2 = PetscFromPointer(y);
  s3 = PetscFromPointer(w);
  (*f2)(ctx,&s1,&s2,&s3,&m,&ierr); CHKERRQ(ierr);
  PetscRmPointer(&s1);
  PetscRmPointer(&s2); 
  PetscRmPointer(&s3); 
  return 0;
}

void pcshellsetapplyrichardson_(PC pc,
         void (*apply)(void*,PetscFortranAddr*,PetscFortranAddr*,PetscFortranAddr*,int*,int*),
         void *ptr, int *__ierr ){
  f2 = apply;
  *__ierr = PCShellSetApplyRichardson(
	(PC)PetscToPointer(pc),ourapplyrichardson,ptr);
}

void mggetcoarsesolve_(PC pc,SLES *sles, int *__ierr ){
  SLES asles;
  *__ierr = MGGetCoarseSolve((PC)PetscToPointer(pc),&asles);
  *(PetscFortranAddr*) sles = PetscFromPointer(asles);
}
void mggetsmoother_(PC pc,int *l,SLES *sles, int *__ierr ){
  SLES asles;
  *__ierr = MGGetSmoother((PC)PetscToPointer(pc),*l,&asles);
  *(PetscFortranAddr*) sles = PetscFromPointer(asles);
}
void mggetsmootherup_(PC pc,int *l,SLES *sles, int *__ierr ){
  SLES asles;
  *__ierr = MGGetSmootherUp((PC)PetscToPointer(pc),*l,&asles);
  *(PetscFortranAddr*) sles = PetscFromPointer(asles);
}
void mggetsmootherdown_(PC pc,int *l,SLES *sles, int *__ierr ){
  SLES asles;
  *__ierr = MGGetSmootherDown((PC)PetscToPointer(pc),*l,&asles);
  *(PetscFortranAddr*) sles = PetscFromPointer(asles);
}

void pcbjacobigetsubsles_(PC pc,int *n_local,int *first_local,PetscFortranAddr *sles, 
                          int *__ierr ){
  SLES *tsles;
  int  i;
  *__ierr = PCBJacobiGetSubSLES(
	(PC)PetscToPointer(pc),n_local,first_local,&tsles);
  for ( i=0; i<*n_local; i++ ){
    sles[i] = PetscFromPointer(tsles[i]);
  }
}

void pcgetoperators_(PC pc,Mat *mat,Mat *pmat,MatStructure *flag, int *__ierr){
  Mat m,p;
  if (FORTRANNULLINTEGER(flag)) flag = PETSC_NULL;
  *__ierr = PCGetOperators((PC)PetscToPointer(pc),&m,&p,flag);
  if (mat) *(PetscFortranAddr*) mat = PetscFromPointer(m);
  if (pmat) *(PetscFortranAddr*) pmat = PetscFromPointer(p);
}

void pcgetfactoredmatrix_(PC pc,Mat *mat, int *__ierr ){
  Mat m;
  *__ierr = PCGetFactoredMatrix((PC)PetscToPointer(pc),&m);
  *(PetscFortranAddr*) mat = PetscFromPointer(m);
}
 
void pcsetoptionsprefix_(PC pc,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = PCSetOptionsPrefix((PC)PetscToPointer(pc),t);
  FREECHAR(prefix,t);
}

void pcappendoptionsprefix_(PC pc,CHAR prefix, int *__ierr,int len ){
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = PCAppendOptionsPrefix((PC)PetscToPointer(pc),t);
  FREECHAR(prefix,t);
}

void pcdestroy_(PC pc, int *__ierr ){
  *__ierr = PCDestroy((PC)PetscToPointer(pc));
  PetscRmPointer(pc);
}

void pccreate_(MPI_Comm comm,PC *newpc, int *__ierr ){
  PC p;
  *__ierr = PCCreate((MPI_Comm)PetscToPointerComm( *(int*)(comm) ),&p);
  *(PetscFortranAddr*) newpc = PetscFromPointer(p);
}

void pcregisterdestroy_(int *__ierr){
  *__ierr = PCRegisterDestroy();
}

void pcgettype_(PC pc,CHAR name,int *__ierr,int len)
{
  char *tname;

  *__ierr = PCGetType((PC)PetscToPointer(pc),&tname);
#if defined(USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  PetscStrncpy(t,tname,len1);
  }
#else
  PetscStrncpy(name,tname,len);
#endif
}

#if defined(__cplusplus)
}
#endif
