#ifndef lint
static char vcid[] = "$Id: zpc.c,v 1.7 1996/01/15 21:41:08 balay Exp balay $";
#endif

#include "zpetsc.h"
#include "sles.h"
#include "mg.h"
#include "pinclude/petscfix.h"

#ifdef FORTRANCAPS
#define pcregisterall_             PCREGISTERALL
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
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define pcregisterall_             pcregisterall
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
#endif

static void (*f1)(void *,int*,int*,int*);
static int ourshellapply(void *ctx,Vec x,Vec y)
{
  int ierr = 0, s1, s2;
  s1 = MPIR_FromPointer(x);
  s2 = MPIR_FromPointer(y);
  (*f1)(ctx,&s1,&s2,&ierr); CHKERRQ(ierr);
  MPIR_RmPointer(s1);
  MPIR_RmPointer(s2); 
  return 0;
}
void pcshellsetapply_(PC pc,void (*apply)(void*,int*,int*,int*),void *ptr,
                      int *__ierr ){
  f1 = apply;
  *__ierr = PCShellSetApply(
	(PC)MPIR_ToPointer( *(int*)(pc) ),ourshellapply,ptr);
}
/* -----------------------------------------------------------------*/
static void (*f2)(void*,int*,int*,int*,int*,int*);
static int ourapplyrichardson(void *ctx,Vec x,Vec y,Vec w,int m)
{
  int ierr = 0, s1,s2,s3;
  s1 = MPIR_FromPointer(x);
  s2 = MPIR_FromPointer(y);
  s3 = MPIR_FromPointer(w);
  (*f2)(ctx,&s1,&s2,&s3,&m,&ierr); CHKERRQ(ierr);
  MPIR_RmPointer(s1);
  MPIR_RmPointer(s2); 
  MPIR_RmPointer(s3); 
  return 0;
}

void pcshellsetapplyrichardson_(PC pc,
                        void (*apply)(void*,int*,int*,int*,int*,int*),
                              void *ptr, int *__ierr ){
  f2 = apply;
  *__ierr = PCShellSetApplyRichardson(
	(PC)MPIR_ToPointer( *(int*)(pc) ),ourapplyrichardson,ptr);
}

void mggetcoarsesolve_(PC pc,SLES *sles, int *__ierr ){
  SLES asles;
  *__ierr = MGGetCoarseSolve((PC)MPIR_ToPointer( *(int*)(pc) ),&asles);
  *(int*) sles = MPIR_FromPointer(asles);
}
void mggetsmoother_(PC pc,int *l,SLES *sles, int *__ierr ){
  SLES asles;
  *__ierr = MGGetSmoother((PC)MPIR_ToPointer( *(int*)(pc) ),*l,&asles);
  *(int*) sles = MPIR_FromPointer(asles);
}
void mggetsmootherup_(PC pc,int *l,SLES *sles, int *__ierr ){
  SLES asles;
  *__ierr = MGGetSmootherUp((PC)MPIR_ToPointer( *(int*)(pc) ),*l,&asles);
  *(int*) sles = MPIR_FromPointer(asles);
}
void mggetsmootherdown_(PC pc,int *l,SLES *sles, int *__ierr ){
  SLES asles;
  *__ierr = MGGetSmootherDown((PC)MPIR_ToPointer( *(int*)(pc) ),*l,&asles);
  *(int*) sles = MPIR_FromPointer(asles);
}

void pcbjacobigetsubsles_(PC pc,int *n_local,int *first_local,int *sles, 
                          int *__ierr ){
  SLES *tsles;
  int  i;
  *__ierr = PCBJacobiGetSubSLES(
	(PC)MPIR_ToPointer( *(int*)(pc) ),n_local,first_local,&tsles);
  for ( i=0; i<*n_local; i++ ){
    sles[i] = MPIR_FromPointer(tsles[i]);
  }
}

void pcgetoperators_(PC pc,Mat *mat,Mat *pmat,MatStructure *flag, int *__ierr){
  Mat m,p;
  *__ierr = PCGetOperators((PC)MPIR_ToPointer( *(int*)(pc) ),&m,&p,flag);
  *(int*) mat = MPIR_FromPointer(m);
  *(int*) pmat = MPIR_FromPointer(p);
}

void pcgetfactoredmatrix_(PC pc,Mat *mat, int *__ierr ){
  Mat m;
  *__ierr = PCGetFactoredMatrix((PC)MPIR_ToPointer( *(int*)(pc) ),&m);
  *(int*) mat = MPIR_FromPointer(m);
}
 
void pcsetoptionsprefix_(PC pc,char *prefix, int *flg, int *__ierr,int len ){
  char *t=0;
  if (prefix[len] != 0) {
    t = (char *) PetscMalloc( (len+1)*sizeof(char) ); 
    PetscStrncpy(t,prefix,len);
    t[len] = 0;
  }
  else t = prefix;
  *__ierr = PCSetOptionsPrefix((PC)MPIR_ToPointer( *(int*)(pc) ),t, flg);
  if( t != prefix) PetscFree(t);
}

void pcappendoptionsprefix_(PC pc,char *prefix, int *flg, int *__ierr,int len ){
  char *t=0;
  if (prefix[len] != 0) {
    t = (char *) PetscMalloc( (len+1)*sizeof(char) ); 
    PetscStrncpy(t,prefix,len);
    t[len] = 0;
  }
  else t = prefix;
  *__ierr = PCAppendOptionsPrefix((PC)MPIR_ToPointer( *(int*)(pc) ),t, flg);
  if( t != prefix) PetscFree(t);
}

void pcdestroy_(PC pc, int *__ierr ){
  *__ierr = PCDestroy((PC)MPIR_ToPointer( *(int*)(pc) ));
  MPIR_RmPointer( *(int*)(pc) );
}

void pccreate_(MPI_Comm comm,PC *newpc, int *__ierr ){
  PC p;
  *__ierr = PCCreate((MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),&p);
  *(int*) newpc = MPIR_FromPointer(p);
}

void pcregisterdestroy_(int *__ierr){
  *__ierr = PCRegisterDestroy();
}

void pcregisterall_(int *__ierr){
  *__ierr = PCRegisterAll();
}

void pcgettype_(PC pc,PCType *type,char *name,int *__ierr,int len)
{
  char *tname;
  if (type == PETSC_NULL_Fortran) type = PETSC_NULL;
  *__ierr = PCGetType((PC)MPIR_ToPointer(*(int*)pc),type,&tname);
  if (name != PETSC_NULL_Fortran) PetscStrncpy(name,tname,len);
}
