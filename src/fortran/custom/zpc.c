#ifndef lint
static char vcid[] = "$Id: zoptions.c,v 1.1 1995/08/21 19:56:20 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "sles.h"
#include "mg.h"
#if defined(HAVE_STRING_H)
#include <string.h>
#endif
#include "pinclude/petscfix.h"

#ifdef FORTRANCAPS
#define pcregisterdestroy_   PCREGISTERDESTROY
#define pcdestroy_           PCDESTROY
#define pccreate_            PCCREATE
#define pcgetoperators_      PCGETOPERATORS
#define pcgetfactoredmatrix_ PCGETFACTOREDMATRIX
#define pcsetoptionsprefix_  PCSETOPTIONSPREFIX
#define pcgetmethodfromcontext_ PCGETMETHODFROMCONTEXT
#define pcbjacobigetsubsles_ PCBJACOBIGETSUBSLES
#define mggetcoarsesolve_    MGGETCOARSESOLVE
#define mggetsmoother_       MGGETSMOOTHER
#define mggetsmootherup_     MGGETSMOOTHERUP
#define mggetsmootherdown_   MGGETSMOOTHERDOWN
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define pcregisterdestroy_   pcregisterdestroy
#define pcdestroy_           pcdestroy
#define pccreate_            pccreate
#define pcgetoperators_      pcgetoperators
#define pcgetfactoredmatrix_ pcgetfactoredmatrix
#define pcsetoptionsprefix_  pcsetoptionsprefix
#define pcgetmethodfromcontext_ pcgetmethodfromcontext
#define pcbjacobigetsubsles_ pcbjacobigetsubsles
#define mggetcoarsesolve_    mggetcoarsesolve
#define mggetsmoother_       mggetsmoother
#define mggetsmootherup_     mggetsmootherup
#define mggetsmootherdown_   mggetsmootherdown
#endif

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
void pcgetmethodfromcontext_(PC pc,PCMethod *method, int *__ierr ){
  *__ierr = PCGetMethodFromContext((PC)MPIR_ToPointer( *(int*)(pc) ),method);
}
void pcgetfactoredmatrix_(PC pc,Mat *mat, int *__ierr ){
  Mat m;
  *__ierr = PCGetFactoredMatrix((PC)MPIR_ToPointer( *(int*)(pc) ),&m);
  *(int*) mat = MPIR_FromPointer(m);
}
void pcsetoptionsprefix_(PC pc,char *prefix, int *__ierr,int len ){
  char *t;
  if (prefix[len] != 0) {
    t = (char *) PETSCMALLOC( (len+1)*sizeof(char) ); 
    strncpy(t,prefix,len);
    t[len] = 0;
  }
  else t = prefix;
  *__ierr = PCSetOptionsPrefix((PC)MPIR_ToPointer( *(int*)(pc) ),t);
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
