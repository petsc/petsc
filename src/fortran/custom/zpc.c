/*$Id: zpc.c,v 1.30 1999/10/04 22:51:03 balay Exp bsmith $*/

#include "src/fortran/custom/zpetsc.h"
#include "sles.h"
#include "mg.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
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
#define pcnullspaceattach_         PCNULLSPACEATTACH
#define pcnullspacecreate_         PCNULLSPACECREATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcnullspacecreate_         pcnullspacecreate
#define pcnullspaceattach_         pcnullspaceattach
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

void PETSC_STDCALL pcnullspacecreate_(MPI_Comm comm,int *has_cnst,int *n,Vec *vecs,PCNullSpace *SP,int *ierr)
{
  *ierr = PCNullSpaceCreate((MPI_Comm)PetscToPointerComm(*(int*)(comm)),*has_cnst, *n, vecs,SP);
}

void PETSC_STDCALL pcnullspaceattach_(PC *pc,PCNullSpace *nullsp,int *__ierr)
{
  *__ierr = PCNullSpaceAttach(*pc,*nullsp);
}

void PETSC_STDCALL pcsettype_(PC *pc,CHAR type PETSC_MIXED_LEN(len), int *__ierr PETSC_END_LEN(len) )
{
  char *t;

  FIXCHAR(type,len,t);
  *__ierr = PCSetType(*pc,t);
  FREECHAR(type,t);
}


static void (*f1)(void *,Vec*,Vec*,int*);
static int ourshellapply(void *ctx,Vec x,Vec y)
{
  int              ierr = 0;
  (*f1)(ctx,&x,&y,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL pcshellsetapply_(PC *pc,void (*apply)(void*,Vec *,Vec *,int*),void *ptr,
                      int *__ierr )
{
  f1 = apply;
  *__ierr = PCShellSetApply(*pc,ourshellapply,ptr);
}

static void (*f9)(void *,int*);
static int ourshellsetup(void *ctx)
{
  int              ierr = 0;

  (*f9)(ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL pcshellsetsetup_(PC *pc,void (*setup)(void*,int*), int *__ierr )
{
  f9 = setup;
  *__ierr = PCShellSetSetUp(*pc,ourshellsetup);
}

/* -----------------------------------------------------------------*/
static void (*f2)(void*,Vec*,Vec*,Vec*,int*,int*);
static int ourapplyrichardson(void *ctx,Vec x,Vec y,Vec w,int m)
{
  int              ierr = 0;

  (*f2)(ctx,&x,&y,&w,&m,&ierr);CHKERRQ(ierr);
  return 0;
}

void PETSC_STDCALL pcshellsetapplyrichardson_(PC *pc,
         void (*apply)(void*,Vec *,Vec *,Vec *,int*,int*),
         void *ptr, int *__ierr )
{
  f2 = apply;
  *__ierr = PCShellSetApplyRichardson(*pc,ourapplyrichardson,ptr);
}

void PETSC_STDCALL mggetcoarsesolve_(PC *pc,SLES *sles, int *__ierr )
{
  *__ierr = MGGetCoarseSolve(*pc,sles);
}

void PETSC_STDCALL mggetsmoother_(PC *pc,int *l,SLES *sles, int *__ierr )
{
  *__ierr = MGGetSmoother(*pc,*l,sles);
}

void PETSC_STDCALL mggetsmootherup_(PC *pc,int *l,SLES *sles, int *__ierr )
{
  *__ierr = MGGetSmootherUp(*pc,*l,sles);
}

void PETSC_STDCALL mggetsmootherdown_(PC *pc,int *l,SLES *sles, int *__ierr )
{
  *__ierr = MGGetSmootherDown(*pc,*l,sles);
}

void PETSC_STDCALL pcbjacobigetsubsles_(PC *pc,int *n_local,int *first_local,SLES *sles,int *__ierr )
{
  SLES *tsles;
  int  i;
  *__ierr = PCBJacobiGetSubSLES(*pc,n_local,first_local,&tsles);
  for ( i=0; i<*n_local; i++ ){
    sles[i] = tsles[i];
  }
}

void PETSC_STDCALL pcasmgetsubsles_(PC *pc,int *n_local,int *first_local,SLES *sles,int *__ierr )
{
  SLES *tsles;
  int  i;
  *__ierr = PCASMGetSubSLES(*pc,n_local,first_local,&tsles);
  for ( i=0; i<*n_local; i++ ){
    sles[i] = tsles[i];
  }
}

void PETSC_STDCALL pcgetoperators_(PC *pc,Mat *mat,Mat *pmat,MatStructure *flag, int *__ierr)
{
  if (FORTRANNULLINTEGER(flag)) flag = PETSC_NULL;
  if (FORTRANNULLOBJECT(mat))   mat = PETSC_NULL;
  if (FORTRANNULLOBJECT(pmat))  pmat = PETSC_NULL;
  *__ierr = PCGetOperators(*pc,mat,pmat,flag);
}

void PETSC_STDCALL pcgetfactoredmatrix_(PC *pc,Mat *mat, int *__ierr )
{
  *__ierr = PCGetFactoredMatrix(*pc,mat);
}
 
void PETSC_STDCALL pcsetoptionsprefix_(PC *pc,CHAR prefix PETSC_MIXED_LEN(len),
                                       int *__ierr PETSC_END_LEN(len) )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = PCSetOptionsPrefix(*pc,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL pcappendoptionsprefix_(PC *pc,CHAR prefix PETSC_MIXED_LEN(len),
                                          int *__ierr PETSC_END_LEN(len) )
{
  char *t;

  FIXCHAR(prefix,len,t);
  *__ierr = PCAppendOptionsPrefix(*pc,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL pcdestroy_(PC *pc, int *__ierr )
{
  *__ierr = PCDestroy(*pc);
}

void PETSC_STDCALL pccreate_(MPI_Comm comm,PC *newpc, int *__ierr )
{
  *__ierr = PCCreate((MPI_Comm)PetscToPointerComm( *(int*)(comm) ),newpc);
}

void PETSC_STDCALL pcregisterdestroy_(int *__ierr)
{
  *__ierr = PCRegisterDestroy();
}

void PETSC_STDCALL pcgettype_(PC *pc,CHAR name PETSC_MIXED_LEN(len),int *__ierr PETSC_END_LEN(len) )
{
  char *tname;

  *__ierr = PCGetType(*pc,&tname);
#if defined(USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  *__ierr = PetscStrncpy(t,tname,len1); if (*__ierr) return;
  }
#else
  *__ierr = PetscStrncpy(name,tname,len);if (*__ierr) return;
#endif
}

void PETSC_STDCALL pcgetoptionsprefix_(PC *pc, CHAR prefix PETSC_MIXED_LEN(len),
                                       int *__ierr PETSC_END_LEN(len) )
{
  char *tname;

  *__ierr = PCGetOptionsPrefix(*pc,&tname);
#if defined(USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *__ierr = PetscStrncpy(t,tname,len1);if (*__ierr) return;
  }
#else
  *__ierr = PetscStrncpy(prefix,tname,len);if (*__ierr) return;
#endif
}



EXTERN_C_END

