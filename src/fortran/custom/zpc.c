
#include "src/fortran/custom/zpetsc.h"
#include "petscksp.h"
#include "petscmg.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define pcmgdefaultresidual_       PCMGDEFAULTRESIDUAL
#define pcmgsetresidual_           PCMGSETRESIDUAL
#define pcasmsetlocalsubdomains_   PCASMSETLOCALSUBDOMAINS
#define pcasmsetglobalsubdomains_  PCASMSETGLOBALSUBDOMAINS
#define pcasmgetlocalsubmatrices_  PCASMGETLOCALSUBMATRICES
#define pcasmgetlocalsubdomains_   PCASMGETLOCALSUBDOMAINS
#define pcregisterdestroy_         PCREGISTERDESTROY
#define pcdestroy_                 PCDESTROY
#define pccreate_                  PCCREATE
#define pcgetoperators_            PCGETOPERATORS
#define pcgetfactoredmatrix_       PCGETFACTOREDMATRIX
#define pcsetoptionsprefix_        PCSETOPTIONSPREFIX
#define pcappendoptionsprefix_     PCAPPENDOPTIONSPREFIX
#define pcbjacobigetsubksp_        PCBJACOBIGETSUBKSP
#define pcasmgetsubksp_            PCASMGETSUBKSP
#define pcmggetcoarsesolve_        PCMGGETCOARSESOLVE
#define pcmggetsmoother_           PCMGGETSMOOTHER
#define pcmggetsmootherup_         PCMGGETSMOOTHERUP
#define pcmggetsmootherdown_       PCMGGETSMOOTHERDOWN
#define pcshellsetapply_           PCSHELLSETAPPLY
#define pcshellsetapplytranspose_  PCSHELLSETAPPLYTRANSPOSE
#define pcshellsetapplyrichardson_ PCSHELLSETAPPLYRICHARDSON
#define pcgettype_                 PCGETTYPE
#define pcsettype_                 PCSETTYPE
#define pcgetoptionsprefix_        PCGETOPTIONSPREFIX
#define matnullspacecreate_        MATNULLSPACECREATE
#define pcview_                    PCVIEW
#define pcmgsetlevels_             PCMGSETLEVELS
#define pccompositesettype_        PCCOMPOSITESETTYPE
#define pccompositeaddpc_          PCCOMPOSITEADDPC
#define pccompositegetpc_          PCCOMPOSITEGETPC
#define pccompositespecialsetalpha_  PCCOMPOSITESETALPHA
#define pcshellsetsetup_           PCSHELLSETSETUP
#define pcilusetmatordering_       PCILUSETMATORDERING
#define pclusetmatordering_        PCLUSETMATORDERING
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define pcmgdefaultresidual_       pcmgdefaultresidual
#define pcmgsetresidual_           pcmgsetresidual
#define pcasmsetlocalsubdomains_   pcasmsetlocalsubdomains
#define pcasmsetglobalsubdomains_  pcasmsetglobalsubdomains
#define pcasmgetlocalsubmatrices_  pcasmgetlocalsubmatrices
#define pcasmgetlocalsubdomains_   pcasmgetlocalsubdomains
#define matnullspacecreate_        matnullspacecreate
#define pcregisterdestroy_         pcregisterdestroy
#define pcdestroy_                 pcdestroy
#define pccreate_                  pccreate
#define pcgetoperators_            pcgetoperators
#define pcgetfactoredmatrix_       pcgetfactoredmatrix
#define pcsetoptionsprefix_        pcsetoptionsprefix
#define pcappendoptionsprefix_     pcappendoptionsprefix
#define pcbjacobigetsubksp_        pcbjacobigetsubksp
#define pcasmgetsubksp_            pcasmgetsubksp
#define pcmggetcoarsesolve_        pcmggetcoarsesolve
#define pcmggetsmoother_           pcmggetsmoother
#define pcmggetsmootherup_         pcmggetsmootherup
#define pcmggetsmootherdown_       pcmggetsmootherdown
#define pcshellsetapplyrichardson_ pcshellsetapplyrichardson
#define pcshellsetapply_           pcshellsetapply
#define pcshellsetapplytranspose_  pcshellsetapplytranspose
#define pcgettype_                 pcgettype
#define pcsettype_                 pcsettype
#define pcgetoptionsprefix_        pcgetoptionsprefix
#define pcview_                    pcview
#define pcmgsetlevels_             pcmgsetlevels
#define pccompositesettype_        pccompositesettype
#define pccompositeaddpc_          pccompositeaddpc
#define pccompositegetpc_          pccompositegetpc
#define pccompositespecialsetalpha_  pccompositespecialsetalpha
#define pcshellsetsetup_           pcshellsetsetup
#define pcilusetmatordering_       pcilusetmatordering
#define pclusetmatordering_        pclusetmatordering
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f2)(void*,Vec*,Vec*,Vec*,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscErrorCode*);
static void (PETSC_STDCALL *f1)(void*,Vec*,Vec*,PetscErrorCode*);
static void (PETSC_STDCALL *f3)(void*,Vec*,Vec*,PetscErrorCode*);
static void (PETSC_STDCALL *f9)(void*,PetscErrorCode*);
EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourapplyrichardson(void *ctx,Vec x,Vec y,Vec w,PetscReal rtol,PetscReal abstol,PetscReal dtol,PetscInt m)
{
  PetscErrorCode ierr = 0;

  (*f2)(ctx,&x,&y,&w,&rtol,&abstol,&dtol,&m,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourshellapply(void *ctx,Vec x,Vec y)
{
  PetscErrorCode ierr = 0;
  (*f1)(ctx,&x,&y,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourshellapplytranspose(void *ctx,Vec x,Vec y)
{
  PetscErrorCode ierr = 0;
  (*f3)(ctx,&x,&y,&ierr);CHKERRQ(ierr);
  return 0;
}

static PetscErrorCode ourshellsetup(void *ctx)
{
  PetscErrorCode ierr = 0;

  (*f9)(ctx,&ierr);CHKERRQ(ierr);
  return 0;
}

typedef PetscErrorCode (*MVVVV)(Mat,Vec,Vec,Vec);
static PetscErrorCode ourresidualfunction(Mat mat,Vec b,Vec x,Vec R)
{
  PetscErrorCode ierr = 0;
  (*(void (PETSC_STDCALL *)(Mat*,Vec*,Vec*,Vec*,PetscErrorCode*))(((PetscObject)mat)->fortran_func_pointers[0]))(&mat,&b,&x,&R,&ierr);
  return 0;
}

EXTERN_C_BEGIN
void PETSC_STDCALL pccompositespecialsetalpha_(PC *pc,PetscScalar *alpha,PetscErrorCode *ierr)
{
  *ierr = PCCompositeSpecialSetAlpha(*pc,*alpha);
}

void PETSC_STDCALL pccompositesettype_(PC *pc,PCCompositeType *type,PetscErrorCode *ierr)
{
  *ierr = PCCompositeSetType(*pc,*type);
}

void PETSC_STDCALL pccompositeaddpc_(PC *pc,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PCCompositeAddPC(*pc,t);
  FREECHAR(type,t);
}

void PETSC_STDCALL pccompositegetpc_(PC *pc,PetscInt *n,PC *subpc,PetscErrorCode *ierr)
{
  *ierr = PCCompositeGetPC(*pc,*n,subpc);
}

void PETSC_STDCALL pcmgsetlevels_(PC *pc,PetscInt *levels,MPI_Comm *comms, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(comms);
  *ierr = PCMGSetLevels(*pc,*levels,comms);
}

void PETSC_STDCALL pcview_(PC *pc,PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PCView(*pc,v);
}

void PETSC_STDCALL matnullspacecreate_(MPI_Comm *comm,PetscTruth *has_cnst,PetscInt *n,Vec *vecs,MatNullSpace *SP,PetscErrorCode *ierr)
{
  *ierr = MatNullSpaceCreate((MPI_Comm)PetscToPointerComm(*comm),*has_cnst,*n,vecs,SP);
}

void PETSC_STDCALL pcsettype_(PC *pc,CHAR type PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type,len,t);
  *ierr = PCSetType(*pc,t);
  FREECHAR(type,t);
}


void PETSC_STDCALL pcshellsetapply_(PC *pc,void (PETSC_STDCALL *apply)(void*,Vec *,Vec *,PetscErrorCode*),
                                    PetscErrorCode *ierr)
{
  f1 = apply;
  *ierr = PCShellSetApply(*pc,ourshellapply);
}

void PETSC_STDCALL pcshellsetapplytranspose_(PC *pc,void (PETSC_STDCALL *applytranspose)(void*,Vec *,Vec *,PetscErrorCode*),
                                             PetscErrorCode *ierr)
{
  f3 = applytranspose;
  *ierr = PCShellSetApplyTranspose(*pc,ourshellapplytranspose);
}


void PETSC_STDCALL pcshellsetsetup_(PC *pc,void (PETSC_STDCALL *setup)(void*,PetscErrorCode*),PetscErrorCode *ierr)
{
  f9 = setup;
  *ierr = PCShellSetSetUp(*pc,ourshellsetup);
}

/* -----------------------------------------------------------------*/

void PETSC_STDCALL pcshellsetapplyrichardson_(PC *pc,
         void (PETSC_STDCALL *apply)(void*,Vec *,Vec *,Vec *,PetscReal*,PetscReal*,PetscReal*,PetscInt*,PetscErrorCode*),
         PetscErrorCode *ierr)
{
  f2 = apply;
  *ierr = PCShellSetApplyRichardson(*pc,ourapplyrichardson);
}

void PETSC_STDCALL pcmggetcoarsesolve_(PC *pc,KSP *ksp,PetscErrorCode *ierr)
{
  *ierr = PCMGGetCoarseSolve(*pc,ksp);
}

void PETSC_STDCALL pcmggetsmoother_(PC *pc,PetscInt *l,KSP *ksp,PetscErrorCode *ierr)
{
  *ierr = PCMGGetSmoother(*pc,*l,ksp);
}

void PETSC_STDCALL pcmggetsmootherup_(PC *pc,PetscInt *l,KSP *ksp,PetscErrorCode *ierr)
{
  *ierr = PCMGGetSmootherUp(*pc,*l,ksp);
}

void PETSC_STDCALL pcmggetsmootherdown_(PC *pc,PetscInt *l,KSP *ksp,PetscErrorCode *ierr)
{
  *ierr = PCMGGetSmootherDown(*pc,*l,ksp);
}

void PETSC_STDCALL pcbjacobigetsubksp_(PC *pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp,PetscErrorCode *ierr)
{
  KSP *tksp;
  PetscInt  i,nloc;
  CHKFORTRANNULLINTEGER(n_local);
  CHKFORTRANNULLINTEGER(first_local);
  *ierr = PCBJacobiGetSubKSP(*pc,&nloc,first_local,&tksp);
  if (n_local) *n_local = nloc;
  for (i=0; i<nloc; i++){
    ksp[i] = tksp[i];
  }
}

void PETSC_STDCALL pcasmgetsubksp_(PC *pc,PetscInt *n_local,PetscInt *first_local,KSP *ksp,PetscErrorCode *ierr)
{
  KSP *tksp;
  PetscInt  i,nloc;
  CHKFORTRANNULLINTEGER(n_local);
  CHKFORTRANNULLINTEGER(first_local);
  *ierr = PCASMGetSubKSP(*pc,&nloc,first_local,&tksp);
  if (n_local) *n_local = nloc;
  for (i=0; i<nloc; i++){
    ksp[i] = tksp[i];
  }
}

void PETSC_STDCALL pcgetoperators_(PC *pc,Mat *mat,Mat *pmat,MatStructure *flag,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(mat);
  CHKFORTRANNULLOBJECT(pmat)
  *ierr = PCGetOperators(*pc,mat,pmat,flag);
}

void PETSC_STDCALL pcgetfactoredmatrix_(PC *pc,Mat *mat,PetscErrorCode *ierr)
{
  *ierr = PCGetFactoredMatrix(*pc,mat);
}
 
void PETSC_STDCALL pcsetoptionsprefix_(PC *pc,CHAR prefix PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PCSetOptionsPrefix(*pc,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL pcappendoptionsprefix_(PC *pc,CHAR prefix PETSC_MIXED_LEN(len),
                                          PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(prefix,len,t);
  *ierr = PCAppendOptionsPrefix(*pc,t);
  FREECHAR(prefix,t);
}

void PETSC_STDCALL pcdestroy_(PC *pc,PetscErrorCode *ierr)
{
  *ierr = PCDestroy(*pc);
}

void PETSC_STDCALL pccreate_(MPI_Comm *comm,PC *newpc,PetscErrorCode *ierr)
{
  *ierr = PCCreate((MPI_Comm)PetscToPointerComm(*comm),newpc);
}

void PETSC_STDCALL pcregisterdestroy_(PetscErrorCode *ierr)
{
  *ierr = PCRegisterDestroy();
}

void PETSC_STDCALL pcgettype_(PC *pc,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = PCGetType(*pc,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  *ierr = PetscStrncpy(t,tname,len1); if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(name,tname,len);if (*ierr) return;
#endif
  FIXRETURNCHAR(name,len);

}

void PETSC_STDCALL pcgetoptionsprefix_(PC *pc,CHAR prefix PETSC_MIXED_LEN(len),
                                       PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *tname;

  *ierr = PCGetOptionsPrefix(*pc,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
    char *t = _fcdtocp(prefix); int len1 = _fcdlen(prefix);
    *ierr = PetscStrncpy(t,tname,len1);if (*ierr) return;
  }
#else
  *ierr = PetscStrncpy(prefix,tname,len);if (*ierr) return;
#endif
}

void PETSC_STDCALL pcasmsetlocalsubdomains_(PC *pc,PetscInt *n,IS *is, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(is);
  *ierr = PCASMSetLocalSubdomains(*pc,*n,is);
}

void PETSC_STDCALL pcasmsettotalsubdomains_(PC *pc,PetscInt *N,IS *is, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(is);
  *ierr = PCASMSetTotalSubdomains(*pc,*N,is);
}

void PETSC_STDCALL pcasmgetlocalsubmatrices_(PC *pc,PetscInt *n,Mat *mat, PetscErrorCode *ierr)
{
  PetscInt nloc,i;
  Mat  *tmat;
  CHKFORTRANNULLOBJECT(mat);
  CHKFORTRANNULLINTEGER(n);
  *ierr = PCASMGetLocalSubmatrices(*pc,&nloc,&tmat);
  if (n) *n = nloc;
  if (mat) {
    for (i=0; i<nloc; i++){
      mat[i] = tmat[i];
    }
  }
}
void PETSC_STDCALL pcasmgetlocalsubdomains_(PC *pc,PetscInt *n,IS *is, PetscErrorCode *ierr)
{
  PetscInt nloc,i;
  IS  *tis;
  CHKFORTRANNULLOBJECT(is);
  CHKFORTRANNULLINTEGER(n);
  *ierr = PCASMGetLocalSubdomains(*pc,&nloc,&tis);
  if (n) *n = nloc;
  if (is) {
    for (i=0; i<nloc; i++){
      is[i] = tis[i];
    }
  }
}

void pcmgdefaultresidual_(Mat *mat,Vec *b,Vec *x,Vec *r, PetscErrorCode *ierr)
{
  *ierr = PCMGDefaultResidual(*mat,*b,*x,*r);
}

void PETSC_STDCALL pcmgsetresidual_(PC *pc,PetscInt *l,PetscErrorCode (*residual)(Mat*,Vec*,Vec*,Vec*,PetscErrorCode*),Mat *mat, PetscErrorCode *ierr)
{
  MVVVV rr;
  if ((FCNVOID)residual == (FCNVOID)pcmgdefaultresidual_) rr = PCMGDefaultResidual;
  else {
    if (!((PetscObject)*mat)->fortran_func_pointers) {
      *ierr = PetscMalloc(1*sizeof(void*),&((PetscObject)*mat)->fortran_func_pointers);
    }
    ((PetscObject)*mat)->fortran_func_pointers[0] = (FCNVOID)residual;
    rr = ourresidualfunction;
  }
  *ierr = PCMGSetResidual(*pc,*l,rr,*mat);
}

void PETSC_STDCALL pcilusetmatordering_(PC *pc,CHAR ordering PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len)){
  char *t;

    FIXCHAR(ordering,len,t);
    *ierr = PCILUSetMatOrdering(*pc,t);
    FREECHAR(ordering,t);
}

void PETSC_STDCALL pclusetmatordering_(PC *pc,CHAR ordering PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len)){
  char *t;

    FIXCHAR(ordering,len,t);
    *ierr = PCLUSetMatOrdering(*pc,t);
    FREECHAR(ordering,t);
}
EXTERN_C_END

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define pchypresettype_ PCHYPRESETTYPE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define pchypresettype_ pchypresettype
#endif

#if defined(PETSC_HAVE_HYPRE) && !defined(PETSC_USE_COMPLEX)

#if defined(__cplusplus)
extern "C" {
#endif
void PETSC_STDCALL  pchypresettype_(PC *pc, CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len) )
{
  char *t;
  FIXCHAR(name,len,t);
  *ierr = PCHYPRESetType(*pc,t);
  FREECHAR(name,t);
}
#if defined(__cplusplus)
}
#endif

#endif


