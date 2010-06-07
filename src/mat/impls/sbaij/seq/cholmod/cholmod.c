#define PETSCMAT_DLL

/*
   Provides an interface to the CHOLMOD 1.7.1 sparse solver

   When build with PETSC_USE_64BIT_INDICES this will use UF_Long as the
   integer type in UMFPACK, otherwise it will use int. This means
   all integers in this file as simply declared as PetscInt. Also it means
   that UMFPACK UL_Long version MUST be built with 64 bit integers when used.

*/
#include "../src/mat/impls/sbaij/seq/sbaij.h"

#if defined(PETSC_USE_COMPLEX)
#  define CHOLMOD_SCALAR_TYPE       CHOLMOD_COMPLEX
#else
#  define CHOLMOD_SCALAR_TYPE       CHOLMOD_REAL
#endif

#if defined(PETSC_USE_64BIT_INDICES)
#  define CHOLMOD_INT_TYPE          CHOLMOD_LONG
#  define cholmod_X_start           cholmod_l_start
#  define cholmod_X_analyze         cholmod_l_analyze
#  define cholmod_X_analyze_p       cholmod_l_analyze_p
#  define cholmod_X_factorize       cholmod_l_factorize
#  define cholmod_X_finish          cholmod_l_finish
#  define cholmod_X_free_factor     cholmod_l_free_factor
#  define cholmod_X_free_dense      cholmod_l_free_dense
#  define cholmod_X_solve           cholmod_l_solve
#else
#  define CHOLMOD_INT_TYPE          CHOLMOD_INT
#  define cholmod_X_start           cholmod_start
#  define cholmod_X_analyze         cholmod_analyze
#  define cholmod_X_analyze_p       cholmod_analyze_p
#  define cholmod_X_factorize       cholmod_factorize
#  define cholmod_X_finish          cholmod_finish
#  define cholmod_X_free_factor     cholmod_free_factor
#  define cholmod_X_free_dense      cholmod_free_dense
#  define cholmod_X_solve           cholmod_solve
#endif

#define UF_long long long
#define UF_long_max LONG_LONG_MAX
#define UF_long_id "%lld"
#undef I  /* complex.h defines I=_Complex_I, but cholmod_core.h uses I as a field member */

EXTERN_C_BEGIN
#include <cholmod.h>
EXTERN_C_END

typedef struct {
  cholmod_sparse *matrix;
  cholmod_factor *factor;
  cholmod_common *common;
} Mat_CHOLMOD;

#undef __FUNCT__
#define __FUNCT__ "CholmodStart"
static PetscErrorCode CholmodStart(Mat F)
{
  PetscErrorCode ierr;
  Mat_CHOLMOD    *chol=(Mat_CHOLMOD*)F->spptr;
  cholmod_common *c;
  PetscTruth     flg;

  PetscFunctionBegin;
  if (chol->common) PetscFunctionReturn(0);
  ierr = PetscMalloc(sizeof(*chol->common),&chol->common);CHKERRQ(ierr);
  ierr = !cholmod_X_start(chol->common);CHKERRQ(ierr);
  c = chol->common;
#define CHOLMOD_OPTION_DOUBLE(name,help) do {                            \
    PetscReal tmp = (PetscReal)c->name;                                  \
    ierr = PetscOptionsReal("-mat_cholmod_" #name,help,"None",tmp,&tmp,0);CHKERRQ(ierr); \
    c->name = (double)tmp;                                               \
  } while (0)
#define CHOLMOD_OPTION_INT(name,help) do {                               \
    PetscInt tmp = (PetscInt)c->name;                                    \
    ierr = PetscOptionsInt("-mat_cholmod_" #name,help,"None",tmp,&tmp,0);CHKERRQ(ierr); \
    c->name = (int)tmp;                                                  \
  } while (0)
#define CHOLMOD_OPTION_SIZE_T(name,help) do {                            \
    PetscInt tmp = (PetscInt)c->name;                                    \
    ierr = PetscOptionsInt("-mat_cholmod_" #name,help,"None",tmp,&tmp,0);CHKERRQ(ierr); \
    if (tmp < 0) SETERRQ(((PetscObject)F)->comm,PETSC_ERR_ARG_OUTOFRANGE,"value must be positive"); \
    c->name = (size_t)tmp;                                               \
  } while (0)
#define CHOLMOD_OPTION_TRUTH(name,help) do {                             \
    PetscTruth tmp = (PetscTruth)!!c->name;                              \
    ierr = PetscOptionsTruth("-mat_cholmod_" #name,help,"None",tmp,&tmp,0);CHKERRQ(ierr); \
    c->name = (int)tmp;                                                  \
  } while (0)

  ierr = PetscOptionsBegin(((PetscObject)F)->comm,((PetscObject)F)->prefix,"CHOLMOD Options","Mat");CHKERRQ(ierr);
  CHOLMOD_OPTION_DOUBLE(dbound,"Minimum absolute value of diagonal entries of D");
  CHOLMOD_OPTION_DOUBLE(grow0,"Global growth ratio when factors are modified");
  CHOLMOD_OPTION_DOUBLE(grow1,"Column growth ratio when factors are modified");
  CHOLMOD_OPTION_SIZE_T(grow2,"Affine column growth constant when factors are modified");
  CHOLMOD_OPTION_SIZE_T(maxrank,"Max rank of update, larger values are faster but use more memory [2,4,8]");
  {
    static const char *const list[] = {"SIMPLICIAL","AUTO","SUPERNODAL","MatCholmodFactorType","MAT_CHOLMOD_FACTOR_",0};
    PetscEnum choice = (PetscEnum)c->supernodal;
    ierr = PetscOptionsEnum("-mat_cholmod_factor","Factorization method","None",list,(PetscEnum)c->supernodal,&choice,0);CHKERRQ(ierr);
    c->supernodal = (int)choice;
  }
  if (c->supernodal) CHOLMOD_OPTION_DOUBLE(supernodal_switch,"flop/nnz_L threshold for switching to supernodal factorization");
  CHOLMOD_OPTION_TRUTH(final_asis,"Leave factors \"as is\"");
  CHOLMOD_OPTION_TRUTH(final_pack,"Pack the columns when finished (use FALSE if the factors will be updated later)");
  if (!c->final_asis) {
    CHOLMOD_OPTION_TRUTH(final_super,"Leave supernodal factors instead of converting to simplicial");
    CHOLMOD_OPTION_TRUTH(final_ll,"Turn LDL' factorization into LL'");
    CHOLMOD_OPTION_TRUTH(final_monotonic,"Ensure columns are monotonic when done");
    CHOLMOD_OPTION_TRUTH(final_resymbol,"Remove numerically zero values resulting from relaxed supernodal amalgamation");
  }
  {
    PetscReal tmp[] = {(PetscReal)c->zrelax[0],(PetscReal)c->zrelax[1],(PetscReal)c->zrelax[2]};
    PetscInt n = 3;
    ierr = PetscOptionsRealArray("-mat_cholmod_zrelax","3 real supernodal relaxed amalgamation parameters","None",tmp,&n,&flg);CHKERRQ(ierr);
    if (flg && n != 3) SETERRQ(((PetscObject)F)->comm,PETSC_ERR_ARG_OUTOFRANGE,"must provide exactly 3 parameters to -mat_cholmod_zrelax");
    if (flg) while (n--) c->zrelax[n] = (double)tmp[n];
  }
  {
    PetscInt n,tmp[] = {(PetscInt)c->nrelax[0],(PetscInt)c->nrelax[1],(PetscInt)c->nrelax[2]};
    ierr = PetscOptionsIntArray("-mat_cholmod_nrelax","3 size_t supernodal relaxed amalgamation parameters","None",tmp,&n,&flg);CHKERRQ(ierr);
    if (flg && n != 3) SETERRQ(((PetscObject)F)->comm,PETSC_ERR_ARG_OUTOFRANGE,"must provide exactly 3 parameters to -mat_cholmod_nrelax");
    if (flg) while (n--) c->nrelax[n] = (size_t)tmp[n];
  }
  CHOLMOD_OPTION_TRUTH(prefer_upper,"Work with upper triangular form (faster when using fill-reducing ordering, slower in natural ordering)");
  CHOLMOD_OPTION_INT(print,"Verbosity level");
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatWrapCholmod_seqsbaij"
static PetscErrorCode MatWrapCholmod_seqsbaij(Mat A,cholmod_sparse *B)
{
  Mat_SeqSBAIJ *sbaij = (Mat_SeqSBAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(B,sizeof(*B));CHKERRQ(ierr);
  /* CHOLMOD uses column alignment, SBAIJ stores the upper factor, so we pass it on as a lower factor, swapping the meaning of row and column */
  B->nrow  = (size_t)A->cmap->n;
  B->ncol  = (size_t)A->rmap->n;
  B->nzmax = (size_t)sbaij->maxnz;
  B->p     = sbaij->i;
  B->i     = sbaij->j;
  B->x     = sbaij->a;
  B->stype = -1;
  B->itype = CHOLMOD_INT_TYPE;
  B->xtype = CHOLMOD_SCALAR_TYPE;
  B->dtype = CHOLMOD_DOUBLE;
  B->sorted = 1;
  B->packed = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecWrapCholmod"
static PetscErrorCode VecWrapCholmod(Vec X,cholmod_dense *Y)
{
  PetscErrorCode ierr;
  PetscScalar *x;
  PetscInt n;

  PetscFunctionBegin;
  ierr = PetscMemzero(Y,sizeof(*Y));CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetSize(X,&n);CHKERRQ(ierr);
  Y->x = (double*)x;
  Y->nrow = n;
  Y->ncol = 1;
  Y->nzmax = n;
  Y->d = n;
  Y->x = (double*)x;
  Y->xtype = CHOLMOD_SCALAR_TYPE;
  Y->dtype = CHOLMOD_DOUBLE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_CHOLMOD"
static PetscErrorCode MatDestroy_CHOLMOD(Mat F)
{
  PetscErrorCode ierr;
  Mat_CHOLMOD    *chol=(Mat_CHOLMOD*)F->spptr;

  PetscFunctionBegin;
  ierr = !cholmod_X_free_factor(&chol->factor,chol->common);CHKERRQ(ierr);
  ierr = !cholmod_X_finish(chol->common);CHKERRQ(ierr);
  ierr = PetscFree(chol->common);CHKERRQ(ierr);
  ierr = PetscFree(chol->matrix);CHKERRQ(ierr);
  ierr = MatDestroy_SeqSBAIJ(F);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_CHOLMOD(Mat,Vec,Vec);

static const char *const CholmodOrderingMethods[] = {"User","AMD","METIS","NESDIS(default)","Natural","NESDIS(small=20000)","NESDIS(small=4,no constrained)","NESDIS()"};

#undef __FUNCT__
#define __FUNCT__ "MatFactorInfo_CHOLMOD"
static PetscErrorCode MatFactorInfo_CHOLMOD(Mat F,PetscViewer viewer)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->spptr;
  const cholmod_common *c = chol->common;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (F->ops->solve != MatSolve_CHOLMOD) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIPrintf(viewer,"CHOLMOD run parameters:\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.dbound            %g  (Smallest absolute value of diagonal entries of D)\n",c->dbound);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.grow0             %g\n",c->grow0);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.grow1             %g\n",c->grow1);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.grow2             %u\n",(unsigned)c->grow2);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.maxrank           %u\n",(unsigned)c->maxrank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.supernodal_switch %g\n",c->supernodal_switch);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.supernodal        %d\n",c->supernodal);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_asis        %d\n",c->final_asis);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_super       %d\n",c->final_super);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_ll          %d\n",c->final_ll);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_pack        %d\n",c->final_pack);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_monotonic   %d\n",c->final_monotonic);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_resymbol    %d\n",c->final_resymbol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.zrelax            [%g,%g,%g]\n",c->zrelax[0],c->zrelax[1],c->zrelax[2]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.nrelax            [%u,%u,%u]\n",(unsigned)c->nrelax[0],(unsigned)c->nrelax[1],(unsigned)c->nrelax[2]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.prefer_upper      %d\n",c->prefer_upper);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.print             %d\n",c->print);CHKERRQ(ierr);
  for (i=0; i<c->nmethods; i++) {
    ierr = PetscViewerASCIIPrintf(viewer,"Ordering method %D%s:\n",i,i==c->selected?" [SELECTED]":"");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  lnz %g, fl %g, prune_dense %g, prune_dense2 %g\n",
        c->method[i].lnz,c->method[i].fl,c->method[i].prune_dense,c->method[i].prune_dense2);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer,"Common.postorder         %d\n",chol->common->postorder);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_CHOLMOD"
static PetscErrorCode MatView_CHOLMOD(Mat F,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscTruth        iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = MatView_SeqSBAIJ(F,viewer);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatFactorInfo_CHOLMOD(F,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_CHOLMOD"
static PetscErrorCode MatSolve_CHOLMOD(Mat F,Vec B,Vec X)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->spptr;
  cholmod_dense  cholB,*cholX;
  PetscScalar    *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecWrapCholmod(B,&cholB);CHKERRQ(ierr);
  cholX = cholmod_X_solve(CHOLMOD_A,chol->factor,&cholB,chol->common);
  if (!cholX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CHOLMOD failed");
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(x,cholX->x,cholX->nrow*sizeof(*x));CHKERRQ(ierr);
  ierr = !cholmod_X_free_dense(&cholX,chol->common);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_CHOLMOD"
static PetscErrorCode MatCholeskyFactorNumeric_CHOLMOD(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->spptr;
  cholmod_sparse cholA;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatWrapCholmod_seqsbaij(A,&cholA);CHKERRQ(ierr);
  ierr = !cholmod_X_factorize(&cholA,chol->factor,chol->common);CHKERRQ(ierr);
  F->ops->solve          = MatSolve_CHOLMOD;
  F->ops->solvetranspose = MatSolve_CHOLMOD;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_CHOLMOD"
static PetscErrorCode MatCholeskyFactorSymbolic_CHOLMOD(Mat F,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_CHOLMOD    *chol = (Mat_CHOLMOD*)F->spptr;
  PetscErrorCode ierr;
  cholmod_sparse cholA;

  PetscFunctionBegin;
  ierr = MatWrapCholmod_seqsbaij(A,&cholA);CHKERRQ(ierr);
  if (perm) {
    const PetscInt *ip;
    ierr = ISGetIndices(perm,&ip);CHKERRQ(ierr);
    chol->factor = cholmod_X_analyze_p(&cholA,(PetscInt*)ip,0,0,chol->common);
    ierr = ISRestoreIndices(perm,&ip);CHKERRQ(ierr);
  } else {
    chol->factor = cholmod_X_analyze(&cholA,chol->common);
  }
  if (!chol->factor) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CHOLMOD factorization failed");
  F->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_CHOLMOD;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_seqsbaij_cholmod"
PetscErrorCode MatFactorGetSolverPackage_seqsbaij_cholmod(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCHOLMOD;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
  MATSOLVERCHOLMOD = "cholmod" - A matrix type providing direct solvers (Cholesky) for sequential matrices
  via the external package CHOLMOD.

  ./configure --download-cholmod to install PETSc to use CHOLMOD

  Consult CHOLMOD documentation for more information about the Common parameters
  which correspond to the options database keys below.

  Options Database Keys:
  -mat_cholmod_dbound <0>: Minimum absolute value of diagonal entries of D (None)
  -mat_cholmod_grow0 <1.2>: Global growth ratio when factors are modified (None)
  -mat_cholmod_grow1 <1.2>: Column growth ratio when factors are modified (None)
  -mat_cholmod_grow2 <5>: Affine column growth constant when factors are modified (None)
  -mat_cholmod_maxrank <8>: Max rank of update, larger values are faster but use more memory [2,4,8] (None)
  -mat_cholmod_factor <AUTO> (choose one of) SIMPLICIAL AUTO SUPERNODAL
  -mat_cholmod_supernodal_switch <40>: flop/nnz_L threshold for switching to supernodal factorization (None)
  -mat_cholmod_final_asis: <TRUE> Leave factors "as is" (None)
  -mat_cholmod_final_pack: <TRUE> Pack the columns when finished (use FALSE if the factors will be updated later) (None)
  -mat_cholmod_zrelax <0.8>: 3 real supernodal relaxed amalgamation parameters (None)
  -mat_cholmod_nrelax <4>: 3 size_t supernodal relaxed amalgamation parameters (None)
  -mat_cholmod_prefer_upper: <TRUE> Work with upper triangular form (faster when using fill-reducing ordering, slower in natural ordering) (None)
  -mat_cholmod_print <3>: Verbosity level (None)

   Level: beginner

.seealso: PCCHOLESKY, PCFactorSetMatSolverPackage(), MatSolverPackage
M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_seqsbaij_cholmod"
PetscErrorCode MatGetFactor_seqsbaij_cholmod(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_CHOLMOD    *chol;
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=A->cmap->n,bs;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"CHOLMOD cannot do %s factorization with SBAIJ, only %s",
                                             MatFactorTypes[ftype],MatFactorTypes[MAT_FACTOR_CHOLESKY]);
  ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
  if (bs != 1) SETERRQ1(((PetscObject)A)->comm,PETSC_ERR_SUP,"CHOLMOD only supports block size=1, given %D",bs);
  /* Create the factorization matrix F */
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(B,1,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscNewLog(B,Mat_CHOLMOD,&chol);CHKERRQ(ierr);
  B->spptr                 = chol;
  B->ops->view             = MatView_CHOLMOD;
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_CHOLMOD;
  B->ops->destroy          = MatDestroy_CHOLMOD;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_seqsbaij_cholmod",MatFactorGetSolverPackage_seqsbaij_cholmod);CHKERRQ(ierr);
  B->factortype            = MAT_FACTOR_CHOLESKY;
  B->assembled             = PETSC_TRUE;  /* required by -ksp_view */
  B->preallocated          = PETSC_TRUE;

  ierr = CholmodStart(B);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END
