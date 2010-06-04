#define PETSCMAT_DLL

/*
   Provides an interface to the CHOLMOD 1.7.1 sparse solver

   When build with PETSC_USE_64BIT_INDICES this will use UF_Long as the
   integer type in UMFPACK, otherwise it will use int. This means
   all integers in this file as simply declared as PetscInt. Also it means
   that UMFPACK UL_Long version MUST be built with 64 bit integers when used.

*/
#include "../src/mat/impls/sbaij/seq/sbaij.h"

#if defined(PETSC_USE_64BIT_INDICES)
#if defined(PETSC_USE_COMPLEX)
#define umfpack_UMF_free_symbolic   umfpack_zl_free_symbolic
#define umfpack_UMF_free_numeric    umfpack_zl_free_numeric
#define umfpack_UMF_wsolve          umfpack_zl_wsolve
#define umfpack_UMF_numeric         umfpack_zl_numeric
#define umfpack_UMF_report_numeric  umfpack_zl_report_numeric
#define umfpack_UMF_report_control  umfpack_zl_report_control
#define umfpack_UMF_report_status   umfpack_zl_report_status
#define umfpack_UMF_report_info     umfpack_zl_report_info
#define umfpack_UMF_report_symbolic umfpack_zl_report_symbolic
#define umfpack_UMF_qsymbolic       umfpack_zl_qsymbolic
#define umfpack_UMF_symbolic        umfpack_zl_symbolic
#define umfpack_UMF_defaults        umfpack_zl_defaults

#else
#define umfpack_UMF_free_symbolic   umfpack_dl_free_symbolic
#define umfpack_UMF_free_numeric    umfpack_dl_free_numeric
#define umfpack_UMF_wsolve          umfpack_dl_wsolve
#define umfpack_UMF_numeric         umfpack_dl_numeric
#define umfpack_UMF_report_numeric  umfpack_dl_report_numeric
#define umfpack_UMF_report_control  umfpack_dl_report_control
#define umfpack_UMF_report_status   umfpack_dl_report_status
#define umfpack_UMF_report_info     umfpack_dl_report_info
#define umfpack_UMF_report_symbolic umfpack_dl_report_symbolic
#define umfpack_UMF_qsymbolic       umfpack_dl_qsymbolic
#define umfpack_UMF_symbolic        umfpack_dl_symbolic
#define umfpack_UMF_defaults        umfpack_dl_defaults
#endif

#else
#if defined(PETSC_USE_COMPLEX)
#define umfpack_UMF_free_symbolic   umfpack_zi_free_symbolic
#define umfpack_UMF_free_numeric    umfpack_zi_free_numeric
#define umfpack_UMF_wsolve          umfpack_zi_wsolve
#define umfpack_UMF_numeric         umfpack_zi_numeric
#define umfpack_UMF_report_numeric  umfpack_zi_report_numeric
#define umfpack_UMF_report_control  umfpack_zi_report_control
#define umfpack_UMF_report_status   umfpack_zi_report_status
#define umfpack_UMF_report_info     umfpack_zi_report_info
#define umfpack_UMF_report_symbolic umfpack_zi_report_symbolic
#define umfpack_UMF_qsymbolic       umfpack_zi_qsymbolic
#define umfpack_UMF_symbolic        umfpack_zi_symbolic
#define umfpack_UMF_defaults        umfpack_zi_defaults

#else
#define CHOLMOD_INT_TYPE          CHOLMOD_INT
#define CHOLMOD_SCALAR_TYPE       CHOLMOD_REAL
#define cholmod_X_analyze         cholmod_analyze
#define cholmod_X_factorize       cholmod_factorize
#define cholmod_X_solve           cholmod_solve
#endif
#endif


#define UF_long long long
#define UF_long_max LONG_LONG_MAX
#define UF_long_id "%lld"

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

  PetscFunctionBegin;
  if (chol->common) PetscFunctionReturn(0);
  ierr = PetscMalloc(sizeof(*chol->common),&chol->common);CHKERRQ(ierr);
  ierr = !cholmod_start(chol->common);CHKERRQ(ierr);
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
  ierr = !cholmod_free_factor(&chol->factor,chol->common);CHKERRQ(ierr);
  ierr = !cholmod_finish(chol->common);CHKERRQ(ierr);
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
  ierr = PetscViewerASCIIPrintf(viewer,"Common.grow2             %ud\n",(unsigned)c->grow2);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.maxrank           %ud\n",(unsigned)c->maxrank);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.supernodal_switch %g\n",c->supernodal_switch);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.supernodal        %d\n",c->supernodal);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_asis        %d\n",c->final_asis);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_super       %d\n",c->final_super);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_ll          %d\n",c->final_ll);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_pack        %d\n",c->final_pack);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_monotonic   %d\n",c->final_monotonic);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.final_resymbol    %d\n",c->final_resymbol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.zrelax            [%g,%g,%g]\n",c->zrelax[0],c->zrelax[1],c->zrelax[2]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.nrelax            [%d,%d,%d]\n",(unsigned)c->nrelax[0],(unsigned)c->nrelax[1],(unsigned)c->nrelax[2]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.prefer_upper      %d\n",c->prefer_upper);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Common.print             %d\n",c->print);CHKERRQ(ierr);
  {
    //char buf[512],*p = buf;
    for (i=0; i<c->nmethods; i++) {
      //ierr = PetscSNPrintf(p,p-buf+sizeof buf,"%s%s",CholmodOrderingMethods[i],(i+1==c->nmethods)?"",",");
      ierr = PetscViewerASCIIPrintf(viewer,"Ordering method %D%s:\n",i,i==c->selected?" [SELECTED]":"");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  lnz %g, fl %g, prune_dense %g, prune_dense2 %g\n",
          c->method[i].lnz,c->method[i].fl,c->method[i].prune_dense,c->method[i].prune_dense2);CHKERRQ(ierr);
    }
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
  cholX = cholmod_solve(CHOLMOD_A,chol->factor,&cholB,chol->common);
  if (!cholX) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"CHOLMOD failed");
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(x,cholX->x,cholX->nrow*sizeof(*x));CHKERRQ(ierr);
  ierr = !cholmod_free_dense(&cholX,chol->common);CHKERRQ(ierr);
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
  ierr = !cholmod_factorize(&cholA,chol->factor,chol->common);CHKERRQ(ierr);
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
  ierr = CholmodStart(F);CHKERRQ(ierr);
  ierr = MatWrapCholmod_seqsbaij(A,&cholA);CHKERRQ(ierr);
  if (perm) {
    const PetscInt *ip;
    ierr = ISGetIndices(perm,&ip);CHKERRQ(ierr);
    chol->factor = cholmod_analyze_p(&cholA,(PetscInt*)ip,0,0,chol->common);
    ierr = ISRestoreIndices(perm,&ip);CHKERRQ(ierr);
  } else {
    chol->factor = cholmod_analyze(&cholA,chol->common);
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
  MATSOLVERCHOLMOD = "umfpack" - A matrix type providing direct solvers (LU) for sequential matrices
  via the external package UMFPACK.

  ./configure --download-umfpack to install PETSc to use UMFPACK

  Consult UMFPACK documentation for more information about the Control parameters
  which correspond to the options database keys below.

  Options Database Keys:
+ -Mat_CHOLMOD_prl - UMFPACK print level: Control[UMFPACK_PRL]
. -Mat_CHOLMOD_strategy <AUTO> (choose one of) AUTO UNSYMMETRIC SYMMETRIC 2BY2
. -Mat_CHOLMOD_dense_col <alpha_c> - UMFPACK dense column threshold: Control[UMFPACK_DENSE_COL]
. -Mat_CHOLMOD_dense_row <0.2>: Control[UMFPACK_DENSE_ROW]
. -Mat_CHOLMOD_amd_dense <10>: Control[UMFPACK_AMD_DENSE]
. -Mat_CHOLMOD_block_size <bs> - UMFPACK block size for BLAS-Level 3 calls: Control[UMFPACK_BLOCK_SIZE]
. -Mat_CHOLMOD_2by2_tolerance <0.01>: Control[UMFPACK_2BY2_TOLERANCE]
. -Mat_CHOLMOD_fixq <0>: Control[UMFPACK_FIXQ]
. -Mat_CHOLMOD_aggressive <1>: Control[UMFPACK_AGGRESSIVE]
. -Mat_CHOLMOD_pivot_tolerance <delta> - UMFPACK partial pivot tolerance: Control[UMFPACK_PIVOT_TOLERANCE]
.  -Mat_CHOLMOD_sym_pivot_tolerance <0.001>: Control[UMFPACK_SYM_PIVOT_TOLERANCE]
.  -Mat_CHOLMOD_scale <NONE> (choose one of) NONE SUM MAX
. -Mat_CHOLMOD_alloc_init <delta> - UMFPACK factorized matrix allocation modifier: Control[UMFPACK_ALLOC_INIT]
.  -Mat_CHOLMOD_droptol <0>: Control[UMFPACK_DROPTOL]
- -Mat_CHOLMOD_irstep <maxit> - UMFPACK maximum number of iterative refinement steps: Control[UMFPACK_IRSTEP]

   Level: beginner

.seealso: PCLU, MATSOLVERSUPERLU, MATSOLVERMUMPS, MATSOLVERSPOOLES, PCFactorSetMatSolverPackage(), MatSolverPackage
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

  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

