#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <StrumpackSparseSolver.h>

/*
  These are only relevant for MATMPIAIJ, not for MATSEQAIJ.
    REPLICATED  - STRUMPACK expects the entire sparse matrix and right-hand side on every process.
    DISTRIBUTED - STRUMPACK expects the sparse matrix and right-hand side to be distributed across the entire MPI communicator.
*/
typedef enum {REPLICATED, DISTRIBUTED} STRUMPACK_MatInputMode;
const char *STRUMPACK_MatInputModes[] = {"REPLICATED","DISTRIBUTED","STRUMPACK_MatInputMode","PETSC_",0};

typedef struct {
  STRUMPACK_SparseSolver S;
  STRUMPACK_MatInputMode MatInputMode;
} STRUMPACK_data;


#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_STRUMPACK"
static PetscErrorCode MatGetDiagonal_STRUMPACK(Mat A,Vec v)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Mat type: STRUMPACK factor");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_STRUMPACK"
static PetscErrorCode MatDestroy_STRUMPACK(Mat A)
{
  STRUMPACK_data *sp = (STRUMPACK_data*)A->spptr;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  /* Deallocate STRUMPACK storage */
  PetscStackCall("STRUMPACK_destroy",STRUMPACK_destroy(&(sp->S)));
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  }

  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverPackage_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetHSSRelCompTol_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetHSSMinSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetColPerm_C",NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSTRUMPACKSetHSSRelCompTol_STRUMPACK"
static PetscErrorCode MatSTRUMPACKSetHSSRelCompTol_STRUMPACK(Mat F,PetscReal rctol)
{
  STRUMPACK_data *sp = (STRUMPACK_data*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_rctol", STRUMPACK_set_rctol(sp->S,rctol));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSTRUMPACKSetHSSRelCompTol"
/*@
  MatSTRUMPACKSetHSSRelCompTol - Set STRUMPACK relative tolerance for HSS compression
   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  rctol - relative compression tolerance

  Options Database:
.   -mat_strumpack_rctol <rctol>

   Level: beginner

   References:
.      STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetHSSRelCompTol(Mat F,PetscReal rctol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(F,rctol,2);
  ierr = PetscTryMethod(F,"MatSTRUMPACKSetHSSRelCompTol_C",(Mat,PetscReal),(F,rctol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSTRUMPACKSetHSSMinSize_STRUMPACK"
static PetscErrorCode MatSTRUMPACKSetHSSMinSize_STRUMPACK(Mat F,PetscInt hssminsize)
{
  STRUMPACK_data *sp = (STRUMPACK_data*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_minimum_HSS_size", STRUMPACK_set_minimum_HSS_size(sp->S,hssminsize));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSTRUMPACKSetHSSMinSize"
/*@
  MatSTRUMPACKSetHSSMinSize - Set STRUMPACK minimum dense matrix size for low-rank approximation
   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  hssminsize - minimum dense matrix size for low-rank approximation

  Options Database:
.   -mat_strumpack_hssminsize <hssminsize>

   Level: beginner

   References:
.      STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetHSSMinSize(Mat F,PetscInt hssminsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,hssminsize,2);
  ierr = PetscTryMethod(F,"MatSTRUMPACKSetHSSMinSize_C",(Mat,PetscInt),(F,hssminsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSTRUMPACKSetColPerm_STRUMPACK"
static PetscErrorCode MatSTRUMPACKSetColPerm_STRUMPACK(Mat F,PetscBool cperm)
{
  STRUMPACK_data *sp = (STRUMPACK_data*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_mc64job",STRUMPACK_set_mc64job(sp->S,cperm ? 5 : 0));
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSTRUMPACKSetColPerm"
/*@
  MatSTRUMPACKSetColPerm - Set whether STRUMPACK should try to permute the columns of the matrix in order to get a nonzero diagonal
   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  cperm - whether or not to permute (internally) the columns of the matrix

  Options Database:
.   -mat_strumpack_colperm <cperm>

   Level: beginner

   References:
.      STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetColPerm(Mat F,PetscBool cperm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(F,cperm,2);
  ierr = PetscTryMethod(F,"MatSTRUMPACKSetColPerm_C",(Mat,PetscBool),(F,cperm));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_STRUMPACK"
static PetscErrorCode MatSolve_STRUMPACK(Mat A,Vec b_mpi,Vec x)
{
  STRUMPACK_data        *sp = (STRUMPACK_data*)A->spptr;
  STRUMPACK_RETURN_CODE sp_err;
  PetscErrorCode        ierr;
  PetscMPIInt           size;
  PetscInt              N=A->cmap->N;
  const PetscScalar     *bptr;
  PetscScalar           *xptr;
  Vec                   x_seq,b_seq;
  IS                    iden;
  VecScatter            scat;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size > 1 && sp->MatInputMode == REPLICATED) {
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x_seq);CHKERRQ(ierr);
    ierr = VecGetArray(x_seq,&xptr);CHKERRQ(ierr);
    /* replicated mat input, convert b to b_seq */
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&b_seq);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&iden);CHKERRQ(ierr);
    ierr = VecScatterCreate(b_mpi,iden,b_seq,iden,&scat);CHKERRQ(ierr);
    ierr = ISDestroy(&iden);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat,b_mpi,b_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat,b_mpi,b_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArrayRead(b_seq,&bptr);CHKERRQ(ierr);
  } else { /* size==1 || distributed mat input */
    ierr = VecGetArray(x,&xptr);CHKERRQ(ierr);
    ierr = VecGetArrayRead(b_mpi,&bptr);CHKERRQ(ierr);
  }

  PetscStackCall("STRUMPACK_solve",sp_err = STRUMPACK_solve(sp->S,(PetscScalar*)bptr,xptr,0));

  if (sp_err != STRUMPACK_SUCCESS) {
    if (sp_err == STRUMPACK_MATRIX_NOT_SET)        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix was not set");
    else if (sp_err == STRUMPACK_REORDERING_ERROR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix reordering failed");
    else                                           SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: solve failed");
  }

  if (size > 1 && sp->MatInputMode == REPLICATED) {
    ierr = VecRestoreArrayRead(b_seq,&bptr);CHKERRQ(ierr);
    ierr = VecDestroy(&b_seq);CHKERRQ(ierr);
    /* convert seq x to mpi x */
    ierr = VecRestoreArray(x_seq,&xptr);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat,x_seq,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat,x_seq,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat);CHKERRQ(ierr);
    ierr = VecDestroy(&x_seq);CHKERRQ(ierr);
  } else {
    ierr = VecRestoreArray(x,&xptr);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(b_mpi,&bptr);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatSolve_STRUMPACK"
static PetscErrorCode MatMatSolve_STRUMPACK(Mat A,Mat B_mpi,Mat X)
{
  PetscErrorCode   ierr;
  PetscBool        flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)B_mpi,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  ierr = PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatMatSolve_STRUMPACK() is not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFactorInfo_STRUMPACK"
static PetscErrorCode MatFactorInfo_STRUMPACK(Mat A,PetscViewer viewer)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* check if matrix is strumpack type */
  if (A->ops->solve != MatSolve_STRUMPACK) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIPrintf(viewer,"STRUMPACK sparse solver!\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_STRUMPACK"
static PetscErrorCode MatView_STRUMPACK(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatFactorInfo_STRUMPACK(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_STRUMPACK"
static PetscErrorCode MatLUFactorNumeric_STRUMPACK(Mat F,Mat A,const MatFactorInfo *info)
{
  STRUMPACK_data        *sp = (STRUMPACK_data*)F->spptr;
  STRUMPACK_RETURN_CODE sp_err;
  Mat                   *tseq,A_seq = NULL;
  Mat_SeqAIJ            *A_d,*A_o;
  Mat_MPIAIJ            *mat;
  PetscErrorCode        ierr;
  PetscInt              M=A->rmap->N,m=A->rmap->n,N=A->cmap->N;
  PetscMPIInt           size;
  IS                    isrow;
  PetscBool             flg;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);

  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
  if (flg) { /* A is MATMPIAIJ */
    if (sp->MatInputMode == REPLICATED) {
      if (size > 1) { /* convert mpi A to seq mat A */
        ierr = ISCreateStride(PETSC_COMM_SELF,M,0,1,&isrow);CHKERRQ(ierr);
        ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&tseq);CHKERRQ(ierr);
        ierr = ISDestroy(&isrow);CHKERRQ(ierr);
        A_seq = *tseq;
        ierr  = PetscFree(tseq);CHKERRQ(ierr);
        A_d   = (Mat_SeqAIJ*)A_seq->data;
      } else { /* size == 1 */
        mat = (Mat_MPIAIJ*)A->data;
        A_d = (Mat_SeqAIJ*)(mat->A)->data;
      }
      PetscStackCall("STRUMPACK_set_csr_matrix",STRUMPACK_set_csr_matrix(sp->S,&N,A_d->i,A_d->j,A_d->a,0));
    } else { /* sp->MatInputMode == DISTRIBUTED */
      mat = (Mat_MPIAIJ*)A->data;
      A_d = (Mat_SeqAIJ*)(mat->A)->data;
      A_o = (Mat_SeqAIJ*)(mat->B)->data;
      PetscStackCall("STRUMPACK_set_MPIAIJ_matrix",STRUMPACK_set_MPIAIJ_matrix(sp->S,&m,A_d->i,A_d->j,A_d->a,A_o->i,A_o->j,A_o->a,mat->garray));
    }
  } else { /* A is MATSEQAIJ */
    A_d = (Mat_SeqAIJ*)A->data;
    PetscStackCall("STRUMPACK_set_csr_matrix",STRUMPACK_set_csr_matrix(sp->S,&N,A_d->i,A_d->j,A_d->a,0));
  }

  /* Reorder and Factor the matrix. */
  /* TODO figure out how to avoid reorder if the matrix values changed, but the pattern remains the same. */
  PetscStackCall("STRUMPACK_reorder",sp_err = STRUMPACK_reorder(sp->S));
  PetscStackCall("STRUMPACK_factor",sp_err = STRUMPACK_factor(sp->S));
  if (sp_err != STRUMPACK_SUCCESS) {
    if (sp_err == STRUMPACK_MATRIX_NOT_SET)        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix was not set");
    else if (sp_err == STRUMPACK_REORDERING_ERROR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix reordering failed");
    else                                           SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: factorization failed");
  }
  if (flg && sp->MatInputMode == REPLICATED && size > 1) {
    ierr = MatDestroy(&A_seq);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_STRUMPACK"
static PetscErrorCode MatLUFactorSymbolic_STRUMPACK(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  F->ops->lufactornumeric = MatLUFactorNumeric_STRUMPACK;
  F->ops->solve           = MatSolve_STRUMPACK;
  F->ops->matsolve        = MatMatSolve_STRUMPACK;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_aij_strumpack"
static PetscErrorCode MatFactorGetSolverPackage_aij_strumpack(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSTRUMPACK;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERSSTRUMPACK = "strumpack" - A solver package providing a direct sparse solver (PCLU)
  and a preconditioner (PCILU) using low-rank compression via the external package STRUMPACK.

  Consult the STRUMPACK-sparse manual for more info.

  Use
     ./configure --download-strumpack
  to have PETSc installed with STRUMPACK

  Use
    -pc_type lu -pc_factor_mat_solver_package strumpack
  to use this as an exact (direct) solver, use
    -pc_type ilu -pc_factor_mat_solver_package strumpack
  to enable low-rank compression (i.e, use as a preconditioner).

  Works with AIJ matrices

  Options Database Keys:
+ -mat_strumpack_rctol <1e-4>           - Relative compression tolerance (None)
. -mat_strumpack_hssminsize <512>       - Minimum size of dense block for HSS compression (None)
. -mat_strumpack_colperm <TRUE>         - Permute matrix to make diagonal nonzeros (None)

 Level: beginner

.seealso: PCLU, PCILU, MATSOLVERSUPERLU_DIST, MATSOLVERMUMPS, PCFactorSetMatSolverPackage(), MatSolverPackage
M*/
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_aij_strumpack"
static PetscErrorCode MatGetFactor_aij_strumpack(Mat A,MatFactorType ftype,Mat *F)
{
  Mat                 B;
  STRUMPACK_data      *sp;
  PetscErrorCode      ierr;
  PetscInt            M=A->rmap->N,N=A->cmap->N;
  PetscMPIInt         size;
  STRUMPACK_INTERFACE iface;
  PetscBool           verb,flg,set;
  PetscReal           rctol;
  PetscInt            hssminsize;
  int                 argc;
  char                **args,*copts,*pname;
  size_t              len;
  const STRUMPACK_PRECISION table[2][2][2] = {{{STRUMPACK_FLOATCOMPLEX_64,STRUMPACK_DOUBLECOMPLEX_64},
                                               {STRUMPACK_FLOAT_64,STRUMPACK_DOUBLE_64}},
                                              {{STRUMPACK_FLOATCOMPLEX,STRUMPACK_DOUBLECOMPLEX},
                                               {STRUMPACK_FLOAT,STRUMPACK_DOUBLE}}};
  const STRUMPACK_PRECISION prec = table[(sizeof(PetscInt)==8)?0:1][(PETSC_SCALAR==PETSC_COMPLEX)?0:1][(PETSC_REAL==PETSC_FLOAT)?0:1];

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  /* Create the factorization matrix */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,NULL);
  ierr = MatMPIAIJSetPreallocation(B,0,NULL,0,NULL);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU) {
    B->ops->lufactorsymbolic  = MatLUFactorSymbolic_STRUMPACK;
    B->ops->ilufactorsymbolic = MatLUFactorSymbolic_STRUMPACK;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");
  B->ops->view             = MatView_STRUMPACK;
  B->ops->destroy          = MatDestroy_STRUMPACK;
  B->ops->getdiagonal      = MatGetDiagonal_STRUMPACK;
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_aij_strumpack);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetHSSRelCompTol_C",MatSTRUMPACKSetHSSRelCompTol_STRUMPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetHSSMinSize_C",MatSTRUMPACKSetHSSMinSize_STRUMPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetColPerm_C",MatSTRUMPACKSetColPerm_STRUMPACK);CHKERRQ(ierr);
  B->factortype = ftype;
  ierr     = PetscNewLog(B,&sp);CHKERRQ(ierr);
  B->spptr = sp;

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"STRUMPACK Options","Mat");CHKERRQ(ierr);
  sp->MatInputMode = DISTRIBUTED;
  ierr = PetscOptionsEnum("-mat_strumpack_matinput","Matrix input mode (replicated or distributed)","None",STRUMPACK_MatInputModes,
                          (PetscEnum)sp->MatInputMode,(PetscEnum*)&sp->MatInputMode,NULL);CHKERRQ(ierr);
  if (sp->MatInputMode == DISTRIBUTED && size == 1) sp->MatInputMode = REPLICATED;
  switch (sp->MatInputMode) {
  case REPLICATED:
    iface = STRUMPACK_MPI;
    break;
  case DISTRIBUTED:
  default:
    iface = STRUMPACK_MPI_DIST;
  }
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&flg);
  if (flg) iface = STRUMPACK_MT;

  verb = PetscLogPrintInfo ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_strumpack_verbose","Print STRUMPACK information","None",verb,&verb,NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetAll(NULL,&copts);CHKERRQ(ierr);
  ierr = PetscStrlen(copts,&len);CHKERRQ(ierr);
  len += PETSC_MAX_PATH_LEN+1;
  ierr = PetscMalloc1(len,&pname);CHKERRQ(ierr);
  /* first string is assumed to be the program name, so add program name to options string */
  ierr = PetscGetProgramName(pname,len);CHKERRQ(ierr);
  ierr = PetscStrcat(pname," ");CHKERRQ(ierr);
  ierr = PetscStrcat(pname,copts);CHKERRQ(ierr);
  ierr = PetscStrToArray(pname,' ',&argc,&args);CHKERRQ(ierr);
  ierr = PetscFree(copts);CHKERRQ(ierr);
  ierr = PetscFree(pname);CHKERRQ(ierr);

  PetscStackCall("STRUMPACK_init",STRUMPACK_init(&(sp->S),PetscObjectComm((PetscObject)A),prec,iface,argc,args,verb));

  PetscStackCall("STRUMPACK_get_rctol",rctol = (PetscReal)STRUMPACK_get_rctol(sp->S));
  ierr = PetscOptionsReal("-mat_strumpack_rctol","Relative compression tolerance","None",rctol,&rctol,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_rctol",STRUMPACK_set_rctol(sp->S,(double)rctol));

  PetscStackCall("STRUMPACK_get_mc64job",flg = (STRUMPACK_get_mc64job(sp->S) == 0) ? PETSC_FALSE : PETSC_TRUE);
  ierr = PetscOptionsBool("-mat_strumpack_colperm","Find a col perm to get nonzero diagonal","None",flg,&flg,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_mc64job",STRUMPACK_set_mc64job(sp->S,flg ? 5 : 0));

  PetscStackCall("STRUMPACK_get_minimum_HSS_size",hssminsize = (PetscInt)STRUMPACK_get_minimum_HSS_size(sp->S));
  ierr = PetscOptionsInt("-mat_strumpack_hssminsize","Minimum size of dense block for HSS compression","None",hssminsize,&hssminsize,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_minimum_HSS_size",STRUMPACK_set_minimum_HSS_size(sp->S,(int)hssminsize));

  PetscOptionsEnd();

  if (ftype == MAT_FACTOR_ILU) {
    /* When enabling HSS compression, the STRUMPACK solver becomes an incomplete                */
    /* (or approximate) LU factorization.                                                       */
    PetscStackCall("STRUMPACK_use_HSS",STRUMPACK_use_HSS(sp->S,1));
    /* Disable the outer iterative solver from STRUMPACK.                                       */
    /* When STRUMPACK is used as a direct solver, it will by default do iterative refinement.   */
    /* When STRUMPACK is used with as an approximate factorization preconditioner (by enabling  */
    /* low-rank compression), it will use it's own GMRES. Here we can disable the               */
    /* outer iterative solver, as PETSc uses STRUMPACK from within a KSP.                       */
    PetscStackCall("STRUMPACK_set_Krylov_solver", STRUMPACK_set_Krylov_solver(sp->S, STRUMPACK_DIRECT));
  }

  /* Allow the user to set or overwrite the above options from the command line                 */
  PetscStackCall("STRUMPACK_set_from_options",STRUMPACK_set_from_options(sp->S));
  ierr = PetscStrToArrayDestroy(argc,args);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolverPackageRegister_STRUMPACK"
PETSC_EXTERN PetscErrorCode MatSolverPackageRegister_STRUMPACK(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverPackageRegister(MATSOLVERSTRUMPACK,MATMPIAIJ,MAT_FACTOR_LU,MatGetFactor_aij_strumpack);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERSTRUMPACK,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_aij_strumpack);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERSTRUMPACK,MATMPIAIJ,MAT_FACTOR_ILU,MatGetFactor_aij_strumpack);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERSTRUMPACK,MATSEQAIJ,MAT_FACTOR_ILU,MatGetFactor_aij_strumpack);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
