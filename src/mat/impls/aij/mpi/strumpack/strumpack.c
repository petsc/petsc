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

extern PetscErrorCode MatFactorInfo_STRUMPACK(Mat,PetscViewer);
extern PetscErrorCode MatLUFactorNumeric_STRUMPACK(Mat,Mat,const MatFactorInfo*);
extern PetscErrorCode MatDestroy_STRUMPACK(Mat);
extern PetscErrorCode MatView_STRUMPACK(Mat,PetscViewer);
extern PetscErrorCode MatSolve_STRUMPACK(Mat,Vec,Vec);
extern PetscErrorCode MatLUFactorSymbolic_STRUMPACK(Mat,Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode MatDestroy_MPIAIJ(Mat);

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_STRUMPACK"
PetscErrorCode MatGetDiagonal_STRUMPACK(Mat A,Vec v)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Mat type: STRUMPACK factor");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_STRUMPACK"
PetscErrorCode MatDestroy_STRUMPACK(Mat A)
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_STRUMPACK"
PetscErrorCode MatSolve_STRUMPACK(Mat A,Vec b_mpi,Vec x)
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
    /* global mat input, convert b to b_seq */
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
PetscErrorCode MatMatSolve_STRUMPACK(Mat A,Mat B_mpi,Mat X)
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
#define __FUNCT__ "MatLUFactorNumeric_STRUMPACK"
PetscErrorCode MatLUFactorNumeric_STRUMPACK(Mat F,Mat A,const MatFactorInfo *info)
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
PetscErrorCode MatLUFactorSymbolic_STRUMPACK(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  F->ops->lufactornumeric = MatLUFactorNumeric_STRUMPACK;
  F->ops->solve           = MatSolve_STRUMPACK;
  F->ops->matsolve        = MatMatSolve_STRUMPACK;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_aij_strumpack"
PetscErrorCode MatFactorGetSolverPackage_aij_strumpack(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSTRUMPACK;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_aij_strumpack"
PETSC_EXTERN PetscErrorCode MatGetFactor_aij_strumpack(Mat A,MatFactorType ftype,Mat *F)
{
  Mat                 B;
  STRUMPACK_data      *sp;
  PetscErrorCode      ierr;
  PetscInt            M=A->rmap->N,N=A->cmap->N;
  PetscMPIInt         size;
  STRUMPACK_INTERFACE iface;
  PetscBool           verb,flg;
  int                 argc;
  char                **args;
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
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_STRUMPACK;
  B->ops->view             = MatView_STRUMPACK;
  B->ops->destroy          = MatDestroy_STRUMPACK;
  B->ops->getdiagonal      = MatGetDiagonal_STRUMPACK;
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_aij_strumpack);CHKERRQ(ierr);
  B->factortype = MAT_FACTOR_LU;
  ierr     = PetscNewLog(B,&sp);CHKERRQ(ierr);
  B->spptr = sp;

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"STRUMPACK Options","Mat");CHKERRQ(ierr);
  sp->MatInputMode = DISTRIBUTED;
  iface = STRUMPACK_MPI_DIST;
  ierr = PetscOptionsEnum("-mat_strumpack_matinput","Matrix input mode (global or distributed)","None",STRUMPACK_MatInputModes,
                          (PetscEnum)sp->MatInputMode,(PetscEnum*)&sp->MatInputMode,NULL);CHKERRQ(ierr);
  if (sp->MatInputMode == DISTRIBUTED && size == 1) sp->MatInputMode = REPLICATED;
  if (sp->MatInputMode == DISTRIBUTED)     iface = STRUMPACK_MPI_DIST;
  else if (sp->MatInputMode == REPLICATED) iface = STRUMPACK_MPI;

  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&flg);
  if (flg) iface = STRUMPACK_MT;

  if (PetscLogPrintInfo) verb = PETSC_TRUE;
  else verb = PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_strumpack_verbose","Print STRUMPACK information","None",verb,&verb,NULL);CHKERRQ(ierr);
  PetscOptionsEnd();

  ierr = PetscGetArgs(&argc,&args);CHKERRQ(ierr);
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(&(sp->S),PetscObjectComm((PetscObject)A),prec,iface,argc,args,verb));
  PetscStackCall("STRUMPACK_set_from_options",STRUMPACK_set_from_options(sp->S));

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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFactorInfo_STRUMPACK"
PetscErrorCode MatFactorInfo_STRUMPACK(Mat A,PetscViewer viewer)
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
PetscErrorCode MatView_STRUMPACK(Mat A,PetscViewer viewer)
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


/*MC
  MATSOLVERSSTRUMPACK - Parallel direct solver package for LU factorization

  Use ./configure --download-strumpack to have PETSc installed with STRUMPACK

  Use -pc_type lu -pc_factor_mat_solver_package strumpack to us this direct solver

   Works with AIJ matrices

.seealso: PCLU

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage

M*/

