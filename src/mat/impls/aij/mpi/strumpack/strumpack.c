#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <StrumpackSparseSolver.h>

/*
    GLOBAL - The sparse matrix and right hand side are all stored initially on process 0. Should be called centralized
    DISTRIBUTED - The sparse matrix and right hand size are initially stored across the entire MPI communicator.
*/
typedef enum {GLOBAL, DISTRIBUTED} STRUMPACK_MatInputMode;
const char *STRUMPACK_MatInputModes[] = {"GLOBAL","DISTRIBUTED","STRUMPACK_MatInputMode","PETSC_",0};

typedef struct {
  STRUMPACK_SparseSolver S;
  STRUMPACK_MatInputMode MatInputMode;
} STRUMPACK_data;

extern PetscErrorCode MatFactorInfo_STRUMPACK_MPI(Mat,PetscViewer);
extern PetscErrorCode MatLUFactorNumeric_STRUMPACK_MPI(Mat,Mat,const MatFactorInfo*);
extern PetscErrorCode MatDestroy_STRUMPACK_MPI(Mat);
extern PetscErrorCode MatView_STRUMPACK_MPI(Mat,PetscViewer);
extern PetscErrorCode MatSolve_STRUMPACK_MPI(Mat,Vec,Vec);
extern PetscErrorCode MatLUFactorSymbolic_STRUMPACK_MPI(Mat,Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode MatDestroy_MPIAIJ(Mat);

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_STRUMPACK_MPI"
PetscErrorCode MatGetDiagonal_STRUMPACK_MPI(Mat A,Vec v)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Mat type: STRUMPACK_MPI factor");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_STRUMPACK_MPI"
PetscErrorCode MatDestroy_STRUMPACK_MPI(Mat A)
{
  STRUMPACK_data *sp = (STRUMPACK_data*)A->spptr;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  /* Deallocate STRUMPACK_MPI storage */
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
#define __FUNCT__ "MatSolve_STRUMPACK_MPI"
PetscErrorCode MatSolve_STRUMPACK_MPI(Mat A,Vec b_mpi,Vec x)
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
  if (size > 1 && sp->MatInputMode == GLOBAL) {
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
  }

  if (size > 1 && sp->MatInputMode == GLOBAL) {
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
#define __FUNCT__ "MatMatSolve_STRUMPACK_MPI"
PetscErrorCode MatMatSolve_STRUMPACK_MPI(Mat A,Mat B_mpi,Mat X)
{
  PetscErrorCode   ierr;
  PetscBool        flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)B_mpi,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  ierr = PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatMatSolve_STRUMPACK_MPI() is not implemented yet");
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_STRUMPACK_MPI"
PetscErrorCode MatLUFactorNumeric_STRUMPACK_MPI(Mat F,Mat A,const MatFactorInfo *info)
{
  STRUMPACK_data        *sp = (STRUMPACK_data*)F->spptr;
  STRUMPACK_RETURN_CODE sp_err;
  Mat                   *tseq,A_seq = NULL;
  Mat_SeqAIJ            *A_d,*A_o;
  PetscErrorCode        ierr;
  PetscInt              M=A->rmap->N,m=A->rmap->n,N=A->cmap->N;
  PetscMPIInt           size;
  IS                    isrow;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);

  if (sp->MatInputMode == GLOBAL) { /* global mat input */
    if (size > 1) { /* convert mpi A to seq mat A */
      ierr = ISCreateStride(PETSC_COMM_SELF,M,0,1,&isrow);CHKERRQ(ierr);
      ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&tseq);CHKERRQ(ierr);
      ierr = ISDestroy(&isrow);CHKERRQ(ierr);
      A_seq = *tseq;
      ierr  = PetscFree(tseq);CHKERRQ(ierr);
      A_d   = (Mat_SeqAIJ*)A_seq->data;
    } else {
      PetscBool flg;
      ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
      if (flg) {
        Mat_MPIAIJ *At = (Mat_MPIAIJ*)A->data;
        A = At->A;
      }
      A_d = (Mat_SeqAIJ*)A->data;
    }
    PetscStackCall("STRUMPACK_set_csr_matrix",STRUMPACK_set_csr_matrix(sp->S,&N,A_d->i,A_d->j,A_d->a,0));
  } else { /* distributed mat input */
    Mat_MPIAIJ *mat = (Mat_MPIAIJ*)A->data;
    A_d = (Mat_SeqAIJ*)(mat->A)->data;
    A_o = (Mat_SeqAIJ*)(mat->B)->data;
    PetscStackCall("STRUMPACK_set_MPIAIJ_matrix",STRUMPACK_set_MPIAIJ_matrix(sp->S,&m,A_d->i,A_d->j,A_d->a,A_o->i,A_o->j,A_o->a,mat->garray));
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

  if (sp->MatInputMode == GLOBAL && size > 1) {
    ierr = MatDestroy(&A_seq);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_STRUMPACK_MPI"
PetscErrorCode MatLUFactorSymbolic_STRUMPACK_MPI(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  F->ops->lufactornumeric = MatLUFactorNumeric_STRUMPACK_MPI;
  F->ops->solve           = MatSolve_STRUMPACK_MPI;
  F->ops->matsolve        = MatMatSolve_STRUMPACK_MPI;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_aij_strumpack_mpi"
PetscErrorCode MatFactorGetSolverPackage_aij_strumpack_mpi(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSTRUMPACK_MPI;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_aij_strumpack_mpi"
PETSC_EXTERN PetscErrorCode MatGetFactor_aij_strumpack_mpi(Mat A,MatFactorType ftype,Mat *F)
{
  Mat                 B;
  STRUMPACK_data      *sp;
  PetscErrorCode      ierr;
  PetscInt            M=A->rmap->N,N=A->cmap->N;
  PetscMPIInt         size;
  STRUMPACK_INTERFACE iface;
  PetscBool           verb;
  PetscInt            argc;
  char                **args;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);

  /* Create the factorization matrix */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,NULL);
  ierr = MatMPIAIJSetPreallocation(B,0,NULL,0,NULL);CHKERRQ(ierr);

  B->ops->lufactorsymbolic = MatLUFactorSymbolic_STRUMPACK_MPI;
  B->ops->view             = MatView_STRUMPACK_MPI;
  B->ops->destroy          = MatDestroy_STRUMPACK_MPI;
  B->ops->getdiagonal      = MatGetDiagonal_STRUMPACK_MPI;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_aij_strumpack_mpi);CHKERRQ(ierr);

  B->factortype = MAT_FACTOR_LU;

  ierr     = PetscNewLog(B,&sp);CHKERRQ(ierr);
  B->spptr = sp;

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"STRUMPACK_MPI Options","Mat");CHKERRQ(ierr);

  sp->MatInputMode = DISTRIBUTED;
  iface = STRUMPACK_MPI_DIST;

  ierr = PetscOptionsEnum("-mat_strumpack_mpi_matinput","Matrix input mode (global or distributed)","None",STRUMPACK_MatInputModes,
                          (PetscEnum)sp->MatInputMode,(PetscEnum*)&sp->MatInputMode,NULL);CHKERRQ(ierr);
  if (sp->MatInputMode == DISTRIBUTED && size == 1) sp->MatInputMode = GLOBAL;
  if (sp->MatInputMode == DISTRIBUTED) {
    iface = STRUMPACK_MPI_DIST;
  } else if (sp->MatInputMode == GLOBAL) {
    iface = STRUMPACK_MPI;
  }
  if (PetscLogPrintInfo) verb = PETSC_TRUE;
  else verb = PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_strumpack_verbose","Print STRUMPACK information","None",verb,&verb,NULL);CHKERRQ(ierr);

  PetscOptionsEnd();

  ierr = PetscGetArgs(&argc,&args);CHKERRQ(ierr);

#if defined(PETSC_USE_64BIT_INDICES)
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(&(sp->S),PetscObjectComm((PetscObject)A),STRUMPACK_FLOATCOMPLEX_64,iface,argc,args,verb));
#else
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(&(sp->S),PetscObjectComm((PetscObject)A),STRUMPACK_DOUBLECOMPLEX_64,iface,argc,args,verb));
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(&(sp->S),PetscObjectComm((PetscObject)A),STRUMPACK_FLOAT_64,iface,argc,args,verb));
#else
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(&(sp->S),PetscObjectComm((PetscObject)A),STRUMPACK_DOUBLE_64,iface,argc,args,verb));
#endif
#endif
#else
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(&(sp->S),PetscObjectComm((PetscObject)A),STRUMPACK_FLOATCOMPLEX,iface,argc,args,verb));
#else
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(&(sp->S),PetscObjectComm((PetscObject)A),STRUMPACK_DOUBLECOMPLEX,iface,argc,args,verb));
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(&(sp->S),PetscObjectComm((PetscObject)A),STRUMPACK_FLOAT,iface,argc,args,verb));
#else
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(&(sp->S),PetscObjectComm((PetscObject)A),STRUMPACK_DOUBLE,iface,argc,args,verb));
#endif
#endif
#endif
  PetscStackCall("STRUMPACK_set_from_options",STRUMPACK_set_from_options(sp->S));

  *F = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolverPackageRegister_STRUMPACK_MPI"
PETSC_EXTERN PetscErrorCode MatSolverPackageRegister_STRUMPACK_MPI(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverPackageRegister(MATSOLVERSTRUMPACK_MPI,MATMPIAIJ,MAT_FACTOR_LU,MatGetFactor_aij_strumpack_mpi);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERSTRUMPACK_MPI,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_aij_strumpack_mpi);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFactorInfo_STRUMPACK_MPI"
PetscErrorCode MatFactorInfo_STRUMPACK_MPI(Mat A,PetscViewer viewer)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* check if matrix is strumpack type */
  if (A->ops->solve != MatSolve_STRUMPACK_MPI) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIPrintf(viewer,"STRUMPACK_MPI sparse solver!\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_STRUMPACK_MPI"
PetscErrorCode MatView_STRUMPACK_MPI(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatFactorInfo_STRUMPACK_MPI(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


/*MC
  MATSOLVERSSTRUMPACK_MPI - Parallel direct solver package for LU factorization

  Use ./configure --download-strumpack --download-parmetis --download-metis --download-ptscotch (and ScaLAPACK?/BLACS?) to have PETSc installed with SuperLU_DIST

  Use -pc_type lu -pc_factor_mat_solver_package strumpack to us this direct solver

   Works with AIJ matrices

.seealso: PCLU

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage

M*/

