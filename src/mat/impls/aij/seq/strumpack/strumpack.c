#include <../src/mat/impls/aij/seq/aij.h>
#include <StrumpackSparseSolver.h>

extern PetscErrorCode MatFactorInfo_STRUMPACK(Mat,PetscViewer);
extern PetscErrorCode MatLUFactorNumeric_STRUMPACK(Mat,Mat,const MatFactorInfo*);
extern PetscErrorCode MatDestroy_STRUMPACK(Mat);
extern PetscErrorCode MatView_STRUMPACK(Mat,PetscViewer);
extern PetscErrorCode MatSolve_STRUMPACK(Mat,Vec,Vec);
extern PetscErrorCode MatMatSolve_STRUMPACK(Mat,Mat,Mat);
extern PetscErrorCode MatSolveTranspose_STRUMPACK(Mat,Vec,Vec);
extern PetscErrorCode MatLUFactorSymbolic_STRUMPACK(Mat,Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode MatDuplicate_STRUMPACK(Mat,MatDuplicateOption,Mat*);

#undef __FUNCT__
#define __FUNCT__ "MatFactorInfo_STRUMPACK"
PetscErrorCode MatFactorInfo_STRUMPACK(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPrintf(viewer,"STRUMPACK sparse solver!\n");CHKERRQ(ierr);
  /* TODO print some more info!? */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_STRUMPACK"
PetscErrorCode MatLUFactorNumeric_STRUMPACK(Mat F,Mat A,const MatFactorInfo *info)
{
  STRUMPACK_SparseSolver *sp = (STRUMPACK_SparseSolver*)(F->spptr);
  STRUMPACK_RETURN_CODE  sp_err;
  PetscErrorCode         ierr;
  PetscInt               m,n;
  Mat_SeqAIJ             *aij = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;

  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  /* check if matrix is square! */
  if (m != n) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Mat type: STRUMPACK only supports square matrices");

  /* Numerical factorization */
  if (F->factortype == MAT_FACTOR_LU) {
    PetscStackCall("STRUMPACK_set_csr_matrix",STRUMPACK_set_csr_matrix(*sp,&m,aij->i,aij->j,aij->a,0));
    PetscStackCall("STRUMPACK_reorder",sp_err=STRUMPACK_reorder(*sp));
    PetscStackCall("STRUMPACK_factor",sp_err=STRUMPACK_factor(*sp));
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");

  if (sp_err != STRUMPACK_SUCCESS) {
    if (sp_err == STRUMPACK_MATRIX_NOT_SET)        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix was not set");
    else if (sp_err == STRUMPACK_REORDERING_ERROR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix reordering failed");
    else                                           SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: factorization failed");
  }

  F->ops->solve          = MatSolve_STRUMPACK;
  F->ops->solvetranspose = NULL;
  F->ops->matsolve       = NULL;
  PetscFunctionReturn(0);
}

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
  STRUMPACK_SparseSolver *sp = (STRUMPACK_SparseSolver*)A->spptr;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (sp) PetscStackCall("STRUMPACK_destroy",STRUMPACK_destroy(sp));
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverPackage_C",NULL);CHKERRQ(ierr);
  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
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


#undef __FUNCT__
#define __FUNCT__ "MatSolve_STRUMPACK_Private"
PetscErrorCode MatSolve_STRUMPACK_Private(Mat A,Vec b,Vec x)
{
  STRUMPACK_SparseSolver *sp = (STRUMPACK_SparseSolver*)(A->spptr);
  const PetscScalar      *barray;
  PetscScalar            *xarray;
  PetscErrorCode         ierr;
  STRUMPACK_RETURN_CODE  sp_err;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(b,&barray);CHKERRQ(ierr);
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
    PetscStackCall("STRUMPACK_solve",sp_err=STRUMPACK_solve(*sp,(PetscScalar*)barray,xarray,0));
    /* }  else if (HSS solve as preconditioner)? */
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");
  ierr = VecRestoreArrayRead(b,&barray);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);
  if (sp_err == STRUMPACK_MATRIX_NOT_SET)        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix was not set");
  else if (sp_err == STRUMPACK_REORDERING_ERROR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix reordering failed");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_STRUMPACK"
PetscErrorCode MatSolve_STRUMPACK(Mat A,Vec b,Vec x)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolve_STRUMPACK_Private(A,b,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveTranspose_STRUMPACK"
PetscErrorCode MatSolveTranspose_STRUMPACK(Mat A,Vec b,Vec x)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatSolveTranspose_STRUMPACK() is not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatSolve_STRUMPACK"
PetscErrorCode MatMatSolve_STRUMPACK(Mat A,Mat B,Mat X)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)B,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  ierr = PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatMatSolve_STRUMPACK() is not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_STRUMPACK"
PetscErrorCode MatLUFactorSymbolic_STRUMPACK(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  F->ops->lufactornumeric = MatLUFactorNumeric_STRUMPACK;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFactorGetSolverPackage_seqaij_strumpack"
PetscErrorCode MatFactorGetSolverPackage_seqaij_strumpack(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSTRUMPACK;
  PetscFunctionReturn(0);
}


/*MC
  MATSOLVERSTRUMPACK = "strumpack" - A solver package providing LU sequential matrices
  via the external package STRUMPACK.

  Use ./configure --download-strumpack to have PETSc installed with STRUMPACK

  Use -pc_type lu -pc_factor_mat_solver_package strumpack to us this direct solver

  Options Database Keys:
 TODO

   Level: beginner

.seealso: PCLU, PCILU, MATSOLVERSUPERLU, MATSOLVERMUMPS, PCFactorSetMatSolverPackage(), MatSolverPackage
M*/

#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_seqaij_strumpack"
PETSC_EXTERN PetscErrorCode MatGetFactor_seqaij_strumpack(Mat A,MatFactorType ftype,Mat *F)
{
  Mat                    B;
  STRUMPACK_SparseSolver *sp;
  PetscErrorCode         ierr;
  PetscBool              verb;
  PetscInt               argc;
  char                   **args;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,NULL);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic  = MatLUFactorSymbolic_STRUMPACK;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");

  B->ops->destroy     = MatDestroy_STRUMPACK;
  B->ops->view        = MatView_STRUMPACK;
  B->ops->getdiagonal = MatGetDiagonal_STRUMPACK;
  B->factortype       = ftype;
  B->assembled        = PETSC_TRUE;           /* required by -ksp_view */
  B->preallocated     = PETSC_TRUE;

  ierr = PetscNewLog(B,&sp);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_seqaij_strumpack);CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"STRUMPACK Options","Mat");CHKERRQ(ierr);
  if (PetscLogPrintInfo) verb = PETSC_TRUE;
  else verb = PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_strumpack_verbose","Print STRUMPACK information","None",verb,&verb,NULL);CHKERRQ(ierr);

  ierr = PetscGetArgs(&argc,&args);CHKERRQ(ierr);

#if defined(PETSC_USE_64BIT_INDICES)
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(sp,0,STRUMPACK_FLOATCOMPLEX_64,STRUMPACK_MT,argc,args,verb));
#else
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(sp,0,STRUMPACK_DOUBLECOMPLEX_64,STRUMPACK_MT,argc,args,verb));
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(sp,0,STRUMPACK_FLOAT_64,STRUMPACK_MT,argc,args,verb));
#else
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(sp,0,STRUMPACK_DOUBLE_64,STRUMPACK_MT,argc,args,verb));
#endif
#endif
#else
#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(sp,0,STRUMPACK_FLOATCOMPLEX,STRUMPACK_MT,argc,args,verb));
#else
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(sp,0,STRUMPACK_DOUBLECOMPLEX,STRUMPACK_MT,argc,args,verb));
#endif
#else
#if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(sp,0,STRUMPACK_FLOAT,STRUMPACK_MT,argc,args,verb));
#else
  PetscStackCall("STRUMPACK_init",STRUMPACK_init(sp,0,STRUMPACK_DOUBLE,STRUMPACK_MT,argc,args,verb));
#endif
#endif
#endif

  PetscOptionsEnd();

  B->spptr = sp;
  *F       = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolverPackageRegister_STRUMPACK"
PETSC_EXTERN PetscErrorCode MatSolverPackageRegister_STRUMPACK(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverPackageRegister(MATSOLVERSTRUMPACK,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_seqaij_strumpack);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
