#include <../src/mat/impls/aij/seq/aij.h>            /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <StrumpackSparseSolver.h>

static PetscErrorCode MatGetDiagonal_STRUMPACK(Mat A,Vec v)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Mat type: STRUMPACK factor");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_STRUMPACK(Mat A)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)A->spptr;
  PetscErrorCode         ierr;
  PetscBool              flg;

  PetscFunctionBegin;
  /* Deallocate STRUMPACK storage */
  PetscStackCall("STRUMPACK_destroy",STRUMPACK_destroy(S));
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

static PetscErrorCode MatSTRUMPACKSetHSSRelCompTol_STRUMPACK(Mat F,PetscReal rctol)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_rctol", STRUMPACK_set_rctol(*S,rctol));
  PetscFunctionReturn(0);
}

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

static PetscErrorCode MatSTRUMPACKSetHSSMinSize_STRUMPACK(Mat F,PetscInt hssminsize)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_minimum_HSS_size", STRUMPACK_set_minimum_HSS_size(*S,hssminsize));
  PetscFunctionReturn(0);
}

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

static PetscErrorCode MatSTRUMPACKSetColPerm_STRUMPACK(Mat F,PetscBool cperm)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_mc64job",STRUMPACK_set_mc64job(*S,cperm ? 5 : 0));
  PetscFunctionReturn(0);
}

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

static PetscErrorCode MatSolve_STRUMPACK(Mat A,Vec b_mpi,Vec x)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)A->spptr;
  STRUMPACK_RETURN_CODE  sp_err;
  PetscErrorCode         ierr;
  const PetscScalar      *bptr;
  PetscScalar            *xptr;

  PetscFunctionBegin;
  ierr = VecGetArray(x,&xptr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(b_mpi,&bptr);CHKERRQ(ierr);

  PetscStackCall("STRUMPACK_solve",sp_err = STRUMPACK_solve(*S,(PetscScalar*)bptr,xptr,0));
  switch (sp_err) {
  case STRUMPACK_SUCCESS: break;
  case STRUMPACK_MATRIX_NOT_SET:   { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix was not set"); break; }
  case STRUMPACK_REORDERING_ERROR: { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix reordering failed"); break; }
  default:                           SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: solve failed");
  }
  ierr = VecRestoreArray(x,&xptr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(b_mpi,&bptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

static PetscErrorCode MatFactorInfo_STRUMPACK(Mat A,PetscViewer viewer)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* check if matrix is strumpack type */
  if (A->ops->solve != MatSolve_STRUMPACK) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIPrintf(viewer,"STRUMPACK sparse solver!\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

static PetscErrorCode MatLUFactorNumeric_STRUMPACK(Mat F,Mat A,const MatFactorInfo *info)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;
  STRUMPACK_RETURN_CODE  sp_err;
  Mat_SeqAIJ             *A_d,*A_o;
  Mat_MPIAIJ             *mat;
  PetscErrorCode         ierr;
  PetscInt               M=A->rmap->N,m=A->rmap->n;
  PetscBool              flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
  if (flg) { /* A is MATMPIAIJ */
    mat = (Mat_MPIAIJ*)A->data;
    A_d = (Mat_SeqAIJ*)(mat->A)->data;
    A_o = (Mat_SeqAIJ*)(mat->B)->data;
    PetscStackCall("STRUMPACK_set_MPIAIJ_matrix",STRUMPACK_set_MPIAIJ_matrix(*S,&m,A_d->i,A_d->j,A_d->a,A_o->i,A_o->j,A_o->a,mat->garray));
  } else { /* A is MATSEQAIJ */
    A_d = (Mat_SeqAIJ*)A->data;
    PetscStackCall("STRUMPACK_set_csr_matrix",STRUMPACK_set_csr_matrix(*S,&M,A_d->i,A_d->j,A_d->a,0));
  }

  /* Reorder and Factor the matrix. */
  /* TODO figure out how to avoid reorder if the matrix values changed, but the pattern remains the same. */
  PetscStackCall("STRUMPACK_reorder",sp_err = STRUMPACK_reorder(*S));
  PetscStackCall("STRUMPACK_factor",sp_err = STRUMPACK_factor(*S));
  switch (sp_err) {
  case STRUMPACK_SUCCESS: break;
  case STRUMPACK_MATRIX_NOT_SET:   { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix was not set"); break; }
  case STRUMPACK_REORDERING_ERROR: { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix reordering failed"); break; }
  default:                           SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: factorization failed");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_STRUMPACK(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  F->ops->lufactornumeric = MatLUFactorNumeric_STRUMPACK;
  F->ops->solve           = MatSolve_STRUMPACK;
  F->ops->matsolve        = MatMatSolve_STRUMPACK;
  PetscFunctionReturn(0);
}

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
static PetscErrorCode MatGetFactor_aij_strumpack(Mat A,MatFactorType ftype,Mat *F)
{
  Mat                    B;
  STRUMPACK_SparseSolver *S;
  PetscErrorCode         ierr;
  PetscInt               M=A->rmap->N,N=A->cmap->N;
  STRUMPACK_INTERFACE    iface;
  PetscBool              verb,flg,set;
  PetscReal              rctol;
  PetscInt               hssminsize;
  int                    argc;
  char                   **args,*copts,*pname;
  size_t                 len;
  const STRUMPACK_PRECISION table[2][2][2] = {{{STRUMPACK_FLOATCOMPLEX_64,STRUMPACK_DOUBLECOMPLEX_64},
                                               {STRUMPACK_FLOAT_64,STRUMPACK_DOUBLE_64}},
                                              {{STRUMPACK_FLOATCOMPLEX,STRUMPACK_DOUBLECOMPLEX},
                                               {STRUMPACK_FLOAT,STRUMPACK_DOUBLE}}};
  const STRUMPACK_PRECISION prec = table[(sizeof(PetscInt)==8)?0:1][(PETSC_SCALAR==PETSC_COMPLEX)?0:1][(PETSC_REAL==PETSC_FLOAT)?0:1];

  PetscFunctionBegin;
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
  B->ops->view        = MatView_STRUMPACK;
  B->ops->destroy     = MatDestroy_STRUMPACK;
  B->ops->getdiagonal = MatGetDiagonal_STRUMPACK;
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_aij_strumpack);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetHSSRelCompTol_C",MatSTRUMPACKSetHSSRelCompTol_STRUMPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetHSSMinSize_C",MatSTRUMPACKSetHSSMinSize_STRUMPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetColPerm_C",MatSTRUMPACKSetColPerm_STRUMPACK);CHKERRQ(ierr);
  B->factortype = ftype;
  B->assembled  = PETSC_TRUE;           /* required by -ksp_view */
  ierr     = PetscNewLog(B,&S);CHKERRQ(ierr);
  B->spptr = S;

  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&flg);
  iface = flg ? STRUMPACK_MT : STRUMPACK_MPI_DIST;

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"STRUMPACK Options","Mat");CHKERRQ(ierr);

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

  PetscStackCall("STRUMPACK_init",STRUMPACK_init(S,PetscObjectComm((PetscObject)A),prec,iface,argc,args,verb));

  PetscStackCall("STRUMPACK_get_rctol",rctol = (PetscReal)STRUMPACK_get_rctol(*S));
  ierr = PetscOptionsReal("-mat_strumpack_rctol","Relative compression tolerance","None",rctol,&rctol,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_rctol",STRUMPACK_set_rctol(*S,(double)rctol));

  PetscStackCall("STRUMPACK_get_mc64job",flg = (STRUMPACK_get_mc64job(*S) == 0) ? PETSC_FALSE : PETSC_TRUE);
  ierr = PetscOptionsBool("-mat_strumpack_colperm","Find a col perm to get nonzero diagonal","None",flg,&flg,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_mc64job",STRUMPACK_set_mc64job(*S,flg ? 5 : 0));

  PetscStackCall("STRUMPACK_get_minimum_HSS_size",hssminsize = (PetscInt)STRUMPACK_get_minimum_HSS_size(*S));
  ierr = PetscOptionsInt("-mat_strumpack_hssminsize","Minimum size of dense block for HSS compression","None",hssminsize,&hssminsize,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_minimum_HSS_size",STRUMPACK_set_minimum_HSS_size(*S,(int)hssminsize));

  PetscOptionsEnd();

  if (ftype == MAT_FACTOR_ILU) {
    /* When enabling HSS compression, the STRUMPACK solver becomes an incomplete                */
    /* (or approximate) LU factorization.                                                       */
    PetscStackCall("STRUMPACK_use_HSS",STRUMPACK_use_HSS(*S,1));
    /* Disable the outer iterative solver from STRUMPACK.                                       */
    /* When STRUMPACK is used as a direct solver, it will by default do iterative refinement.   */
    /* When STRUMPACK is used with as an approximate factorization preconditioner (by enabling  */
    /* low-rank compression), it will use it's own GMRES. Here we can disable the               */
    /* outer iterative solver, as PETSc uses STRUMPACK from within a KSP.                       */
    PetscStackCall("STRUMPACK_set_Krylov_solver", STRUMPACK_set_Krylov_solver(*S, STRUMPACK_DIRECT));
  }

  /* Allow the user to set or overwrite the above options from the command line                 */
  PetscStackCall("STRUMPACK_set_from_options",STRUMPACK_set_from_options(*S));
  ierr = PetscStrToArrayDestroy(argc,args);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

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
