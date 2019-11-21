/*
        Provides an interface to the SuperLU_DIST sparse solver
*/

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

EXTERN_C_BEGIN
#if defined(PETSC_USE_COMPLEX)
#include <superlu_zdefs.h>
#else
#include <superlu_ddefs.h>
#endif
EXTERN_C_END

typedef struct {
  int_t                  nprow,npcol,*row,*col;
  gridinfo_t             grid;
  superlu_dist_options_t options;
  SuperMatrix            A_sup;
  ScalePermstruct_t      ScalePermstruct;
  LUstruct_t             LUstruct;
  int                    StatPrint;
  SOLVEstruct_t          SOLVEstruct;
  fact_t                 FactPattern;
  MPI_Comm               comm_superlu;
#if defined(PETSC_USE_COMPLEX)
  doublecomplex          *val;
#else
  double                 *val;
#endif
  PetscBool              matsolve_iscalled,matmatsolve_iscalled;
  PetscBool              CleanUpSuperLU_Dist;  /* Flag to clean up (non-global) SuperLU objects during Destroy */
} Mat_SuperLU_DIST;


PetscErrorCode MatSuperluDistGetDiagU_SuperLU_DIST(Mat F,PetscScalar *diagU)
{
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)F->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscStackCall("SuperLU_DIST:pzGetDiagU",pzGetDiagU(F->rmap->N,&lu->LUstruct,&lu->grid,(doublecomplex*)diagU));
#else
  PetscStackCall("SuperLU_DIST:pdGetDiagU",pdGetDiagU(F->rmap->N,&lu->LUstruct,&lu->grid,diagU));
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatSuperluDistGetDiagU(Mat F,PetscScalar *diagU)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  ierr = PetscTryMethod(F,"MatSuperluDistGetDiagU_C",(Mat,PetscScalar*),(F,diagU));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SuperLU_DIST(Mat A)
{
  PetscErrorCode   ierr;
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)A->data;

  PetscFunctionBegin;
  if (lu->CleanUpSuperLU_Dist) {
    /* Deallocate SuperLU_DIST storage */
    PetscStackCall("SuperLU_DIST:Destroy_CompRowLoc_Matrix_dist",Destroy_CompRowLoc_Matrix_dist(&lu->A_sup));
    if (lu->options.SolveInitialized) {
#if defined(PETSC_USE_COMPLEX)
      PetscStackCall("SuperLU_DIST:zSolveFinalize",zSolveFinalize(&lu->options, &lu->SOLVEstruct));
#else
      PetscStackCall("SuperLU_DIST:dSolveFinalize",dSolveFinalize(&lu->options, &lu->SOLVEstruct));
#endif
    }
    PetscStackCall("SuperLU_DIST:Destroy_LU",Destroy_LU(A->cmap->N, &lu->grid, &lu->LUstruct));
    PetscStackCall("SuperLU_DIST:ScalePermstructFree",ScalePermstructFree(&lu->ScalePermstruct));
    PetscStackCall("SuperLU_DIST:LUstructFree",LUstructFree(&lu->LUstruct));

    /* Release the SuperLU_DIST process grid. */
    PetscStackCall("SuperLU_DIST:superlu_gridexit",superlu_gridexit(&lu->grid));
    ierr = MPI_Comm_free(&(lu->comm_superlu));CHKERRQ(ierr);
  }
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSuperluDistGetDiagU_C",NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SuperLU_DIST(Mat A,Vec b_mpi,Vec x)
{
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)A->data;
  PetscErrorCode   ierr;
  PetscInt         m=A->rmap->n;
  SuperLUStat_t    stat;
  double           berr[1];
  PetscScalar      *bptr=NULL;
  int              info; /* SuperLU_Dist info code is ALWAYS an int, even with long long indices */
  static PetscBool cite = PETSC_FALSE;

  PetscFunctionBegin;
  if (lu->options.Fact != FACTORED) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"SuperLU_DIST options.Fact must equal FACTORED");
  ierr = PetscCitationsRegister("@article{lidemmel03,\n  author = {Xiaoye S. Li and James W. Demmel},\n  title = {{SuperLU_DIST}: A Scalable Distributed-Memory Sparse Direct\n           Solver for Unsymmetric Linear Systems},\n  journal = {ACM Trans. Mathematical Software},\n  volume = {29},\n  number = {2},\n  pages = {110-140},\n  year = 2003\n}\n",&cite);CHKERRQ(ierr);

  if (lu->options.SolveInitialized && !lu->matsolve_iscalled) {
    /* see comments in MatMatSolve() */
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:zSolveFinalize",zSolveFinalize(&lu->options, &lu->SOLVEstruct));
#else
    PetscStackCall("SuperLU_DIST:dSolveFinalize",dSolveFinalize(&lu->options, &lu->SOLVEstruct));
#endif
    lu->options.SolveInitialized = NO;
  }
  ierr = VecCopy(b_mpi,x);CHKERRQ(ierr);
  ierr = VecGetArray(x,&bptr);CHKERRQ(ierr);

  PetscStackCall("SuperLU_DIST:PStatInit",PStatInit(&stat));        /* Initialize the statistics variables. */
#if defined(PETSC_USE_COMPLEX)
  PetscStackCall("SuperLU_DIST:pzgssvx",pzgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,(doublecomplex*)bptr,m,1,&lu->grid,&lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));
#else
  PetscStackCall("SuperLU_DIST:pdgssvx",pdgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,bptr,m,1,&lu->grid,&lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));
#endif
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"pdgssvx fails, info: %d\n",info);

  if (lu->options.PrintStat) PetscStackCall("SuperLU_DIST:PStatPrint",PStatPrint(&lu->options, &stat, &lu->grid));  /* Print the statistics. */
  PetscStackCall("SuperLU_DIST:PStatFree",PStatFree(&stat));

  ierr = VecRestoreArray(x,&bptr);CHKERRQ(ierr);
  lu->matsolve_iscalled    = PETSC_TRUE;
  lu->matmatsolve_iscalled = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SuperLU_DIST(Mat A,Mat B_mpi,Mat X)
{
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)A->data;
  PetscErrorCode   ierr;
  PetscInt         m=A->rmap->n,nrhs;
  SuperLUStat_t    stat;
  double           berr[1];
  PetscScalar      *bptr;
  int              info; /* SuperLU_Dist info code is ALWAYS an int, even with long long indices */
  PetscBool        flg;

  PetscFunctionBegin;
  if (lu->options.Fact != FACTORED) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"SuperLU_DIST options.Fact must equal FACTORED");
  ierr = PetscObjectTypeCompareAny((PetscObject)B_mpi,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  if (X != B_mpi) {
    ierr = PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
    if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");
  }

  if (lu->options.SolveInitialized && !lu->matmatsolve_iscalled) {
    /* communication pattern of SOLVEstruct is unlikely created for matmatsolve,
       thus destroy it and create a new SOLVEstruct.
       Otherwise it may result in memory corruption or incorrect solution
       See src/mat/examples/tests/ex125.c */
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:zSolveFinalize",zSolveFinalize(&lu->options, &lu->SOLVEstruct));
#else
    PetscStackCall("SuperLU_DIST:dSolveFinalize",dSolveFinalize(&lu->options, &lu->SOLVEstruct));
#endif
    lu->options.SolveInitialized = NO;
  }
  if (X != B_mpi) {
    ierr = MatCopy(B_mpi,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }

  ierr = MatGetSize(B_mpi,NULL,&nrhs);CHKERRQ(ierr);

  PetscStackCall("SuperLU_DIST:PStatInit",PStatInit(&stat));        /* Initialize the statistics variables. */
  ierr = MatDenseGetArray(X,&bptr);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
  PetscStackCall("SuperLU_DIST:pzgssvx",pzgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,(doublecomplex*)bptr,m,nrhs,&lu->grid, &lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));
#else
  PetscStackCall("SuperLU_DIST:pdgssvx",pdgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,bptr,m,nrhs,&lu->grid,&lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));
#endif

  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"pdgssvx fails, info: %d\n",info);
  ierr = MatDenseRestoreArray(X,&bptr);CHKERRQ(ierr);

  if (lu->options.PrintStat) PetscStackCall("SuperLU_DIST:PStatPrint",PStatPrint(&lu->options, &stat, &lu->grid));  /* Print the statistics. */
  PetscStackCall("SuperLU_DIST:PStatFree",PStatFree(&stat));
  lu->matsolve_iscalled    = PETSC_FALSE;
  lu->matmatsolve_iscalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*
  input:
   F:        numeric Cholesky factor
  output:
   nneg:     total number of negative pivots
   nzero:    total number of zero pivots
   npos:     (global dimension of F) - nneg - nzero
*/
static PetscErrorCode MatGetInertia_SuperLU_DIST(Mat F,PetscInt *nneg,PetscInt *nzero,PetscInt *npos)
{
  PetscErrorCode   ierr;
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)F->data;
  PetscScalar      *diagU=NULL;
  PetscInt         M,i,neg=0,zero=0,pos=0;
  PetscReal        r;

  PetscFunctionBegin;
  if (!F->assembled) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix factor F is not assembled");
  if (lu->options.RowPerm != NOROWPERM) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must set NOROWPERM");
  ierr = MatGetSize(F,&M,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(M,&diagU);CHKERRQ(ierr);
  ierr = MatSuperluDistGetDiagU(F,diagU);CHKERRQ(ierr);
  for (i=0; i<M; i++) {
#if defined(PETSC_USE_COMPLEX)
    r = PetscImaginaryPart(diagU[i])/10.0;
    if (r< -PETSC_MACHINE_EPSILON || r>PETSC_MACHINE_EPSILON) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"diagU[%d]=%g + i %g is non-real",i,PetscRealPart(diagU[i]),r*10.0);
    r = PetscRealPart(diagU[i]);
#else
    r = diagU[i];
#endif
    if (r > 0) {
      pos++;
    } else if (r < 0) {
      neg++;
    } else zero++;
  }

  ierr = PetscFree(diagU);CHKERRQ(ierr);
  if (nneg)  *nneg  = neg;
  if (nzero) *nzero = zero;
  if (npos)  *npos  = pos;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_SuperLU_DIST(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_SuperLU_DIST  *lu = (Mat_SuperLU_DIST*)F->data;
  Mat               Aloc;
  const PetscScalar *av;
  const PetscInt    *ai=NULL,*aj=NULL;
  PetscInt          nz,dummy;
  int               sinfo;   /* SuperLU_Dist info flag is always an int even with long long indices */
  SuperLUStat_t     stat;
  double            *berr=0;
  PetscBool         ismpiaij,isseqaij,flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  if (ismpiaij) {
    ierr = MatMPIAIJGetLocalMat(A,MAT_INITIAL_MATRIX,&Aloc);CHKERRQ(ierr);
  } else if (isseqaij) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    Aloc = A;
  } else SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for type %s",((PetscObject)A)->type_name);

  ierr = MatGetRowIJ(Aloc,0,PETSC_FALSE,PETSC_FALSE,&dummy,&ai,&aj,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GetRowIJ failed");
  ierr = MatSeqAIJGetArrayRead(Aloc,&av);CHKERRQ(ierr);
  nz   = ai[Aloc->rmap->n];

  /* Allocations for A_sup */
  if (lu->options.Fact == DOFACT) { /* first numeric factorization */
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:zallocateA_dist",zallocateA_dist(Aloc->rmap->n, nz, &lu->val, &lu->col, &lu->row));
#else
    PetscStackCall("SuperLU_DIST:dallocateA_dist",dallocateA_dist(Aloc->rmap->n, nz, &lu->val, &lu->col, &lu->row));
#endif
  } else { /* successive numeric factorization, sparsity pattern and perm_c are reused. */
    if (lu->FactPattern == SamePattern_SameRowPerm) {
      lu->options.Fact = SamePattern_SameRowPerm; /* matrix has similar numerical values */
    } else if (lu->FactPattern == SamePattern) {
      PetscStackCall("SuperLU_DIST:Destroy_LU",Destroy_LU(A->rmap->N, &lu->grid, &lu->LUstruct)); /* Deallocate L and U matrices. */
      lu->options.Fact = SamePattern;
    } else if (lu->FactPattern == DOFACT) {
      PetscStackCall("SuperLU_DIST:Destroy_CompRowLoc_Matrix_dist",Destroy_CompRowLoc_Matrix_dist(&lu->A_sup));
      PetscStackCall("SuperLU_DIST:Destroy_LU",Destroy_LU(A->rmap->N, &lu->grid, &lu->LUstruct));
      lu->options.Fact = DOFACT;

#if defined(PETSC_USE_COMPLEX)
      PetscStackCall("SuperLU_DIST:zallocateA_dist",zallocateA_dist(Aloc->rmap->n, nz, &lu->val, &lu->col, &lu->row));
#else
      PetscStackCall("SuperLU_DIST:dallocateA_dist",dallocateA_dist(Aloc->rmap->n, nz, &lu->val, &lu->col, &lu->row));
#endif
    } else {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"options.Fact must be one of SamePattern SamePattern_SameRowPerm DOFACT");
    }
  }

  /* Copy AIJ matrix to superlu_dist matrix */
  ierr = PetscArraycpy(lu->row,ai,Aloc->rmap->n+1);CHKERRQ(ierr);
  ierr = PetscArraycpy(lu->col,aj,nz);CHKERRQ(ierr);
  ierr = PetscArraycpy(lu->val,av,nz);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(Aloc,0,PETSC_FALSE,PETSC_FALSE,&dummy,&ai,&aj,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"RestoreRowIJ failed");
  ierr = MatSeqAIJRestoreArrayRead(Aloc,&av);CHKERRQ(ierr);
  ierr = MatDestroy(&Aloc);CHKERRQ(ierr);

  /* Create and setup A_sup */
  if (lu->options.Fact == DOFACT) {
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:zCreate_CompRowLoc_Matrix_dist",zCreate_CompRowLoc_Matrix_dist(&lu->A_sup, A->rmap->N, A->cmap->N, nz, A->rmap->n, A->rmap->rstart, lu->val, lu->col, lu->row, SLU_NR_loc, SLU_Z, SLU_GE));
#else
    PetscStackCall("SuperLU_DIST:dCreate_CompRowLoc_Matrix_dist",dCreate_CompRowLoc_Matrix_dist(&lu->A_sup, A->rmap->N, A->cmap->N, nz, A->rmap->n, A->rmap->rstart, lu->val, lu->col, lu->row, SLU_NR_loc, SLU_D, SLU_GE));
#endif
  }

  /* Factor the matrix. */
  PetscStackCall("SuperLU_DIST:PStatInit",PStatInit(&stat));   /* Initialize the statistics variables. */
#if defined(PETSC_USE_COMPLEX)
  PetscStackCall("SuperLU_DIST:pzgssvx",pzgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, A->rmap->n, 0, &lu->grid, &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &sinfo));
#else
  PetscStackCall("SuperLU_DIST:pdgssvx",pdgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, A->rmap->n, 0, &lu->grid, &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &sinfo));
#endif

  if (sinfo > 0) {
    if (A->erroriffailure) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot in row %D",sinfo);
    } else {
      if (sinfo <= lu->A_sup.ncol) {
        F->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
        ierr = PetscInfo1(F,"U(i,i) is exactly zero, i= %D\n",sinfo);CHKERRQ(ierr);
      } else if (sinfo > lu->A_sup.ncol) {
        /*
         number of bytes allocated when memory allocation
         failure occurred, plus A->ncol.
         */
        F->factorerrortype = MAT_FACTOR_OUTMEMORY;
        ierr = PetscInfo1(F,"Number of bytes allocated when memory allocation fails %D\n",sinfo);CHKERRQ(ierr);
      }
    }
  } else if (sinfo < 0) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB, "info = %D, argument in p*gssvx() had an illegal value", sinfo);
  }

  if (lu->options.PrintStat) {
    PetscStackCall("SuperLU_DIST:PStatPrint",PStatPrint(&lu->options, &stat, &lu->grid));  /* Print the statistics. */
  }
  PetscStackCall("SuperLU_DIST:PStatFree",PStatFree(&stat));
  F->assembled     = PETSC_TRUE;
  F->preallocated  = PETSC_TRUE;
  lu->options.Fact = FACTORED; /* The factored form of A is supplied. Local option used by this func. only */
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
static PetscErrorCode MatLUFactorSymbolic_SuperLU_DIST(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)F->data;
  PetscInt         M   = A->rmap->N,N=A->cmap->N;

  PetscFunctionBegin;
  /* Initialize the SuperLU process grid. */
  PetscStackCall("SuperLU_DIST:superlu_gridinit",superlu_gridinit(lu->comm_superlu, lu->nprow, lu->npcol, &lu->grid));

  /* Initialize ScalePermstruct and LUstruct. */
  PetscStackCall("SuperLU_DIST:ScalePermstructInit",ScalePermstructInit(M, N, &lu->ScalePermstruct));
  PetscStackCall("SuperLU_DIST:LUstructInit",LUstructInit(N, &lu->LUstruct));
  F->ops->lufactornumeric = MatLUFactorNumeric_SuperLU_DIST;
  F->ops->solve           = MatSolve_SuperLU_DIST;
  F->ops->matsolve        = MatMatSolve_SuperLU_DIST;
  F->ops->getinertia      = NULL;

  if (A->symmetric || A->hermitian) F->ops->getinertia = MatGetInertia_SuperLU_DIST;
  lu->CleanUpSuperLU_Dist = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorSymbolic_SuperLU_DIST(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatLUFactorSymbolic_SuperLU_DIST(F,A,r,r,info);CHKERRQ(ierr);
  F->ops->choleskyfactornumeric = MatLUFactorNumeric_SuperLU_DIST;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverType_aij_superlu_dist(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSUPERLU_DIST;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_Info_SuperLU_DIST(Mat A,PetscViewer viewer)
{
  Mat_SuperLU_DIST       *lu=(Mat_SuperLU_DIST*)A->data;
  superlu_dist_options_t options;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* check if matrix is superlu_dist type */
  if (A->ops->solve != MatSolve_SuperLU_DIST) PetscFunctionReturn(0);

  options = lu->options;
  ierr    = PetscViewerASCIIPrintf(viewer,"SuperLU_DIST run parameters:\n");CHKERRQ(ierr);
  ierr    = PetscViewerASCIIPrintf(viewer,"  Process grid nprow %D x npcol %D \n",lu->nprow,lu->npcol);CHKERRQ(ierr);
  ierr    = PetscViewerASCIIPrintf(viewer,"  Equilibrate matrix %s \n",PetscBools[options.Equil != NO]);CHKERRQ(ierr);
  ierr    = PetscViewerASCIIPrintf(viewer,"  Replace tiny pivots %s \n",PetscBools[options.ReplaceTinyPivot != NO]);CHKERRQ(ierr);
  ierr    = PetscViewerASCIIPrintf(viewer,"  Use iterative refinement %s \n",PetscBools[options.IterRefine == SLU_DOUBLE]);CHKERRQ(ierr);
  ierr    = PetscViewerASCIIPrintf(viewer,"  Processors in row %d col partition %d \n",lu->nprow,lu->npcol);CHKERRQ(ierr);

  switch (options.RowPerm) {
  case NOROWPERM:
    ierr = PetscViewerASCIIPrintf(viewer,"  Row permutation NOROWPERM\n");CHKERRQ(ierr);
    break;
  case LargeDiag_MC64:
    ierr = PetscViewerASCIIPrintf(viewer,"  Row permutation LargeDiag_MC64\n");CHKERRQ(ierr);
    break;
  case LargeDiag_AWPM:
    ierr = PetscViewerASCIIPrintf(viewer,"  Row permutation LargeDiag_AWPM\n");CHKERRQ(ierr);
    break;
  case MY_PERMR:
    ierr = PetscViewerASCIIPrintf(viewer,"  Row permutation MY_PERMR\n");CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown column permutation");
  }

  switch (options.ColPerm) {
  case NATURAL:
    ierr = PetscViewerASCIIPrintf(viewer,"  Column permutation NATURAL\n");CHKERRQ(ierr);
    break;
  case MMD_AT_PLUS_A:
    ierr = PetscViewerASCIIPrintf(viewer,"  Column permutation MMD_AT_PLUS_A\n");CHKERRQ(ierr);
    break;
  case MMD_ATA:
    ierr = PetscViewerASCIIPrintf(viewer,"  Column permutation MMD_ATA\n");CHKERRQ(ierr);
    break;
  /*  Even though this is called METIS, the SuperLU_DIST code sets this by default if PARMETIS is defined, not METIS */
  case METIS_AT_PLUS_A:
    ierr = PetscViewerASCIIPrintf(viewer,"  Column permutation METIS_AT_PLUS_A\n");CHKERRQ(ierr);
    break;
  case PARMETIS:
    ierr = PetscViewerASCIIPrintf(viewer,"  Column permutation PARMETIS\n");CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown column permutation");
  }

  ierr = PetscViewerASCIIPrintf(viewer,"  Parallel symbolic factorization %s \n",PetscBools[options.ParSymbFact != NO]);CHKERRQ(ierr);

  if (lu->FactPattern == SamePattern) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Repeated factorization SamePattern\n");CHKERRQ(ierr);
  } else if (lu->FactPattern == SamePattern_SameRowPerm) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Repeated factorization SamePattern_SameRowPerm\n");CHKERRQ(ierr);
  } else if (lu->FactPattern == DOFACT) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Repeated factorization DOFACT\n");CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown factorization pattern");
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SuperLU_DIST(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatView_Info_SuperLU_DIST(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetFactor_aij_superlu_dist(Mat A,MatFactorType ftype,Mat *F)
{
  Mat                    B;
  Mat_SuperLU_DIST       *lu;
  PetscErrorCode         ierr;
  PetscInt               M=A->rmap->N,N=A->cmap->N,indx;
  PetscMPIInt            size;
  superlu_dist_options_t options;
  PetscBool              flg;
  const char             *colperm[]     = {"NATURAL","MMD_AT_PLUS_A","MMD_ATA","METIS_AT_PLUS_A","PARMETIS"};
  const char             *rowperm[]     = {"NOROWPERM","LargeDiag_MC64","LargeDiag_AWPM","MY_PERMR"};
  const char             *factPattern[] = {"SamePattern","SamePattern_SameRowPerm","DOFACT"};
  PetscBool              set;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,M,N);CHKERRQ(ierr);
  ierr = PetscStrallocpy("superlu_dist",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  B->ops->getinfo = MatGetInfo_External;
  B->ops->view    = MatView_SuperLU_DIST;
  B->ops->destroy = MatDestroy_SuperLU_DIST;

  /* Set the default input options:
     options.Fact              = DOFACT;
     options.Equil             = YES;
     options.ParSymbFact       = NO;
     options.ColPerm           = METIS_AT_PLUS_A;
     options.RowPerm           = LargeDiag_MC64;
     options.ReplaceTinyPivot  = YES;
     options.IterRefine        = DOUBLE;
     options.Trans             = NOTRANS;
     options.SolveInitialized  = NO; -hold the communication pattern used MatSolve() and MatMatSolve()
     options.RefineInitialized = NO;
     options.PrintStat         = YES;
     options.SymPattern        = NO;
  */
  set_default_options_dist(&options);

  if (ftype == MAT_FACTOR_LU) {
    B->factortype = MAT_FACTOR_LU;
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_SuperLU_DIST;
  } else {
    B->factortype = MAT_FACTOR_CHOLESKY;
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SuperLU_DIST;
    options.SymPattern = YES;
  }

  /* set solvertype */
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERSUPERLU_DIST,&B->solvertype);CHKERRQ(ierr);

  ierr    = PetscNewLog(B,&lu);CHKERRQ(ierr);
  B->data = lu;

  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)A),&(lu->comm_superlu));CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  /* Default num of process columns and rows */
  lu->nprow = (int_t) (0.5 + PetscSqrtReal((PetscReal)size));
  if (!lu->nprow) lu->nprow = 1;
  while (lu->nprow > 0) {
    lu->npcol = (int_t) (size/lu->nprow);
    if (size == lu->nprow * lu->npcol) break;
    lu->nprow--;
  }

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"SuperLU_Dist Options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_superlu_dist_r","Number rows in processor partition","None",lu->nprow,(PetscInt*)&lu->nprow,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_superlu_dist_c","Number columns in processor partition","None",lu->npcol,(PetscInt*)&lu->npcol,NULL);CHKERRQ(ierr);
  if (size != lu->nprow * lu->npcol) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number of processes %d must equal to nprow %d * npcol %d",size,lu->nprow,lu->npcol);

  ierr = PetscOptionsBool("-mat_superlu_dist_equil","Equilibrate matrix","None",options.Equil ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set && !flg) options.Equil = NO;

  ierr = PetscOptionsEList("-mat_superlu_dist_rowperm","Row permutation","None",rowperm,4,rowperm[1],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (indx) {
    case 0:
      options.RowPerm = NOROWPERM;
      break;
    case 1:
      options.RowPerm = LargeDiag_MC64;
      break;
    case 2:
      options.RowPerm = LargeDiag_AWPM;
      break;
    case 3:
      options.RowPerm = MY_PERMR;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown row permutation");
    }
  }

  ierr = PetscOptionsEList("-mat_superlu_dist_colperm","Column permutation","None",colperm,5,colperm[3],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (indx) {
    case 0:
      options.ColPerm = NATURAL;
      break;
    case 1:
      options.ColPerm = MMD_AT_PLUS_A;
      break;
    case 2:
      options.ColPerm = MMD_ATA;
      break;
    case 3:
      options.ColPerm = METIS_AT_PLUS_A;
      break;
    case 4:
      options.ColPerm = PARMETIS;   /* only works for np>1 */
      break;
    default:
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown column permutation");
    }
  }

  options.ReplaceTinyPivot = NO;
  ierr = PetscOptionsBool("-mat_superlu_dist_replacetinypivot","Replace tiny pivots","None",options.ReplaceTinyPivot ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set && flg) options.ReplaceTinyPivot = YES;

  options.ParSymbFact = NO;
  ierr = PetscOptionsBool("-mat_superlu_dist_parsymbfact","Parallel symbolic factorization","None",PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set && flg && size>1) {
#if defined(PETSC_HAVE_PARMETIS)
    options.ParSymbFact = YES;
    options.ColPerm     = PARMETIS;   /* in v2.2, PARMETIS is forced for ParSymbFact regardless of user ordering setting */
#else
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"parsymbfact needs PARMETIS");
#endif
  }

  lu->FactPattern = SamePattern;
  ierr = PetscOptionsEList("-mat_superlu_dist_fact","Sparsity pattern for repeated matrix factorization","None",factPattern,3,factPattern[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (indx) {
    case 0:
      lu->FactPattern = SamePattern;
      break;
    case 1:
      lu->FactPattern = SamePattern_SameRowPerm;
      break;
    case 2:
      lu->FactPattern = DOFACT;
      break;
    }
  }

  options.IterRefine = NOREFINE;
  ierr               = PetscOptionsBool("-mat_superlu_dist_iterrefine","Use iterative refinement","None",options.IterRefine == NOREFINE ? PETSC_FALSE : PETSC_TRUE ,&flg,&set);CHKERRQ(ierr);
  if (set) {
    if (flg) options.IterRefine = SLU_DOUBLE;
    else options.IterRefine = NOREFINE;
  }

  if (PetscLogPrintInfo) options.PrintStat = YES;
  else options.PrintStat = NO;
  ierr = PetscOptionsBool("-mat_superlu_dist_statprint","Print factorization information","None",(PetscBool)options.PrintStat,(PetscBool*)&options.PrintStat,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  lu->options              = options;
  lu->options.Fact         = DOFACT;
  lu->matsolve_iscalled    = PETSC_FALSE;
  lu->matmatsolve_iscalled = PETSC_FALSE;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_aij_superlu_dist);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSuperluDistGetDiagU_C",MatSuperluDistGetDiagU_SuperLU_DIST);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_SuperLU_DIST(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERSUPERLU_DIST,MATMPIAIJ,MAT_FACTOR_LU,MatGetFactor_aij_superlu_dist);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERSUPERLU_DIST,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_aij_superlu_dist);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERSUPERLU_DIST,MATMPIAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_aij_superlu_dist);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERSUPERLU_DIST,MATSEQAIJ,MAT_FACTOR_CHOLESKY,MatGetFactor_aij_superlu_dist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERSUPERLU_DIST - Parallel direct solver package for LU factorization

  Use ./configure --download-superlu_dist --download-parmetis --download-metis --download-ptscotch  to have PETSc installed with SuperLU_DIST

  Use -pc_type lu -pc_factor_mat_solver_type superlu_dist to use this direct solver

   Works with AIJ matrices

  Options Database Keys:
+ -mat_superlu_dist_r <n> - number of rows in processor partition
. -mat_superlu_dist_c <n> - number of columns in processor partition
. -mat_superlu_dist_equil - equilibrate the matrix
. -mat_superlu_dist_rowperm <NOROWPERM,LargeDiag_MC64,LargeDiag_AWPM,MY_PERMR> - row permutation
. -mat_superlu_dist_colperm <MMD_AT_PLUS_A,MMD_ATA,NATURAL> - column permutation
. -mat_superlu_dist_replacetinypivot - replace tiny pivots
. -mat_superlu_dist_fact <SamePattern> - (choose one of) SamePattern SamePattern_SameRowPerm DOFACT
. -mat_superlu_dist_iterrefine - use iterative refinement
- -mat_superlu_dist_statprint - print factorization information

   Level: beginner

.seealso: PCLU

.seealso: PCFactorSetMatSolverType(), MatSolverType

M*/
