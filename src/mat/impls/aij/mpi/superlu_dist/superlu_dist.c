/*
        Provides an interface to the SuperLU_DIST sparse solver
*/

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscpkg_version.h>

EXTERN_C_BEGIN
#if defined(PETSC_USE_COMPLEX)
#define CASTDOUBLECOMPLEX (doublecomplex*)
#include <superlu_zdefs.h>
#define LUstructInit zLUstructInit
#define ScalePermstructInit zScalePermstructInit
#define ScalePermstructFree zScalePermstructFree
#define LUstructFree zLUstructFree
#define Destroy_LU zDestroy_LU
#define ScalePermstruct_t zScalePermstruct_t
#define LUstruct_t zLUstruct_t
#define SOLVEstruct_t zSOLVEstruct_t
#define SolveFinalize zSolveFinalize
#define pgssvx pzgssvx
#define Create_CompRowLoc_Matrix_dist zCreate_CompRowLoc_Matrix_dist
#define pGetDiagU pzGetDiagU
#define allocateA_dist zallocateA_dist
#define SLU_ZD SLU_Z
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
#define DeAllocLlu_3d zDeAllocLlu_3d
#define DeAllocGlu_3d zDeAllocGlu_3d
#define Destroy_A3d_gathered_on_2d zDestroy_A3d_gathered_on_2d
#define pgssvx3d pzgssvx3d
#endif
#else
#define CASTDOUBLECOMPLEX
#include <superlu_ddefs.h>
#define LUstructInit dLUstructInit
#define ScalePermstructInit dScalePermstructInit
#define ScalePermstructFree dScalePermstructFree
#define LUstructFree dLUstructFree
#define Destroy_LU dDestroy_LU
#define ScalePermstruct_t dScalePermstruct_t
#define LUstruct_t dLUstruct_t
#define SOLVEstruct_t dSOLVEstruct_t
#define SolveFinalize dSolveFinalize
#define pgssvx pdgssvx
#define Create_CompRowLoc_Matrix_dist dCreate_CompRowLoc_Matrix_dist
#define pGetDiagU pdGetDiagU
#define allocateA_dist dallocateA_dist
#define SLU_ZD SLU_D
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
#define DeAllocLlu_3d dDeAllocLlu_3d
#define DeAllocGlu_3d dDeAllocGlu_3d
#define Destroy_A3d_gathered_on_2d dDestroy_A3d_gathered_on_2d
#define pgssvx3d pdgssvx3d
#endif
#endif
EXTERN_C_END

typedef struct {
  int_t                  nprow,npcol,*row,*col;
  gridinfo_t             grid;
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
  PetscBool              use3d;
  int_t                  npdep; /* replication factor, must be power of two */
  gridinfo3d_t           grid3d;
#endif
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
  PetscStackCall("SuperLU_DIST:pGetDiagU",pGetDiagU(F->rmap->N,&lu->LUstruct,&lu->grid,CASTDOUBLECOMPLEX diagU));
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

/*  This allows reusing the Superlu_DIST communicator and grid when only a single SuperLU_DIST matrix is used at a time */
typedef struct {
  MPI_Comm     comm;
  PetscBool    busy;
  gridinfo_t   grid;
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
  PetscBool    use3d;
  gridinfo3d_t grid3d;
#endif
} PetscSuperLU_DIST;
static PetscMPIInt Petsc_Superlu_dist_keyval = MPI_KEYVAL_INVALID;

PETSC_EXTERN PetscMPIInt MPIAPI Petsc_Superlu_dist_keyval_Delete_Fn(MPI_Comm comm,PetscMPIInt keyval,void *attr_val,void *extra_state)
{
  PetscErrorCode    ierr;
  PetscSuperLU_DIST *context = (PetscSuperLU_DIST *) attr_val;

  PetscFunctionBegin;
  if (keyval != Petsc_Superlu_dist_keyval) SETERRMPI(PETSC_COMM_SELF,PETSC_ERR_ARG_CORRUPT,"Unexpected keyval");
  ierr = PetscInfo(NULL,"Removing Petsc_Superlu_dist_keyval attribute from communicator that is being freed\n");CHKERRQ(ierr);
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
  if (context->use3d) {
    PetscStackCall("SuperLU_DIST:superlu_gridexit3d",superlu_gridexit3d(&context->grid3d));
  } else
#endif
    PetscStackCall("SuperLU_DIST:superlu_gridexit",superlu_gridexit(&context->grid));
  ierr = MPI_Comm_free(&context->comm);CHKERRMPI(ierr);
  ierr = PetscFree(context);CHKERRQ(ierr);
  PetscFunctionReturn(MPI_SUCCESS);
}

/*
   Performs MPI_Comm_free_keyval() on Petsc_Superlu_dist_keyval but keeps the global variable for
   users who do not destroy all PETSc objects before PetscFinalize().

   The value Petsc_Superlu_dist_keyval is retained so that Petsc_Superlu_dist_keyval_Delete_Fn()
   can still check that the keyval associated with the MPI communicator is correct when the MPI
   communicator is destroyed.

   This is called in PetscFinalize()
*/
static PetscErrorCode Petsc_Superlu_dist_keyval_free(void)
{
  PetscErrorCode ierr;
  PetscMPIInt    Petsc_Superlu_dist_keyval_temp = Petsc_Superlu_dist_keyval;

  PetscFunctionBegin;
  ierr = PetscInfo(NULL,"Freeing Petsc_Superlu_dist_keyval\n");CHKERRQ(ierr);
  ierr = MPI_Comm_free_keyval(&Petsc_Superlu_dist_keyval_temp);CHKERRMPI(ierr);
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
      PetscStackCall("SuperLU_DIST:SolveFinalize",SolveFinalize(&lu->options, &lu->SOLVEstruct));
    }
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
    if (lu->use3d) {
      if (lu->grid3d.zscp.Iam == 0) {
        PetscStackCall("SuperLU_DIST:Destroy_LU",Destroy_LU(A->cmap->N, &lu->grid3d.grid2d, &lu->LUstruct));
      } else {
        PetscStackCall("SuperLU_DIST:DeAllocLlu_3d",DeAllocLlu_3d(lu->A_sup.ncol, &lu->LUstruct, &lu->grid3d));
        PetscStackCall("SuperLU_DIST:DeAllocGlu_3d",DeAllocGlu_3d(&lu->LUstruct));
      }
      PetscStackCall("SuperLU_DIST:Destroy_A3d_gathered_on_2d",Destroy_A3d_gathered_on_2d(&lu->SOLVEstruct, &lu->grid3d));
    } else
#endif
      PetscStackCall("SuperLU_DIST:Destroy_LU",Destroy_LU(A->cmap->N, &lu->grid, &lu->LUstruct));
    PetscStackCall("SuperLU_DIST:ScalePermstructFree",ScalePermstructFree(&lu->ScalePermstruct));
    PetscStackCall("SuperLU_DIST:LUstructFree",LUstructFree(&lu->LUstruct));

    /* Release the SuperLU_DIST process grid only if the matrix has its own copy, that is it is not in the communicator context */
    if (lu->comm_superlu) {
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
      if (lu->use3d) {
        PetscStackCall("SuperLU_DIST:superlu_gridexit3d",superlu_gridexit3d(&lu->grid3d));
      } else
#endif
        PetscStackCall("SuperLU_DIST:superlu_gridexit",superlu_gridexit(&lu->grid));
    }
  }
  /*
   * We always need to release the communicator that was created in MatGetFactor_aij_superlu_dist.
   * lu->CleanUpSuperLU_Dist was turned on in MatLUFactorSymbolic_SuperLU_DIST. There are some use
   * cases where we only create a matrix but do not solve mat. In these cases, lu->CleanUpSuperLU_Dist
   * is off, and the communicator was not released or marked as "not busy " in the old code.
   * Here we try to release comm regardless.
  */
  if (lu->comm_superlu) {
    ierr = PetscCommRestoreComm(PetscObjectComm((PetscObject)A),&lu->comm_superlu);CHKERRQ(ierr);
  } else {
    PetscSuperLU_DIST *context;
    MPI_Comm          comm;
    PetscMPIInt       flg;

    ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_get_attr(comm,Petsc_Superlu_dist_keyval,&context,&flg);CHKERRMPI(ierr);
    PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Communicator does not have expected Petsc_Superlu_dist_keyval attribute");
    context->busy = PETSC_FALSE;
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
  PetscCheck(lu->options.Fact == FACTORED,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"SuperLU_DIST options.Fact must equal FACTORED");
  ierr = PetscCitationsRegister("@article{lidemmel03,\n  author = {Xiaoye S. Li and James W. Demmel},\n  title = {{SuperLU_DIST}: A Scalable Distributed-Memory Sparse Direct\n           Solver for Unsymmetric Linear Systems},\n  journal = {ACM Trans. Mathematical Software},\n  volume = {29},\n  number = {2},\n  pages = {110-140},\n  year = 2003\n}\n",&cite);CHKERRQ(ierr);

  if (lu->options.SolveInitialized && !lu->matsolve_iscalled) {
    /* see comments in MatMatSolve() */
    PetscStackCall("SuperLU_DIST:SolveFinalize",SolveFinalize(&lu->options, &lu->SOLVEstruct));
    lu->options.SolveInitialized = NO;
  }
  ierr = VecCopy(b_mpi,x);CHKERRQ(ierr);
  ierr = VecGetArray(x,&bptr);CHKERRQ(ierr);

  PetscStackCall("SuperLU_DIST:PStatInit",PStatInit(&stat));        /* Initialize the statistics variables. */
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
  if (lu->use3d)
    PetscStackCall("SuperLU_DIST:pgssvx3d",pgssvx3d(&lu->options,&lu->A_sup,&lu->ScalePermstruct,CASTDOUBLECOMPLEX bptr,m,1,&lu->grid3d,&lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));
  else
#endif
    PetscStackCall("SuperLU_DIST:pgssvx",pgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,CASTDOUBLECOMPLEX bptr,m,1,&lu->grid,&lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));
  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"pdgssvx fails, info: %d",info);

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
  PetscCheck(lu->options.Fact == FACTORED,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"SuperLU_DIST options.Fact must equal FACTORED");
  ierr = PetscObjectTypeCompareAny((PetscObject)B_mpi,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  if (X != B_mpi) {
    ierr = PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
    PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");
  }

  if (lu->options.SolveInitialized && !lu->matmatsolve_iscalled) {
    /* communication pattern of SOLVEstruct is unlikely created for matmatsolve,
       thus destroy it and create a new SOLVEstruct.
       Otherwise it may result in memory corruption or incorrect solution
       See src/mat/tests/ex125.c */
    PetscStackCall("SuperLU_DIST:SolveFinalize",SolveFinalize(&lu->options, &lu->SOLVEstruct));
    lu->options.SolveInitialized = NO;
  }
  if (X != B_mpi) {
    ierr = MatCopy(B_mpi,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }

  ierr = MatGetSize(B_mpi,NULL,&nrhs);CHKERRQ(ierr);

  PetscStackCall("SuperLU_DIST:PStatInit",PStatInit(&stat));        /* Initialize the statistics variables. */
  ierr = MatDenseGetArray(X,&bptr);CHKERRQ(ierr);

#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
  if (lu->use3d)
    PetscStackCall("SuperLU_DIST:pgssvx3d",pgssvx3d(&lu->options,&lu->A_sup,&lu->ScalePermstruct,CASTDOUBLECOMPLEX bptr,m,nrhs,&lu->grid3d,&lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));
  else
#endif
    PetscStackCall("SuperLU_DIST:pgssvx",pgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,CASTDOUBLECOMPLEX bptr,m,nrhs,&lu->grid,&lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));

  PetscCheck(!info,PETSC_COMM_SELF,PETSC_ERR_LIB,"pdgssvx fails, info: %d",info);
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
  PetscCheck(F->assembled,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix factor F is not assembled");
  PetscCheck(lu->options.RowPerm == NOROWPERM,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must set NOROWPERM");
  ierr = MatGetSize(F,&M,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(M,&diagU);CHKERRQ(ierr);
  ierr = MatSuperluDistGetDiagU(F,diagU);CHKERRQ(ierr);
  for (i=0; i<M; i++) {
#if defined(PETSC_USE_COMPLEX)
    r = PetscImaginaryPart(diagU[i])/10.0;
    PetscCheck(r > -PETSC_MACHINE_EPSILON && r < PETSC_MACHINE_EPSILON,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"diagU[%" PetscInt_FMT "]=%g + i %g is non-real",i,(double)PetscRealPart(diagU[i]),(double)(r*10.0));
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
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for type %s",((PetscObject)A)->type_name);

  ierr = MatGetRowIJ(Aloc,0,PETSC_FALSE,PETSC_FALSE,&dummy,&ai,&aj,&flg);CHKERRQ(ierr);
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"GetRowIJ failed");
  ierr = MatSeqAIJGetArrayRead(Aloc,&av);CHKERRQ(ierr);
  nz   = ai[Aloc->rmap->n];

  /* Allocations for A_sup */
  if (lu->options.Fact == DOFACT) { /* first numeric factorization */
    PetscStackCall("SuperLU_DIST:allocateA_dist",allocateA_dist(Aloc->rmap->n, nz, &lu->val, &lu->col, &lu->row));
  } else { /* successive numeric factorization, sparsity pattern and perm_c are reused. */
    if (lu->FactPattern == SamePattern_SameRowPerm) {
      lu->options.Fact = SamePattern_SameRowPerm; /* matrix has similar numerical values */
    } else if (lu->FactPattern == SamePattern) {
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
      if (lu->use3d) {
        if (lu->grid3d.zscp.Iam == 0) {
          PetscStackCall("SuperLU_DIST:Destroy_LU",Destroy_LU(A->cmap->N, &lu->grid3d.grid2d, &lu->LUstruct));
          PetscStackCall("SuperLU_DIST:SolveFinalize",SolveFinalize(&lu->options, &lu->SOLVEstruct));
        } else {
          PetscStackCall("SuperLU_DIST:DeAllocLlu_3d",DeAllocLlu_3d(lu->A_sup.ncol, &lu->LUstruct, &lu->grid3d));
          PetscStackCall("SuperLU_DIST:DeAllocGlu_3d",DeAllocGlu_3d(&lu->LUstruct));
        }
      } else
#endif
        PetscStackCall("SuperLU_DIST:Destroy_LU",Destroy_LU(A->rmap->N, &lu->grid, &lu->LUstruct));
      lu->options.Fact = SamePattern;
    } else if (lu->FactPattern == DOFACT) {
      PetscStackCall("SuperLU_DIST:Destroy_CompRowLoc_Matrix_dist",Destroy_CompRowLoc_Matrix_dist(&lu->A_sup));
      PetscStackCall("SuperLU_DIST:Destroy_LU",Destroy_LU(A->rmap->N, &lu->grid, &lu->LUstruct));
      lu->options.Fact = DOFACT;
      PetscStackCall("SuperLU_DIST:allocateA_dist",allocateA_dist(Aloc->rmap->n, nz, &lu->val, &lu->col, &lu->row));
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"options.Fact must be one of SamePattern SamePattern_SameRowPerm DOFACT");
  }

  /* Copy AIJ matrix to superlu_dist matrix */
  ierr = PetscArraycpy(lu->row,ai,Aloc->rmap->n+1);CHKERRQ(ierr);
  ierr = PetscArraycpy(lu->col,aj,nz);CHKERRQ(ierr);
  ierr = PetscArraycpy(lu->val,av,nz);CHKERRQ(ierr);
  ierr = MatRestoreRowIJ(Aloc,0,PETSC_FALSE,PETSC_FALSE,&dummy,&ai,&aj,&flg);CHKERRQ(ierr);
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"RestoreRowIJ failed");
  ierr = MatSeqAIJRestoreArrayRead(Aloc,&av);CHKERRQ(ierr);
  ierr = MatDestroy(&Aloc);CHKERRQ(ierr);

  /* Create and setup A_sup */
  if (lu->options.Fact == DOFACT) {
    PetscStackCall("SuperLU_DIST:Create_CompRowLoc_Matrix_dist",Create_CompRowLoc_Matrix_dist(&lu->A_sup, A->rmap->N, A->cmap->N, nz, A->rmap->n, A->rmap->rstart, lu->val, lu->col, lu->row, SLU_NR_loc, SLU_ZD, SLU_GE));
  }

  /* Factor the matrix. */
  PetscStackCall("SuperLU_DIST:PStatInit",PStatInit(&stat));   /* Initialize the statistics variables. */
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
  if (lu->use3d) {
    PetscStackCall("SuperLU_DIST:pgssvx3d",pgssvx3d(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, A->rmap->n, 0, &lu->grid3d, &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &sinfo));
  } else
#endif
    PetscStackCall("SuperLU_DIST:pgssvx",pgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, A->rmap->n, 0, &lu->grid, &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &sinfo));
  if (sinfo > 0) {
    PetscCheck(!A->erroriffailure,PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot in row %d",sinfo);
    else {
      if (sinfo <= lu->A_sup.ncol) {
        F->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
        ierr = PetscInfo(F,"U(i,i) is exactly zero, i= %d\n",sinfo);CHKERRQ(ierr);
      } else if (sinfo > lu->A_sup.ncol) {
        /*
         number of bytes allocated when memory allocation
         failure occurred, plus A->ncol.
         */
        F->factorerrortype = MAT_FACTOR_OUTMEMORY;
        ierr = PetscInfo(F,"Number of bytes allocated when memory allocation fails %d\n",sinfo);CHKERRQ(ierr);
      }
    }
  } else PetscCheck(sinfo >= 0,PETSC_COMM_SELF,PETSC_ERR_LIB, "info = %d, argument in p*gssvx() had an illegal value", sinfo);

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
  /* would love to use superlu 'IFMT' macro but it looks like it's inconsistently applied, the
   * format spec for int64_t is set to %d for whatever reason */
  ierr = PetscViewerASCIIPrintf(viewer,"  Process grid nprow %lld x npcol %lld \n",(long long)lu->nprow,(long long)lu->npcol);CHKERRQ(ierr);
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
  if (lu->use3d) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Using 3d decomposition with npdep %lld \n",(long long)lu->npdep);CHKERRQ(ierr);
  }
#endif

  ierr = PetscViewerASCIIPrintf(viewer,"  Equilibrate matrix %s \n",PetscBools[options.Equil != NO]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Replace tiny pivots %s \n",PetscBools[options.ReplaceTinyPivot != NO]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Use iterative refinement %s \n",PetscBools[options.IterRefine == SLU_DOUBLE]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Processors in row %lld col partition %lld \n",(long long)lu->nprow,(long long)lu->npcol);CHKERRQ(ierr);

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

  B->trivialsymbolic = PETSC_TRUE;
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
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);

  {
    PetscMPIInt       flg;
    MPI_Comm          comm;
    PetscSuperLU_DIST *context = NULL;

    ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
    if (Petsc_Superlu_dist_keyval == MPI_KEYVAL_INVALID) {
      ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,Petsc_Superlu_dist_keyval_Delete_Fn,&Petsc_Superlu_dist_keyval,(void*)0);CHKERRMPI(ierr);
      ierr = PetscRegisterFinalize(Petsc_Superlu_dist_keyval_free);CHKERRQ(ierr);
    }
    ierr = MPI_Comm_get_attr(comm,Petsc_Superlu_dist_keyval,&context,&flg);CHKERRMPI(ierr);
    if (!flg || context->busy) {
      if (!flg) {
        ierr = PetscNew(&context);CHKERRQ(ierr);
        context->busy = PETSC_TRUE;
        ierr = MPI_Comm_dup(comm,&context->comm);CHKERRMPI(ierr);
        ierr = MPI_Comm_set_attr(comm,Petsc_Superlu_dist_keyval,context);CHKERRMPI(ierr);
      } else {
        ierr = PetscCommGetComm(PetscObjectComm((PetscObject)A),&lu->comm_superlu);CHKERRQ(ierr);
      }

      /* Default number of process columns and rows */
      lu->nprow = (int_t) (0.5 + PetscSqrtReal((PetscReal)size));
      if (!lu->nprow) lu->nprow = 1;
      while (lu->nprow > 0) {
        lu->npcol = (int_t) (size/lu->nprow);
        if (size == lu->nprow * lu->npcol) break;
        lu->nprow--;
      }
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
      lu->use3d = PETSC_FALSE;
      lu->npdep = 1;
#endif
      ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"SuperLU_Dist Options","Mat");CHKERRQ(ierr);
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
      ierr = PetscOptionsBool("-mat_superlu_dist_3d","Use SuperLU_DIST 3D distribution","None",lu->use3d,&lu->use3d,NULL);CHKERRQ(ierr);
      if (lu->use3d) {
        PetscInt t;
        ierr = PetscOptionsInt("-mat_superlu_dist_d","Number of z entries in processor partition","None",lu->npdep,(PetscInt*)&lu->npdep,NULL);CHKERRQ(ierr);
        t = (PetscInt) PetscLog2Real((PetscReal)lu->npdep);
        PetscCheck(PetscPowInt(2,t) == lu->npdep,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_OUTOFRANGE,"-mat_superlu_dist_d %lld must be a power of 2",(long long)lu->npdep);
        if (lu->npdep > 1) {
          lu->nprow = (int_t) (0.5 + PetscSqrtReal((PetscReal)(size/lu->npdep)));
          if (!lu->nprow) lu->nprow = 1;
          while (lu->nprow > 0) {
            lu->npcol = (int_t) (size/(lu->npdep*lu->nprow));
            if (size == lu->nprow * lu->npcol * lu->npdep) break;
            lu->nprow--;
          }
        }
      }
#endif
      ierr = PetscOptionsInt("-mat_superlu_dist_r","Number rows in processor partition","None",lu->nprow,(PetscInt*)&lu->nprow,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsInt("-mat_superlu_dist_c","Number columns in processor partition","None",lu->npcol,(PetscInt*)&lu->npcol,NULL);CHKERRQ(ierr);
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
      PetscCheck(size == lu->nprow*lu->npcol*lu->npdep,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number of processes %d must equal to nprow %lld * npcol %lld * npdep %lld",size,(long long)lu->nprow,(long long)lu->npcol,(long long)lu->npdep);
#else
      PetscCheck(size == lu->nprow*lu->npcol,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number of processes %d must equal to nprow %lld * npcol %lld",size,(long long)lu->nprow,(long long)lu->npcol);
#endif
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
      if (lu->use3d) {
        PetscStackCall("SuperLU_DIST:superlu_gridinit3d",superlu_gridinit3d(context ? context->comm : lu->comm_superlu, lu->nprow, lu->npcol,lu->npdep, &lu->grid3d));
        if (context) {context->grid3d = lu->grid3d; context->use3d = lu->use3d;}
      } else {
#endif
        PetscStackCall("SuperLU_DIST:superlu_gridinit",superlu_gridinit(context ? context->comm : lu->comm_superlu, lu->nprow, lu->npcol, &lu->grid));
        if (context) context->grid = lu->grid;
#if PETSC_PKG_SUPERLU_DIST_VERSION_GE(7,2,0)
      }
#endif
      ierr = PetscInfo(NULL,"Duplicating a communicator for SuperLU_DIST and calling superlu_gridinit()\n");CHKERRQ(ierr);
      if (flg) {
        ierr = PetscInfo(NULL,"Communicator attribute already in use so not saving communicator and SuperLU_DIST grid in communicator attribute \n");CHKERRQ(ierr);
      } else {
        ierr = PetscInfo(NULL,"Storing communicator and SuperLU_DIST grid in communicator attribute\n");CHKERRQ(ierr);
      }
    } else {
      ierr = PetscInfo(NULL,"Reusing communicator and superlu_gridinit() for SuperLU_DIST from communicator attribute.");CHKERRQ(ierr);
      context->busy = PETSC_TRUE;
      lu->grid      = context->grid;
    }
  }

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"SuperLU_Dist Options","Mat");CHKERRQ(ierr);
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
. -mat_superlu_dist_3d - use 3d partition, requires SuperLU_DIST 7.2 or later
. -mat_superlu_dist_d <n> - depth in 3d partition (valid only if -mat_superlu_dist_3d) is provided
. -mat_superlu_dist_equil - equilibrate the matrix
. -mat_superlu_dist_rowperm <NOROWPERM,LargeDiag_MC64,LargeDiag_AWPM,MY_PERMR> - row permutation
. -mat_superlu_dist_colperm <NATURAL,MMD_AT_PLUS_A,MMD_ATA,METIS_AT_PLUS_A,PARMETIS> - column permutation
. -mat_superlu_dist_replacetinypivot - replace tiny pivots
. -mat_superlu_dist_fact <SamePattern> - (choose one of) SamePattern SamePattern_SameRowPerm DOFACT
. -mat_superlu_dist_iterrefine - use iterative refinement
- -mat_superlu_dist_statprint - print factorization information

  Notes:
    If PETSc was configured with --with-cuda than this solver will automatically use the GPUs.

  Level: beginner

.seealso: PCLU

.seealso: PCFactorSetMatSolverType(), MatSolverType

M*/
