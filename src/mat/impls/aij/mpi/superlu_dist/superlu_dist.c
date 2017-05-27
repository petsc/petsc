
/*
        Provides an interface to the SuperLU_DIST sparse solver
*/

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#if defined(PETSC_HAVE_STDLIB_H) /* This is to get around weird problem with SuperLU on cray */
#include <stdlib.h>
#endif

#if defined(PETSC_USE_64BIT_INDICES)
/* ugly SuperLU_Dist variable telling it to use long long int */
#define _LONGINT
#endif

EXTERN_C_BEGIN
#if defined(PETSC_USE_COMPLEX)
#include <superlu_zdefs.h>
#else
#include <superlu_ddefs.h>
#endif
EXTERN_C_END

/*
    GLOBAL - The sparse matrix and right hand side are all stored initially on process 0. Should be called centralized
    DISTRIBUTED - The sparse matrix and right hand size are initially stored across the entire MPI communicator.
*/
typedef enum {GLOBAL,DISTRIBUTED} SuperLU_MatInputMode;
const char *SuperLU_MatInputModes[] = {"GLOBAL","DISTRIBUTED","SuperLU_MatInputMode","PETSC_",0};

typedef struct {
  int_t                  nprow,npcol,*row,*col;
  gridinfo_t             grid;
  superlu_dist_options_t options;
  SuperMatrix            A_sup;
  ScalePermstruct_t      ScalePermstruct;
  LUstruct_t             LUstruct;
  int                    StatPrint;
  SuperLU_MatInputMode   MatInputMode;
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
  Mat_SuperLU_DIST  *lu= (Mat_SuperLU_DIST*)F->data;

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
  PetscErrorCode    ierr;

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
    if (lu->MatInputMode == GLOBAL) {
      PetscStackCall("SuperLU_DIST:Destroy_CompCol_Matrix_dist",Destroy_CompCol_Matrix_dist(&lu->A_sup));
    } else {
      PetscStackCall("SuperLU_DIST:Destroy_CompRowLoc_Matrix_dist",Destroy_CompRowLoc_Matrix_dist(&lu->A_sup));
      if (lu->options.SolveInitialized) {
#if defined(PETSC_USE_COMPLEX)
        PetscStackCall("SuperLU_DIST:zSolveFinalize",zSolveFinalize(&lu->options, &lu->SOLVEstruct));
#else
        PetscStackCall("SuperLU_DIST:dSolveFinalize",dSolveFinalize(&lu->options, &lu->SOLVEstruct));
#endif
      }
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
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverPackage_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSuperluDistGetDiagU_C",NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SuperLU_DIST(Mat A,Vec b_mpi,Vec x)
{
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)A->data;
  PetscErrorCode   ierr;
  PetscMPIInt      size;
  PetscInt         m=A->rmap->n,M=A->rmap->N,N=A->cmap->N;
  SuperLUStat_t    stat;
  double           berr[1];
  PetscScalar      *bptr;
  PetscInt         nrhs=1;
  Vec              x_seq;
  IS               iden;
  VecScatter       scat;
  int              info; /* SuperLU_Dist info code is ALWAYS an int, even with long long indices */
  static PetscBool cite = PETSC_FALSE;

  PetscFunctionBegin;
  if (lu->options.Fact != FACTORED) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"SuperLU_DIST options.Fact mush equal FACTORED");
  ierr = PetscCitationsRegister("@article{lidemmel03,\n  author = {Xiaoye S. Li and James W. Demmel},\n  title = {{SuperLU_DIST}: A Scalable Distributed-Memory Sparse Direct\n           Solver for Unsymmetric Linear Systems},\n  journal = {ACM Trans. Mathematical Software},\n  volume = {29},\n  number = {2},\n  pages = {110-140},\n  year = 2003\n}\n",&cite);CHKERRQ(ierr);

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size > 1 && lu->MatInputMode == GLOBAL) {
    /* global mat input, convert b to x_seq */
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x_seq);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&iden);CHKERRQ(ierr);
    ierr = VecScatterCreate(b_mpi,iden,x_seq,iden,&scat);CHKERRQ(ierr);
    ierr = ISDestroy(&iden);CHKERRQ(ierr);

    ierr = VecScatterBegin(scat,b_mpi,x_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat,b_mpi,x_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(x_seq,&bptr);CHKERRQ(ierr);
  } else { /* size==1 || distributed mat input */
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
  }

  PetscStackCall("SuperLU_DIST:PStatInit",PStatInit(&stat));        /* Initialize the statistics variables. */
  if (lu->MatInputMode == GLOBAL) {
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:pzgssvx_ABglobal",pzgssvx_ABglobal(&lu->options,&lu->A_sup,&lu->ScalePermstruct,(doublecomplex*)bptr,M,nrhs,&lu->grid,&lu->LUstruct,berr,&stat,&info));
#else
    PetscStackCall("SuperLU_DIST:pdgssvx_ABglobal",pdgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct,bptr,M,nrhs,&lu->grid,&lu->LUstruct,berr,&stat,&info));
#endif
  } else { /* distributed mat input */
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:pzgssvx",pzgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,(doublecomplex*)bptr,m,nrhs,&lu->grid,&lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));
#else
    PetscStackCall("SuperLU_DIST:pdgssvx",pdgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,bptr,m,nrhs,&lu->grid,&lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));
#endif
  }
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"pdgssvx fails, info: %d\n",info);

  if (lu->options.PrintStat) PStatPrint(&lu->options, &stat, &lu->grid);      /* Print the statistics. */
  PetscStackCall("SuperLU_DIST:PStatFree",PStatFree(&stat));

  if (size > 1 && lu->MatInputMode == GLOBAL) {
    /* convert seq x to mpi x */
    ierr = VecRestoreArray(x_seq,&bptr);CHKERRQ(ierr);
    ierr = VecScatterBegin(scat,x_seq,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scat,x_seq,x,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scat);CHKERRQ(ierr);
    ierr = VecDestroy(&x_seq);CHKERRQ(ierr);
  } else {
    ierr = VecRestoreArray(x,&bptr);CHKERRQ(ierr);

    lu->matsolve_iscalled    = PETSC_TRUE;
    lu->matmatsolve_iscalled = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SuperLU_DIST(Mat A,Mat B_mpi,Mat X)
{
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)A->data;
  PetscErrorCode   ierr;
  PetscMPIInt      size;
  PetscInt         M=A->rmap->N,m=A->rmap->n,nrhs;
  SuperLUStat_t    stat;
  double           berr[1];
  PetscScalar      *bptr;
  int              info; /* SuperLU_Dist info code is ALWAYS an int, even with long long indices */
  PetscBool        flg;

  PetscFunctionBegin;
  if (lu->options.Fact != FACTORED) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"SuperLU_DIST options.Fact mush equal FACTORED");
  ierr = PetscObjectTypeCompareAny((PetscObject)B_mpi,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  ierr = PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  if (size > 1 && lu->MatInputMode == GLOBAL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatInputMode=GLOBAL for nproc %d>1 is not supported",size);
  /* size==1 or distributed mat input */
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
  ierr = MatCopy(B_mpi,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = MatGetSize(B_mpi,NULL,&nrhs);CHKERRQ(ierr);

  PetscStackCall("SuperLU_DIST:PStatInit",PStatInit(&stat));        /* Initialize the statistics variables. */
  ierr = MatDenseGetArray(X,&bptr);CHKERRQ(ierr);
  if (lu->MatInputMode == GLOBAL) { /* size == 1 */
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:pzgssvx_ABglobal",pzgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct,(doublecomplex*)bptr, M, nrhs,&lu->grid, &lu->LUstruct, berr, &stat, &info));
#else
    PetscStackCall("SuperLU_DIST:pdgssvx_ABglobal",pdgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct,bptr, M, nrhs, &lu->grid, &lu->LUstruct, berr, &stat, &info));
#endif
  } else { /* distributed mat input */
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:pzgssvx",pzgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,(doublecomplex*)bptr,m,nrhs,&lu->grid, &lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));
#else
    PetscStackCall("SuperLU_DIST:pdgssvx",pdgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,bptr,m,nrhs,&lu->grid,&lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info));
#endif
  }
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"pdgssvx fails, info: %d\n",info);
  ierr = MatDenseRestoreArray(X,&bptr);CHKERRQ(ierr);

  if (lu->options.PrintStat) PStatPrint(&lu->options, &stat, &lu->grid); /* Print the statistics. */
  PetscStackCall("SuperLU_DIST:PStatFree",PStatFree(&stat));
  lu->matsolve_iscalled    = PETSC_FALSE;
  lu->matmatsolve_iscalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}


static PetscErrorCode MatLUFactorNumeric_SuperLU_DIST(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat              *tseq,A_seq = NULL;
  Mat_SeqAIJ       *aa,*bb;
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)(F)->data;
  PetscErrorCode   ierr;
  PetscInt         M=A->rmap->N,N=A->cmap->N,i,*ai,*aj,*bi,*bj,nz,rstart,*garray,
                   m=A->rmap->n, colA_start,j,jcol,jB,countA,countB,*bjj,*ajj;
  int              sinfo;   /* SuperLU_Dist info flag is always an int even with long long indices */
  PetscMPIInt      size;
  SuperLUStat_t    stat;
  double           *berr=0;
  IS               isrow;
#if defined(PETSC_USE_COMPLEX)
  doublecomplex    *av, *bv;
#else
  double           *av, *bv;
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);

  if (lu->MatInputMode == GLOBAL) { /* global mat input */
    if (size > 1) { /* convert mpi A to seq mat A */
      ierr = ISCreateStride(PETSC_COMM_SELF,M,0,1,&isrow);CHKERRQ(ierr);
      ierr = MatCreateSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&tseq);CHKERRQ(ierr);
      ierr = ISDestroy(&isrow);CHKERRQ(ierr);

      A_seq = *tseq;
      ierr  = PetscFree(tseq);CHKERRQ(ierr);
      aa    = (Mat_SeqAIJ*)A_seq->data;
    } else {
      PetscBool flg;
      ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
      if (flg) {
        Mat_MPIAIJ *At = (Mat_MPIAIJ*)A->data;
        A = At->A;
      }
      aa =  (Mat_SeqAIJ*)A->data;
    }

    /* Convert Petsc NR matrix to SuperLU_DIST NC.
       Note: memories of lu->val, col and row are allocated by CompRow_to_CompCol_dist()! */
    if (lu->options.Fact != DOFACT) {/* successive numeric factorization, sparsity pattern is reused. */
      PetscStackCall("SuperLU_DIST:Destroy_CompCol_Matrix_dist",Destroy_CompCol_Matrix_dist(&lu->A_sup));
      if (lu->FactPattern == SamePattern_SameRowPerm) {
        lu->options.Fact = SamePattern_SameRowPerm; /* matrix has similar numerical values */
      } else { /* lu->FactPattern == SamePattern */
        PetscStackCall("SuperLU_DIST:Destroy_LU",Destroy_LU(N, &lu->grid, &lu->LUstruct));
        lu->options.Fact = SamePattern;
      }
    }
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:zCompRow_to_CompCol_dist",zCompRow_to_CompCol_dist(M,N,aa->nz,(doublecomplex*)aa->a,(int_t*)aa->j,(int_t*)aa->i,&lu->val,&lu->col, &lu->row));
#else
    PetscStackCall("SuperLU_DIST:dCompRow_to_CompCol_dist",dCompRow_to_CompCol_dist(M,N,aa->nz,aa->a,(int_t*)aa->j,(int_t*)aa->i,&lu->val, &lu->col, &lu->row));
#endif

    /* Create compressed column matrix A_sup. */
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:zCreate_CompCol_Matrix_dist",zCreate_CompCol_Matrix_dist(&lu->A_sup, M, N, aa->nz, lu->val, lu->col, lu->row, SLU_NC, SLU_Z, SLU_GE));
#else
    PetscStackCall("SuperLU_DIST:dCreate_CompCol_Matrix_dist",dCreate_CompCol_Matrix_dist(&lu->A_sup, M, N, aa->nz, lu->val, lu->col, lu->row, SLU_NC, SLU_D, SLU_GE));
#endif
  } else { /* distributed mat input */
    Mat_MPIAIJ *mat = (Mat_MPIAIJ*)A->data;
    aa=(Mat_SeqAIJ*)(mat->A)->data;
    bb=(Mat_SeqAIJ*)(mat->B)->data;
    ai=aa->i; aj=aa->j;
    bi=bb->i; bj=bb->j;
#if defined(PETSC_USE_COMPLEX)
    av=(doublecomplex*)aa->a;
    bv=(doublecomplex*)bb->a;
#else
    av=aa->a;
    bv=bb->a;
#endif
    rstart = A->rmap->rstart;
    nz     = aa->nz + bb->nz;
    garray = mat->garray;

    if (lu->options.Fact == DOFACT) { /* first numeric factorization */
#if defined(PETSC_USE_COMPLEX)
      PetscStackCall("SuperLU_DIST:zallocateA_dist",zallocateA_dist(m, nz, &lu->val, &lu->col, &lu->row));
#else
      PetscStackCall("SuperLU_DIST:dallocateA_dist",dallocateA_dist(m, nz, &lu->val, &lu->col, &lu->row));
#endif
    } else { /* successive numeric factorization, sparsity pattern and perm_c are reused. */
      if (lu->FactPattern == SamePattern_SameRowPerm) {
        lu->options.Fact = SamePattern_SameRowPerm; /* matrix has similar numerical values */
      } else if (lu->FactPattern == SamePattern) {
        PetscStackCall("SuperLU_DIST:Destroy_LU",Destroy_LU(N, &lu->grid, &lu->LUstruct)); /* Deallocate L and U matrices. */
        lu->options.Fact = SamePattern;
      } else if (lu->FactPattern == DOFACT) {
        PetscStackCall("SuperLU_DIST:Destroy_CompRowLoc_Matrix_dist",Destroy_CompRowLoc_Matrix_dist(&lu->A_sup));
        PetscStackCall("SuperLU_DIST:Destroy_LU",Destroy_LU(N, &lu->grid, &lu->LUstruct));
        lu->options.Fact = DOFACT;

#if defined(PETSC_USE_COMPLEX)
      PetscStackCall("SuperLU_DIST:zallocateA_dist",zallocateA_dist(m, nz, &lu->val, &lu->col, &lu->row));
#else
      PetscStackCall("SuperLU_DIST:dallocateA_dist",dallocateA_dist(m, nz, &lu->val, &lu->col, &lu->row));
#endif
      } else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"options.Fact must be one of SamePattern SamePattern_SameRowPerm DOFACT");
      }
    }
    nz = 0;
    for (i=0; i<m; i++) {
      lu->row[i] = nz;
      countA     = ai[i+1] - ai[i];
      countB     = bi[i+1] - bi[i];
      if (aj) {
        ajj = aj + ai[i]; /* ptr to the beginning of this row */
      } else {
        ajj = NULL;
      }
      bjj = bj + bi[i];

      /* B part, smaller col index */
      if (aj) {
        colA_start = rstart + ajj[0]; /* the smallest global col index of A */
      } else { /* superlu_dist does not require matrix has diagonal entries, thus aj=NULL would work */
        colA_start = rstart;
      }
      jB         = 0;
      for (j=0; j<countB; j++) {
        jcol = garray[bjj[j]];
        if (jcol > colA_start) {
          jB = j;
          break;
        }
        lu->col[nz]   = jcol;
        lu->val[nz++] = *bv++;
        if (j==countB-1) jB = countB;
      }

      /* A part */
      for (j=0; j<countA; j++) {
        lu->col[nz]   = rstart + ajj[j];
        lu->val[nz++] = *av++;
      }

      /* B part, larger col index */
      for (j=jB; j<countB; j++) {
        lu->col[nz]   = garray[bjj[j]];
        lu->val[nz++] = *bv++;
      }
    }
    lu->row[m] = nz;

    if (lu->options.Fact == DOFACT) {
#if defined(PETSC_USE_COMPLEX)
      PetscStackCall("SuperLU_DIST:zCreate_CompRowLoc_Matrix_dist",zCreate_CompRowLoc_Matrix_dist(&lu->A_sup, M, N, nz, m, rstart,lu->val, lu->col, lu->row, SLU_NR_loc, SLU_Z, SLU_GE));
#else
      PetscStackCall("SuperLU_DIST:dCreate_CompRowLoc_Matrix_dist",dCreate_CompRowLoc_Matrix_dist(&lu->A_sup, M, N, nz, m, rstart,lu->val, lu->col, lu->row, SLU_NR_loc, SLU_D, SLU_GE));
#endif
    }
  }

  /* Factor the matrix. */
  PetscStackCall("SuperLU_DIST:PStatInit",PStatInit(&stat));   /* Initialize the statistics variables. */
  if (lu->MatInputMode == GLOBAL) { /* global mat input */
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:pzgssvx_ABglobal",pzgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0,&lu->grid, &lu->LUstruct, berr, &stat, &sinfo));
#else
    PetscStackCall("SuperLU_DIST:pdgssvx_ABglobal",pdgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0,&lu->grid, &lu->LUstruct, berr, &stat, &sinfo));
#endif
  } else { /* distributed mat input */
#if defined(PETSC_USE_COMPLEX)
    PetscStackCall("SuperLU_DIST:pzgssvx",pzgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, m, 0, &lu->grid,&lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &sinfo));
#else
    PetscStackCall("SuperLU_DIST:pdgssvx",pdgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, m, 0, &lu->grid,&lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &sinfo));
#endif
  }
  
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

  if (lu->MatInputMode == GLOBAL && size > 1) {
    ierr = MatDestroy(&A_seq);CHKERRQ(ierr);
  }

  if (lu->options.PrintStat) {
    PStatPrint(&lu->options, &stat, &lu->grid);  /* Print the statistics. */
  }
  PetscStackCall("SuperLU_DIST:PStatFree",PStatFree(&stat));
  (F)->assembled    = PETSC_TRUE;
  (F)->preallocated = PETSC_TRUE;
  lu->options.Fact  = FACTORED; /* The factored form of A is supplied. Local option used by this func. only */
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
  lu->CleanUpSuperLU_Dist = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverPackage_aij_superlu_dist(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSUPERLU_DIST;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorInfo_SuperLU_DIST(Mat A,PetscViewer viewer)
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
  ierr    = PetscViewerASCIIPrintf(viewer,"  Matrix input mode %d \n",lu->MatInputMode);CHKERRQ(ierr);
  ierr    = PetscViewerASCIIPrintf(viewer,"  Replace tiny pivots %s \n",PetscBools[options.ReplaceTinyPivot != NO]);CHKERRQ(ierr);
  ierr    = PetscViewerASCIIPrintf(viewer,"  Use iterative refinement %s \n",PetscBools[options.IterRefine == SLU_DOUBLE]);CHKERRQ(ierr);
  ierr    = PetscViewerASCIIPrintf(viewer,"  Processors in row %d col partition %d \n",lu->nprow,lu->npcol);CHKERRQ(ierr);
  ierr    = PetscViewerASCIIPrintf(viewer,"  Row permutation %s \n",(options.RowPerm == NOROWPERM) ? "NATURAL" : "LargeDiag");CHKERRQ(ierr);

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
      ierr = MatFactorInfo_SuperLU_DIST(A,viewer);CHKERRQ(ierr);
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
  const char             *rowperm[]     = {"LargeDiag","NATURAL"};
  const char             *factPattern[] = {"SamePattern","SamePattern_SameRowPerm","DOFACT"};
  PetscBool              set;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,M,N);CHKERRQ(ierr);
  ierr = PetscStrallocpy("superlu_dist",&((PetscObject)B)->type_name);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  B->ops->getinfo          = MatGetInfo_External;
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_SuperLU_DIST;
  B->ops->view             = MatView_SuperLU_DIST;
  B->ops->destroy          = MatDestroy_SuperLU_DIST;

  B->factortype = MAT_FACTOR_LU;

  /* set solvertype */
  ierr = PetscFree(B->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERSUPERLU_DIST,&B->solvertype);CHKERRQ(ierr);

  ierr    = PetscNewLog(B,&lu);CHKERRQ(ierr);
  B->data = lu;

  /* Set the default input options:
     options.Fact              = DOFACT;
     options.Equil             = YES;
     options.ParSymbFact       = NO;
     options.ColPerm           = METIS_AT_PLUS_A;
     options.RowPerm           = LargeDiag;
     options.ReplaceTinyPivot  = YES;
     options.IterRefine        = DOUBLE;
     options.Trans             = NOTRANS;
     options.SolveInitialized  = NO; -hold the communication pattern used MatSolve() and MatMatSolve()
     options.RefineInitialized = NO;
     options.PrintStat         = YES;
  */
  set_default_options_dist(&options);

  ierr = MPI_Comm_dup(PetscObjectComm((PetscObject)A),&(lu->comm_superlu));CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRQ(ierr);
  /* Default num of process columns and rows */
  lu->npcol = (int_t) (0.5 + PetscSqrtReal((PetscReal)size));
  if (!lu->npcol) lu->npcol = 1;
  while (lu->npcol > 0) {
    lu->nprow = (int_t) (size/lu->npcol);
    if (size == lu->nprow * lu->npcol) break;
    lu->npcol--;
  }

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"SuperLU_Dist Options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_superlu_dist_r","Number rows in processor partition","None",lu->nprow,(PetscInt*)&lu->nprow,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_superlu_dist_c","Number columns in processor partition","None",lu->npcol,(PetscInt*)&lu->npcol,NULL);CHKERRQ(ierr);
  if (size != lu->nprow * lu->npcol) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number of processes %d must equal to nprow %d * npcol %d",size,lu->nprow,lu->npcol);

  lu->MatInputMode = DISTRIBUTED;

  ierr = PetscOptionsEnum("-mat_superlu_dist_matinput","Matrix input mode (global or distributed)","None",SuperLU_MatInputModes,(PetscEnum)lu->MatInputMode,(PetscEnum*)&lu->MatInputMode,NULL);CHKERRQ(ierr);
  if (lu->MatInputMode == DISTRIBUTED && size == 1) lu->MatInputMode = GLOBAL;

  ierr = PetscOptionsBool("-mat_superlu_dist_equil","Equilibrate matrix","None",options.Equil ? PETSC_TRUE : PETSC_FALSE,&flg,&set);CHKERRQ(ierr);
  if (set && !flg) options.Equil = NO;

  ierr = PetscOptionsEList("-mat_superlu_dist_rowperm","Row permutation","None",rowperm,2,rowperm[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) {
    switch (indx) {
    case 0:
      options.RowPerm = LargeDiag;
      break;
    case 1:
      options.RowPerm = NOROWPERM;
      break;
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
    if (lu->MatInputMode == GLOBAL) {
#if defined(PETSC_USE_INFO)
      ierr = PetscInfo(A,"Warning: '-mat_superlu_dist_parsymbfact' is ignored because MatInputMode=GLOBAL\n");CHKERRQ(ierr);
#endif
    } else {
#if defined(PETSC_HAVE_PARMETIS)
      options.ParSymbFact = YES;
      options.ColPerm     = PARMETIS;   /* in v2.2, PARMETIS is forced for ParSymbFact regardless of user ordering setting */
#else
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"parsymbfact needs PARMETIS");
#endif
    }
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

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverPackage_C",MatFactorGetSolverPackage_aij_superlu_dist);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSuperluDistGetDiagU_C",MatSuperluDistGetDiagU_SuperLU_DIST);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverPackageRegister_SuperLU_DIST(void)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatSolverPackageRegister(MATSOLVERSUPERLU_DIST,MATMPIAIJ,  MAT_FACTOR_LU,MatGetFactor_aij_superlu_dist);CHKERRQ(ierr);
  ierr = MatSolverPackageRegister(MATSOLVERSUPERLU_DIST,MATSEQAIJ,  MAT_FACTOR_LU,MatGetFactor_aij_superlu_dist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERSUPERLU_DIST - Parallel direct solver package for LU factorization

  Use ./configure --download-superlu_dist --download-parmetis --download-metis --download-ptscotch  to have PETSc installed with SuperLU_DIST

  Use -pc_type lu -pc_factor_mat_solver_package superlu_dist to us this direct solver

   Works with AIJ matrices

  Options Database Keys:
+ -mat_superlu_dist_r <n> - number of rows in processor partition
. -mat_superlu_dist_c <n> - number of columns in processor partition
. -mat_superlu_dist_matinput <0,1> - matrix input mode; 0=global, 1=distributed
. -mat_superlu_dist_equil - equilibrate the matrix
. -mat_superlu_dist_rowperm <LargeDiag,NATURAL> - row permutation
. -mat_superlu_dist_colperm <MMD_AT_PLUS_A,MMD_ATA,NATURAL> - column permutation
. -mat_superlu_dist_replacetinypivot - replace tiny pivots
. -mat_superlu_dist_fact <SamePattern> - (choose one of) SamePattern SamePattern_SameRowPerm DOFACT
. -mat_superlu_dist_iterrefine - use iterative refinement
- -mat_superlu_dist_statprint - print factorization information

   Level: beginner

.seealso: PCLU

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage

M*/
