
/* 
        Provides an interface to the SuperLU_DIST_2.2 sparse solver
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
  int_t                   nprow,npcol,*row,*col;
  gridinfo_t              grid;
  superlu_options_t       options;
  SuperMatrix             A_sup;
  ScalePermstruct_t       ScalePermstruct;
  LUstruct_t              LUstruct;
  int                     StatPrint;
  SuperLU_MatInputMode    MatInputMode;
  SOLVEstruct_t           SOLVEstruct; 
  fact_t                  FactPattern;
  MPI_Comm                comm_superlu;
#if defined(PETSC_USE_COMPLEX)
  doublecomplex           *val;
#else
  double                  *val;
#endif
  PetscBool               matsolve_iscalled,matmatsolve_iscalled;
  PetscBool  CleanUpSuperLU_Dist; /* Flag to clean up (non-global) SuperLU objects during Destroy */
} Mat_SuperLU_DIST;

extern PetscErrorCode MatFactorInfo_SuperLU_DIST(Mat,PetscViewer);
extern PetscErrorCode MatLUFactorNumeric_SuperLU_DIST(Mat,Mat,const MatFactorInfo *);
extern PetscErrorCode MatDestroy_SuperLU_DIST(Mat);
extern PetscErrorCode MatView_SuperLU_DIST(Mat,PetscViewer);
extern PetscErrorCode MatSolve_SuperLU_DIST(Mat,Vec,Vec);
extern PetscErrorCode MatLUFactorSymbolic_SuperLU_DIST(Mat,Mat,IS,IS,const MatFactorInfo *);
extern PetscErrorCode MatDestroy_MPIAIJ(Mat);

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SuperLU_DIST"
PetscErrorCode MatDestroy_SuperLU_DIST(Mat A)
{
  PetscErrorCode   ierr;
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)A->spptr;
  PetscBool        flg;

  PetscFunctionBegin;
  if (lu && lu->CleanUpSuperLU_Dist) {
    /* Deallocate SuperLU_DIST storage */
    if (lu->MatInputMode == GLOBAL) {
      Destroy_CompCol_Matrix_dist(&lu->A_sup);
    } else {
      Destroy_CompRowLoc_Matrix_dist(&lu->A_sup);
      if ( lu->options.SolveInitialized ) {
#if defined(PETSC_USE_COMPLEX)
        zSolveFinalize(&lu->options, &lu->SOLVEstruct);
#else
        dSolveFinalize(&lu->options, &lu->SOLVEstruct);
#endif
      }
    }
    Destroy_LU(A->cmap->N, &lu->grid, &lu->LUstruct);
    ScalePermstructFree(&lu->ScalePermstruct);
    LUstructFree(&lu->LUstruct);

    /* Release the SuperLU_DIST process grid. */
    superlu_gridexit(&lu->grid);

    ierr = MPI_Comm_free(&(lu->comm_superlu));CHKERRQ(ierr);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SuperLU_DIST"
PetscErrorCode MatSolve_SuperLU_DIST(Mat A,Vec b_mpi,Vec x)
{
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)A->spptr;
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

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
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
    if (lu->options.SolveInitialized && !lu->matsolve_iscalled){
      /* see comments in MatMatSolve() */
#if defined(PETSC_USE_COMPLEX)
      zSolveFinalize(&lu->options, &lu->SOLVEstruct);
#else
      dSolveFinalize(&lu->options, &lu->SOLVEstruct);
#endif
      lu->options.SolveInitialized = NO; 
    }
    ierr = VecCopy(b_mpi,x);CHKERRQ(ierr);
    ierr = VecGetArray(x,&bptr);CHKERRQ(ierr);
  }
 
  if (lu->options.Fact != FACTORED) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"SuperLU_DIST options.Fact mush equal FACTORED");

  PStatInit(&stat);        /* Initialize the statistics variables. */
  if (lu->MatInputMode == GLOBAL) { 
#if defined(PETSC_USE_COMPLEX)
    pzgssvx_ABglobal(&lu->options,&lu->A_sup,&lu->ScalePermstruct,(doublecomplex*)bptr,M,nrhs, 
                   &lu->grid,&lu->LUstruct,berr,&stat,&info);
#else
    pdgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct,bptr,M,nrhs, 
                   &lu->grid,&lu->LUstruct,berr,&stat,&info);
#endif 
  } else { /* distributed mat input */
#if defined(PETSC_USE_COMPLEX)
    pzgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,(doublecomplex*)bptr,m,nrhs,&lu->grid,
	    &lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info);
#else
    pdgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,bptr,m,nrhs,&lu->grid,
	    &lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info);
#endif
  }
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"pdgssvx fails, info: %d\n",info);

  if (lu->options.PrintStat) {
     PStatPrint(&lu->options, &stat, &lu->grid);     /* Print the statistics. */
  }
  PStatFree(&stat);
 
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

#undef __FUNCT__  
#define __FUNCT__ "MatMatSolve_SuperLU_DIST"
PetscErrorCode MatMatSolve_SuperLU_DIST(Mat A,Mat B_mpi,Mat X)
{
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)A->spptr;
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
  ierr = PetscTypeCompareAny((PetscObject)B_mpi,&flg,MATSEQDENSE,MATMPIDENSE,PETSC_NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  ierr = PetscTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,PETSC_NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");
  
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  if (size > 1 && lu->MatInputMode == GLOBAL) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatInputMode=GLOBAL for nproc %d>1 is not supported",size);
  /* size==1 or distributed mat input */
  if (lu->options.SolveInitialized && !lu->matmatsolve_iscalled){
    /* communication pattern of SOLVEstruct is unlikely created for matmatsolve, 
       thus destroy it and create a new SOLVEstruct.
       Otherwise it may result in memory corruption or incorrect solution 
       See src/mat/examples/tests/ex125.c */
#if defined(PETSC_USE_COMPLEX)
    zSolveFinalize(&lu->options, &lu->SOLVEstruct);
#else
    dSolveFinalize(&lu->options, &lu->SOLVEstruct);
#endif
    lu->options.SolveInitialized  = NO; 
  }
  ierr = MatCopy(B_mpi,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = MatGetSize(B_mpi,PETSC_NULL,&nrhs);CHKERRQ(ierr);
  
  PStatInit(&stat);        /* Initialize the statistics variables. */
  ierr = MatGetArray(X,&bptr);CHKERRQ(ierr);
  if (lu->MatInputMode == GLOBAL) { /* size == 1 */
#if defined(PETSC_USE_COMPLEX)
    pzgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct,(doublecomplex*)bptr, M, nrhs, 
                   &lu->grid, &lu->LUstruct, berr, &stat, &info);
#else
    pdgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct,bptr, M, nrhs, 
                   &lu->grid, &lu->LUstruct, berr, &stat, &info);
#endif 
  } else { /* distributed mat input */
#if defined(PETSC_USE_COMPLEX)
    pzgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,(doublecomplex*)bptr,m,nrhs,&lu->grid,
	    &lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info);
#else
    pdgssvx(&lu->options,&lu->A_sup,&lu->ScalePermstruct,bptr,m,nrhs,&lu->grid,
	    &lu->LUstruct,&lu->SOLVEstruct,berr,&stat,&info);
#endif
  }
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"pdgssvx fails, info: %d\n",info);
  ierr = MatRestoreArray(X,&bptr);CHKERRQ(ierr);

  if (lu->options.PrintStat) {
     PStatPrint(&lu->options, &stat, &lu->grid); /* Print the statistics. */
  }
  PStatFree(&stat);
  lu->matsolve_iscalled    = PETSC_FALSE;
  lu->matmatsolve_iscalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}


#undef __FUNCT__   
#define __FUNCT__ "MatLUFactorNumeric_SuperLU_DIST"
PetscErrorCode MatLUFactorNumeric_SuperLU_DIST(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat              *tseq,A_seq = PETSC_NULL;
  Mat_SeqAIJ       *aa,*bb;
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)(F)->spptr;
  PetscErrorCode   ierr;
  PetscInt         M=A->rmap->N,N=A->cmap->N,i,*ai,*aj,*bi,*bj,nz,rstart,*garray,
                   m=A->rmap->n, colA_start,j,jcol,jB,countA,countB,*bjj,*ajj;
  int              sinfo; /* SuperLU_Dist info flag is always an int even with long long indices */
  PetscMPIInt      size;
  SuperLUStat_t    stat;
  double           *berr=0;
  IS               isrow;
  PetscLogDouble   time0,time,time_min,time_max; 
  Mat              F_diag=PETSC_NULL;
#if defined(PETSC_USE_COMPLEX)
  doublecomplex    *av, *bv; 
#else
  double           *av, *bv; 
#endif

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  
  if (lu->options.PrintStat) { /* collect time for mat conversion */
    ierr = MPI_Barrier(((PetscObject)A)->comm);CHKERRQ(ierr);
    ierr = PetscGetTime(&time0);CHKERRQ(ierr);  
  }

  if (lu->MatInputMode == GLOBAL) { /* global mat input */
    if (size > 1) { /* convert mpi A to seq mat A */
      ierr = ISCreateStride(PETSC_COMM_SELF,M,0,1,&isrow);CHKERRQ(ierr);  
      ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&tseq);CHKERRQ(ierr);
      ierr = ISDestroy(&isrow);CHKERRQ(ierr);
   
      A_seq = *tseq;
      ierr = PetscFree(tseq);CHKERRQ(ierr);
      aa =  (Mat_SeqAIJ*)A_seq->data;
    } else {
      PetscBool  flg;
      ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
      if (flg) {
        Mat_MPIAIJ *At = (Mat_MPIAIJ*)A->data;
        A = At->A;
      }
      aa =  (Mat_SeqAIJ*)A->data;
    }

    /* Convert Petsc NR matrix to SuperLU_DIST NC. 
       Note: memories of lu->val, col and row are allocated by CompRow_to_CompCol_dist()! */
    if (lu->options.Fact != DOFACT) {/* successive numeric factorization, sparsity pattern is reused. */
      Destroy_CompCol_Matrix_dist(&lu->A_sup);
      if (lu->FactPattern == SamePattern_SameRowPerm){
        lu->options.Fact = SamePattern_SameRowPerm; /* matrix has similar numerical values */
      } else { /* lu->FactPattern == SamePattern */
        Destroy_LU(N, &lu->grid, &lu->LUstruct); 
        lu->options.Fact = SamePattern;    
      }
    }
#if defined(PETSC_USE_COMPLEX)
    zCompRow_to_CompCol_dist(M,N,aa->nz,(doublecomplex*)aa->a,aa->j,aa->i,&lu->val,&lu->col, &lu->row);
#else
    dCompRow_to_CompCol_dist(M,N,aa->nz,aa->a,aa->j,aa->i,&lu->val, &lu->col, &lu->row);
#endif

    /* Create compressed column matrix A_sup. */
#if defined(PETSC_USE_COMPLEX)
    zCreate_CompCol_Matrix_dist(&lu->A_sup, M, N, aa->nz, lu->val, lu->col, lu->row, SLU_NC, SLU_Z, SLU_GE);
#else
    dCreate_CompCol_Matrix_dist(&lu->A_sup, M, N, aa->nz, lu->val, lu->col, lu->row, SLU_NC, SLU_D, SLU_GE);  
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
   
    if (lu->options.Fact == DOFACT) {/* first numeric factorization */
#if defined(PETSC_USE_COMPLEX)
      zallocateA_dist(m, nz, &lu->val, &lu->col, &lu->row);
#else
      dallocateA_dist(m, nz, &lu->val, &lu->col, &lu->row);
#endif
    } else { /* successive numeric factorization, sparsity pattern and perm_c are reused. */
      /* Destroy_CompRowLoc_Matrix_dist(&lu->A_sup); */ /* this leads to crash! However, see SuperLU_DIST_2.5/EXAMPLE/pzdrive2.c */
      if (lu->FactPattern == SamePattern_SameRowPerm){
        lu->options.Fact = SamePattern_SameRowPerm; /* matrix has similar numerical values */
      } else {
        Destroy_LU(N, &lu->grid, &lu->LUstruct); /* Deallocate storage associated with the L and U matrices. */
        lu->options.Fact = SamePattern;
      }
    }
    nz = 0;
    for ( i=0; i<m; i++ ) {
      lu->row[i] = nz;
      countA = ai[i+1] - ai[i];
      countB = bi[i+1] - bi[i];
      ajj = aj + ai[i];  /* ptr to the beginning of this row */
      bjj = bj + bi[i];  

      /* B part, smaller col index */   
      colA_start = rstart + ajj[0]; /* the smallest global col index of A */  
      jB = 0;
      for (j=0; j<countB; j++){
        jcol = garray[bjj[j]];
        if (jcol > colA_start) {
          jB = j;
          break;
        }
        lu->col[nz] = jcol; 
        lu->val[nz++] = *bv++;
        if (j==countB-1) jB = countB; 
      }

      /* A part */
      for (j=0; j<countA; j++){
        lu->col[nz] = rstart + ajj[j]; 
        lu->val[nz++] = *av++;
      }

      /* B part, larger col index */      
      for (j=jB; j<countB; j++){
        lu->col[nz] = garray[bjj[j]];
        lu->val[nz++] = *bv++;
      }
    } 
    lu->row[m] = nz;
#if defined(PETSC_USE_COMPLEX)
    zCreate_CompRowLoc_Matrix_dist(&lu->A_sup, M, N, nz, m, rstart,lu->val, lu->col, lu->row, SLU_NR_loc, SLU_Z, SLU_GE);
#else
    dCreate_CompRowLoc_Matrix_dist(&lu->A_sup, M, N, nz, m, rstart,lu->val, lu->col, lu->row, SLU_NR_loc, SLU_D, SLU_GE);
#endif
  }
  if (lu->options.PrintStat) {
    ierr = PetscGetTime(&time);CHKERRQ(ierr);  
    time0 = time - time0;
  }

  /* Factor the matrix. */
  PStatInit(&stat);   /* Initialize the statistics variables. */
  if (lu->MatInputMode == GLOBAL) { /* global mat input */
#if defined(PETSC_USE_COMPLEX)
    pzgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0,&lu->grid, &lu->LUstruct, berr, &stat, &sinfo);
#else
    pdgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0,&lu->grid, &lu->LUstruct, berr, &stat, &sinfo);
#endif 
  } else { /* distributed mat input */
#if defined(PETSC_USE_COMPLEX)
    pzgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, m, 0, &lu->grid,&lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &sinfo);
    if (sinfo) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"pzgssvx fails, info: %d\n",sinfo);
#else
    pdgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, m, 0, &lu->grid,&lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &sinfo);
    if (sinfo) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"pdgssvx fails, info: %d\n",sinfo);
#endif
  }

  if (lu->MatInputMode == GLOBAL && size > 1){
    ierr = MatDestroy(&A_seq);CHKERRQ(ierr);
  }

  if (lu->options.PrintStat) {
    ierr = MPI_Reduce(&time0,&time_max,1,MPI_DOUBLE,MPI_MAX,0,((PetscObject)A)->comm);
    ierr = MPI_Reduce(&time0,&time_min,1,MPI_DOUBLE,MPI_MIN,0,((PetscObject)A)->comm);
    ierr = MPI_Reduce(&time0,&time,1,MPI_DOUBLE,MPI_SUM,0,((PetscObject)A)->comm);
    time = time/size; /* average time */
    ierr = PetscPrintf(((PetscObject)A)->comm, "        Mat conversion(PETSc->SuperLU_DIST) time (max/min/avg): \n                              %g / %g / %g\n",time_max,time_min,time);CHKERRQ(ierr);
    PStatPrint(&lu->options, &stat, &lu->grid);  /* Print the statistics. */
  }
  PStatFree(&stat);  
  if (size > 1){
    F_diag = ((Mat_MPIAIJ *)(F)->data)->A;
    F_diag->assembled = PETSC_TRUE; 
  }
  (F)->assembled    = PETSC_TRUE;
  (F)->preallocated = PETSC_TRUE;
  lu->options.Fact  = FACTORED; /* The factored form of A is supplied. Local option used by this func. only */
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SuperLU_DIST"
PetscErrorCode MatLUFactorSymbolic_SuperLU_DIST(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_SuperLU_DIST  *lu = (Mat_SuperLU_DIST*)F->spptr;   
  PetscInt          M=A->rmap->N,N=A->cmap->N;

  PetscFunctionBegin;
  /* Initialize the SuperLU process grid. */
  superlu_gridinit(lu->comm_superlu, lu->nprow, lu->npcol, &lu->grid);

  /* Initialize ScalePermstruct and LUstruct. */
  ScalePermstructInit(M, N, &lu->ScalePermstruct);
  LUstructInit(M, N, &lu->LUstruct); 
  F->ops->lufactornumeric = MatLUFactorNumeric_SuperLU_DIST;
  F->ops->solve           = MatSolve_SuperLU_DIST;
  F->ops->matsolve        = MatMatSolve_SuperLU_DIST;
  lu->CleanUpSuperLU_Dist = PETSC_TRUE;
  PetscFunctionReturn(0); 
}

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_aij_superlu_dist"
PetscErrorCode MatFactorGetSolverPackage_aij_superlu_dist(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSUPERLU_DIST;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_aij_superlu_dist"
PetscErrorCode MatGetFactor_aij_superlu_dist(Mat A,MatFactorType ftype,Mat *F)
{
  Mat               B;
  Mat_SuperLU_DIST  *lu;   
  PetscErrorCode    ierr;
  PetscInt          M=A->rmap->N,N=A->cmap->N,indx;
  PetscMPIInt       size;
  superlu_options_t options;
  PetscBool         flg;
  const char        *colperm[] = {"NATURAL","MMD_AT_PLUS_A","MMD_ATA","METIS_AT_PLUS_A","PARMETIS"}; 
  const char        *rowperm[] = {"LargeDiag","NATURAL"}; 
  const char        *factPattern[] = {"SamePattern","SamePattern_SameRowPerm"};

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);
  ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactorsymbolic = MatLUFactorSymbolic_SuperLU_DIST;
  B->ops->view             = MatView_SuperLU_DIST;
  B->ops->destroy          = MatDestroy_SuperLU_DIST;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_aij_superlu_dist",MatFactorGetSolverPackage_aij_superlu_dist);CHKERRQ(ierr);
  B->factortype            = MAT_FACTOR_LU;  

  ierr = PetscNewLog(B,Mat_SuperLU_DIST,&lu);CHKERRQ(ierr);
  B->spptr = lu;

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

  ierr = MPI_Comm_dup(((PetscObject)A)->comm,&(lu->comm_superlu));CHKERRQ(ierr);
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&size);CHKERRQ(ierr);
  /* Default num of process columns and rows */
  lu->npcol = PetscMPIIntCast((PetscInt)(0.5 + PetscSqrtReal((PetscReal)size))); 
  if (!lu->npcol) lu->npcol = 1;
  while (lu->npcol > 0) {
    lu->nprow = PetscMPIIntCast(size/lu->npcol);  
    if (size == lu->nprow * lu->npcol) break;
    lu->npcol --;
  }    
  
  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"SuperLU_Dist Options","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_superlu_dist_r","Number rows in processor partition","None",lu->nprow,&lu->nprow,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_superlu_dist_c","Number columns in processor partition","None",lu->npcol,&lu->npcol,PETSC_NULL);CHKERRQ(ierr);
    if (size != lu->nprow * lu->npcol) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Number of processes %d must equal to nprow %d * npcol %d",size,lu->nprow,lu->npcol);
  
    lu->MatInputMode = DISTRIBUTED;
    ierr = PetscOptionsEnum("-mat_superlu_dist_matinput","Matrix input mode (global or distributed)","None",SuperLU_MatInputModes,(PetscEnum)lu->MatInputMode,(PetscEnum*)&lu->MatInputMode,PETSC_NULL);CHKERRQ(ierr);
    if(lu->MatInputMode == DISTRIBUTED && size == 1) lu->MatInputMode = GLOBAL;

    ierr = PetscOptionsBool("-mat_superlu_dist_equil","Equilibrate matrix","None",PETSC_TRUE,&flg,0);CHKERRQ(ierr); 
    if (!flg) {
      options.Equil = NO;
    }

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
        options.ColPerm = PARMETIS; /* only works for np>1 */
        break;
      default:
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Unknown column permutation");
      }
    }

    ierr = PetscOptionsBool("-mat_superlu_dist_replacetinypivot","Replace tiny pivots","None",PETSC_TRUE,&flg,0);CHKERRQ(ierr); 
    if (!flg) {
      options.ReplaceTinyPivot = NO;
    }

    options.ParSymbFact = NO;
    ierr = PetscOptionsBool("-mat_superlu_dist_parsymbfact","Parallel symbolic factorization","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr); 
    if (flg){
#ifdef PETSC_HAVE_PARMETIS
      options.ParSymbFact = YES;
      options.ColPerm     = PARMETIS; /* in v2.2, PARMETIS is forced for ParSymbFact regardless of user ordering setting */
#else
      printf("parsymbfact needs PARMETIS");
#endif
    }

    lu->FactPattern = SamePattern_SameRowPerm;
    ierr = PetscOptionsEList("-mat_superlu_dist_fact","Sparsity pattern for repeated matrix factorization","None",factPattern,2,factPattern[1],&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      switch (indx) {
      case 0:
        lu->FactPattern = SamePattern;
        break;
      case 1:
        lu->FactPattern = SamePattern_SameRowPerm;
        break;
      }
    } 
    
    options.IterRefine = NOREFINE;
    ierr = PetscOptionsBool("-mat_superlu_dist_iterrefine","Use iterative refinement","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
    if (flg) {
      options.IterRefine = SLU_DOUBLE;
    }

    if (PetscLogPrintInfo) {
      options.PrintStat = YES; 
    } else {
      options.PrintStat = NO;
    }
    ierr = PetscOptionsBool("-mat_superlu_dist_statprint","Print factorization information","None",
                              (PetscBool)options.PrintStat,(PetscBool*)&options.PrintStat,0);CHKERRQ(ierr); 
  PetscOptionsEnd();

  lu->options              = options; 
  lu->options.Fact         = DOFACT;
  lu->matsolve_iscalled    = PETSC_FALSE;
  lu->matmatsolve_iscalled = PETSC_FALSE;
  *F = B;
  PetscFunctionReturn(0); 
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqaij_superlu_dist"
PetscErrorCode MatGetFactor_seqaij_superlu_dist(Mat A,MatFactorType ftype,Mat *F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetFactor_aij_superlu_dist(A,ftype,F);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_mpiaij_superlu_dist"
PetscErrorCode MatGetFactor_mpiaij_superlu_dist(Mat A,MatFactorType ftype,Mat *F)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatGetFactor_aij_superlu_dist(A,ftype,F);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatFactorInfo_SuperLU_DIST"
PetscErrorCode MatFactorInfo_SuperLU_DIST(Mat A,PetscViewer viewer)
{
  Mat_SuperLU_DIST  *lu=(Mat_SuperLU_DIST*)A->spptr;
  superlu_options_t options;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* check if matrix is superlu_dist type */
  if (A->ops->solve != MatSolve_SuperLU_DIST) PetscFunctionReturn(0);

  options = lu->options;
  ierr = PetscViewerASCIIPrintf(viewer,"SuperLU_DIST run parameters:\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Process grid nprow %D x npcol %D \n",lu->nprow,lu->npcol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Equilibrate matrix %s \n",PetscBools[options.Equil != NO]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Matrix input mode %d \n",lu->MatInputMode);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Replace tiny pivots %s \n",PetscBools[options.ReplaceTinyPivot != NO]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Use iterative refinement %s \n",PetscBools[options.IterRefine == SLU_DOUBLE]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Processors in row %d col partition %d \n",lu->nprow,lu->npcol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Row permutation %s \n",(options.RowPerm == NOROWPERM) ? "NATURAL": "LargeDiag");CHKERRQ(ierr);
 
  switch(options.ColPerm){
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
  
  if (lu->FactPattern == SamePattern){
    ierr = PetscViewerASCIIPrintf(viewer,"  Repeated factorization SamePattern\n");CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer,"  Repeated factorization SamePattern_SameRowPerm\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_SuperLU_DIST"
PetscErrorCode MatView_SuperLU_DIST(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatFactorInfo_SuperLU_DIST(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


/*MC
  MATSOLVERSUPERLU_DIST - Parallel direct solver package for LU factorization

   Works with AIJ matrices  

  Options Database Keys:
+ -mat_superlu_dist_r <n> - number of rows in processor partition
. -mat_superlu_dist_c <n> - number of columns in processor partition
. -mat_superlu_dist_matinput <0,1> - matrix input mode; 0=global, 1=distributed
. -mat_superlu_dist_equil - equilibrate the matrix
. -mat_superlu_dist_rowperm <LargeDiag,NATURAL> - row permutation
. -mat_superlu_dist_colperm <MMD_AT_PLUS_A,MMD_ATA,NATURAL> - column permutation
. -mat_superlu_dist_replacetinypivot - replace tiny pivots
. -mat_superlu_dist_fact <SamePattern> - (choose one of) SamePattern SamePattern_SameRowPerm
. -mat_superlu_dist_iterrefine - use iterative refinement
- -mat_superlu_dist_statprint - print factorization information

   Level: beginner

.seealso: PCLU

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage

M*/

