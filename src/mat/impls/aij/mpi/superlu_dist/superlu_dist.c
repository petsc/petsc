#define PETSCMAT_DLL

/* 
        Provides an interface to the SuperLU_DIST_2.0 sparse solver
*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"
#if defined(PETSC_HAVE_STDLIB_H) /* This is to get arround weird problem with SuperLU on cray */
#include "stdlib.h"
#endif

EXTERN_C_BEGIN 
#if defined(PETSC_USE_COMPLEX)
#include "superlu_zdefs.h"
#else
#include "superlu_ddefs.h"
#endif
EXTERN_C_END 

typedef enum { GLOBAL,DISTRIBUTED
} SuperLU_MatInputMode;

typedef struct {
  int_t                   nprow,npcol,*row,*col;
  gridinfo_t              grid;
  superlu_options_t       options;
  SuperMatrix             A_sup;
  ScalePermstruct_t       ScalePermstruct;
  LUstruct_t              LUstruct;
  int                     StatPrint;
  int                     MatInputMode;
  SOLVEstruct_t           SOLVEstruct; 
  fact_t                  FactPattern;
  MPI_Comm                comm_superlu;
#if defined(PETSC_USE_COMPLEX)
  doublecomplex           *val;
#else
  double                  *val;
#endif

  /* A few function pointers for inheritance */
  PetscErrorCode (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);
  PetscErrorCode (*MatView)(Mat,PetscViewer);
  PetscErrorCode (*MatAssemblyEnd)(Mat,MatAssemblyType);
  PetscErrorCode (*MatLUFactorSymbolic)(Mat,IS,IS,MatFactorInfo*,Mat*);
  PetscErrorCode (*MatDestroy)(Mat);

  /* Flag to clean up (non-global) SuperLU objects during Destroy */
  PetscTruth CleanUpSuperLU_Dist;
} Mat_SuperLU_DIST;

EXTERN PetscErrorCode MatDuplicate_SuperLU_DIST(Mat,MatDuplicateOption,Mat*);

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SuperLU_DIST_Base"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SuperLU_DIST_Base(Mat A,MatType type,MatReuse reuse,Mat *newmat) 
{
  PetscErrorCode   ierr;
  Mat              B=*newmat;
  Mat_SuperLU_DIST *lu=(Mat_SuperLU_DIST *)A->spptr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  /* Reset the original function pointers */
  B->ops->duplicate        = lu->MatDuplicate;
  B->ops->view             = lu->MatView;
  B->ops->assemblyend      = lu->MatAssemblyEnd;
  B->ops->lufactorsymbolic = lu->MatLUFactorSymbolic;
  B->ops->destroy          = lu->MatDestroy;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaij_superlu_dist_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_superlu_dist_seqaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_mpiaij_superlu_dist_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_superlu_dist_mpiaij_C","",PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)B,type);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SuperLU_DIST"
PetscErrorCode MatDestroy_SuperLU_DIST(Mat A)
{
  PetscErrorCode   ierr;
  PetscMPIInt      size;
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)A->spptr; 
    
  PetscFunctionBegin;
  if (lu->CleanUpSuperLU_Dist) {
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
    Destroy_LU(A->cmap.N, &lu->grid, &lu->LUstruct);
    ScalePermstructFree(&lu->ScalePermstruct);
    LUstructFree(&lu->LUstruct);
    
    /* Release the SuperLU_DIST process grid. */
    superlu_gridexit(&lu->grid);
    
    ierr = MPI_Comm_free(&(lu->comm_superlu));CHKERRQ(ierr);
  }

  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatConvert_SuperLU_DIST_Base(A,MATSEQAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  } else {
    ierr = MatConvert_SuperLU_DIST_Base(A,MATMPIAIJ,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  }
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SuperLU_DIST"
PetscErrorCode MatSolve_SuperLU_DIST(Mat A,Vec b_mpi,Vec x)
{
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)A->spptr;
  PetscErrorCode   ierr;
  PetscMPIInt      size;
  PetscInt         m=A->rmap.N, N=A->cmap.N; 
  SuperLUStat_t    stat;  
  double           berr[1];
  PetscScalar      *bptr;  
  PetscInt         info, nrhs=1;
  Vec              x_seq;
  IS               iden;
  VecScatter       scat;
  
  PetscFunctionBegin;
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size > 1) {  
    if (lu->MatInputMode == GLOBAL) { /* global mat input, convert b to x_seq */
      ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&iden);CHKERRQ(ierr);
      ierr = VecScatterCreate(b_mpi,iden,x_seq,iden,&scat);CHKERRQ(ierr);
      ierr = ISDestroy(iden);CHKERRQ(ierr);

      ierr = VecScatterBegin(b_mpi,x_seq,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
      ierr = VecScatterEnd(b_mpi,x_seq,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
      ierr = VecGetArray(x_seq,&bptr);CHKERRQ(ierr); 
    } else { /* distributed mat input */
      ierr = VecCopy(b_mpi,x);CHKERRQ(ierr);
      ierr = VecGetArray(x,&bptr);CHKERRQ(ierr);
    }
  } else { /* size == 1 */
    ierr = VecCopy(b_mpi,x);CHKERRQ(ierr);
    ierr = VecGetArray(x,&bptr);CHKERRQ(ierr); 
  }
 
  if (lu->options.Fact != FACTORED) 
    SETERRQ(PETSC_ERR_ARG_WRONG,"SuperLU_DIST options.Fact mush equal FACTORED");

  PStatInit(&stat);        /* Initialize the statistics variables. */
  if (lu->MatInputMode == GLOBAL) { 
#if defined(PETSC_USE_COMPLEX)
    pzgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct,(doublecomplex*)bptr, m, nrhs, 
                   &lu->grid, &lu->LUstruct, berr, &stat, &info);
#else
    pdgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct,bptr, m, nrhs, 
                   &lu->grid, &lu->LUstruct, berr, &stat, &info);
#endif 
  } else { /* distributed mat input */
#if defined(PETSC_USE_COMPLEX)
    pzgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, (doublecomplex*)bptr, A->rmap.N, nrhs, &lu->grid,
	    &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &info);
    if (info) SETERRQ1(PETSC_ERR_LIB,"pzgssvx fails, info: %d\n",info);
#else
    pdgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, bptr, A->rmap.N, nrhs, &lu->grid,
	    &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &info);
    if (info) SETERRQ1(PETSC_ERR_LIB,"pdgssvx fails, info: %d\n",info);
#endif
  }
  if (lu->options.PrintStat) {
     PStatPrint(&lu->options, &stat, &lu->grid);     /* Print the statistics. */
  }
  PStatFree(&stat);
 
  if (size > 1) {    
    if (lu->MatInputMode == GLOBAL){ /* convert seq x to mpi x */
      ierr = VecRestoreArray(x_seq,&bptr);CHKERRQ(ierr);
      ierr = VecScatterBegin(x_seq,x,INSERT_VALUES,SCATTER_REVERSE,scat);CHKERRQ(ierr);
      ierr = VecScatterEnd(x_seq,x,INSERT_VALUES,SCATTER_REVERSE,scat);CHKERRQ(ierr);
      ierr = VecScatterDestroy(scat);CHKERRQ(ierr);
      ierr = VecDestroy(x_seq);CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray(x,&bptr);CHKERRQ(ierr);
    }
  } else {
    ierr = VecRestoreArray(x,&bptr);CHKERRQ(ierr); 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatLUFactorNumeric_SuperLU_DIST"
PetscErrorCode MatLUFactorNumeric_SuperLU_DIST(Mat A,MatFactorInfo *info,Mat *F)
{
  Mat              *tseq,A_seq = PETSC_NULL;
  Mat_SeqAIJ       *aa,*bb;
  Mat_SuperLU_DIST *lu = (Mat_SuperLU_DIST*)(*F)->spptr;
  PetscErrorCode   ierr;
  PetscInt         M=A->rmap.N,N=A->cmap.N,sinfo,i,*ai,*aj,*bi,*bj,nz,rstart,*garray,
                   m=A->rmap.n, irow,colA_start,j,jcol,jB,countA,countB,*bjj,*ajj;
  PetscMPIInt      size,rank;
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
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(A->comm,&rank);CHKERRQ(ierr);
  
  if (lu->options.PrintStat) { /* collect time for mat conversion */
    ierr = MPI_Barrier(A->comm);CHKERRQ(ierr);
    ierr = PetscGetTime(&time0);CHKERRQ(ierr);  
  }

  if (lu->MatInputMode == GLOBAL) { /* global mat input */
    if (size > 1) { /* convert mpi A to seq mat A */
      ierr = ISCreateStride(PETSC_COMM_SELF,M,0,1,&isrow);CHKERRQ(ierr);  
      ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&tseq);CHKERRQ(ierr);
      ierr = ISDestroy(isrow);CHKERRQ(ierr);
   
      A_seq = *tseq;
      ierr = PetscFree(tseq);CHKERRQ(ierr);
      aa =  (Mat_SeqAIJ*)A_seq->data;
    } else {
      aa =  (Mat_SeqAIJ*)A->data;
    }

    /* Convert Petsc NR matrix to SuperLU_DIST NC. 
       Note: memories of lu->val, col and row are allocated by CompRow_to_CompCol_dist()! */
    if (lu->options.Fact != DOFACT) {/* successive numeric factorization, sparsity pattern is reused. */
      if (lu->FactPattern == SamePattern_SameRowPerm){
        Destroy_CompCol_Matrix_dist(&lu->A_sup);
        /* Destroy_LU(N, &lu->grid, &lu->LUstruct); Crash! Comment it out does not lead to mem leak. */
        lu->options.Fact = SamePattern_SameRowPerm; /* matrix has similar numerical values */
      } else {
        Destroy_CompCol_Matrix_dist(&lu->A_sup); 
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
    rstart = A->rmap.rstart;
    nz     = aa->nz + bb->nz;
    garray = mat->garray;
   
    if (lu->options.Fact == DOFACT) {/* first numeric factorization */
#if defined(PETSC_USE_COMPLEX)
      zallocateA_dist(m, nz, &lu->val, &lu->col, &lu->row);
#else
      dallocateA_dist(m, nz, &lu->val, &lu->col, &lu->row);
#endif
    } else { /* successive numeric factorization, sparsity pattern and perm_c are reused. */
      if (lu->FactPattern == SamePattern_SameRowPerm){
        /* Destroy_LU(N, &lu->grid, &lu->LUstruct); Crash! Comment it out does not lead to mem leak. */
        lu->options.Fact = SamePattern_SameRowPerm; /* matrix has similar numerical values */
      } else {
        Destroy_LU(N, &lu->grid, &lu->LUstruct); /* Deallocate storage associated with the L and U matrices. */
        lu->options.Fact = SamePattern;
      }
    }
    nz = 0; irow = rstart;   
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
    zCreate_CompRowLoc_Matrix_dist(&lu->A_sup, M, N, nz, m, rstart,
				   lu->val, lu->col, lu->row, SLU_NR_loc, SLU_Z, SLU_GE);
#else
    dCreate_CompRowLoc_Matrix_dist(&lu->A_sup, M, N, nz, m, rstart,
				   lu->val, lu->col, lu->row, SLU_NR_loc, SLU_D, SLU_GE);
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
    pzgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0, 
                   &lu->grid, &lu->LUstruct, berr, &stat, &sinfo);
#else
    pdgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0, 
                   &lu->grid, &lu->LUstruct, berr, &stat, &sinfo);
#endif 
  } else { /* distributed mat input */
#if defined(PETSC_USE_COMPLEX)
    pzgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0, &lu->grid,
	    &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &sinfo);
    if (sinfo) SETERRQ1(PETSC_ERR_LIB,"pzgssvx fails, info: %d\n",sinfo);
#else
    pdgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0, &lu->grid,
	    &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &sinfo);
    if (sinfo) SETERRQ1(PETSC_ERR_LIB,"pdgssvx fails, info: %d\n",sinfo);
#endif
  }

  if (lu->MatInputMode == GLOBAL && size > 1){
    ierr = MatDestroy(A_seq);CHKERRQ(ierr);
  }

  if (lu->options.PrintStat) {
    if (size > 1){
      ierr = MPI_Reduce(&time0,&time_max,1,MPI_DOUBLE,MPI_MAX,0,A->comm);
      ierr = MPI_Reduce(&time0,&time_min,1,MPI_DOUBLE,MPI_MIN,0,A->comm);
      ierr = MPI_Reduce(&time0,&time,1,MPI_DOUBLE,MPI_SUM,0,A->comm);
      time = time/size; /* average time */
      if (!rank)
        ierr = PetscPrintf(PETSC_COMM_SELF, "        Mat conversion(PETSc->SuperLU_DIST) time (max/min/avg): \n \
                              %g / %g / %g\n",time_max,time_min,time);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF, "        Mat conversion(PETSc->SuperLU_DIST) time: \n \
                              %g\n",time0);
    }
    
    PStatPrint(&lu->options, &stat, &lu->grid);  /* Print the statistics. */
  }
  PStatFree(&stat);  
  if (size > 1){
    F_diag = ((Mat_MPIAIJ *)(*F)->data)->A;
    F_diag->assembled = PETSC_TRUE; 
  }
  (*F)->assembled  = PETSC_TRUE;
  lu->options.Fact = FACTORED; /* The factored form of A is supplied. Local option used by this func. only */
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SuperLU_DIST"
PetscErrorCode MatLUFactorSymbolic_SuperLU_DIST(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat               B;
  Mat_SuperLU_DIST  *lu;   
  PetscErrorCode    ierr;
  PetscInt          M=A->rmap.N,N=A->cmap.N,indx;
  PetscMPIInt       size;
  superlu_options_t options;
  PetscTruth        flg;
  const char        *pctype[] = {"MMD_AT_PLUS_A","NATURAL","MMD_ATA"}; 
  const char        *prtype[] = {"LargeDiag","NATURAL"}; 
  const char        *factPattern[] = {"SamePattern","SamePattern_SameRowPerm"};

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = MatCreate(A->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap.n,A->cmap.n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);
  ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactornumeric  = MatLUFactorNumeric_SuperLU_DIST;
  B->ops->solve            = MatSolve_SuperLU_DIST;
  B->factor                = FACTOR_LU;  

  lu = (Mat_SuperLU_DIST*)(B->spptr);

  /*   Set the default input options:
        options.Fact = DOFACT;
        options.Equil = YES;
        options.ColPerm = MMD_AT_PLUS_A;
        options.RowPerm = LargeDiag;
        options.ReplaceTinyPivot = YES;
        options.Trans = NOTRANS;
        options.IterRefine = DOUBLE;
        options.SolveInitialized = NO;
        options.RefineInitialized = NO;
        options.PrintStat = YES;
  */
  set_default_options_dist(&options);

  ierr = MPI_Comm_dup(A->comm,&(lu->comm_superlu));CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  
  ierr = PetscOptionsBegin(A->comm,A->prefix,"SuperLU_Dist Options","Mat");CHKERRQ(ierr);
    lu->npcol = (PetscMPIInt)(PetscSqrtScalar(size)); /* Default num of process columns */
    if (!lu->npcol) lu->npcol = 1;
    lu->nprow = (PetscMPIInt)(size/lu->npcol);        /* Default num of process rows */
    ierr = PetscOptionsInt("-mat_superlu_dist_r","Number rows in processor partition","None",lu->nprow,&lu->nprow,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_superlu_dist_c","Number columns in processor partition","None",lu->npcol,&lu->npcol,PETSC_NULL);CHKERRQ(ierr);
    if (size != lu->nprow * lu->npcol) 
      SETERRQ3(PETSC_ERR_ARG_SIZ,"Number of processes %d must equal to nprow %d * npcol %d",size,lu->nprow,lu->npcol);
  
    lu->MatInputMode = DISTRIBUTED;
    ierr = PetscOptionsInt("-mat_superlu_dist_matinput","Matrix input mode (0: GLOBAL; 1: DISTRIBUTED)","None",lu->MatInputMode,&lu->MatInputMode,PETSC_NULL);CHKERRQ(ierr);
    if(lu->MatInputMode == DISTRIBUTED && size == 1) lu->MatInputMode = GLOBAL;

    ierr = PetscOptionsTruth("-mat_superlu_dist_equil","Equilibrate matrix","None",PETSC_TRUE,&flg,0);CHKERRQ(ierr); 
    if (!flg) {
      options.Equil = NO;
    }

    ierr = PetscOptionsEList("-mat_superlu_dist_rowperm","Row permutation","None",prtype,2,prtype[0],&indx,&flg);CHKERRQ(ierr);
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

    ierr = PetscOptionsEList("-mat_superlu_dist_colperm","Column permutation","None",pctype,3,pctype[0],&indx,&flg);CHKERRQ(ierr);
    if (flg) {
      switch (indx) {
      case 0:
        options.ColPerm = MMD_AT_PLUS_A;
        break;
      case 1:
        options.ColPerm = NATURAL;
        break;
      case 2:
        options.ColPerm = MMD_ATA;
        break;
      }
    }

    ierr = PetscOptionsTruth("-mat_superlu_dist_replacetinypivot","Replace tiny pivots","None",PETSC_TRUE,&flg,0);CHKERRQ(ierr); 
    if (!flg) {
      options.ReplaceTinyPivot = NO;
    }

    lu->FactPattern = SamePattern;
    ierr = PetscOptionsEList("-mat_superlu_dist_fact","Sparsity pattern for repeated matrix factorization","None",factPattern,2,factPattern[0],&indx,&flg);CHKERRQ(ierr);
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
    ierr = PetscOptionsTruth("-mat_superlu_dist_iterrefine","Use iterative refinement","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
    if (flg) {
      options.IterRefine = DOUBLE;    
    }

    if (PetscLogPrintInfo) {
      options.PrintStat = YES; 
    } else {
      options.PrintStat = NO;
    }
    ierr = PetscOptionsTruth("-mat_superlu_dist_statprint","Print factorization information","None",
                              (PetscTruth)options.PrintStat,(PetscTruth*)&options.PrintStat,0);CHKERRQ(ierr); 
  PetscOptionsEnd();

  /* Initialize the SuperLU process grid. */
  superlu_gridinit(lu->comm_superlu, lu->nprow, lu->npcol, &lu->grid);

  /* Initialize ScalePermstruct and LUstruct. */
  ScalePermstructInit(M, N, &lu->ScalePermstruct);
  LUstructInit(M, N, &lu->LUstruct); 

  lu->options             = options; 
  lu->options.Fact        = DOFACT;
  lu->CleanUpSuperLU_Dist = PETSC_TRUE;
  *F = B;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SuperLU_DIST"
PetscErrorCode MatAssemblyEnd_SuperLU_DIST(Mat A,MatAssemblyType mode) {
  PetscErrorCode   ierr;
  Mat_SuperLU_DIST *lu=(Mat_SuperLU_DIST*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);
  lu->MatLUFactorSymbolic  = A->ops->lufactorsymbolic;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_SuperLU_DIST;
  PetscFunctionReturn(0);
}

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
  ierr = PetscViewerASCIIPrintf(viewer,"  Equilibrate matrix %s \n",PetscTruths[options.Equil != NO]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Matrix input mode %d \n",lu->MatInputMode);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Replace tiny pivots %s \n",PetscTruths[options.ReplaceTinyPivot != NO]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Use iterative refinement %s \n",PetscTruths[options.IterRefine == DOUBLE]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Processors in row %d col partition %d \n",lu->nprow,lu->npcol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Row permutation %s \n",(options.RowPerm == NOROWPERM) ? "NATURAL": "LargeDiag");CHKERRQ(ierr);
  if (options.ColPerm == NATURAL) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Column permutation NATURAL\n");CHKERRQ(ierr);
  } else if (options.ColPerm == MMD_AT_PLUS_A) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Column permutation MMD_AT_PLUS_A\n");CHKERRQ(ierr);
  } else if (options.ColPerm == MMD_ATA) {
    ierr = PetscViewerASCIIPrintf(viewer,"  Column permutation MMD_ATA\n");CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Unknown column permutation");
  }
  
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
  PetscTruth        iascii;
  PetscViewerFormat format;
  Mat_SuperLU_DIST  *lu=(Mat_SuperLU_DIST*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatView)(A,viewer);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatFactorInfo_SuperLU_DIST(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_Base_SuperLU_DIST"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_Base_SuperLU_DIST(Mat A,MatType type,MatReuse reuse,Mat *newmat) 
{
  /* This routine is only called to convert to MATSUPERLU_DIST */
  /* from MATSEQAIJ if A has a single process communicator */
  /* or MATMPIAIJ otherwise, so we will ignore 'MatType type'. */
  PetscErrorCode   ierr;
  PetscMPIInt      size;
  MPI_Comm         comm;
  Mat              B=*newmat;
  Mat_SuperLU_DIST *lu;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = PetscNew(Mat_SuperLU_DIST,&lu);CHKERRQ(ierr);

  lu->MatDuplicate         = A->ops->duplicate;
  lu->MatView              = A->ops->view;
  lu->MatAssemblyEnd       = A->ops->assemblyend;
  lu->MatLUFactorSymbolic  = A->ops->lufactorsymbolic;
  lu->MatDestroy           = A->ops->destroy;
  lu->CleanUpSuperLU_Dist  = PETSC_FALSE;

  B->spptr                 = (void*)lu;
  B->ops->duplicate        = MatDuplicate_SuperLU_DIST;
  B->ops->view             = MatView_SuperLU_DIST;
  B->ops->assemblyend      = MatAssemblyEnd_SuperLU_DIST;
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_SuperLU_DIST;
  B->ops->destroy          = MatDestroy_SuperLU_DIST;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_superlu_dist_C",
                                             "MatConvert_Base_SuperLU_DIST",MatConvert_Base_SuperLU_DIST);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_superlu_dist_seqaij_C",
                                             "MatConvert_SuperLU_DIST_Base",MatConvert_SuperLU_DIST_Base);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_mpiaij_superlu_dist_C",
                                             "MatConvert_Base_SuperLU_DIST",MatConvert_Base_SuperLU_DIST);CHKERRQ(ierr);
    ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_superlu_dist_mpiaij_C",
                                             "MatConvert_SuperLU_DIST_Base",MatConvert_SuperLU_DIST_Base);CHKERRQ(ierr);
  }
  ierr = PetscInfo(0,"Using SuperLU_DIST for SeqAIJ LU factorization and solves.\n");CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSUPERLU_DIST);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_SuperLU_DIST"
PetscErrorCode MatDuplicate_SuperLU_DIST(Mat A, MatDuplicateOption op, Mat *M) {
  PetscErrorCode   ierr;
  Mat_SuperLU_DIST *lu=(Mat_SuperLU_DIST *)A->spptr;

  PetscFunctionBegin;
  ierr = (*lu->MatDuplicate)(A,op,M);CHKERRQ(ierr);
  ierr = PetscMemcpy((*M)->spptr,lu,sizeof(Mat_SuperLU_DIST));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATSUPERLU_DIST - MATSUPERLU_DIST = "superlu_dist" - A matrix type providing direct solvers (LU) for parallel matrices 
  via the external package SuperLU_DIST.

  If SuperLU_DIST is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes SuperLU_DIST solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATSUPERLU_DIST).

  This matrix inherits from MATSEQAIJ when constructed with a single process communicator,
  and from MATMPIAIJ otherwise.  As a result, for single process communicators, 
  MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported 
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.  One can also call MatConvert for an inplace
  conversion to or from the MATSEQAIJ or MATMPIAIJ type (depending on the communicator size)
  without data copy.

  Options Database Keys:
+ -mat_type superlu_dist - sets the matrix type to "superlu_dist" during a call to MatSetFromOptions()
. -mat_superlu_dist_r <n> - number of rows in processor partition
. -mat_superlu_dist_c <n> - number of columns in processor partition
. -mat_superlu_dist_matinput <0,1> - matrix input mode; 0=global, 1=distributed
. -mat_superlu_dist_equil - equilibrate the matrix
. -mat_superlu_dist_rowperm <LargeDiag,NATURAL> - row permutation
. -mat_superlu_dist_colperm <MMD_AT_PLUS_A,MMD_ATA,NATURAL> - column permutation
. -mat_superlu_dist_replacetinypivot - replace tiny pivots
. -mat_superlu_dist_fact <SamePattern> (choose one of) SamePattern SamePattern_SameRowPerm
. -mat_superlu_dist_iterrefine - use iterative refinement
- -mat_superlu_dist_statprint - print factorization information

   Level: beginner

.seealso: PCLU
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SuperLU_DIST"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SuperLU_DIST(Mat A) 
{
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of SeqAIJ or MPIAIJ */
  /*   and SuperLU_DIST types */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSUPERLU_DIST);CHKERRQ(ierr);
  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  } else {
    ierr   = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
    /*  A_diag = 0x0 ???  -- do we need it?
    Mat A_diag = ((Mat_MPIAIJ *)A->data)->A;
    ierr = MatConvert_Base_SuperLU_DIST(A_diag,MATSUPERLU_DIST,MAT_REUSE_MATRIX,&A_diag);CHKERRQ(ierr);
    */
  }
  ierr = MatConvert_Base_SuperLU_DIST(A,MATSUPERLU_DIST,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

