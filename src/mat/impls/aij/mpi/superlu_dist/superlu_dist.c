/*$Id: superlu_DIST.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
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
  MatStructure            flg;
  MPI_Comm                comm_superlu;
#if defined(PETSC_USE_COMPLEX)
  doublecomplex           *val;
#else
  double                  *val;
#endif

  /* A few function pointers for inheritance */
  int (*MatView)(Mat,PetscViewer);
  int (*MatAssemblyEnd)(Mat,MatAssemblyType);
  int (*MatDestroy)(Mat);

  /* Flag to clean up (non-global) SuperLU objects during Destroy */
  PetscTruth CleanUpSuperLUDist;
} Mat_MPIAIJ_SuperLU_DIST;

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_SuperLU_DIST"
int MatDestroy_MPIAIJ_SuperLU_DIST(Mat A)
{
  Mat_MPIAIJ              *a  = (Mat_MPIAIJ*)A->data; 
  Mat_MPIAIJ_SuperLU_DIST *lu = (Mat_MPIAIJ_SuperLU_DIST*)A->spptr; 
  int                     ierr, size=a->size,(*destroy)(Mat);
    
  PetscFunctionBegin;
  if (lu->CleanUpSuperLUDist) {
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
    Destroy_LU(A->N, &lu->grid, &lu->LUstruct);
    ScalePermstructFree(&lu->ScalePermstruct);
    LUstructFree(&lu->LUstruct);

    /* Release the SuperLU_DIST process grid. */
    superlu_gridexit(&lu->grid);
    
    ierr = MPI_Comm_free(&(lu->comm_superlu));CHKERRQ(ierr);
  }
  destroy = lu->MatDestroy;
  ierr = PetscFree(lu);CHKERRQ(ierr); 
  ierr = (*destroy)(A);CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_MPIAIJ_Spooles_DIST"
int MatView_MPIAIJ_SuperLU_DIST(Mat A,PetscViewer viewer)
{
  int                     ierr;
  PetscTruth              isascii;
  PetscViewerFormat       format;
  Mat_MPIAIJ_SuperLU_DIST *lu=(Mat_MPIAIJ_SuperLU_DIST*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatView)(A,viewer);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      ierr = MatMPIAIJFactorInfo_SuperLu(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_MPIAIJ_SuperLU_DIST"
int MatAssemblyEnd_MPIAIJ_SuperLU_DIST(Mat A,MatAssemblyType mode) {
  int                     ierr;
  Mat_MPIAIJ_SuperLU_DIST *lu=(Mat_MPIAIJ_SuperLU_DIST*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);
  ierr = MatUseSuperLU_DIST_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MPIAIJ_SuperLU_DIST"
int MatSolve_MPIAIJ_SuperLU_DIST(Mat A,Vec b_mpi,Vec x)
{
  Mat_MPIAIJ              *aa = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJ_SuperLU_DIST *lu = (Mat_MPIAIJ_SuperLU_DIST*)A->spptr;
  int                     ierr, size=aa->size;
  int                     m=A->M, N=A->N; 
  SuperLUStat_t           stat;  
  double                  berr[1];
  PetscScalar             *bptr;  
  int                     info, nrhs=1;
  Vec                     x_seq;
  IS                      iden;
  VecScatter              scat;
  PetscLogDouble          time0,time,time_min,time_max; 
  
  PetscFunctionBegin;
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
 
  lu->options.Fact = FACTORED; /* The factored form of A is supplied. Local option used by this func. only.*/

  PStatInit(&stat);        /* Initialize the statistics variables. */
  if (lu->StatPrint) {
    ierr = MPI_Barrier(A->comm);CHKERRQ(ierr); /* to be removed */
    ierr = PetscGetTime(&time0);CHKERRQ(ierr);  /* to be removed */
  }
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
    pzgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, (doublecomplex*)bptr, A->M, nrhs, &lu->grid,
	    &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &info);
    if (info) SETERRQ1(1,"pzgssvx fails, info: %d\n",info);
#else
    pdgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, bptr, A->M, nrhs, &lu->grid,
	    &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &info);
    if (info) SETERRQ1(1,"pdgssvx fails, info: %d\n",info);
#endif
  }
  if (lu->StatPrint) {
    ierr = PetscGetTime(&time);CHKERRQ(ierr);  /* to be removed */
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
  if (lu->StatPrint) {
    time0 = time - time0;
    ierr = MPI_Reduce(&time0,&time_max,1,MPI_DOUBLE,MPI_MAX,0,A->comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&time0,&time_min,1,MPI_DOUBLE,MPI_MIN,0,A->comm);CHKERRQ(ierr);
    ierr = MPI_Reduce(&time0,&time,1,MPI_DOUBLE,MPI_SUM,0,A->comm);CHKERRQ(ierr);
    time = time/size; /* average time */
    ierr = PetscPrintf(A->comm, "  Time for superlu_dist solve (max/min/avg): %g / %g / %g\n\n",time_max,time_min,time);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatLUFactorNumeric_MPIAIJ_SuperLU_DIST"
int MatLUFactorNumeric_MPIAIJ_SuperLU_DIST(Mat A,Mat *F)
{
  Mat_MPIAIJ              *fac = (Mat_MPIAIJ*)(*F)->data,*mat;
  Mat                     *tseq,A_seq = PETSC_NULL;
  Mat_SeqAIJ              *aa,*bb;
  Mat_MPIAIJ_SuperLU_DIST *lu = (Mat_MPIAIJ_SuperLU_DIST*)(*F)->spptr;
  int                     M=A->M,N=A->N,info,ierr,size=fac->size,i,*ai,*aj,*bi,*bj,nz,rstart,*garray,
                          m=A->m, irow,colA_start,j,jcol,jB,countA,countB,*bjj,*ajj;
  SuperLUStat_t           stat;
  double                  *berr=0;
  IS                      isrow;
  PetscLogDouble          time0[2],time[2],time_min[2],time_max[2]; 
#if defined(PETSC_USE_COMPLEX)
  doublecomplex           *av, *bv; 
#else
  double                  *av, *bv; 
#endif

  PetscFunctionBegin;
  if (lu->StatPrint) {
    ierr = MPI_Barrier(A->comm);CHKERRQ(ierr);
    ierr = PetscGetTime(&time0[0]);CHKERRQ(ierr);  
  }

  if (lu->MatInputMode == GLOBAL) { /* global mat input */
    if (size > 1) { /* convert mpi A to seq mat A */
      ierr = ISCreateStride(PETSC_COMM_SELF,M,0,1,&isrow); CHKERRQ(ierr);  
      ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&tseq); CHKERRQ(ierr);
      ierr = ISDestroy(isrow);CHKERRQ(ierr);
   
      A_seq = *tseq;
      ierr = PetscFree(tseq);CHKERRQ(ierr);
      aa =  (Mat_SeqAIJ*)A_seq->data;
    } else {
      aa =  (Mat_SeqAIJ*)A->data;
    }

    /* Allocate storage, then convert Petsc NR matrix to SuperLU_DIST NC */
    if (lu->flg == DIFFERENT_NONZERO_PATTERN) {/* first numeric factorization */
#if defined(PETSC_USE_COMPLEX)
      zallocateA_dist(N, aa->nz, &lu->val, &lu->col, &lu->row);
#else
      dallocateA_dist(N, aa->nz, &lu->val, &lu->col, &lu->row);
#endif
    } else { /* successive numeric factorization, sparsity pattern is reused. */
      Destroy_CompCol_Matrix_dist(&lu->A_sup); 
      Destroy_LU(N, &lu->grid, &lu->LUstruct); 
      lu->options.Fact = SamePattern; 
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
    mat =  (Mat_MPIAIJ*)A->data;  
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
    rstart = mat->rstart;
    nz     = aa->nz + bb->nz;
    garray = mat->garray;
    rstart = mat->rstart;

    if (lu->flg == DIFFERENT_NONZERO_PATTERN) {/* first numeric factorization */ 
#if defined(PETSC_USE_COMPLEX)
      zallocateA_dist(m, nz, &lu->val, &lu->col, &lu->row);
#else
      dallocateA_dist(m, nz, &lu->val, &lu->col, &lu->row);
#endif
    } else { /* successive numeric factorization, sparsity pattern and perm_c are reused. */
      /* Destroy_CompRowLoc_Matrix_dist(&lu->A_sup);  */ /* crash! */
      Destroy_LU(N, &lu->grid, &lu->LUstruct); 
      lu->options.Fact = SamePattern; 
    }
    nz = 0; jB = 0; irow = mat->rstart;   
    for ( i=0; i<m; i++ ) {
      lu->row[i] = nz;
      countA = ai[i+1] - ai[i];
      countB = bi[i+1] - bi[i];
      ajj = aj + ai[i];  /* ptr to the beginning of this row */
      bjj = bj + bi[i];  
 
      /* B part, smaller col index */   
      colA_start = mat->rstart + ajj[0]; /* the smallest global col index of A */  
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
        lu->col[nz] = mat->rstart + ajj[j]; 
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
  if (lu->StatPrint) {
    ierr = PetscGetTime(&time[0]);CHKERRQ(ierr);  
    time0[0] = time[0] - time0[0];
  }

  /* Factor the matrix. */
  PStatInit(&stat);   /* Initialize the statistics variables. */

  if (lu->StatPrint) {
    ierr = MPI_Barrier(A->comm);CHKERRQ(ierr);
    ierr = PetscGetTime(&time0[1]);CHKERRQ(ierr);  
  }

  if (lu->MatInputMode == GLOBAL) { /* global mat input */
#if defined(PETSC_USE_COMPLEX)
    pzgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0, 
                   &lu->grid, &lu->LUstruct, berr, &stat, &info);
#else
    pdgssvx_ABglobal(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0, 
                   &lu->grid, &lu->LUstruct, berr, &stat, &info);
#endif 
  } else { /* distributed mat input */
#if defined(PETSC_USE_COMPLEX)
    pzgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0, &lu->grid,
	    &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &info);
    if (info) SETERRQ1(1,"pzgssvx fails, info: %d\n",info);
#else
    pdgssvx(&lu->options, &lu->A_sup, &lu->ScalePermstruct, 0, M, 0, &lu->grid,
	    &lu->LUstruct, &lu->SOLVEstruct, berr, &stat, &info);
    if (info) SETERRQ1(1,"pdgssvx fails, info: %d\n",info);
#endif
  }
  if (lu->StatPrint) {
    ierr = PetscGetTime(&time[1]);CHKERRQ(ierr);  /* to be removed */
    time0[1] = time[1] - time0[1];
    if (lu->StatPrint) PStatPrint(&lu->options, &stat, &lu->grid);  /* Print the statistics. */
  }
  PStatFree(&stat);  

  if (lu->MatInputMode == GLOBAL && size > 1){
    ierr = MatDestroy(A_seq);CHKERRQ(ierr);
  }

  if (lu->StatPrint) {
    ierr = MPI_Reduce(time0,time_max,2,MPI_DOUBLE,MPI_MAX,0,A->comm);
    ierr = MPI_Reduce(time0,time_min,2,MPI_DOUBLE,MPI_MIN,0,A->comm);
    ierr = MPI_Reduce(time0,time,2,MPI_DOUBLE,MPI_SUM,0,A->comm);
    for (i=0; i<2; i++) time[i] = time[i]/size; /* average time */
    ierr = PetscPrintf(A->comm, "  Time for mat conversion (max/min/avg):    %g / %g / %g\n",time_max[0],time_min[0],time[0]);
    ierr = PetscPrintf(A->comm, "  Time for superlu_dist fact (max/min/avg): %g / %g / %g\n\n",time_max[1],time_min[1],time[1]);
  }
  (*F)->assembled = PETSC_TRUE;
  lu->flg         = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIAIJ_SuperLU_DIST"
int MatLUFactorSymbolic_MPIAIJ_SuperLU_DIST(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat                     B;
  Mat_MPIAIJ_SuperLU_DIST *lu;   
  int                     ierr,M=A->M,N=A->N,size;
  superlu_options_t       options;
  char                    buff[32];
  PetscTruth              flg;
  char                    *ptype[] = {"MMD_AT_PLUS_A","NATURAL","MMD_ATA","COLAMD"}; 
  char                    *prtype[] = {"LargeDiag","NATURAL"}; 
  PetscFunctionBegin;
	
  ierr = PetscNew(Mat_MPIAIJ_SuperLU_DIST,&lu);CHKERRQ(ierr); 

  /* Create the factorization matrix */
  ierr = MatCreate(A->comm,A->m,A->n,M,N,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSUPERLUDIST);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);
  ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactornumeric  = MatLUFactorNumeric_MPIAIJ_SuperLU_DIST;
  B->ops->solve            = MatSolve_MPIAIJ_SuperLU_DIST;
  B->ops->destroy          = MatDestroy_MPIAIJ_SuperLU_DIST;  
  B->factor                = FACTOR_LU;  
  B->spptr                 = (void*)lu;

  /* Set the input options */
  set_default_options(&options);
  lu->MatInputMode = GLOBAL;
  ierr = MPI_Comm_dup(A->comm,&(lu->comm_superlu));CHKERRQ(ierr);

  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  lu->nprow = size/2;               /* Default process rows.      */
  if (lu->nprow == 0) lu->nprow = 1;
  lu->npcol = size/lu->nprow;           /* Default process columns.   */

  ierr = PetscOptionsBegin(A->comm,A->prefix,"SuperLU_Dist Options","Mat");CHKERRQ(ierr);
  
    ierr = PetscOptionsInt("-mat_aij_superlu_dist_r","Number rows in processor partition","None",lu->nprow,&lu->nprow,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_aij_superlu_dist_c","Number columns in processor partition","None",lu->npcol,&lu->npcol,PETSC_NULL);CHKERRQ(ierr);
    if (size != lu->nprow * lu->npcol) SETERRQ(1,"Number of processes should be equal to nprow*npcol");
  
    ierr = PetscOptionsInt("-mat_aij_superlu_dist_matinput","Matrix input mode (0: GLOBAL; 1: DISTRIBUTED)","None",lu->MatInputMode,&lu->MatInputMode,PETSC_NULL);CHKERRQ(ierr);
    if(lu->MatInputMode == DISTRIBUTED && size == 1) lu->MatInputMode = GLOBAL;

    ierr = PetscOptionsLogical("-mat_aij_superlu_dist_equil","Equilibrate matrix","None",PETSC_TRUE,&flg,0);CHKERRQ(ierr); 
    if (!flg) {
      options.Equil = NO;
    }

    ierr = PetscOptionsEList("-mat_aij_superlu_dist_rowperm","Row permutation","None",prtype,2,prtype[0],buff,32,&flg);CHKERRQ(ierr);
    while (flg) {
      ierr = PetscStrcmp(buff,"LargeDiag",&flg);CHKERRQ(ierr);
      if (flg) {
        options.RowPerm = LargeDiag;
        break;
      }
      ierr = PetscStrcmp(buff,"NATURAL",&flg);CHKERRQ(ierr);
      if (flg) {
        options.RowPerm = NOROWPERM;
        break;
      }
      SETERRQ1(1,"Unknown row permutation %s",buff);
    }

    ierr = PetscOptionsEList("-mat_aij_superlu_dist_colperm","Column permutation","None",ptype,4,ptype[0],buff,32,&flg);CHKERRQ(ierr);
    while (flg) {
      ierr = PetscStrcmp(buff,"MMD_AT_PLUS_A",&flg);CHKERRQ(ierr);
      if (flg) {
        options.ColPerm = MMD_AT_PLUS_A;
        break;
      }
      ierr = PetscStrcmp(buff,"NATURAL",&flg);CHKERRQ(ierr);
      if (flg) {
        options.ColPerm = NATURAL;
        break;
      }
      ierr = PetscStrcmp(buff,"MMD_ATA",&flg);CHKERRQ(ierr);
      if (flg) {
        options.ColPerm = MMD_ATA;
        break;
      }
      ierr = PetscStrcmp(buff,"COLAMD",&flg);CHKERRQ(ierr);
      if (flg) {
        options.ColPerm = COLAMD;
        break;
      }
      SETERRQ1(1,"Unknown column permutation %s",buff);
    }

    ierr = PetscOptionsLogical("-mat_aij_superlu_dist_replacetinypivot","Replace tiny pivots","None",PETSC_TRUE,&flg,0);CHKERRQ(ierr); 
    if (!flg) {
      options.ReplaceTinyPivot = NO;
    }

    options.IterRefine = NOREFINE;
    ierr = PetscOptionsLogical("-mat_aij_superlu_dist_iterrefine","Use iterative refinement","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
    if (flg) {
      options.IterRefine = DOUBLE;    
    }

    if (PetscLogPrintInfo) {
      lu->StatPrint = (int)PETSC_TRUE; 
    } else {
      lu->StatPrint = (int)PETSC_FALSE; 
    }
    ierr = PetscOptionsLogical("-mat_aij_superlu_dist_statprint","Print factorization information","None",
                              (PetscTruth)lu->StatPrint,(PetscTruth*)&lu->StatPrint,0);CHKERRQ(ierr); 
  PetscOptionsEnd();

  /* Initialize the SuperLU process grid. */
  superlu_gridinit(lu->comm_superlu, lu->nprow, lu->npcol, &lu->grid);

  /* Initialize ScalePermstruct and LUstruct. */
  ScalePermstructInit(M, N, &lu->ScalePermstruct);
  LUstructInit(M, N, &lu->LUstruct); 

  lu->options            = options;
  lu->flg                = DIFFERENT_NONZERO_PATTERN;
  lu->CleanUpSuperLUDist = PETSC_TRUE;
  *F = B;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseSuperLU_DIST_MPIAIJ"
int MatUseSuperLU_DIST_MPIAIJ(Mat A)
{
  PetscFunctionBegin;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_MPIAIJ_SuperLU_DIST;
  A->ops->lufactornumeric  = MatLUFactorNumeric_MPIAIJ_SuperLU_DIST;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMPIAIJFactorInfo_SuperLu"
int MatMPIAIJFactorInfo_SuperLu(Mat A,PetscViewer viewer)
{
  Mat_MPIAIJ_SuperLU_DIST *lu= (Mat_MPIAIJ_SuperLU_DIST*)A->spptr;
  superlu_options_t       options;
  int                     ierr;
  char                    *colperm;

  PetscFunctionBegin;
  /* check if matrix is superlu_dist type */
  if (A->ops->solve != MatSolve_MPIAIJ_SuperLU_DIST) PetscFunctionReturn(0);

  options = lu->options;
  ierr = PetscViewerASCIIPrintf(viewer,"SuperLU_DIST run parameters:\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Equilibrate matrix %s \n",(options.Equil != NO) ? "true": "false");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Replace tiny pivots %s \n",(options.ReplaceTinyPivot != NO) ? "true": "false");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Use iterative refinement %s \n",(options.IterRefine == DOUBLE) ? "true": "false");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Processors in row %d col partition %d \n",lu->nprow,lu->npcol);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Row permutation %s \n",(options.RowPerm == NOROWPERM) ? "NATURAL": "LargeDiag");CHKERRQ(ierr);
  if (options.ColPerm == NATURAL) {
    colperm = "NATURAL";
  } else if (options.ColPerm == MMD_AT_PLUS_A) {
    colperm = "MMD_AT_PLUS_A";
  } else if (options.ColPerm == MMD_ATA) {
    colperm = "MMD_ATA";
  } else if (options.ColPerm == COLAMD) {
    colperm = "COLAMD";
  } else {
    SETERRQ(1,"Unknown column permutation");
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  Column permutation %s \n",colperm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_MPIAIJ_SuperLU_DIST"
int MatCreate_MPIAIJ_SuperLU_DIST(Mat A) {
  int                     ierr,size;
  MPI_Comm                comm;
  Mat_MPIAIJ_SuperLU_DIST *lu;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);CHKERRQ(ierr);
  if (size == 1) {
    ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(A,MATMPIAIJ);CHKERRQ(ierr);
  }
  ierr = MatUseSuperLU_SeqAIJ(A);CHKERRQ(ierr);

  ierr                   = PetscNew(Mat_MPIAIJ_SuperLU_DIST,&lu);CHKERRQ(ierr);
  lu->MatView            = A->ops->view;
  lu->MatAssemblyEnd     = A->ops->assemblyend;
  lu->MatDestroy         = A->ops->destroy;
  lu->CleanUpSuperLUDist = PETSC_FALSE;
  A->spptr               = (void*)lu;
  A->ops->view           = MatView_MPIAIJ_SuperLU_DIST;
  A->ops->assemblyend    = MatAssemblyEnd_MPIAIJ_SuperLU_DIST;
  A->ops->destroy        = MatDestroy_MPIAIJ_SuperLU_DIST;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatLoad_MPIAIJ_SuperLU_DIST"
int MatLoad_MPIAIJ_SuperLU_DIST(PetscViewer viewer,MatType type,Mat *A) {
  int      ierr,size,(*r)(PetscViewer,MatType,Mat*);
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscFListFind(comm,MatLoadList,MATSEQAIJ,(void(**)(void))&r);CHKERRQ(ierr);
  } else {
    ierr = PetscFListFind(comm,MatLoadList,MATMPIAIJ,(void(**)(void))&r);CHKERRQ(ierr);
  }
  ierr = (*r)(viewer,type,A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
