/*$Id: superlu_DIST.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
        Provides an interface to the SuperLU_DIST sparse solver
*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"

#if defined(PETSC_HAVE_SUPERLUDIST) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)

EXTERN_C_BEGIN
#include "superlu_ddefs.h"
EXTERN_C_END

typedef struct {
  int_t                   nprow,npcol;
  gridinfo_t              grid;
  superlu_options_t       options;
  SuperMatrix             A_sup;
  ScalePermstruct_t       ScalePermstruct;
  LUstruct_t              LUstruct;
  int                     StatPrint;
} Mat_MPIAIJ_SuperLU_DIST;

extern int MatDestroy_MPIAIJ(Mat);
extern int MatDestroy_SeqAIJ(Mat);

#if !defined(PETSC_HAVE_SUPERLU)
/* SuperLU function: Convert a row compressed storage into a column compressed storage. */
#undef __FUNCT__  
#define __FUNCT__ "dCompRow_to_CompCol"
void dCompRow_to_CompCol(int m, int n, int nnz,
                    double *a, int *colind, int *rowptr,
                    double **at, int **rowind, int **colptr)
{
    register int i, j, col, relpos;
    int *marker;

    /* Allocate storage for another copy of the matrix. */
    *at = (double *) doubleMalloc_dist(nnz);
    *rowind = (int *) intMalloc_dist(nnz);
    *colptr = (int *) intMalloc_dist(n+1);
    marker = (int *) intCalloc_dist(n);

    /* Get counts of each column of A, and set up column pointers */
    for (i = 0; i < m; ++i)
        for (j = rowptr[i]; j < rowptr[i+1]; ++j) ++marker[colind[j]];
    (*colptr)[0] = 0;
    for (j = 0; j < n; ++j) {
        (*colptr)[j+1] = (*colptr)[j] + marker[j];
        marker[j] = (*colptr)[j];
    }

    /* Transfer the matrix into the compressed column storage. */
    for (i = 0; i < m; ++i) {
        for (j = rowptr[i]; j < rowptr[i+1]; ++j) {
            col = colind[j];
            relpos = marker[col];
            (*rowind)[relpos] = i;
            (*at)[relpos] = a[j];
            ++marker[col];
        }
    }

    SUPERLU_FREE(marker);
}
#else
EXTERN_C_BEGIN
extern void dCompRow_to_CompCol(int,int,int,double*,int*,int*,double**,int**,int**);
EXTERN_C_END
#endif /* PETSC_HAVE_SUPERLU*/

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_SuperLU_DIST"
int MatDestroy_MPIAIJ_SuperLU_DIST(Mat A)
{
  Mat_MPIAIJ         *a  = (Mat_MPIAIJ*)A->data; 
  Mat_MPIAIJ_SuperLU_DIST *lu = (Mat_MPIAIJ_SuperLU_DIST*)a->spptr; 
  int                ierr, size=a->size;
    
  PetscFunctionBegin;
  /* Deallocate SuperLU_DIST storage */
  Destroy_CompCol_Matrix_dist(&lu->A_sup);
  Destroy_LU(A->N, &lu->grid, &lu->LUstruct);
  ScalePermstructFree(&lu->ScalePermstruct);
  LUstructFree(&lu->LUstruct);

  /* Release the SuperLU_DIST process grid. */
  superlu_gridexit(&lu->grid);

  ierr = PetscFree(lu);CHKERRQ(ierr); 
  
  if (size == 1){
    ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  } 
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MPIAIJ_SuperLU_DIST"
int MatSolve_MPIAIJ_SuperLU_DIST(Mat A,Vec b_mpi,Vec x)
{
  Mat_MPIAIJ              *aa = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJ_SuperLU_DIST *lu = (Mat_MPIAIJ_SuperLU_DIST*)aa->spptr;
  int                     ierr, size=aa->size;
  int_t                   m=A->M, N=A->N; 
  superlu_options_t       options=lu->options;
  SuperLUStat_t           stat;
  double                  berr[1],*bptr;  
  int                     info, nrhs=1;
  Vec                     x_seq;
  IS                      iden;
  VecScatter              scat;
  PetscLogDouble          time0,time,time_min,time_max; 
  
  PetscFunctionBegin;
  if (size > 1) {  /* convert mpi vector b to seq vector x_seq */
    ierr = VecCreateSeq(PETSC_COMM_SELF,N,&x_seq);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,N,0,1,&iden);CHKERRQ(ierr);
    ierr = VecScatterCreate(b_mpi,iden,x_seq,iden,&scat);CHKERRQ(ierr);
    ierr = ISDestroy(iden);CHKERRQ(ierr);

    ierr = VecScatterBegin(b_mpi,x_seq,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
    ierr = VecScatterEnd(b_mpi,x_seq,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
    ierr = VecGetArray(x_seq,&bptr);CHKERRQ(ierr); 
  } else {
    ierr = VecCopy(b_mpi,x);CHKERRQ(ierr);
    ierr = VecGetArray(x,&bptr);CHKERRQ(ierr); 
  }
 
  options.Fact = FACTORED; /* The factored form of A is supplied. Local option used by this func. only.*/

  PStatInit(&stat);        /* Initialize the statistics variables. */
  if (lu->StatPrint) {
    ierr = MPI_Barrier(A->comm);CHKERRQ(ierr); /* to be removed */
    ierr = PetscGetTime(&time0);CHKERRQ(ierr);  /* to be removed */
  }
  pdgssvx_ABglobal(&options, &lu->A_sup, &lu->ScalePermstruct, bptr, m, nrhs, 
                   &lu->grid, &lu->LUstruct, berr, &stat, &info);
  if (lu->StatPrint) {
    ierr = PetscGetTime(&time);CHKERRQ(ierr);  /* to be removed */
     PStatPrint(&stat, &lu->grid);     /* Print the statistics. */
  }
  PStatFree(&stat);
 
  if (size > 1) {    /* convert seq x to mpi x */
    ierr = VecRestoreArray(x_seq,&bptr);CHKERRQ(ierr);
    ierr = VecScatterBegin(x_seq,x,INSERT_VALUES,SCATTER_REVERSE,scat);CHKERRQ(ierr);
    ierr = VecScatterEnd(x_seq,x,INSERT_VALUES,SCATTER_REVERSE,scat);CHKERRQ(ierr);
    ierr = VecScatterDestroy(scat);CHKERRQ(ierr);
    ierr = VecDestroy(x_seq);CHKERRQ(ierr);
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
  Mat_MPIAIJ              *fac = (Mat_MPIAIJ*)(*F)->data;
  Mat                     *tseq,A_seq = PETSC_NULL;
  Mat_SeqAIJ              *aa;
  Mat_MPIAIJ_SuperLU_DIST *lu = (Mat_MPIAIJ_SuperLU_DIST*)fac->spptr;
  int                     M=A->M,N=A->N,info,ierr,size=fac->size,i;
  SuperLUStat_t           stat;
  double                  *berr=0, *bptr=0;
  int_t                   *asub, *xa;
  double                  *a; 
  SuperMatrix             A_sup;
  IS                      isrow;
  PetscLogDouble          time0[2],time[2],time_min[2],time_max[2]; 

  PetscFunctionBegin;
  if (lu->StatPrint) {
    ierr = MPI_Barrier(A->comm);CHKERRQ(ierr);
    ierr = PetscGetTime(&time0[0]);CHKERRQ(ierr);  
  }

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

  /* Allocate storage for compressed column representation. */
  dallocateA_dist(N, aa->nz, &a, &asub, &xa);
  
  /* Convert Petsc NR matrix storage to SuperLU_DIST NC storage */
  dCompRow_to_CompCol(M,N,aa->nz,aa->a,aa->j,aa->i,&a, &asub, &xa);

  if (lu->StatPrint) {
    ierr = PetscGetTime(&time[0]);CHKERRQ(ierr);  /* to be removed */
    time0[0] = time[0] - time0[0];
  }

  /* Create compressed column matrix A_sup. */
  dCreate_CompCol_Matrix_dist(&A_sup, M, N, aa->nz, a, asub, xa, NC, D, GE);  

  /* Factor the matrix. */
  PStatInit(&stat);                /* Initialize the statistics variables. */

  if (lu->StatPrint) {
    ierr = MPI_Barrier(A->comm);CHKERRQ(ierr);
    ierr = PetscGetTime(&time0[1]);CHKERRQ(ierr);  
  }
  pdgssvx_ABglobal(&lu->options, &A_sup, &lu->ScalePermstruct, bptr, M, 0, 
                   &lu->grid, &lu->LUstruct, berr, &stat, &info);  
  if (lu->StatPrint) {
    ierr = PetscGetTime(&time[1]);CHKERRQ(ierr);  /* to be removed */
    time0[1] = time[1] - time0[1];
    if (lu->StatPrint) PStatPrint(&stat, &lu->grid);        /* Print the statistics. */
  }
  PStatFree(&stat);  

  lu->A_sup        = A_sup;
  lu->options.Fact = SamePattern; /* Sparsity pattern of A and perm_c can be reused. */
  if (size > 1){
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
  (*F)->assembled             = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIAIJ_SuperLU_DIST"
int MatLUFactorSymbolic_MPIAIJ_SuperLU_DIST(Mat A,IS r,IS c,MatLUInfo *info,Mat *F)
{
  Mat_MPIAIJ              *fac;
  Mat_MPIAIJ_SuperLU_DIST *lu;   /* ptr to Mat_MPIAIJ_SuperLU_DIST */
  int                     ierr,M=A->M,N=A->N,size;
  gridinfo_t              grid; 
  superlu_options_t       options;
  ScalePermstruct_t       ScalePermstruct;
  LUstruct_t              LUstruct;
  char                    buff[32];
  PetscTruth              flg;
  char                    *ptype[] = {"MMD_AT_PLUS_A","NATURAL","MMD_ATA","COLAMD"}; 
  char                    *prtype[] = {"LargeDiag","NATURAL"}; 
  PetscFunctionBegin;
	
  /* Create the factorization matrix F */ 
  ierr = MatCreateMPIAIJ(A->comm,PETSC_DECIDE,PETSC_DECIDE,M,N,0,PETSC_NULL,0,PETSC_NULL,F);CHKERRQ(ierr);

  (*F)->ops->lufactornumeric  = MatLUFactorNumeric_MPIAIJ_SuperLU_DIST;
  (*F)->ops->solve            = MatSolve_MPIAIJ_SuperLU_DIST;
  (*F)->ops->destroy          = MatDestroy_MPIAIJ_SuperLU_DIST;  
  (*F)->factor                = FACTOR_LU;  
  fac                         = (Mat_MPIAIJ*)(*F)->data; 

  ierr                        = PetscNew(Mat_MPIAIJ_SuperLU_DIST,&lu);CHKERRQ(ierr); 
  fac->spptr                  = (void*)lu;

  /* Set the input options */
  set_default_options(&options);

  ierr = MPI_Comm_size(A->comm,&size);CHKERRQ(ierr);
  lu->nprow = size/2;               /* Default process rows.      */
  if (lu->nprow == 0) lu->nprow = 1;
  lu->npcol = size/lu->nprow;           /* Default process columns.   */

  ierr = PetscOptionsBegin(A->comm,A->prefix,"SuperLU_Dist Options","Mat");CHKERRQ(ierr);
  
    ierr = PetscOptionsInt("-mat_aij_superlu_dist_r","Number rows in processor partition","None",lu->nprow,&lu->nprow,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_aij_superlu_dist_c","Number columns in processor partition","None",lu->npcol,&lu->npcol,PETSC_NULL);CHKERRQ(ierr);
    if (size != lu->nprow * lu->npcol) SETERRQ(1,"Number of processes should be equal to nprow*npcol");
  
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
  superlu_gridinit(A->comm, lu->nprow, lu->npcol, &grid);

  /* Initialize ScalePermstruct and LUstruct. */
  ScalePermstructInit(M, N, &ScalePermstruct);
  LUstructInit(M, N, &LUstruct);

  lu->ScalePermstruct = ScalePermstruct;
  lu->LUstruct        = LUstruct;
  lu->options         = options;
  lu->grid            = grid;
  fac->size           = size;

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

int MatMPIAIJFactorInfo_SuperLu(Mat A,PetscViewer viewer)
{
  Mat_MPIAIJ              *fac = (Mat_MPIAIJ*)(A)->data;
  Mat_MPIAIJ_SuperLU_DIST *lu;
  superlu_options_t       options;
  int                     ierr;
  char                    *colperm;

  PetscFunctionBegin;
  /* check if matrix is superlu_dist type */
  if (A->ops->solve != MatSolve_MPIAIJ_SuperLU_DIST) PetscFunctionReturn(0);

  lu      = (Mat_MPIAIJ_SuperLU_DIST*)fac->spptr;
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

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseSuperLU_DIST_MPIAIJ"
int MatUseSuperLU_DIST_MPIAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif


