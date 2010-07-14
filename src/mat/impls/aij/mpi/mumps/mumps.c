#define PETSCMAT_DLL

/* 
    Provides an interface to the MUMPS sparse solver
*/
#include "../src/mat/impls/aij/seq/aij.h"  /*I  "petscmat.h"  I*/
#include "../src/mat/impls/aij/mpi/mpiaij.h"
#include "../src/mat/impls/sbaij/seq/sbaij.h"
#include "../src/mat/impls/sbaij/mpi/mpisbaij.h"

EXTERN_C_BEGIN 
#if defined(PETSC_USE_COMPLEX)
#include "zmumps_c.h"
#else
#include "dmumps_c.h" 
#endif
EXTERN_C_END 
#define JOB_INIT -1
#define JOB_END -2
/* macros s.t. indices match MUMPS documentation */
#define ICNTL(I) icntl[(I)-1] 
#define CNTL(I) cntl[(I)-1] 
#define INFOG(I) infog[(I)-1]
#define INFO(I) info[(I)-1]
#define RINFOG(I) rinfog[(I)-1]
#define RINFO(I) rinfo[(I)-1]

typedef struct {
#if defined(PETSC_USE_COMPLEX)
  ZMUMPS_STRUC_C id;
#else
  DMUMPS_STRUC_C id;
#endif
  MatStructure   matstruc;
  PetscMPIInt    myid,size;
  PetscInt       *irn,*jcn,sym,nSolve;
  PetscScalar    *val;
  MPI_Comm       comm_mumps;
  VecScatter     scat_rhs, scat_sol;
  PetscTruth     isAIJ,CleanUpMUMPS;
  Vec            b_seq,x_seq;
  PetscErrorCode (*MatDestroy)(Mat);
} Mat_MUMPS;

EXTERN PetscErrorCode MatDuplicate_MUMPS(Mat,MatDuplicateOption,Mat*);

/* convert Petsc mpiaij matrix to triples: row[nz], col[nz], val[nz] */
/*
  input: 
    A       - matrix in mpiaij or mpisbaij (bs=1) format
    shift   - 0: C style output triple; 1: Fortran style output triple.
    valOnly - FALSE: spaces are allocated and values are set for the triple  
              TRUE:  only the values in v array are updated
  output:     
    nnz     - dim of r, c, and v (number of local nonzero entries of A)
    r, c, v - row and col index, matrix values (matrix triples) 
 */
PetscErrorCode MatConvertToTriples(Mat A,int shift,PetscTruth valOnly,int *nnz,int **r, int **c, PetscScalar **v) 
{
  PetscInt       *ai, *aj, *bi, *bj, rstart,nz, *garray;
  PetscErrorCode ierr;
  PetscInt       i,j,jj,jB,irow,m=A->rmap->n,*ajj,*bjj,countA,countB,colA_start,jcol;
  PetscInt       *row,*col;
  PetscScalar    *av, *bv,*val;
  PetscTruth     isAIJ;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&isAIJ);CHKERRQ(ierr);
  if (isAIJ){
    Mat_MPIAIJ    *mat =  (Mat_MPIAIJ*)A->data;
    Mat_SeqAIJ    *aa=(Mat_SeqAIJ*)(mat->A)->data;
    Mat_SeqAIJ    *bb=(Mat_SeqAIJ*)(mat->B)->data;
    nz = aa->nz + bb->nz;
    ai=aa->i; aj=aa->j; bi=bb->i; bj=bb->j; rstart= A->rmap->rstart;
    garray = mat->garray;
    av=aa->a; bv=bb->a;
   
  } else {
    Mat_MPISBAIJ  *mat =  (Mat_MPISBAIJ*)A->data;
    Mat_SeqSBAIJ  *aa=(Mat_SeqSBAIJ*)(mat->A)->data;
    Mat_SeqBAIJ    *bb=(Mat_SeqBAIJ*)(mat->B)->data;
    if (A->rmap->bs > 1) SETERRQ1(PETSC_ERR_SUP," bs=%d is not supported yet\n", A->rmap->bs);
    nz = aa->nz + bb->nz;
    ai=aa->i; aj=aa->j; bi=bb->i; bj=bb->j; rstart= A->rmap->rstart;
    garray = mat->garray;
    av=aa->a; bv=bb->a;
  }

  if (!valOnly){ 
    ierr = PetscMalloc(nz*sizeof(PetscInt) ,&row);CHKERRQ(ierr);
    ierr = PetscMalloc(nz*sizeof(PetscInt),&col);CHKERRQ(ierr);
    ierr = PetscMalloc(nz*sizeof(PetscScalar),&val);CHKERRQ(ierr);
    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v; 
  }
  *nnz = nz; 

  jj = 0; irow = rstart;   
  for ( i=0; i<m; i++ ) {
    ajj = aj + ai[i];                 /* ptr to the beginning of this row */      
    countA = ai[i+1] - ai[i];
    countB = bi[i+1] - bi[i];
    bjj = bj + bi[i];  

    /* get jB, the starting local col index for the 2nd B-part */
    colA_start = rstart + ajj[0]; /* the smallest col index for A */                      
    j=-1;
    do {
      j++;
      if (j == countB) break;
      jcol = garray[bjj[j]];
    } while (jcol < colA_start);
    jB = j;
  
    /* B-part, smaller col index */   
    colA_start = rstart + ajj[0]; /* the smallest col index for A */  
    for (j=0; j<jB; j++){
      jcol = garray[bjj[j]];
      if (!valOnly){ 
        row[jj] = irow + shift; col[jj] = jcol + shift; 

      }
      val[jj++] = *bv++;
    }
    /* A-part */
    for (j=0; j<countA; j++){
      if (!valOnly){
        row[jj] = irow + shift; col[jj] = rstart + ajj[j] + shift; 
      }
      val[jj++] = *av++;
    }
    /* B-part, larger col index */      
    for (j=jB; j<countB; j++){
      if (!valOnly){
        row[jj] = irow + shift; col[jj] = garray[bjj[j]] + shift;
      }
      val[jj++] = *bv++;
    }
    irow++;
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MUMPS"
PetscErrorCode MatDestroy_MUMPS(Mat A)
{
  Mat_MUMPS      *lu=(Mat_MUMPS*)A->spptr; 
  PetscErrorCode ierr;
  PetscTruth     isSeqAIJ,isSeqSBAIJ;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ);CHKERRQ(ierr);

  if (lu->CleanUpMUMPS) {
    /* Terminate instance, deallocate memories */
    if (lu->id.sol_loc){ierr = PetscFree2(lu->id.sol_loc,lu->id.isol_loc);CHKERRQ(ierr);}
    if (lu->scat_rhs){ierr = VecScatterDestroy(lu->scat_rhs);CHKERRQ(ierr);}
    if (lu->b_seq) {ierr = VecDestroy(lu->b_seq);CHKERRQ(ierr);}
    if (lu->nSolve && lu->scat_sol){ierr = VecScatterDestroy(lu->scat_sol);CHKERRQ(ierr);}
    if (lu->nSolve && lu->x_seq){ierr = VecDestroy(lu->x_seq);CHKERRQ(ierr);}
    /* val is reused for SeqAIJ/SBAIJ - but malloced for MPIAIJ/SBAIJ */
    if (!(isSeqAIJ || isSeqSBAIJ) && lu->val){ierr = PetscFree(lu->val);CHKERRQ(ierr);}
    lu->id.job=JOB_END; 
#if defined(PETSC_USE_COMPLEX)
    zmumps_c(&lu->id); 
#else
    dmumps_c(&lu->id); 
#endif
    ierr = PetscFree(lu->irn);CHKERRQ(ierr);
    ierr = PetscFree(lu->jcn);CHKERRQ(ierr);    
    ierr = MPI_Comm_free(&(lu->comm_mumps));CHKERRQ(ierr);
  }
  /* clear composed functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatFactorGetSolverPackage_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatMumpsSetIcntl_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = (lu->MatDestroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MUMPS"
PetscErrorCode MatSolve_MUMPS(Mat A,Vec b,Vec x) 
{
  Mat_MUMPS      *lu=(Mat_MUMPS*)A->spptr; 
  PetscScalar    *array;
  Vec            x_seq;
  IS             is_iden,is_petsc;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin; 
  lu->id.nrhs = 1;
  x_seq = lu->b_seq;
  if (lu->size > 1){
    /* MUMPS only supports centralized rhs. Scatter b into a seqential rhs vector */
    ierr = VecScatterBegin(lu->scat_rhs,b,x_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(lu->scat_rhs,b,x_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (!lu->myid) {ierr = VecGetArray(x_seq,&array);CHKERRQ(ierr);}
  } else {  /* size == 1 */
    ierr = VecCopy(b,x);CHKERRQ(ierr);
    ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  }
  if (!lu->myid) { /* define rhs on the host */
    lu->id.nrhs = 1;
#if defined(PETSC_USE_COMPLEX)
    lu->id.rhs = (mumps_double_complex*)array;
#else
    lu->id.rhs = array;
#endif
  }
  if (lu->size == 1){
    ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
  } else if (!lu->myid){
    ierr = VecRestoreArray(x_seq,&array);CHKERRQ(ierr); 
  }

  if (lu->size > 1){
    /* distributed solution */
    lu->id.ICNTL(21) = 1; 
    if (!lu->nSolve){
      /* Create x_seq=sol_loc for repeated use */ 
      PetscInt    lsol_loc;
      PetscScalar *sol_loc;
      lsol_loc = lu->id.INFO(23); /* length of sol_loc */
      ierr = PetscMalloc2(lsol_loc,PetscScalar,&sol_loc,lsol_loc,PetscInt,&lu->id.isol_loc);CHKERRQ(ierr);
      lu->id.lsol_loc = lsol_loc;
#if defined(PETSC_USE_COMPLEX)
      lu->id.sol_loc  = (mumps_double_complex*)sol_loc;
#else
      lu->id.sol_loc  = sol_loc;
#endif
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,lsol_loc,sol_loc,&lu->x_seq);CHKERRQ(ierr);
    }
  }

  /* solve phase */
  /*-------------*/
  lu->id.job = 3;
#if defined(PETSC_USE_COMPLEX)
  zmumps_c(&lu->id); 
#else
  dmumps_c(&lu->id); 
#endif
  if (lu->id.INFOG(1) < 0) {   
    SETERRQ1(PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d\n",lu->id.INFOG(1));
  }

  if (lu->size > 1) { /* convert mumps distributed solution to petsc mpi x */
    if (!lu->nSolve){ /* create scatter scat_sol */
      ierr = ISCreateStride(PETSC_COMM_SELF,lu->id.lsol_loc,0,1,&is_iden);CHKERRQ(ierr); /* from */
      for (i=0; i<lu->id.lsol_loc; i++){
        lu->id.isol_loc[i] -= 1; /* change Fortran style to C style */
      }
      ierr = ISCreateGeneral(PETSC_COMM_SELF,lu->id.lsol_loc,lu->id.isol_loc,&is_petsc);CHKERRQ(ierr);  /* to */
      ierr = VecScatterCreate(lu->x_seq,is_iden,x,is_petsc,&lu->scat_sol);CHKERRQ(ierr);
      ierr = ISDestroy(is_iden);CHKERRQ(ierr);
      ierr = ISDestroy(is_petsc);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(lu->scat_sol,lu->x_seq,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(lu->scat_sol,lu->x_seq,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  } 
  lu->nSolve++;
  PetscFunctionReturn(0);
}

#if !defined(PETSC_USE_COMPLEX)
/* 
  input:
   F:        numeric factor
  output:
   nneg:     total number of negative pivots
   nzero:    0
   npos:     (global dimension of F) - nneg
*/

#undef __FUNCT__  
#define __FUNCT__ "MatGetInertia_SBAIJMUMPS"
PetscErrorCode MatGetInertia_SBAIJMUMPS(Mat F,int *nneg,int *nzero,int *npos)
{ 
  Mat_MUMPS      *lu =(Mat_MUMPS*)F->spptr; 
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)F)->comm,&size);CHKERRQ(ierr);
  /* MUMPS 4.3.1 calls ScaLAPACK when ICNTL(13)=0 (default), which does not offer the possibility to compute the inertia of a dense matrix. Set ICNTL(13)=1 to skip ScaLAPACK */
  if (size > 1 && lu->id.ICNTL(13) != 1){
    SETERRQ1(PETSC_ERR_ARG_WRONG,"ICNTL(13)=%d. -mat_mumps_icntl_13 must be set as 1 for correct global matrix inertia\n",lu->id.INFOG(13));
  }
  if (nneg){  
    if (!lu->myid){
      *nneg = lu->id.INFOG(12);
    } 
    ierr = MPI_Bcast(nneg,1,MPI_INT,0,lu->comm_mumps);CHKERRQ(ierr);
  }
  if (nzero) *nzero = 0;  
  if (npos)  *npos  = F->rmap->N - (*nneg);
  PetscFunctionReturn(0);
}
#endif /* !defined(PETSC_USE_COMPLEX) */

#undef __FUNCT__   
#define __FUNCT__ "MatFactorNumeric_MUMPS"
PetscErrorCode MatFactorNumeric_MUMPS(Mat F,Mat A,const MatFactorInfo *info) 
{
  Mat_MUMPS      *lu =(Mat_MUMPS*)(F)->spptr; 
  PetscErrorCode ierr;
  PetscInt       rnz,nnz,nz=0,i,M=A->rmap->N,*ai,*aj,icntl;
  PetscTruth     valOnly,flg;
  Mat            F_diag; 
  IS             is_iden;
  Vec            b;
  PetscTruth     isSeqAIJ,isSeqSBAIJ;

  PetscFunctionBegin; 	
  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ);CHKERRQ(ierr);
  if (lu->matstruc == DIFFERENT_NONZERO_PATTERN){ 
    (F)->ops->solve   = MatSolve_MUMPS;

    /* Initialize a MUMPS instance */
    ierr = MPI_Comm_rank(((PetscObject)A)->comm, &lu->myid);
    ierr = MPI_Comm_size(((PetscObject)A)->comm,&lu->size);CHKERRQ(ierr);
    lu->id.job = JOB_INIT; 
    ierr = MPI_Comm_dup(((PetscObject)A)->comm,&(lu->comm_mumps));CHKERRQ(ierr);
    lu->id.comm_fortran = MPI_Comm_c2f(lu->comm_mumps);

    /* Set mumps options */
    ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"MUMPS Options","Mat");CHKERRQ(ierr);
    lu->id.par=1;  /* host participates factorizaton and solve */
    lu->id.sym=lu->sym; 
    if (lu->sym == 2){
      ierr = PetscOptionsInt("-mat_mumps_sym","SYM: (1,2)","None",lu->id.sym,&icntl,&flg);CHKERRQ(ierr); 
      if (flg && icntl == 1) lu->id.sym=icntl;  /* matrix is spd */
    }
#if defined(PETSC_USE_COMPLEX)
    zmumps_c(&lu->id); 
#else
    dmumps_c(&lu->id); 
#endif
 
    if (isSeqAIJ || isSeqSBAIJ){
      lu->id.ICNTL(18) = 0;   /* centralized assembled matrix input */
    } else {
      lu->id.ICNTL(18) = 3;   /* distributed assembled matrix input */
    }

    icntl=-1;
    lu->id.ICNTL(4) = 0;  /* level of printing; overwrite mumps default ICNTL(4)=2 */
    ierr = PetscOptionsInt("-mat_mumps_icntl_4","ICNTL(4): level of printing (0 to 4)","None",lu->id.ICNTL(4),&icntl,&flg);CHKERRQ(ierr);
    if ((flg && icntl > 0) || PetscLogPrintInfo) {
      lu->id.ICNTL(4)=icntl; /* and use mumps default icntl(i), i=1,2,3 */
    } else { /* no output */
      lu->id.ICNTL(1) = 0;  /* error message, default= 6 */
      lu->id.ICNTL(2) = 0;  /* output stream for diagnostic printing, statistics, and warning. default=0 */
      lu->id.ICNTL(3) = 0; /* output stream for global information, default=6 */
    }
    ierr = PetscOptionsInt("-mat_mumps_icntl_6","ICNTL(6): column permutation and/or scaling to get a zero-free diagonal (0 to 7)","None",lu->id.ICNTL(6),&lu->id.ICNTL(6),PETSC_NULL);CHKERRQ(ierr);
    icntl=-1;
    ierr = PetscOptionsInt("-mat_mumps_icntl_7","ICNTL(7): matrix ordering (0 to 7)","None",lu->id.ICNTL(7),&icntl,&flg);CHKERRQ(ierr);
    if (flg) {
      if (icntl== 1){
        SETERRQ(PETSC_ERR_SUP,"pivot order be set by the user in PERM_IN -- not supported by the PETSc/MUMPS interface\n");
      } else {
        lu->id.ICNTL(7) = icntl;
      }
    } 
    ierr = PetscOptionsInt("-mat_mumps_icntl_8","ICNTL(8): scaling strategy (-2 to 7 or 77)","None",lu->id.ICNTL(8),&lu->id.ICNTL(8),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_9","ICNTL(9): A or A^T x=b to be solved. 1: A; otherwise: A^T","None",lu->id.ICNTL(9),&lu->id.ICNTL(9),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_10","ICNTL(10): max num of refinements","None",lu->id.ICNTL(10),&lu->id.ICNTL(10),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_11","ICNTL(11): statistics related to the linear system solved (via -ksp_view)","None",lu->id.ICNTL(11),&lu->id.ICNTL(11),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_12","ICNTL(12): efficiency control: defines the ordering strategy with scaling constraints (0 to 3","None",lu->id.ICNTL(12),&lu->id.ICNTL(12),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_13","ICNTL(13): efficiency control: with or without ScaLAPACK","None",lu->id.ICNTL(13),&lu->id.ICNTL(13),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_14","ICNTL(14): percentage of estimated workspace increase","None",lu->id.ICNTL(14),&lu->id.ICNTL(14),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_19","ICNTL(19): Schur complement","None",lu->id.ICNTL(19),&lu->id.ICNTL(19),PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsInt("-mat_mumps_icntl_22","ICNTL(22): in-core/out-of-core facility (0 or 1)","None",lu->id.ICNTL(22),&lu->id.ICNTL(22),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_23","ICNTL(23): max size of the working memory (MB) that can allocate per processor","None",lu->id.ICNTL(23),&lu->id.ICNTL(23),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_24","ICNTL(24): detection of null pivot rows (0 or 1)","None",lu->id.ICNTL(24),&lu->id.ICNTL(24),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_25","ICNTL(25): computation of a null space basis","None",lu->id.ICNTL(25),&lu->id.ICNTL(25),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_26","ICNTL(26): Schur options for right-hand side or solution vector","None",lu->id.ICNTL(26),&lu->id.ICNTL(26),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_mumps_icntl_27","ICNTL(27): experimental parameter","None",lu->id.ICNTL(27),&lu->id.ICNTL(27),PETSC_NULL);CHKERRQ(ierr);

    ierr = PetscOptionsReal("-mat_mumps_cntl_1","CNTL(1): relative pivoting threshold","None",lu->id.CNTL(1),&lu->id.CNTL(1),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_mumps_cntl_2","CNTL(2): stopping criterion of refinement","None",lu->id.CNTL(2),&lu->id.CNTL(2),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_mumps_cntl_3","CNTL(3): absolute pivoting threshold","None",lu->id.CNTL(3),&lu->id.CNTL(3),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_mumps_cntl_4","CNTL(4): value for static pivoting","None",lu->id.CNTL(4),&lu->id.CNTL(4),PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_mumps_cntl_5","CNTL(5): fixation for null pivots","None",lu->id.CNTL(5),&lu->id.CNTL(5),PETSC_NULL);CHKERRQ(ierr);
    PetscOptionsEnd();
  }

  /* define matrix A */
  switch (lu->id.ICNTL(18)){
  case 0:  /* centralized assembled matrix input (size=1) */
    if (!lu->myid) {
      if (isSeqAIJ){
        Mat_SeqAIJ   *aa = (Mat_SeqAIJ*)A->data;
        nz               = aa->nz;
        ai = aa->i; aj = aa->j; lu->val = aa->a;
      } else if (isSeqSBAIJ) {
        Mat_SeqSBAIJ *aa = (Mat_SeqSBAIJ*)A->data;
        nz                  =  aa->nz;
        ai = aa->i; aj = aa->j; lu->val = aa->a;
      } else {
        SETERRQ(PETSC_ERR_SUP,"No mumps factorization for this matrix type");
      }
      if (lu->matstruc == DIFFERENT_NONZERO_PATTERN){ /* first numeric factorization, get irn and jcn */
        ierr = PetscMalloc(nz*sizeof(PetscInt),&lu->irn);CHKERRQ(ierr);
        ierr = PetscMalloc(nz*sizeof(PetscInt),&lu->jcn);CHKERRQ(ierr); 
        nz = 0;
        for (i=0; i<M; i++){
          rnz = ai[i+1] - ai[i];
          while (rnz--) {  /* Fortran row/col index! */
            lu->irn[nz] = i+1; lu->jcn[nz] = (*aj)+1; aj++; nz++; 
          }
        }
      }
    }
    break;
  case 3:  /* distributed assembled matrix input (size>1) */
    if (lu->matstruc == DIFFERENT_NONZERO_PATTERN){
      valOnly = PETSC_FALSE; 
    } else {
      valOnly = PETSC_TRUE; /* only update mat values, not row and col index */
    }
    ierr = MatConvertToTriples(A,1,valOnly, &nnz, &lu->irn, &lu->jcn, &lu->val);CHKERRQ(ierr);
    break;
  default: SETERRQ(PETSC_ERR_SUP,"Matrix input format is not supported by MUMPS.");
  }

  /* analysis phase */
  /*----------------*/  
  if (lu->matstruc == DIFFERENT_NONZERO_PATTERN){ 
    lu->id.job = 1; 

    lu->id.n = M;
    switch (lu->id.ICNTL(18)){
    case 0:  /* centralized assembled matrix input */
      if (!lu->myid) {
        lu->id.nz =nz; lu->id.irn=lu->irn; lu->id.jcn=lu->jcn;
        if (lu->id.ICNTL(6)>1){
#if defined(PETSC_USE_COMPLEX)
          lu->id.a = (mumps_double_complex*)lu->val; 
#else
          lu->id.a = lu->val; 
#endif
        }
      }
      break;
    case 3:  /* distributed assembled matrix input (size>1) */ 
      lu->id.nz_loc = nnz; 
      lu->id.irn_loc=lu->irn; lu->id.jcn_loc=lu->jcn;
      if (lu->id.ICNTL(6)>1) {
#if defined(PETSC_USE_COMPLEX)
        lu->id.a_loc = (mumps_double_complex*)lu->val;
#else
        lu->id.a_loc = lu->val;
#endif
      }      
      /* MUMPS only supports centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
      if (!lu->myid){
        ierr = VecCreateSeq(PETSC_COMM_SELF,A->cmap->N,&lu->b_seq);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF,A->cmap->N,0,1,&is_iden);CHKERRQ(ierr);
      } else {
        ierr = VecCreateSeq(PETSC_COMM_SELF,0,&lu->b_seq);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&is_iden);CHKERRQ(ierr);
      }
      ierr = VecCreate(((PetscObject)A)->comm,&b);CHKERRQ(ierr);
      ierr = VecSetSizes(b,A->rmap->n,PETSC_DECIDE);CHKERRQ(ierr);
      ierr = VecSetFromOptions(b);CHKERRQ(ierr);

      ierr = VecScatterCreate(b,is_iden,lu->b_seq,is_iden,&lu->scat_rhs);CHKERRQ(ierr);
      ierr = ISDestroy(is_iden);CHKERRQ(ierr);
      ierr = VecDestroy(b);CHKERRQ(ierr);    
      break;
    }    
#if defined(PETSC_USE_COMPLEX)
    zmumps_c(&lu->id); 
#else
    dmumps_c(&lu->id); 
#endif
    if (lu->id.INFOG(1) < 0) { 
      SETERRQ1(PETSC_ERR_LIB,"Error reported by MUMPS in analysis phase: INFOG(1)=%d\n",lu->id.INFOG(1)); 
    }
  }

  /* numerical factorization phase */
  /*-------------------------------*/
  lu->id.job = 2;
  if(!lu->id.ICNTL(18)) { 
    if (!lu->myid) {
#if defined(PETSC_USE_COMPLEX)
      lu->id.a = (mumps_double_complex*)lu->val; 
#else
      lu->id.a = lu->val; 
#endif
    }
  } else {
#if defined(PETSC_USE_COMPLEX)
    lu->id.a_loc = (mumps_double_complex*)lu->val; 
#else
    lu->id.a_loc = lu->val; 
#endif
  }
#if defined(PETSC_USE_COMPLEX)
  zmumps_c(&lu->id); 
#else
  dmumps_c(&lu->id); 
#endif
  if (lu->id.INFOG(1) < 0) {
    if (lu->id.INFO(1) == -13) {
      SETERRQ1(PETSC_ERR_LIB,"Error reported by MUMPS in numerical factorization phase: Cannot allocate required memory %d megabytes\n",lu->id.INFO(2)); 
    } else {
      SETERRQ2(PETSC_ERR_LIB,"Error reported by MUMPS in numerical factorization phase: INFO(1)=%d, INFO(2)=%d\n",lu->id.INFO(1),lu->id.INFO(2)); 
    }
  }

  if (!lu->myid && lu->id.ICNTL(16) > 0){
    SETERRQ1(PETSC_ERR_LIB,"  lu->id.ICNTL(16):=%d\n",lu->id.INFOG(16)); 
  }

  if (lu->size > 1){
    if ((F)->factor == MAT_FACTOR_LU){
      F_diag = ((Mat_MPIAIJ *)(F)->data)->A;
    } else {
      F_diag = ((Mat_MPISBAIJ *)(F)->data)->A;
    }
    F_diag->assembled = PETSC_TRUE;
    if (lu->nSolve){
      ierr = VecScatterDestroy(lu->scat_sol);CHKERRQ(ierr);  
      ierr = PetscFree2(lu->id.sol_loc,lu->id.isol_loc);CHKERRQ(ierr);
      ierr = VecDestroy(lu->x_seq);CHKERRQ(ierr);
    }
  }
  (F)->assembled   = PETSC_TRUE;
  lu->matstruc      = SAME_NONZERO_PATTERN;
  lu->CleanUpMUMPS  = PETSC_TRUE;
  lu->nSolve        = 0;
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_AIJMUMPS"
PetscErrorCode MatLUFactorSymbolic_AIJMUMPS(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_MUMPS      *lu = (Mat_MUMPS*)F->spptr;   

  PetscFunctionBegin;
  lu->sym                  = 0;
  lu->matstruc             = DIFFERENT_NONZERO_PATTERN;
  F->ops->lufactornumeric  = MatFactorNumeric_MUMPS;
  PetscFunctionReturn(0); 
}


/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SBAIJMUMPS"
PetscErrorCode MatCholeskyFactorSymbolic_SBAIJMUMPS(Mat F,Mat A,IS r,const MatFactorInfo *info) 
{
  Mat_MUMPS      *lu = (Mat_MUMPS*)(F)->spptr;   

  PetscFunctionBegin;
  lu->sym                          = 2;
  lu->matstruc                     = DIFFERENT_NONZERO_PATTERN;
  (F)->ops->choleskyfactornumeric = MatFactorNumeric_MUMPS;
#if !defined(PETSC_USE_COMPLEX)
  (F)->ops->getinertia            = MatGetInertia_SBAIJMUMPS;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatFactorInfo_MUMPS"
PetscErrorCode MatFactorInfo_MUMPS(Mat A,PetscViewer viewer) 
{
  Mat_MUMPS      *lu=(Mat_MUMPS*)A->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* check if matrix is mumps type */
  if (A->ops->solve != MatSolve_MUMPS) PetscFunctionReturn(0);

  ierr = PetscViewerASCIIPrintf(viewer,"MUMPS run parameters:\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  SYM (matrix type):                  %d \n",lu->id.sym);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  PAR (host participation):           %d \n",lu->id.par);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(1) (output for error):        %d \n",lu->id.ICNTL(1));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(2) (output of diagnostic msg):%d \n",lu->id.ICNTL(2));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(3) (output for global info):  %d \n",lu->id.ICNTL(3));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(4) (level of printing):       %d \n",lu->id.ICNTL(4));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(5) (input mat struct):        %d \n",lu->id.ICNTL(5));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(6) (matrix prescaling):       %d \n",lu->id.ICNTL(6));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(7) (matrix ordering):         %d \n",lu->id.ICNTL(7));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(8) (scalling strategy):       %d \n",lu->id.ICNTL(8));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(9) (A/A^T x=b is solved):     %d \n",lu->id.ICNTL(9));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(10) (max num of refinements): %d \n",lu->id.ICNTL(10));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(11) (error analysis):         %d \n",lu->id.ICNTL(11));CHKERRQ(ierr);  
  if (!lu->myid && lu->id.ICNTL(11)>0) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(4) (inf norm of input mat):        %g\n",lu->id.RINFOG(4));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(5) (inf norm of solution):         %g\n",lu->id.RINFOG(5));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(6) (inf norm of residual):         %g\n",lu->id.RINFOG(6));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(7),RINFOG(8) (backward error est): %g, %g\n",lu->id.RINFOG(7),lu->id.RINFOG(8));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(9) (error estimate):               %g \n",lu->id.RINFOG(9));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"        RINFOG(10),RINFOG(11)(condition numbers): %g, %g\n",lu->id.RINFOG(10),lu->id.RINFOG(11));CHKERRQ(ierr);
  
  }
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(12) (efficiency control):                         %d \n",lu->id.ICNTL(12));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(13) (efficiency control):                         %d \n",lu->id.ICNTL(13));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(14) (percentage of estimated workspace increase): %d \n",lu->id.ICNTL(14));CHKERRQ(ierr);
  /* ICNTL(15-17) not used */
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(18) (input mat struct):                           %d \n",lu->id.ICNTL(18));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(19) (Shur complement info):                       %d \n",lu->id.ICNTL(19));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(20) (rhs sparse pattern):                         %d \n",lu->id.ICNTL(20));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(21) (solution struct):                            %d \n",lu->id.ICNTL(21));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(22) (in-core/out-of-core facility):               %d \n",lu->id.ICNTL(22));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(23) (max size of memory can be allocated locally):%d \n",lu->id.ICNTL(23));CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(24) (detection of null pivot rows):               %d \n",lu->id.ICNTL(24));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(25) (computation of a null space basis):          %d \n",lu->id.ICNTL(25));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(26) (Schur options for rhs or solution):          %d \n",lu->id.ICNTL(26));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(27) (experimental parameter):                     %d \n",lu->id.ICNTL(27));CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(1) (relative pivoting threshold):      %g \n",lu->id.CNTL(1));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(2) (stopping criterion of refinement): %g \n",lu->id.CNTL(2));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(3) (absolute pivoting threshold):      %g \n",lu->id.CNTL(3));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(4) (value of static pivoting):         %g \n",lu->id.CNTL(4));CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(5) (fixation for null pivots):         %g \n",lu->id.CNTL(5));CHKERRQ(ierr);

  /* infomation local to each processor */
  if (!lu->myid) {ierr = PetscPrintf(PETSC_COMM_SELF, "      RINFO(1) (local estimated flops for the elimination after analysis): \n");CHKERRQ(ierr);}
  ierr = PetscSynchronizedPrintf(((PetscObject)A)->comm,"             [%d] %g \n",lu->myid,lu->id.RINFO(1));CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(((PetscObject)A)->comm);
  if (!lu->myid) {ierr = PetscPrintf(PETSC_COMM_SELF, "      RINFO(2) (local estimated flops for the assembly after factorization): \n");CHKERRQ(ierr);}
  ierr = PetscSynchronizedPrintf(((PetscObject)A)->comm,"             [%d]  %g \n",lu->myid,lu->id.RINFO(2));CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(((PetscObject)A)->comm);
  if (!lu->myid) {ierr = PetscPrintf(PETSC_COMM_SELF, "      RINFO(3) (local estimated flops for the elimination after factorization): \n");CHKERRQ(ierr);}
  ierr = PetscSynchronizedPrintf(((PetscObject)A)->comm,"             [%d]  %g \n",lu->myid,lu->id.RINFO(3));CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(((PetscObject)A)->comm);

  if (!lu->myid) {ierr = PetscPrintf(PETSC_COMM_SELF, "      INFO(15) (estimated size of (in MB) MUMPS internal data for running numerical factorization): \n");CHKERRQ(ierr);}
  ierr = PetscSynchronizedPrintf(((PetscObject)A)->comm,"             [%d] %d \n",lu->myid,lu->id.INFO(15));CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(((PetscObject)A)->comm);

  if (!lu->myid) {ierr = PetscPrintf(PETSC_COMM_SELF, "      INFO(16) (size of (in MB) MUMPS internal data used during numerical factorization): \n");CHKERRQ(ierr);}
  ierr = PetscSynchronizedPrintf(((PetscObject)A)->comm,"             [%d] %d \n",lu->myid,lu->id.INFO(16));CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(((PetscObject)A)->comm);

  if (!lu->myid) {ierr = PetscPrintf(PETSC_COMM_SELF, "      INFO(23) (num of pivots eliminated on this processor after factorization): \n");CHKERRQ(ierr);}
  ierr = PetscSynchronizedPrintf(((PetscObject)A)->comm,"             [%d] %d \n",lu->myid,lu->id.INFO(23));CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(((PetscObject)A)->comm);

  if (!lu->myid){ /* information from the host */
    ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(1) (global estimated flops for the elimination after analysis): %g \n",lu->id.RINFOG(1));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(2) (global estimated flops for the assembly after factorization): %g \n",lu->id.RINFOG(2));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(3) (global estimated flops for the elimination after factorization): %g \n",lu->id.RINFOG(3));CHKERRQ(ierr);

    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(3) (estimated real workspace for factors on all processors after analysis): %d \n",lu->id.INFOG(3));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(4) (estimated integer workspace for factors on all processors after analysis): %d \n",lu->id.INFOG(4));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(5) (estimated maximum front size in the complete tree): %d \n",lu->id.INFOG(5));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(6) (number of nodes in the complete tree): %d \n",lu->id.INFOG(6));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(7) (ordering option effectively uese after analysis): %d \n",lu->id.INFOG(7));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(8) (structural symmetry in percent of the permuted matrix after analysis): %d \n",lu->id.INFOG(8));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(9) (total real/complex workspace to store the matrix factors after factorization): %d \n",lu->id.INFOG(9));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(10) (total integer space store the matrix factors after factorization): %d \n",lu->id.INFOG(10));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(11) (order of largest frontal matrix after factorization): %d \n",lu->id.INFOG(11));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(12) (number of off-diagonal pivots): %d \n",lu->id.INFOG(12));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(13) (number of delayed pivots after factorization): %d \n",lu->id.INFOG(13));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(14) (number of memory compress after factorization): %d \n",lu->id.INFOG(14));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(15) (number of steps of iterative refinement after solution): %d \n",lu->id.INFOG(15));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(16) (estimated size (in MB) of all MUMPS internal data for factorization after analysis: value on the most memory consuming processor): %d \n",lu->id.INFOG(16));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(17) (estimated size of all MUMPS internal data for factorization after analysis: sum over all processors): %d \n",lu->id.INFOG(17));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(18) (size of all MUMPS internal data allocated during factorization: value on the most memory consuming processor): %d \n",lu->id.INFOG(18));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(19) (size of all MUMPS internal data allocated during factorization: sum over all processors): %d \n",lu->id.INFOG(19));CHKERRQ(ierr);
     ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(20) (estimated number of entries in the factors): %d \n",lu->id.INFOG(20));CHKERRQ(ierr);
     ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(21) (size in MB of memory effectively used during factorization - value on the most memory consuming processor): %d \n",lu->id.INFOG(21));CHKERRQ(ierr);
     ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(22) (size in MB of memory effectively used during factorization - sum over all processors): %d \n",lu->id.INFOG(22));CHKERRQ(ierr);
     ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(23) (after analysis: value of ICNTL(6) effectively used): %d \n",lu->id.INFOG(23));CHKERRQ(ierr);
     ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(24) (after analysis: value of ICNTL(12) effectively used): %d \n",lu->id.INFOG(24));CHKERRQ(ierr);
     ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(25) (after factorization: number of pivots modified by static pivoting): %d \n",lu->id.INFOG(25));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_MUMPS"
PetscErrorCode MatView_MUMPS(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscTruth        iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
    ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO){
      ierr = MatFactorInfo_MUMPS(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_MUMPS"
PetscErrorCode MatGetInfo_MUMPS(Mat A,MatInfoType flag,MatInfo *info)
{
    Mat_MUMPS  *lu =(Mat_MUMPS*)A->spptr;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = lu->id.INFOG(20);
  info->nz_used           = lu->id.INFOG(20);
  info->nz_unneeded       = 0.0;
  info->assemblies        = 0.0;
  info->mallocs           = 0.0;
  info->memory            = 0.0;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

/*MC
  MAT_SOLVER_MUMPS -  A matrix type providing direct solvers (LU and Cholesky) for
  distributed and sequential matrices via the external package MUMPS. 

  Works with MATAIJ and MATSBAIJ matrices

  Options Database Keys:
+ -mat_mumps_sym <0,1,2> - 0 the matrix is unsymmetric, 1 symmetric positive definite, 2 symmetric
. -mat_mumps_icntl_4 <0,...,4> - print level
. -mat_mumps_icntl_6 <0,...,7> - matrix prescaling options (see MUMPS User's Guide)
. -mat_mumps_icntl_7 <0,...,7> - matrix orderings (see MUMPS User's Guide)
. -mat_mumps_icntl_9 <1,2> - A or A^T x=b to be solved: 1 denotes A, 2 denotes A^T
. -mat_mumps_icntl_10 <n> - maximum number of iterative refinements
. -mat_mumps_icntl_11 <n> - error analysis, a positive value returns statistics during -ksp_view
. -mat_mumps_icntl_12 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_icntl_13 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_icntl_14 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_icntl_15 <n> - efficiency control (see MUMPS User's Guide)
. -mat_mumps_cntl_1 <delta> - relative pivoting threshold
. -mat_mumps_cntl_2 <tol> - stopping criterion for refinement
- -mat_mumps_cntl_3 <adelta> - absolute pivoting threshold

  Level: beginner

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage

M*/

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_mumps"
PetscErrorCode MatFactorGetSolverPackage_mumps(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MAT_SOLVER_MUMPS;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN 
/*
    The seq and mpi versions of this function are the same 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqaij_mumps"
PetscErrorCode MatGetFactor_seqaij_mumps(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_LU) {
    SETERRQ(PETSC_ERR_SUP,"Cannot use PETSc AIJ matrices with MUMPS Cholesky, use SBAIJ matrix");
  }
  /* Create the factorization matrix */
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMUMPS;
  B->ops->view             = MatView_MUMPS;
  B->ops->getinfo          = MatGetInfo_MUMPS;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_mumps",MatFactorGetSolverPackage_mumps);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMumpsSetIcntl_C","MatMumpsSetIcntl",MatMumpsSetIcntl);CHKERRQ(ierr);
  B->factor                = MAT_FACTOR_LU;  

  ierr = PetscNewLog(B,Mat_MUMPS,&mumps);CHKERRQ(ierr);
  mumps->CleanUpMUMPS              = PETSC_FALSE;
  mumps->isAIJ                     = PETSC_TRUE;
  mumps->scat_rhs                  = PETSC_NULL;
  mumps->scat_sol                  = PETSC_NULL;
  mumps->nSolve                    = 0;
  mumps->MatDestroy                = B->ops->destroy;
  B->ops->destroy                  = MatDestroy_MUMPS;
  B->spptr                         = (void*)mumps;

  *F = B;
  PetscFunctionReturn(0); 
}
EXTERN_C_END

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_mpiaij_mumps"
PetscErrorCode MatGetFactor_mpiaij_mumps(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_LU) {
    SETERRQ(PETSC_ERR_SUP,"Cannot use PETSc AIJ matrices with MUMPS Cholesky, use SBAIJ matrix");
  }
  /* Create the factorization matrix */
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMUMPS;
  B->ops->view             = MatView_MUMPS;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_mumps",MatFactorGetSolverPackage_mumps);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMumpsSetIcntl_C","MatMumpsSetIcntl",MatMumpsSetIcntl);CHKERRQ(ierr);
  B->factor                = MAT_FACTOR_LU;  

  ierr = PetscNewLog(B,Mat_MUMPS,&mumps);CHKERRQ(ierr);
  mumps->CleanUpMUMPS              = PETSC_FALSE;
  mumps->isAIJ                     = PETSC_TRUE;
  mumps->scat_rhs                  = PETSC_NULL;
  mumps->scat_sol                  = PETSC_NULL;
  mumps->nSolve                    = 0;
  mumps->MatDestroy                = B->ops->destroy;
  B->ops->destroy                  = MatDestroy_MUMPS;
  B->spptr                         = (void*)mumps;

  *F = B;
  PetscFunctionReturn(0); 
}
EXTERN_C_END

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqsbaij_mumps"
PetscErrorCode MatGetFactor_seqsbaij_mumps(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) {
    SETERRQ(PETSC_ERR_SUP,"Cannot use PETSc SBAIJ matrices with MUMPS LU, use AIJ matrix");
  }
  /* Create the factorization matrix */ 
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(B,1,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(B,1,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SBAIJMUMPS;
  B->ops->view                   = MatView_MUMPS;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_mumps",MatFactorGetSolverPackage_mumps);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMumpsSetIcntl_C","MatMumpsSetIcntl",MatMumpsSetIcntl);CHKERRQ(ierr);
  B->factor                      = MAT_FACTOR_CHOLESKY;

  ierr = PetscNewLog(B,Mat_MUMPS,&mumps);CHKERRQ(ierr);
  mumps->CleanUpMUMPS              = PETSC_FALSE;
  mumps->isAIJ                     = PETSC_TRUE;
  mumps->scat_rhs                  = PETSC_NULL;
  mumps->scat_sol                  = PETSC_NULL;
  mumps->nSolve                    = 0;
  mumps->MatDestroy                = B->ops->destroy;
  B->ops->destroy                  = MatDestroy_MUMPS;
  B->spptr                         = (void*)mumps;

  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_mpisbaij_mumps"
PetscErrorCode MatGetFactor_mpisbaij_mumps(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) {
    SETERRQ(PETSC_ERR_SUP,"Cannot use PETSc SBAIJ matrices with MUMPS LU, use AIJ matrix");
  }
  /* Create the factorization matrix */ 
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(B,1,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(B,1,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SBAIJMUMPS;
  B->ops->view                   = MatView_MUMPS;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_mumps",MatFactorGetSolverPackage_mumps);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMumpsSetIcntl_C","MatMumpsSetIcntl",MatMumpsSetIcntl);CHKERRQ(ierr);
  B->factor                      = MAT_FACTOR_CHOLESKY;

  ierr = PetscNewLog(B,Mat_MUMPS,&mumps);CHKERRQ(ierr);
  mumps->CleanUpMUMPS              = PETSC_FALSE;
  mumps->isAIJ                     = PETSC_TRUE;
  mumps->scat_rhs                  = PETSC_NULL;
  mumps->scat_sol                  = PETSC_NULL;
  mumps->nSolve                    = 0;
  mumps->MatDestroy                = B->ops->destroy;
  B->ops->destroy                  = MatDestroy_MUMPS;
  B->spptr                         = (void*)mumps;

  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------------------------*/
/*@
  MatMumpsSetIcntl - Set MUMPS parameter ICNTL()

   Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
.  idx - index of MUMPS parameter array ICNTL()
-  icntl - value of MUMPS ICNTL(imumps)

  Options Database:
.   -mat_mumps_icntl_<idx> <icntl>

   Level: beginner

   References: MUMPS Users' Guide 

.seealso: MatGetFactor()
@*/
#undef __FUNCT__   
#define __FUNCT__ "MatMumpsSetIcntl"
PetscErrorCode MatMumpsSetIcntl(Mat F,PetscInt idx,PetscInt icntl)
{
  Mat_MUMPS      *lu =(Mat_MUMPS*)(F)->spptr; 

  PetscFunctionBegin; 
  lu->id.ICNTL(idx) = icntl;
  PetscFunctionReturn(0);
}

