
/* 
    Provides an interface to the MUMPS sparse solver
*/

#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I  "petscmat.h"  I*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>

EXTERN_C_BEGIN 
#if defined(PETSC_USE_COMPLEX)
#include <zmumps_c.h>
#else
#include <dmumps_c.h> 
#endif
EXTERN_C_END 
#define JOB_INIT -1
#define JOB_FACTSYMBOLIC 1
#define JOB_FACTNUMERIC 2
#define JOB_SOLVE 3
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
  PetscInt       *irn,*jcn,nz,sym,nSolve;
  PetscScalar    *val;
  MPI_Comm       comm_mumps;
  VecScatter     scat_rhs, scat_sol;
  PetscBool      isAIJ,CleanUpMUMPS;
  Vec            b_seq,x_seq;
  PetscErrorCode (*Destroy)(Mat);
  PetscErrorCode (*ConvertToTriples)(Mat, int, MatReuse, int*, int**, int**, PetscScalar**);
} Mat_MUMPS;

extern PetscErrorCode MatDuplicate_MUMPS(Mat,MatDuplicateOption,Mat*);


/* MatConvertToTriples_A_B */
/*convert Petsc matrix to triples: row[nz], col[nz], val[nz] */
/*
  input: 
    A       - matrix in aij,baij or sbaij (bs=1) format
    shift   - 0: C style output triple; 1: Fortran style output triple.
    reuse   - MAT_INITIAL_MATRIX: spaces are allocated and values are set for the triple  
              MAT_REUSE_MATRIX:   only the values in v array are updated
  output:     
    nnz     - dim of r, c, and v (number of local nonzero entries of A)
    r, c, v - row and col index, matrix values (matrix triples) 
 */

#undef __FUNCT__
#define __FUNCT__ "MatConvertToTriples_seqaij_seqaij"
PetscErrorCode MatConvertToTriples_seqaij_seqaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v) 
{
  const PetscInt   *ai,*aj,*ajj,M=A->rmap->n;
  PetscInt         nz,rnz,i,j;
  PetscErrorCode   ierr;
  PetscInt         *row,*col;
  Mat_SeqAIJ       *aa=(Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  *v=aa->a;
  if (reuse == MAT_INITIAL_MATRIX){
    nz = aa->nz; ai = aa->i; aj = aa->j;
    *nnz = nz;
    ierr = PetscMalloc(2*nz*sizeof(PetscInt), &row);CHKERRQ(ierr);
    col  = row + nz;

    nz = 0;
    for(i=0; i<M; i++) {
      rnz = ai[i+1] - ai[i];
      ajj = aj + ai[i];
      for(j=0; j<rnz; j++) {
	row[nz] = i+shift; col[nz++] = ajj[j] + shift;
      }
    } 
    *r = row; *c = col;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertToTriples_seqbaij_seqaij"
PetscErrorCode MatConvertToTriples_seqbaij_seqaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v) 
{
  Mat_SeqBAIJ        *aa=(Mat_SeqBAIJ*)A->data;
  const PetscInt     *ai,*aj,*ajj,bs=A->rmap->bs,bs2=aa->bs2,M=A->rmap->N/bs;
  PetscInt           nz,idx=0,rnz,i,j,k,m;
  PetscErrorCode     ierr;
  PetscInt           *row,*col;

  PetscFunctionBegin;
  *v = aa->a;
  if (reuse == MAT_INITIAL_MATRIX){
    ai = aa->i; aj = aa->j;
    nz = bs2*aa->nz;
    *nnz = nz;
    ierr = PetscMalloc(2*nz*sizeof(PetscInt), &row);CHKERRQ(ierr);
    col  = row + nz;

    for(i=0; i<M; i++) {
      ajj = aj + ai[i];
      rnz = ai[i+1] - ai[i];
      for(k=0; k<rnz; k++) {
	for(j=0; j<bs; j++) {
	  for(m=0; m<bs; m++) {
	    row[idx]     = i*bs + m + shift;
	    col[idx++]   = bs*(ajj[k]) + j + shift;
	  }
	}
      }
    }
    *r = row; *c = col;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertToTriples_seqsbaij_seqsbaij"
PetscErrorCode MatConvertToTriples_seqsbaij_seqsbaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v) 
{
  const PetscInt   *ai, *aj,*ajj,M=A->rmap->n;
  PetscInt         nz,rnz,i,j;
  PetscErrorCode   ierr;
  PetscInt         *row,*col;
  Mat_SeqSBAIJ     *aa=(Mat_SeqSBAIJ*)A->data;

  PetscFunctionBegin;
  *v = aa->a;
  if (reuse == MAT_INITIAL_MATRIX){ 
    nz = aa->nz;ai=aa->i; aj=aa->j;*v=aa->a;
    *nnz = nz;
    ierr = PetscMalloc(2*nz*sizeof(PetscInt), &row);CHKERRQ(ierr);
    col  = row + nz;

    nz = 0;
    for(i=0; i<M; i++) {
      rnz = ai[i+1] - ai[i];
      ajj = aj + ai[i];
      for(j=0; j<rnz; j++) {
	row[nz] = i+shift; col[nz++] = ajj[j] + shift;
      }
    } 
    *r = row; *c = col;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertToTriples_seqaij_seqsbaij"
PetscErrorCode MatConvertToTriples_seqaij_seqsbaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v) 
{
  const PetscInt     *ai,*aj,*ajj,*adiag,M=A->rmap->n;
  PetscInt           nz,rnz,i,j;
  const PetscScalar  *av,*v1;
  PetscScalar        *val;
  PetscErrorCode     ierr;
  PetscInt           *row,*col;
  Mat_SeqSBAIJ       *aa=(Mat_SeqSBAIJ*)A->data;

  PetscFunctionBegin;
  ai=aa->i; aj=aa->j;av=aa->a;
  adiag=aa->diag;
  if (reuse == MAT_INITIAL_MATRIX){
    nz = M + (aa->nz-M)/2;
    *nnz = nz;
    ierr = PetscMalloc((2*nz*sizeof(PetscInt)+nz*sizeof(PetscScalar)), &row);CHKERRQ(ierr);
    col  = row + nz;
    val  = (PetscScalar*)(col + nz);

    nz = 0;
    for(i=0; i<M; i++) {
      rnz = ai[i+1] - adiag[i];
      ajj  = aj + adiag[i];
      v1   = av + adiag[i];
      for(j=0; j<rnz; j++) {
	row[nz] = i+shift; col[nz] = ajj[j] + shift; val[nz++] = v1[j];
      }
    } 
    *r = row; *c = col; *v = val;
  } else {
    nz = 0; val = *v;
    for(i=0; i <M; i++) {
      rnz = ai[i+1] - adiag[i];
      ajj = aj + adiag[i];
      v1  = av + adiag[i];
      for(j=0; j<rnz; j++) {
	val[nz++] = v1[j];
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertToTriples_mpisbaij_mpisbaij"
PetscErrorCode MatConvertToTriples_mpisbaij_mpisbaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v) 
{
  const PetscInt     *ai, *aj, *bi, *bj,*garray,m=A->rmap->n,*ajj,*bjj;
  PetscErrorCode     ierr;
  PetscInt           rstart,nz,i,j,jj,irow,countA,countB;
  PetscInt           *row,*col;
  const PetscScalar  *av, *bv,*v1,*v2;
  PetscScalar        *val;
  Mat_MPISBAIJ       *mat =  (Mat_MPISBAIJ*)A->data;
  Mat_SeqSBAIJ       *aa=(Mat_SeqSBAIJ*)(mat->A)->data;
  Mat_SeqBAIJ        *bb=(Mat_SeqBAIJ*)(mat->B)->data;

  PetscFunctionBegin;
  ai=aa->i; aj=aa->j; bi=bb->i; bj=bb->j; rstart= A->rmap->rstart;
  garray = mat->garray;
  av=aa->a; bv=bb->a;  

  if (reuse == MAT_INITIAL_MATRIX){
    nz = aa->nz + bb->nz;
    *nnz = nz;
    ierr = PetscMalloc((2*nz*sizeof(PetscInt)+nz*sizeof(PetscScalar)), &row);CHKERRQ(ierr);
    col  = row + nz;
    val  = (PetscScalar*)(col + nz);

    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v; 
  }

  jj = 0; irow = rstart;   
  for ( i=0; i<m; i++ ) {
    ajj    = aj + ai[i];                 /* ptr to the beginning of this row */      
    countA = ai[i+1] - ai[i];
    countB = bi[i+1] - bi[i];
    bjj    = bj + bi[i];
    v1     = av + ai[i];
    v2     = bv + bi[i];

    /* A-part */
    for (j=0; j<countA; j++){
      if (reuse == MAT_INITIAL_MATRIX) {
        row[jj] = irow + shift; col[jj] = rstart + ajj[j] + shift; 
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for(j=0; j < countB; j++){
      if (reuse == MAT_INITIAL_MATRIX) {
	row[jj] = irow + shift; col[jj] = garray[bjj[j]] + shift;
      }
      val[jj++] = v2[j];
    }
    irow++;
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertToTriples_mpiaij_mpiaij"
PetscErrorCode MatConvertToTriples_mpiaij_mpiaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v) 
{
  const PetscInt     *ai, *aj, *bi, *bj,*garray,m=A->rmap->n,*ajj,*bjj;
  PetscErrorCode     ierr;
  PetscInt           rstart,nz,i,j,jj,irow,countA,countB;
  PetscInt           *row,*col;
  const PetscScalar  *av, *bv,*v1,*v2;
  PetscScalar        *val;
  Mat_MPIAIJ         *mat =  (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ         *aa=(Mat_SeqAIJ*)(mat->A)->data;
  Mat_SeqAIJ         *bb=(Mat_SeqAIJ*)(mat->B)->data;

  PetscFunctionBegin;
  ai=aa->i; aj=aa->j; bi=bb->i; bj=bb->j; rstart= A->rmap->rstart;
  garray = mat->garray;
  av=aa->a; bv=bb->a;  

  if (reuse == MAT_INITIAL_MATRIX){
    nz = aa->nz + bb->nz;
    *nnz = nz;
    ierr = PetscMalloc((2*nz*sizeof(PetscInt)+nz*sizeof(PetscScalar)), &row);CHKERRQ(ierr);
    col  = row + nz;
    val  = (PetscScalar*)(col + nz);

    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v; 
  }

  jj = 0; irow = rstart;   
  for ( i=0; i<m; i++ ) {
    ajj    = aj + ai[i];                 /* ptr to the beginning of this row */      
    countA = ai[i+1] - ai[i];
    countB = bi[i+1] - bi[i];
    bjj    = bj + bi[i];
    v1     = av + ai[i];
    v2     = bv + bi[i];

    /* A-part */
    for (j=0; j<countA; j++){
      if (reuse == MAT_INITIAL_MATRIX){
        row[jj] = irow + shift; col[jj] = rstart + ajj[j] + shift;
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for(j=0; j < countB; j++){
      if (reuse == MAT_INITIAL_MATRIX){
	row[jj] = irow + shift; col[jj] = garray[bjj[j]] + shift;
      }
      val[jj++] = v2[j];
    }
    irow++;
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertToTriples_mpibaij_mpiaij"
PetscErrorCode MatConvertToTriples_mpibaij_mpiaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v) 
{
  Mat_MPIBAIJ        *mat =  (Mat_MPIBAIJ*)A->data;
  Mat_SeqBAIJ        *aa=(Mat_SeqBAIJ*)(mat->A)->data;
  Mat_SeqBAIJ        *bb=(Mat_SeqBAIJ*)(mat->B)->data;
  const PetscInt     *ai = aa->i, *bi = bb->i, *aj = aa->j, *bj = bb->j,*ajj, *bjj;
  const PetscInt     *garray = mat->garray,mbs=mat->mbs,rstart=A->rmap->rstart;
  const PetscInt     bs = A->rmap->bs,bs2=mat->bs2;
  PetscErrorCode     ierr;
  PetscInt           nz,i,j,k,n,jj,irow,countA,countB,idx;
  PetscInt           *row,*col;
  const PetscScalar  *av=aa->a, *bv=bb->a,*v1,*v2;
  PetscScalar        *val;

  PetscFunctionBegin;

  if (reuse == MAT_INITIAL_MATRIX) {
    nz = bs2*(aa->nz + bb->nz);
    *nnz = nz;
    ierr = PetscMalloc((2*nz*sizeof(PetscInt)+nz*sizeof(PetscScalar)), &row);CHKERRQ(ierr);
    col  = row + nz;
    val  = (PetscScalar*)(col + nz);

    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v; 
  }

  jj = 0; irow = rstart;   
  for ( i=0; i<mbs; i++ ) {       
    countA = ai[i+1] - ai[i];
    countB = bi[i+1] - bi[i];
    ajj    = aj + ai[i];
    bjj    = bj + bi[i];
    v1     = av + bs2*ai[i];
    v2     = bv + bs2*bi[i];

    idx = 0;
    /* A-part */
    for (k=0; k<countA; k++){
      for (j=0; j<bs; j++) {
	for (n=0; n<bs; n++) {
	  if (reuse == MAT_INITIAL_MATRIX){
	    row[jj] = irow + n + shift; 
	    col[jj] = rstart + bs*ajj[k] + j + shift;
	  }
	  val[jj++] = v1[idx++];
	}
      }
    }

    idx = 0;
    /* B-part */
    for(k=0; k<countB; k++){
      for (j=0; j<bs; j++) {
	for (n=0; n<bs; n++) {
	  if (reuse == MAT_INITIAL_MATRIX){
	    row[jj] = irow + n + shift; 
	    col[jj] = bs*garray[bjj[k]] + j + shift;
	  }
	  val[jj++] = v2[idx++];
	}
      }
    }
    irow += bs;
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConvertToTriples_mpiaij_mpisbaij"
PetscErrorCode MatConvertToTriples_mpiaij_mpisbaij(Mat A,int shift,MatReuse reuse,int *nnz,int **r, int **c, PetscScalar **v) 
{
  const PetscInt     *ai, *aj,*adiag, *bi, *bj,*garray,m=A->rmap->n,*ajj,*bjj;
  PetscErrorCode     ierr;
  PetscInt           rstart,nz,nza,nzb,i,j,jj,irow,countA,countB;
  PetscInt           *row,*col;
  const PetscScalar  *av, *bv,*v1,*v2;
  PetscScalar        *val;
  Mat_MPIAIJ         *mat =  (Mat_MPIAIJ*)A->data;
  Mat_SeqAIJ         *aa=(Mat_SeqAIJ*)(mat->A)->data;
  Mat_SeqAIJ         *bb=(Mat_SeqAIJ*)(mat->B)->data;

  PetscFunctionBegin;
  ai=aa->i; aj=aa->j; adiag=aa->diag;
  bi=bb->i; bj=bb->j; garray = mat->garray;
  av=aa->a; bv=bb->a;
  rstart = A->rmap->rstart;

  if (reuse == MAT_INITIAL_MATRIX) {
    nza = 0;    /* num of upper triangular entries in mat->A, including diagonals */
    nzb = 0;    /* num of upper triangular entries in mat->B */ 
    for(i=0; i<m; i++){
      nza    += (ai[i+1] - adiag[i]); 
      countB  = bi[i+1] - bi[i]; 
      bjj     = bj + bi[i];
      for (j=0; j<countB; j++){
        if (garray[bjj[j]] > rstart) nzb++;
      }
    }
    
    nz = nza + nzb; /* total nz of upper triangular part of mat */
    *nnz = nz;
    ierr = PetscMalloc((2*nz*sizeof(PetscInt)+nz*sizeof(PetscScalar)), &row);CHKERRQ(ierr);
    col  = row + nz;
    val  = (PetscScalar*)(col + nz);

    *r = row; *c = col; *v = val;
  } else {
    row = *r; col = *c; val = *v; 
  }

  jj = 0; irow = rstart;   
  for ( i=0; i<m; i++ ) {
    ajj    = aj + adiag[i];                 /* ptr to the beginning of the diagonal of this row */
    v1     = av + adiag[i];
    countA = ai[i+1] - adiag[i];
    countB = bi[i+1] - bi[i];
    bjj    = bj + bi[i];
    v2     = bv + bi[i];

     /* A-part */
    for (j=0; j<countA; j++){
      if (reuse == MAT_INITIAL_MATRIX) {
        row[jj] = irow + shift; col[jj] = rstart + ajj[j] + shift; 
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for(j=0; j < countB; j++){
      if (garray[bjj[j]] > rstart) {
	if (reuse == MAT_INITIAL_MATRIX) {
	  row[jj] = irow + shift; col[jj] = garray[bjj[j]] + shift;
	}
	val[jj++] = v2[j];
      }
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

  PetscFunctionBegin;
  if (lu && lu->CleanUpMUMPS) {
    /* Terminate instance, deallocate memories */
    ierr = PetscFree2(lu->id.sol_loc,lu->id.isol_loc);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&lu->scat_rhs);CHKERRQ(ierr);
    ierr = VecDestroy(&lu->b_seq);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&lu->scat_sol);CHKERRQ(ierr);
    ierr = VecDestroy(&lu->x_seq);CHKERRQ(ierr);
    ierr=PetscFree(lu->id.perm_in);CHKERRQ(ierr);
    ierr = PetscFree(lu->irn);CHKERRQ(ierr);
    lu->id.job=JOB_END;
#if defined(PETSC_USE_COMPLEX)
    zmumps_c(&lu->id);
#else
    dmumps_c(&lu->id);
#endif
    ierr = MPI_Comm_free(&(lu->comm_mumps));CHKERRQ(ierr);
  }
  if (lu && lu->Destroy) {
    ierr = (lu->Destroy)(A);CHKERRQ(ierr);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatFactorGetSolverPackage_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatMumpsSetIcntl_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MUMPS"
PetscErrorCode MatSolve_MUMPS(Mat A,Vec b,Vec x) 
{
  Mat_MUMPS      *lu=(Mat_MUMPS*)A->spptr; 
  PetscScalar    *array;
  Vec            b_seq;
  IS             is_iden,is_petsc;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin; 
  lu->id.nrhs = 1;
  b_seq = lu->b_seq;
  if (lu->size > 1){
    /* MUMPS only supports centralized rhs. Scatter b into a seqential rhs vector */
    ierr = VecScatterBegin(lu->scat_rhs,b,b_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(lu->scat_rhs,b,b_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (!lu->myid) {ierr = VecGetArray(b_seq,&array);CHKERRQ(ierr);}
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

  /* solve phase */
  /*-------------*/
  lu->id.job = JOB_SOLVE;
#if defined(PETSC_USE_COMPLEX)
  zmumps_c(&lu->id); 
#else
  dmumps_c(&lu->id); 
#endif
  if (lu->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in solve phase: INFOG(1)=%d\n",lu->id.INFOG(1));

  if (lu->size > 1) { /* convert mumps distributed solution to petsc mpi x */
    if (!lu->nSolve){ /* create scatter scat_sol */
      ierr = ISCreateStride(PETSC_COMM_SELF,lu->id.lsol_loc,0,1,&is_iden);CHKERRQ(ierr); /* from */
      for (i=0; i<lu->id.lsol_loc; i++){
        lu->id.isol_loc[i] -= 1; /* change Fortran style to C style */
      }
      ierr = ISCreateGeneral(PETSC_COMM_SELF,lu->id.lsol_loc,lu->id.isol_loc,PETSC_COPY_VALUES,&is_petsc);CHKERRQ(ierr);  /* to */
      ierr = VecScatterCreate(lu->x_seq,is_iden,x,is_petsc,&lu->scat_sol);CHKERRQ(ierr);
      ierr = ISDestroy(&is_iden);CHKERRQ(ierr);
      ierr = ISDestroy(&is_petsc);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(lu->scat_sol,lu->x_seq,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(lu->scat_sol,lu->x_seq,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  } 
  lu->nSolve++;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveTranspose_MUMPS"
PetscErrorCode MatSolveTranspose_MUMPS(Mat A,Vec b,Vec x) 
{
  Mat_MUMPS      *lu=(Mat_MUMPS*)A->spptr; 
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  lu->id.ICNTL(9) = 0;
  ierr = MatSolve_MUMPS(A,b,x);CHKERRQ(ierr);
  lu->id.ICNTL(9) = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatSolve_MUMPS"
PetscErrorCode MatMatSolve_MUMPS(Mat A,Mat B,Mat X) 
{
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscTypeCompareAny((PetscObject)B,&flg,MATSEQDENSE,MATMPIDENSE,PETSC_NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  ierr = PetscTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,PETSC_NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatMatSolve_MUMPS() is not implemented yet");
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
  if (size > 1 && lu->id.ICNTL(13) != 1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"ICNTL(13)=%d. -mat_mumps_icntl_13 must be set as 1 for correct global matrix inertia\n",lu->id.INFOG(13));
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
  Mat_MUMPS       *lu =(Mat_MUMPS*)(F)->spptr;
  PetscErrorCode  ierr;
  Mat             F_diag; 
  PetscBool       isMPIAIJ;

  PetscFunctionBegin;
  ierr = (*lu->ConvertToTriples)(A, 1, MAT_REUSE_MATRIX, &lu->nz, &lu->irn, &lu->jcn, &lu->val);CHKERRQ(ierr);

  /* numerical factorization phase */
  /*-------------------------------*/
  lu->id.job = JOB_FACTNUMERIC;
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
    if (lu->id.INFO(1) == -13) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in numerical factorization phase: Cannot allocate required memory %d megabytes\n",lu->id.INFO(2)); 
    else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in numerical factorization phase: INFO(1)=%d, INFO(2)=%d\n",lu->id.INFO(1),lu->id.INFO(2)); 
  }
  if (!lu->myid && lu->id.ICNTL(16) > 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"  lu->id.ICNTL(16):=%d\n",lu->id.INFOG(16)); 

  if (lu->size > 1){
    ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&isMPIAIJ);CHKERRQ(ierr);
    if(isMPIAIJ) {
      F_diag = ((Mat_MPIAIJ *)(F)->data)->A;
    } else {
      F_diag = ((Mat_MPISBAIJ *)(F)->data)->A;
    }
    F_diag->assembled = PETSC_TRUE;
    if (lu->nSolve){
      ierr = VecScatterDestroy(&lu->scat_sol);CHKERRQ(ierr);  
      ierr = PetscFree2(lu->id.sol_loc,lu->id.isol_loc);CHKERRQ(ierr);
      ierr = VecDestroy(&lu->x_seq);CHKERRQ(ierr);
    }
  }
  (F)->assembled   = PETSC_TRUE; 
  lu->matstruc     = SAME_NONZERO_PATTERN;
  lu->CleanUpMUMPS = PETSC_TRUE;
  lu->nSolve       = 0;
 
  if (lu->size > 1){
    /* distributed solution */
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
      ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,lsol_loc,sol_loc,&lu->x_seq);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/* Sets MUMPS options from the options database */
#undef __FUNCT__
#define __FUNCT__ "PetscSetMUMPSFromOptions"
PetscErrorCode PetscSetMUMPSFromOptions(Mat F, Mat A)
{
  Mat_MUMPS        *mumps = (Mat_MUMPS*)F->spptr;
  PetscErrorCode   ierr;
  PetscInt         icntl;
  PetscBool        flg;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"MUMPS Options","Mat");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_1","ICNTL(1): output stream for error messages","None",mumps->id.ICNTL(1),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(1) = icntl;
  ierr = PetscOptionsInt("-mat_mumps_icntl_2","ICNTL(2): output stream for diagnostic printing, statistics, and warning","None",mumps->id.ICNTL(2),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(2) = icntl;
  ierr = PetscOptionsInt("-mat_mumps_icntl_3","ICNTL(3): output stream for global information, collected on the host","None",mumps->id.ICNTL(3),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(3) = icntl;

  ierr = PetscOptionsInt("-mat_mumps_icntl_4","ICNTL(4): level of printing (0 to 4)","None",mumps->id.ICNTL(4),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(4) = icntl; 
  if (mumps->id.ICNTL(4) || PetscLogPrintInfo ) mumps->id.ICNTL(3) = 6; /* resume MUMPS default id.ICNTL(3) = 6 */
  
  ierr = PetscOptionsInt("-mat_mumps_icntl_6","ICNTL(6): permuting and/or scaling the matrix (0 to 7)","None",mumps->id.ICNTL(6),&icntl,&flg);CHKERRQ(ierr);
  if (flg) mumps->id.ICNTL(6) = icntl;

  ierr = PetscOptionsInt("-mat_mumps_icntl_7","ICNTL(7): matrix ordering (0 to 7). 3=Scotch, 4=PORD, 5=Metis","None",mumps->id.ICNTL(7),&icntl,&flg);CHKERRQ(ierr);
  if (flg) {
    if (icntl== 1 && mumps->size > 1){
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"pivot order be set by the user in PERM_IN -- not supported by the PETSc/MUMPS interface\n");
    } else {
      mumps->id.ICNTL(7) = icntl;
    }
  } 
  
  ierr = PetscOptionsInt("-mat_mumps_icntl_8","ICNTL(8): scaling strategy (-2 to 8 or 77)","None",mumps->id.ICNTL(8),&mumps->id.ICNTL(8),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_10","ICNTL(10): max num of refinements","None",mumps->id.ICNTL(10),&mumps->id.ICNTL(10),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_11","ICNTL(11): statistics related to the linear system solved (via -ksp_view)","None",mumps->id.ICNTL(11),&mumps->id.ICNTL(11),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_12","ICNTL(12): efficiency control: defines the ordering strategy with scaling constraints (0 to 3)","None",mumps->id.ICNTL(12),&mumps->id.ICNTL(12),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_13","ICNTL(13): efficiency control: with or without ScaLAPACK","None",mumps->id.ICNTL(13),&mumps->id.ICNTL(13),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_14","ICNTL(14): percentage of estimated workspace increase","None",mumps->id.ICNTL(14),&mumps->id.ICNTL(14),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_19","ICNTL(19): Schur complement","None",mumps->id.ICNTL(19),&mumps->id.ICNTL(19),PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-mat_mumps_icntl_22","ICNTL(22): in-core/out-of-core facility (0 or 1)","None",mumps->id.ICNTL(22),&mumps->id.ICNTL(22),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_23","ICNTL(23): max size of the working memory (MB) that can allocate per processor","None",mumps->id.ICNTL(23),&mumps->id.ICNTL(23),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_24","ICNTL(24): detection of null pivot rows (0 or 1)","None",mumps->id.ICNTL(24),&mumps->id.ICNTL(24),PETSC_NULL);CHKERRQ(ierr);
  if (mumps->id.ICNTL(24)){
    mumps->id.ICNTL(13) = 1; /* turn-off ScaLAPACK to help with the correct detection of null pivots */
  }

  ierr = PetscOptionsInt("-mat_mumps_icntl_25","ICNTL(25): computation of a null space basis","None",mumps->id.ICNTL(25),&mumps->id.ICNTL(25),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_26","ICNTL(26): Schur options for right-hand side or solution vector","None",mumps->id.ICNTL(26),&mumps->id.ICNTL(26),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_27","ICNTL(27): experimental parameter","None",mumps->id.ICNTL(27),&mumps->id.ICNTL(27),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_28","ICNTL(28): use 1 for sequential analysis and ictnl(7) ordering, or 2 for parallel analysis and ictnl(29) ordering","None",mumps->id.ICNTL(28),&mumps->id.ICNTL(28),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_29","ICNTL(29): parallel ordering 1 = ptscotch 2 = parmetis","None",mumps->id.ICNTL(29),&mumps->id.ICNTL(29),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_30","ICNTL(30): compute user-specified set of entries in inv(A)","None",mumps->id.ICNTL(30),&mumps->id.ICNTL(30),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_31","ICNTL(31): factors can be discarded in the solve phase","None",mumps->id.ICNTL(31),&mumps->id.ICNTL(31),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_mumps_icntl_33","ICNTL(33): compute determinant","None",mumps->id.ICNTL(33),&mumps->id.ICNTL(33),PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsReal("-mat_mumps_cntl_1","CNTL(1): relative pivoting threshold","None",mumps->id.CNTL(1),&mumps->id.CNTL(1),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_2","CNTL(2): stopping criterion of refinement","None",mumps->id.CNTL(2),&mumps->id.CNTL(2),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_3","CNTL(3): absolute pivoting threshold","None",mumps->id.CNTL(3),&mumps->id.CNTL(3),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_4","CNTL(4): value for static pivoting","None",mumps->id.CNTL(4),&mumps->id.CNTL(4),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_mumps_cntl_5","CNTL(5): fixation for null pivots","None",mumps->id.CNTL(5),&mumps->id.CNTL(5),PETSC_NULL);CHKERRQ(ierr);
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}
 
#undef __FUNCT__
#define __FUNCT__ "PetscInitializeMUMPS"
PetscErrorCode PetscInitializeMUMPS(Mat A,Mat_MUMPS* mumps)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(((PetscObject)A)->comm, &mumps->myid);
  ierr = MPI_Comm_size(((PetscObject)A)->comm,&mumps->size);CHKERRQ(ierr); 
  ierr = MPI_Comm_dup(((PetscObject)A)->comm,&(mumps->comm_mumps));CHKERRQ(ierr);
  mumps->id.comm_fortran = MPI_Comm_c2f(mumps->comm_mumps);

  mumps->id.job = JOB_INIT;
  mumps->id.par = 1;  /* host participates factorizaton and solve */
  mumps->id.sym = mumps->sym; 
#if defined(PETSC_USE_COMPLEX)
  zmumps_c(&mumps->id); 
#else
  dmumps_c(&mumps->id); 
#endif

  mumps->CleanUpMUMPS = PETSC_FALSE;
  mumps->scat_rhs     = PETSC_NULL;
  mumps->scat_sol     = PETSC_NULL;
  mumps->nSolve       = 0;

  /* set PETSc-MUMPS default options - override MUMPS default */
  mumps->id.ICNTL(3) = 0;
  mumps->id.ICNTL(4) = 0;
  if (mumps->size == 1){
    mumps->id.ICNTL(18) = 0;   /* centralized assembled matrix input */
  } else {
    mumps->id.ICNTL(18) = 3;   /* distributed assembled matrix input */
    mumps->id.ICNTL(21) = 1;   /* distributed solution */
  }
  PetscFunctionReturn(0);
}  
  
/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_AIJMUMPS"
PetscErrorCode MatLUFactorSymbolic_AIJMUMPS(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_MUMPS          *lu = (Mat_MUMPS*)F->spptr;
  PetscErrorCode     ierr;
  Vec                b;
  IS                 is_iden;
  const PetscInt     M = A->rmap->N;

  PetscFunctionBegin;
  lu->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set MUMPS options from the options database */
  ierr = PetscSetMUMPSFromOptions(F,A);CHKERRQ(ierr);
 
  ierr = (*lu->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, &lu->nz, &lu->irn, &lu->jcn, &lu->val);CHKERRQ(ierr);

  /* analysis phase */
  /*----------------*/  
  lu->id.job = JOB_FACTSYMBOLIC; 
  lu->id.n = M;
  switch (lu->id.ICNTL(18)){
  case 0:  /* centralized assembled matrix input */
    if (!lu->myid) {
      lu->id.nz =lu->nz; lu->id.irn=lu->irn; lu->id.jcn=lu->jcn;
      if (lu->id.ICNTL(6)>1){
#if defined(PETSC_USE_COMPLEX)
        lu->id.a = (mumps_double_complex*)lu->val; 
#else
        lu->id.a = lu->val; 
#endif
      }
      if (lu->id.ICNTL(7) == 1){ /* use user-provide matrix ordering */
        if (!lu->myid) {
          const PetscInt *idx;
          PetscInt i,*perm_in;
          ierr = PetscMalloc(M*sizeof(PetscInt),&perm_in);CHKERRQ(ierr);
          ierr = ISGetIndices(r,&idx);CHKERRQ(ierr);
          lu->id.perm_in = perm_in;
          for (i=0; i<M; i++) perm_in[i] = idx[i]+1; /* perm_in[]: start from 1, not 0! */
          ierr = ISRestoreIndices(r,&idx);CHKERRQ(ierr);
        }
      }
    }
    break;
  case 3:  /* distributed assembled matrix input (size>1) */ 
    lu->id.nz_loc = lu->nz; 
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
    ierr = ISDestroy(&is_iden);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);    
    break;
    }    
#if defined(PETSC_USE_COMPLEX)
  zmumps_c(&lu->id); 
#else
  dmumps_c(&lu->id); 
#endif
  if (lu->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in analysis phase: INFOG(1)=%d\n",lu->id.INFOG(1)); 
  
  F->ops->lufactornumeric  = MatFactorNumeric_MUMPS;
  F->ops->solve            = MatSolve_MUMPS;
  F->ops->solvetranspose   = MatSolveTranspose_MUMPS;
  F->ops->matsolve         = MatMatSolve_MUMPS;
  PetscFunctionReturn(0); 
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_BAIJMUMPS"
PetscErrorCode MatLUFactorSymbolic_BAIJMUMPS(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{

  Mat_MUMPS       *lu = (Mat_MUMPS*)F->spptr;
  PetscErrorCode  ierr;
  Vec             b;
  IS              is_iden;
  const PetscInt  M = A->rmap->N;

  PetscFunctionBegin;
  lu->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set MUMPS options from the options database */
  ierr = PetscSetMUMPSFromOptions(F,A);CHKERRQ(ierr);

  ierr = (*lu->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, &lu->nz, &lu->irn, &lu->jcn, &lu->val);CHKERRQ(ierr);

  /* analysis phase */
  /*----------------*/  
  lu->id.job = JOB_FACTSYMBOLIC; 
  lu->id.n = M;
  switch (lu->id.ICNTL(18)){
  case 0:  /* centralized assembled matrix input */
    if (!lu->myid) {
      lu->id.nz =lu->nz; lu->id.irn=lu->irn; lu->id.jcn=lu->jcn;
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
    lu->id.nz_loc = lu->nz; 
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
    ierr = ISDestroy(&is_iden);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);    
    break;
    }    
#if defined(PETSC_USE_COMPLEX)
  zmumps_c(&lu->id); 
#else
  dmumps_c(&lu->id); 
#endif
  if (lu->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in analysis phase: INFOG(1)=%d\n",lu->id.INFOG(1)); 
 
  F->ops->lufactornumeric  = MatFactorNumeric_MUMPS;
  F->ops->solve            = MatSolve_MUMPS;
  F->ops->solvetranspose   = MatSolveTranspose_MUMPS;
  PetscFunctionReturn(0); 
}

/* Note the Petsc r permutation and factor info are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_MUMPS"
PetscErrorCode MatCholeskyFactorSymbolic_MUMPS(Mat F,Mat A,IS r,const MatFactorInfo *info) 
{
  Mat_MUMPS          *lu = (Mat_MUMPS*)F->spptr;
  PetscErrorCode     ierr;
  Vec                b;
  IS                 is_iden;
  const PetscInt     M = A->rmap->N;

  PetscFunctionBegin;
  lu->matstruc = DIFFERENT_NONZERO_PATTERN;

  /* Set MUMPS options from the options database */
  ierr = PetscSetMUMPSFromOptions(F,A);CHKERRQ(ierr);

  ierr = (*lu->ConvertToTriples)(A, 1 , MAT_INITIAL_MATRIX, &lu->nz, &lu->irn, &lu->jcn, &lu->val);CHKERRQ(ierr);

  /* analysis phase */
  /*----------------*/  
  lu->id.job = JOB_FACTSYMBOLIC; 
  lu->id.n = M;
  switch (lu->id.ICNTL(18)){
  case 0:  /* centralized assembled matrix input */
    if (!lu->myid) {
      lu->id.nz =lu->nz; lu->id.irn=lu->irn; lu->id.jcn=lu->jcn;
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
    lu->id.nz_loc = lu->nz; 
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
    ierr = ISDestroy(&is_iden);CHKERRQ(ierr);
    ierr = VecDestroy(&b);CHKERRQ(ierr);    
    break;
    }    
#if defined(PETSC_USE_COMPLEX)
  zmumps_c(&lu->id); 
#else
  dmumps_c(&lu->id); 
#endif
  if (lu->id.INFOG(1) < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by MUMPS in analysis phase: INFOG(1)=%d\n",lu->id.INFOG(1)); 

  F->ops->choleskyfactornumeric = MatFactorNumeric_MUMPS;
  F->ops->solve                 = MatSolve_MUMPS;
  F->ops->solvetranspose        = MatSolve_MUMPS;
#if !defined(PETSC_USE_COMPLEX)
  F->ops->getinertia            = MatGetInertia_SBAIJMUMPS;
#else
  F->ops->getinertia            = PETSC_NULL;
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_MUMPS"
PetscErrorCode MatView_MUMPS(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;
  Mat_MUMPS         *lu=(Mat_MUMPS*)A->spptr;

  PetscFunctionBegin;
  /* check if matrix is mumps type */
  if (A->ops->solve != MatSolve_MUMPS) PetscFunctionReturn(0);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO){
      ierr = PetscViewerASCIIPrintf(viewer,"MUMPS run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  SYM (matrix type):                   %d \n",lu->id.sym);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  PAR (host participation):            %d \n",lu->id.par);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(1) (output for error):         %d \n",lu->id.ICNTL(1));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(2) (output of diagnostic msg): %d \n",lu->id.ICNTL(2));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(3) (output for global info):   %d \n",lu->id.ICNTL(3));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(4) (level of printing):        %d \n",lu->id.ICNTL(4));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(5) (input mat struct):         %d \n",lu->id.ICNTL(5));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(6) (matrix prescaling):        %d \n",lu->id.ICNTL(6));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(7) (sequentia matrix ordering):%d \n",lu->id.ICNTL(7));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(8) (scalling strategy):        %d \n",lu->id.ICNTL(8));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(10) (max num of refinements):  %d \n",lu->id.ICNTL(10));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(11) (error analysis):          %d \n",lu->id.ICNTL(11));CHKERRQ(ierr);  
      if (lu->id.ICNTL(11)>0) {
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(4) (inf norm of input mat):        %g\n",lu->id.RINFOG(4));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(5) (inf norm of solution):         %g\n",lu->id.RINFOG(5));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(6) (inf norm of residual):         %g\n",lu->id.RINFOG(6));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(7),RINFOG(8) (backward error est): %g, %g\n",lu->id.RINFOG(7),lu->id.RINFOG(8));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(9) (error estimate):               %g \n",lu->id.RINFOG(9));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"    RINFOG(10),RINFOG(11)(condition numbers): %g, %g\n",lu->id.RINFOG(10),lu->id.RINFOG(11));CHKERRQ(ierr);
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
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(28) (use parallel or sequential ordering):        %d \n",lu->id.ICNTL(28));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(29) (parallel ordering):                          %d \n",lu->id.ICNTL(29));CHKERRQ(ierr);
      
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(30) (user-specified set of entries in inv(A)):    %d \n",lu->id.ICNTL(30));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(31) (factors is discarded in the solve phase):    %d \n",lu->id.ICNTL(31));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  ICNTL(33) (compute determinant):                        %d \n",lu->id.ICNTL(33));CHKERRQ(ierr);
      
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(1) (relative pivoting threshold):      %g \n",lu->id.CNTL(1));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(2) (stopping criterion of refinement): %g \n",lu->id.CNTL(2));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(3) (absolute pivoting threshold):      %g \n",lu->id.CNTL(3));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(4) (value of static pivoting):         %g \n",lu->id.CNTL(4));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  CNTL(5) (fixation for null pivots):         %g \n",lu->id.CNTL(5));CHKERRQ(ierr);
      
      /* infomation local to each processor */
      ierr = PetscViewerASCIIPrintf(viewer, "  RINFO(1) (local estimated flops for the elimination after analysis): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %g \n",lu->myid,lu->id.RINFO(1));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);
      ierr = PetscViewerASCIIPrintf(viewer, "  RINFO(2) (local estimated flops for the assembly after factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d]  %g \n",lu->myid,lu->id.RINFO(2));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);
      ierr = PetscViewerASCIIPrintf(viewer, "  RINFO(3) (local estimated flops for the elimination after factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d]  %g \n",lu->myid,lu->id.RINFO(3));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);
      
      ierr = PetscViewerASCIIPrintf(viewer, "  INFO(15) (estimated size of (in MB) MUMPS internal data for running numerical factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  [%d] %d \n",lu->myid,lu->id.INFO(15));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);
      
      ierr = PetscViewerASCIIPrintf(viewer, "  INFO(16) (size of (in MB) MUMPS internal data used during numerical factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d \n",lu->myid,lu->id.INFO(16));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);
      
      ierr = PetscViewerASCIIPrintf(viewer, "  INFO(23) (num of pivots eliminated on this processor after factorization): \n");CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"    [%d] %d \n",lu->myid,lu->id.INFO(23));CHKERRQ(ierr);
      ierr = PetscViewerFlush(viewer);
      ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_FALSE);CHKERRQ(ierr);

      if (!lu->myid){ /* information from the host */
        ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(1) (global estimated flops for the elimination after analysis): %g \n",lu->id.RINFOG(1));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(2) (global estimated flops for the assembly after factorization): %g \n",lu->id.RINFOG(2));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  RINFOG(3) (global estimated flops for the elimination after factorization): %g \n",lu->id.RINFOG(3));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  (RINFOG(12) RINFOG(13))*2^INFOG(34) (determinant): (%g,%g)*(2^%d)\n",lu->id.RINFOG(12),lu->id.RINFOG(13),lu->id.INFOG(34));CHKERRQ(ierr);
        
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(3) (estimated real workspace for factors on all processors after analysis): %d \n",lu->id.INFOG(3));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(4) (estimated integer workspace for factors on all processors after analysis): %d \n",lu->id.INFOG(4));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(5) (estimated maximum front size in the complete tree): %d \n",lu->id.INFOG(5));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(6) (number of nodes in the complete tree): %d \n",lu->id.INFOG(6));CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(viewer,"  INFOG(7) (ordering option effectively use after analysis): %d \n",lu->id.INFOG(7));CHKERRQ(ierr);
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
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_MUMPS"
PetscErrorCode MatGetInfo_MUMPS(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_MUMPS  *mumps =(Mat_MUMPS*)A->spptr;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = mumps->id.INFOG(20);
  info->nz_used           = mumps->id.INFOG(20);
  info->nz_unneeded       = 0.0;
  info->assemblies        = 0.0;
  info->mallocs           = 0.0;
  info->memory            = 0.0;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatMumpsSetIcntl_MUMPS"
PetscErrorCode MatMumpsSetIcntl_MUMPS(Mat F,PetscInt icntl,PetscInt ival)
{
  Mat_MUMPS *lu =(Mat_MUMPS*)F->spptr;

  PetscFunctionBegin;
  lu->id.ICNTL(icntl) = ival;
  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatMumpsSetIcntl"
/*@
  MatMumpsSetIcntl - Set MUMPS parameter ICNTL()

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-MUMPS interface
.  icntl - index of MUMPS parameter array ICNTL()
-  ival - value of MUMPS ICNTL(icntl)

  Options Database:
.   -mat_mumps_icntl_<icntl> <ival>

   Level: beginner

   References: MUMPS Users' Guide 

.seealso: MatGetFactor()
@*/
PetscErrorCode MatMumpsSetIcntl(Mat F,PetscInt icntl,PetscInt ival)
{
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  PetscValidLogicalCollectiveInt(F,icntl,2);
  PetscValidLogicalCollectiveInt(F,ival,3);
  ierr = PetscTryMethod(F,"MatMumpsSetIcntl_C",(Mat,PetscInt,PetscInt),(F,icntl,ival));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERMUMPS -  A matrix type providing direct solvers (LU and Cholesky) for
  distributed and sequential matrices via the external package MUMPS. 

  Works with MATAIJ and MATSBAIJ matrices

  Options Database Keys:
+ -mat_mumps_icntl_4 <0,...,4> - print level
. -mat_mumps_icntl_6 <0,...,7> - matrix prescaling options (see MUMPS User's Guide)
. -mat_mumps_icntl_7 <0,...,7> - matrix orderings (see MUMPS User's Guidec)
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
  *type = MATSOLVERMUMPS;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN 
/* MatGetFactor for Seq and MPI AIJ matrices */
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_aij_mumps"
PetscErrorCode MatGetFactor_aij_mumps(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqAIJ;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  if (isSeqAIJ) {
    ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PetscNewLog(B,Mat_MUMPS,&mumps);CHKERRQ(ierr);
  B->ops->view             = MatView_MUMPS;
  B->ops->getinfo          = MatGetInfo_MUMPS;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_mumps",MatFactorGetSolverPackage_mumps);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMumpsSetIcntl_C","MatMumpsSetIcntl_MUMPS",MatMumpsSetIcntl_MUMPS);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMUMPS;
    B->factortype = MAT_FACTOR_LU;
    if (isSeqAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqaij_seqaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpiaij_mpiaij;
    mumps->sym = 0;
  } else {
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MUMPS;
    B->factortype = MAT_FACTOR_CHOLESKY;
    if (isSeqAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqaij_seqsbaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpiaij_mpisbaij;
    if (A->spd_set && A->spd) mumps->sym = 1;
    else                      mumps->sym = 2;
  }

  mumps->isAIJ        = PETSC_TRUE;
  mumps->Destroy      = B->ops->destroy;
  B->ops->destroy     = MatDestroy_MUMPS;
  B->spptr            = (void*)mumps;
  ierr = PetscInitializeMUMPS(A,mumps);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0); 
}
EXTERN_C_END


EXTERN_C_BEGIN 
/* MatGetFactor for Seq and MPI SBAIJ matrices */
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_sbaij_mumps"
PetscErrorCode MatGetFactor_sbaij_mumps(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqSBAIJ;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Cannot use PETSc SBAIJ matrices with MUMPS LU, use AIJ matrix");
  if(A->rmap->bs > 1) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_SUP,"Cannot use PETSc SBAIJ matrices with block size > 1 with MUMPS Cholesky, use AIJ matrix instead"); 
  ierr = PetscTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ);CHKERRQ(ierr);
  /* Create the factorization matrix */ 
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = PetscNewLog(B,Mat_MUMPS,&mumps);CHKERRQ(ierr);
  if (isSeqSBAIJ) {
    ierr = MatSeqSBAIJSetPreallocation(B,1,0,PETSC_NULL);CHKERRQ(ierr);
    mumps->ConvertToTriples = MatConvertToTriples_seqsbaij_seqsbaij;
  } else {
    ierr = MatMPISBAIJSetPreallocation(B,1,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
    mumps->ConvertToTriples = MatConvertToTriples_mpisbaij_mpisbaij;
  }

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MUMPS;
  B->ops->view                   = MatView_MUMPS;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_mumps",MatFactorGetSolverPackage_mumps);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMumpsSetIcntl_C","MatMumpsSetIcntl",MatMumpsSetIcntl);CHKERRQ(ierr);
  B->factortype                  = MAT_FACTOR_CHOLESKY;
  if (A->spd_set && A->spd) mumps->sym = 1;
  else                      mumps->sym = 2;

  mumps->isAIJ        = PETSC_FALSE;
  mumps->Destroy      = B->ops->destroy;
  B->ops->destroy     = MatDestroy_MUMPS;
  B->spptr            = (void*)mumps;
  ierr = PetscInitializeMUMPS(A,mumps);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_baij_mumps"
PetscErrorCode MatGetFactor_baij_mumps(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_MUMPS      *mumps;
  PetscBool      isSeqBAIJ;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = PetscTypeCompare((PetscObject)A,MATSEQBAIJ,&isSeqBAIJ);CHKERRQ(ierr);
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  if (isSeqBAIJ) {
    ierr = MatSeqBAIJSetPreallocation(B,A->rmap->bs,0,PETSC_NULL);CHKERRQ(ierr);
  } else {
    ierr = MatMPIBAIJSetPreallocation(B,A->rmap->bs,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = PetscNewLog(B,Mat_MUMPS,&mumps);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_BAIJMUMPS;
    B->factortype = MAT_FACTOR_LU;
    if (isSeqBAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqbaij_seqaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpibaij_mpiaij;
    mumps->sym = 0;
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot use PETSc BAIJ matrices with MUMPS Cholesky, use SBAIJ or AIJ matrix instead\n");
  }

  B->ops->view             = MatView_MUMPS;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_mumps",MatFactorGetSolverPackage_mumps);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMumpsSetIcntl_C","MatMumpsSetIcntl_MUMPS",MatMumpsSetIcntl_MUMPS);CHKERRQ(ierr);

  mumps->isAIJ        = PETSC_TRUE;
  mumps->Destroy      = B->ops->destroy;
  B->ops->destroy     = MatDestroy_MUMPS;
  B->spptr            = (void*)mumps;
  ierr = PetscInitializeMUMPS(A,mumps);CHKERRQ(ierr);

  *F = B;
  PetscFunctionReturn(0); 
}
EXTERN_C_END

