/*$Id: mumps.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/
/* 
    Provides an interface to the MUMPS_4.2_beta sparse solver
*/

#include "src/mat/impls/aij/seq/aij.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"

#if defined(PETSC_HAVE_MUMPS) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
EXTERN_C_BEGIN 
#include "dmumps_c.h" 
EXTERN_C_END 
#define JOB_INIT -1
#define JOB_END -2
#define ICNTL(I) icntl[(I)-1] /* macro s.t. indices match documentation */
#define INFOG(I) infog[(I)-1]

typedef struct {
  DMUMPS_STRUC_C id;
  MatStructure   matstruc;
  Mat            A_seq;
  int            myid,size,*irn,*jcn;
  PetscScalar    *val;
} Mat_MPIAIJ_MUMPS;

/* convert Petsc mpiaij matrix to triples: row[nz], col[nz], val[nz] */
int MatConvertToTriples(Mat A,int shift,int *nnz,int **r, int **c, PetscScalar **v)
{
  Mat_MPIAIJ   *mat =  (Mat_MPIAIJ*)A->data;  
  Mat_SeqAIJ   *aa=(Mat_SeqAIJ*)(mat->A)->data;
  Mat_SeqAIJ   *bb=(Mat_SeqAIJ*)(mat->B)->data;
  int          *ai=aa->i, *aj=aa->j, *bi=bb->i, *bj=bb->j, rstart= mat->rstart,
               nz = aa->nz + bb->nz, *garray = mat->garray;
  int          ierr,i,j,jj,jB,irow,m=A->m,*ajj,*bjj,countA,countB,colA_start,jcol,myid;
  int          *row,*col;
  PetscScalar *av=aa->a, *bv=bb->a,*val;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(A->comm, &myid);
  if (!(*r)){ 
    ierr = PetscMalloc(nz*sizeof(int),&row);CHKERRQ(ierr);
    ierr = PetscMalloc(nz*sizeof(int),&col);CHKERRQ(ierr);
  }
  if (!(*v)){
    ierr = PetscMalloc(nz*sizeof(PetscScalar),&val);CHKERRQ(ierr);
  } 

  jj = 0; jB = 0; irow = rstart;   
  for ( i=0; i<m; i++ ) {
    ajj = aj + ai[i];                 /* ptr to the beginning of this row */      
    countA = ai[i+1] - ai[i];
    countB = bi[i+1] - bi[i];
    bjj = bj + bi[i];  
  
    /* if (lu->options.symflag == SPOOLES_NONSYMMETRIC ){ */
    /* B part, smaller col index */   
    colA_start = rstart + ajj[0]; /* the smallest col index for A */  
    for (j=0; j<countB; j++){
      jcol = garray[bjj[j]];
      if (jcol > colA_start) {
        jB = j;
        break;
      }
      row[jj] = irow + shift; col[jj] = jcol + shift; 
      val[jj++] = *bv++;
      if (j==countB-1) jB = countB; 
    }
  
    /* A part */
    for (j=0; j<countA; j++){
      row[jj] = irow + shift; col[jj] = rstart + ajj[j] + shift; 
      val[jj++] = *av++;
    }
    /* B part, larger col index */      
    for (j=jB; j<countB; j++){
      row[jj] = irow + shift; col[jj] = garray[bjj[j]] + shift;
      val[jj++] = *bv++;
    }
    irow++;
  } 
  
  *nnz = nz; *r = row; *c = col; *v = val;
  PetscFunctionReturn(0);
}

extern int MatDestroy_MPIAIJ(Mat);
extern int MatDestroy_SeqAIJ(Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_MUMPS"
int MatDestroy_MPIAIJ_MUMPS(Mat A)
{
  Mat_MPIAIJ_MUMPS *lu = (Mat_MPIAIJ_MUMPS*)A->spptr; 
  int              ierr,size=lu->size;

  PetscFunctionBegin; 
  /* Terminate instance, deallocate memories */
  lu->id.job=JOB_END; dmumps_c(&lu->id); 
  if (lu->irn) { ierr = PetscFree(lu->irn);CHKERRQ(ierr);}
  if (lu->jcn) { ierr = PetscFree(lu->jcn);CHKERRQ(ierr);}
  if (lu->val) { ierr = PetscFree(lu->val);CHKERRQ(ierr);}
  if (size>1 && lu->id.ICNTL(18) == 0) {ierr = MatDestroy(lu->A_seq);CHKERRQ(ierr);}
  
  ierr = PetscFree(lu);CHKERRQ(ierr); 
  if (size == 1){
    ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  } 
  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_MPIAIJ_MUMPS"
int MatSolve_MPIAIJ_MUMPS(Mat A,Vec b,Vec x)
{
  Mat_MPIAIJ_MUMPS *lu = (Mat_MPIAIJ_MUMPS*)A->spptr; 
  PetscScalar      *rhs,*array;
  Vec              x_seq;
  IS               iden;
  VecScatter       scat;
  int              ierr;

  PetscFunctionBegin; 
  PetscPrintf(A->comm," ... Solve_\n");
  if (lu->size > 1){
    if (!lu->myid){
      ierr = VecCreateSeq(PETSC_COMM_SELF,A->N,&x_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,A->N,0,1,&iden);CHKERRQ(ierr);
    } else {
      ierr = VecCreateSeq(PETSC_COMM_SELF,0,&x_seq);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&iden);CHKERRQ(ierr);
    }
    ierr = VecScatterCreate(b,iden,x_seq,iden,&scat);CHKERRQ(ierr);
    ierr = ISDestroy(iden);CHKERRQ(ierr);

    ierr = VecScatterBegin(b,x_seq,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
    ierr = VecScatterEnd(b,x_seq,INSERT_VALUES,SCATTER_FORWARD,scat);CHKERRQ(ierr);
    if (!lu->myid) {ierr = VecGetArray(x_seq,&array);CHKERRQ(ierr);}
    /* printf("[%d] rhs is : \n",myid);
       ierr = VecView(x_seq,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr); */
  } else {  /* size == 1 */
    ierr = VecCopy(b,x);CHKERRQ(ierr);
    ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  }
  if (!lu->myid) { /* define rhs on the host */
    lu->id.rhs = array;
  }

  /* solve phase */
  lu->id.job=3;
  dmumps_c(&lu->id);
  if (lu->id.INFOG(1) < 0) {   
    SETERRQ1(1,"Error reported by MUMPS in solve phase: INFOG(1)=%d\n",lu->id.INFOG(1));
  }

  /* convert mumps solution x_seq to petsc mpi x */
  if (lu->size > 1) {
    if (!lu->myid){
      ierr = VecRestoreArray(x_seq,&array);CHKERRQ(ierr);
    }
    ierr = VecScatterBegin(x_seq,x,INSERT_VALUES,SCATTER_REVERSE,scat);CHKERRQ(ierr);
    ierr = VecScatterEnd(x_seq,x,INSERT_VALUES,SCATTER_REVERSE,scat);CHKERRQ(ierr);
    ierr = VecScatterDestroy(scat);CHKERRQ(ierr);
    ierr = VecDestroy(x_seq);CHKERRQ(ierr);
  } else {
    ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
  } 
   
  PetscFunctionReturn(0);
}

#undef __FUNCT__   
#define __FUNCT__ "MatLUFactorNumeric_MPIAIJ_MUMPS"
int MatLUFactorNumeric_MPIAIJ_MUMPS(Mat A,Mat *F)
{
  Mat_MPIAIJ_MUMPS *lu = (Mat_MPIAIJ_MUMPS*)(*F)->spptr; 
  int           rnz,*aj,nnz,ierr,nz,i;
  PetscScalar   *av;
  Mat_SeqAIJ    *aa,*bb;
  Mat_MPIAIJ    *mat;
  IS            isrow;
  Mat           *tseq;

  PetscFunctionBegin; 	
  PetscPrintf(A->comm," ... FactorNumeric_, id.par: %d, id.ICNTL(18): %d\n", lu->id.par,lu->id.ICNTL(18));
  switch (lu->id.ICNTL(18)){
  case 0:  /* centralized assembled matrix input */
    if (lu->size > 1){ /* convert mpi A into seq mat in the host */
      if (lu->myid == 0){
        ierr = ISCreateStride(PETSC_COMM_SELF,A->M,0,1,&isrow); CHKERRQ(ierr);  
      } else {
        ierr = ISCreateStride(PETSC_COMM_SELF,0,0,1,&isrow); CHKERRQ(ierr);
      }  
      ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&tseq);CHKERRQ(ierr); /* called by all proc? */
      ierr = ISDestroy(isrow);CHKERRQ(ierr);
   
      lu->A_seq = *tseq;
      ierr = PetscFree(tseq);CHKERRQ(ierr);
      aa =  (Mat_SeqAIJ*)(lu->A_seq)->data;
      /*
      printf("[%d], A_seq->M: %d\n",lu->myid, A_seq->M);
      ierr = MatView(A_seq,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);*/
    }
 
    if (!lu->myid) {
      if (lu->size == 1) {
        aa   = (Mat_SeqAIJ*)(A)->data;
      } else {
        aa   = (Mat_SeqAIJ*)(lu->A_seq)->data;
      }
      nz   = aa->nz;
      ierr = PetscMalloc(nz*sizeof(int),&lu->irn);CHKERRQ(ierr);
      ierr = PetscMalloc(nz*sizeof(int),&lu->jcn);CHKERRQ(ierr);
      ierr = PetscMalloc(nz*sizeof(PetscScalar),&lu->val);CHKERRQ(ierr);
      aj   = aa->j;
      av   = aa->a;

      nz = 0;
      for (i=0; i<A->M; i++){
        rnz = aa->i[i+1] - aa->i[i];
        while (rnz--) {
          lu->irn[nz] = i+1; lu->jcn[nz] = (*aj)+1; lu->val[nz] = *av; /* Fortran row/col index! */
          nz++; aj++; av++;
        }
      }
      lu->id.n = A->M; lu->id.nz =nz; lu->id.irn=lu->irn; lu->id.jcn=lu->jcn;
      /* lu->id.a = lu->val; */
    }
    break;
  case 3:  /* distributed assembled matrix input (size>1) */
    lu->irn=0; lu->jcn=0; lu->val=0;  /* first numeric factorization, space will be allocated by MatConvertToTriples() */
    ierr = MatConvertToTriples(A, 1, &nnz, &lu->irn, &lu->jcn, &lu->val);CHKERRQ(ierr);
    lu->id.n = A->M;    lu->id.nz_loc = nnz; 
    lu->id.irn_loc=lu->irn; lu->id.jcn_loc=lu->jcn;
    break;
  default: SETERRQ(PETSC_ERR_SUP,"Matrix input format is not supported by MUMPS.");
  }

  /* analysis phase */
  lu->id.job=1;
  dmumps_c(&lu->id);
  if (lu->id.INFOG(1) < 0) { 
    SETERRQ1(1,"Error reported by MUMPS in analysis phase: INFOG(1)=%d\n",lu->id.INFOG(1)); 
  }

  /* numerical factorization phase */
  if(lu->id.ICNTL(18) == 0) {
    if (lu->myid == 0) lu->id.a = lu->val; 
  } else {
    lu->id.a_loc = lu->val; 
  }
  lu->id.job=2;
  dmumps_c(&lu->id);
  if (lu->id.INFOG(1) < 0) {
    SETERRQ1(1,"1, Error reported by MUMPS in numerical factorization phase: INFOG(1)=%d\n",lu->id.INFOG(1)); 
  }

  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_MPIAIJ_MUMPS"
int MatLUFactorSymbolic_MPIAIJ_MUMPS(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat_MPIAIJ              *fac;
  Mat_MPIAIJ_MUMPS        *lu;   
  int                     ierr,M=A->M,N=A->N,size,myid,icntl=-1;
  PetscTruth              flg;

  PetscFunctionBegin; 
  PetscPrintf(A->comm," ... Symbolic_\n");	
  ierr = PetscNew(Mat_MPIAIJ_MUMPS,&lu);CHKERRQ(ierr); 

  /* Create the factorization matrix F */ 
  ierr = MatCreateMPIAIJ(A->comm,A->m,A->n,M,N,0,PETSC_NULL,0,PETSC_NULL,F);CHKERRQ(ierr);

  (*F)->ops->lufactornumeric  = MatLUFactorNumeric_MPIAIJ_MUMPS;
  (*F)->ops->solve            = MatSolve_MPIAIJ_MUMPS;
  (*F)->ops->destroy          = MatDestroy_MPIAIJ_MUMPS;  
  (*F)->factor                = FACTOR_LU;  
  (*F)->spptr                  = (void*)lu;
  fac                         = (Mat_MPIAIJ*)(*F)->data; 
  
  /* Initialize a MUMPS instance */
  ierr = MPI_Comm_rank(A->comm, &lu->myid);
  ierr = MPI_Comm_size(A->comm,&lu->size);CHKERRQ(ierr);
  lu->id.job = JOB_INIT; 
  lu->id.comm_fortran = A->comm;

  lu->id.par=1;  /* host participates factorizaton and solve */
  ierr = PetscOptionsInt("-mat_mumps_par","id.par: ","None",lu->id.par,&icntl,PETSC_NULL);CHKERRQ(ierr);
  if (icntl == 0) lu->id.par=icntl; /* host does not participates factorization and solve -- error for distributed mat input */
 
  lu->id.sym=0;  /* matrix symmetry - 0: unsymmetric; 1: spd; 2: general symmetric. */
  dmumps_c(&lu->id);

  /* output control */
  ierr = PetscOptionsInt("-mat_mumps_icntl_4","ICNTL(4): level of printing","None",lu->id.ICNTL(4),&icntl,&flg);CHKERRQ(ierr);
  if (flg && icntl > 0) {
    lu->id.ICNTL(4)=icntl; 
    printf("[%d] icntl(4): %d\n",lu->myid,lu->id.ICNTL(4));
  } else { /* no output */
    lu->id.ICNTL(1) = 0;  /* error message, default= 6 */
    lu->id.ICNTL(2) = -1; /* output stream for diagnostic printing, statistics, and warning. default=0 */
    lu->id.ICNTL(3) = -1; /* output stream for global information, default=6 */
    lu->id.ICNTL(4) = 0;  /* level of printing, 0 - 4, default=2 */
  }
  
  if (lu->size == 1){
    lu->id.ICNTL(18) = 0;   /* centralized assembled matrix input */
  } else {
    lu->id.ICNTL(18) = 3;   /* distributed assembled matrix input -- default */
    ierr = PetscOptionsInt("-mat_mumps_icntl_18","ICNTL(18): ","None",lu->id.ICNTL(18),&icntl,&flg);CHKERRQ(ierr);
    if (flg){
      switch (icntl){
      case 0: lu->id.ICNTL(18) = 0; break;
      case 3: break;
      default: PetscPrintf(PETSC_COMM_WORLD," id.ICNTL(18)=%d is not supported by the PETSc interface! Default value %d is used.\n", icntl,lu->id.ICNTL(18)); break;
      }
    }
  }
  
  lu->matstruc     = DIFFERENT_NONZERO_PATTERN; 

  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseMUMPS_MPIAIJ"
int MatUseMUMPS_MPIAIJ(Mat A)
{
  PetscFunctionBegin;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_MPIAIJ_MUMPS;
  A->ops->lufactornumeric  = MatLUFactorNumeric_MPIAIJ_MUMPS; 
  PetscFunctionReturn(0);
}

int MatMPIAIJFactorInfo_MUMPS(Mat A,PetscViewer viewer)
{

  PetscFunctionBegin;
  /* check if matrix is mumps type */
  if (A->ops->solve != MatSolve_MPIAIJ_MUMPS) PetscFunctionReturn(0); 

  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseMUMPS_MPIAIJ"
int MatUseMUMPS_MPIAIJ(Mat A)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#endif


