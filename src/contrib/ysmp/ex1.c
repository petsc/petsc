#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.1 1995/12/02 04:34:14 bsmith Exp bsmith $";
#endif

static char help[] = 
"Demonstrates how a user may provide their own matrix operations.\n\
In this example we use the factorization and solve routines of the old\n\
Yale Sparse Matrix Package, YSMP.\n\
Input arguments are:\n\
  -f <input_file> : file to load.\n\
  -ysmp : use YSMP factorization and solve.\n";

#include "mat.h"
#include "sles.h"

extern int SetUpForYsmp(Mat);

int main(int argc,char **args)
{
  int        ierr, its;
  PetscTruth set;
  double     time, norm;
  Scalar     zero = 0.0, none = -1.0;
  Vec        x, b, u;
  Mat        A;
  MatType    mtype;
  SLES       sles;
  char       file[128]; 
  Viewer     fd;

  PetscInitialize(&argc,&args,0,help);

  OptionsSetValue("-mat_aij_oneindex",PETSC_NULL);
  OptionsSetValue("-pc_method","lu");
  OptionsSetValue("-ksp_method","preonly");
  /* Read matrix and RHS */
  OptionsGetString(PETSC_NULL,"-f",file,127);
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,file,BINARY_RDONLY,&fd); CHKERRA(ierr);
  ierr = MatGetTypeFromOptions(PETSC_COMM_WORLD,0,&mtype,&set); CHKERRQ(ierr);
  ierr = MatLoad(fd,mtype,&A); CHKERRA(ierr);
  ierr = VecLoad(fd,&b); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);

  if (OptionsHasName(PETSC_NULL,"-ysmp")) {
    ierr = SetUpForYsmp(A); CHKERRA(ierr);
  }

  /* Set up solution */
  ierr = VecDuplicate(b,&x); CHKERRA(ierr);
  ierr = VecDuplicate(b,&u); CHKERRA(ierr);
  ierr = VecSet(&zero,x); CHKERRA(ierr);

  /* Solve system */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles); CHKERRA(ierr);
  ierr = SLESSetOperators(sles,A,A,ALLMAT_DIFFERENT_NONZERO_PATTERN); CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles); CHKERRA(ierr);
  time = PetscGetTime();
  ierr = SLESSolve(sles,b,x,&its); CHKERRA(ierr);
  time = PetscGetTime()-time;

  /* Show result */
  ierr = MatMult(A,x,u);
  ierr = VecAXPY(&none,b,u); CHKERRA(ierr);
  ierr = VecNorm(u,NORM_2,&norm); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3d\n",its);
  if (norm < 1.e-10) {
    PetscPrintf(PETSC_COMM_WORLD,"Residual norm < 1.e-10\n");
  } else {
    PetscPrintf(PETSC_COMM_WORLD,"Residual norm = %10.4e\n",norm);
  }
  /* PetscPrintf(PETSC_COMM_WORLD,"Time for solve = %5.2f seconds\n",time); */

  /* Cleanup */
  ierr = SLESDestroy(sles); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(b); CHKERRA(ierr);
  ierr = VecDestroy(u); CHKERRA(ierr);
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

/* -----------------------------------------------------------------------*/
/*
    This sets up Petsc Mat_SEQAIJ to use YSMP for the matrix LU factorization 
   and solve.
*/
#include "aij.h"
typedef struct {
   int    first;
   int    *IL,*JL,*IU,*JU;
   int    JLMAX,JUMAX;
   Scalar *L,*U,*D,*ROW;
   IS     icol;
   Mat    A;
} Mat_SeqAIJ_YSMP;
 
#if defined(FORTRANCAPS) 
#define nsf NSF
#define nnf NNF
#define nns NNS
#elif defined(FORTRANUNDERSCORE)
#define nsf nsf_
#define nnf nnf_
#define nns nns_
#endif

extern void nsf(int*,int*,int*,int*,int*,int*,int*,int*,int*,int*,int*, 
                int*,int*,int*);
extern void nnf(int*,int*,int*,int*,int*,int*, Scalar*,Scalar*,Scalar*,int*,int*,Scalar*,
                int*,Scalar*,int*,int*,Scalar*,int*,Scalar*,Scalar*,int*);


int MatSolve_SeqAIJ_YSMP(Mat BB,Vec bb, Vec xx)
{
  Mat_SeqAIJ      *b = (Mat_SeqAIJ*) BB->data;
  Mat_SeqAIJ_YSMP *ysmp = (Mat_SeqAIJ_YSMP *)b->spptr;
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)ysmp->A->data;
  int             N = a->n,FLAG,*IC,*R,*C,*IA = a->i, *JA = a->j,ierr;
  int             *IL = ysmp->IL, *JL = ysmp->JL, *IU = ysmp->IU, *JU = ysmp->JU;
  Scalar          *A = a->a, *U = ysmp->U, *L = ysmp->L,*Z,*B,*D = ysmp->D;
  Scalar          *TMP,*ROW;
  IS              icol = ysmp->icol, col = b->col, row = b->row;
  int             LMAX = ysmp->JLMAX, UMAX = ysmp->JUMAX;

  if (BB->factor != FACTOR_LU) SETERRQ(1,0,"MatSolve_SeqAIJ:Not for unfactored matrix");

  /* get row and column permutations */
  ISGetIndices(row,&R); ISGetIndices(icol,&IC); ISGetIndices(col,&C);

  ierr = VecGetArray(bb,&B); CHKERRQ(ierr);
  ierr = VecGetArray(xx,&Z); CHKERRQ(ierr);
  TMP  = b->solve_work;
  ROW  = ysmp->ROW;

  /* first time through do both numeric factor and solve */  
  if (!ysmp->first) {
    nnf(&N, R,C,IC, IA,JA, A, Z, B, IL,JL,L,&LMAX, D, IU,JU,U,&UMAX,ROW, TMP, &FLAG);
    if (FLAG) {
      SETERRQ2(1,0,"MatSolve_SeqAIJ_YSMP:Error in numeric factorization N %d FLAG %d\n",N,FLAG);
    }
  }
  else {
    SETERRQ(1,0,"Haven't yet set for additional solve ");
  }
  return 0;
}


int MatDestroy_SeqAIJ_YSMP(PetscObject obj)
{
  Mat        A  = (Mat) obj;
  Mat_SeqAIJ *a = (Mat_SeqAIJ *) A->data;

  if (--A->refct > 0) PetscFunctionReturn(0);

  if (A->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->mapping); CHKERRQ(ierr);
  }
  if (A->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->bmapping); CHKERRQ(ierr);
  }
#if defined(USE_PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d, NZ=%d",a->m,a->n,a->nz);
#endif

  /* should also destroy YSMP specific stuff */

  PetscFree(a->solve_work);
  PetscFree(a); 
  ierr = PetscHeaderDestroy(mat);CHKERRQ(ierr);
  return 0;
}

int MatLUFactorSymbolic_SeqAIJ_YSMP(Mat A ,IS row,IS col,double f,Mat *B)
{
  Mat_SeqAIJ_YSMP *ysmp;
  Mat_SeqAIJ      *aij = (Mat_SeqAIJ*) A->data,*b;
  IS              icol;
  int             *R,*IC, N = aij->n,*IA = aij->i, *JA = aij->j,*C;
  int             *IL, *JL,JLMAX,*IU,*JU,JUMAX,FLAG,*Q,*IM,ierr,i;

  ierr = ISInvertPermutation(col,&icol); CHKERRQ(ierr);
  ISGetIndices(row,&R); ISGetIndices(icol,&IC); ISGetIndices(col,&C);

  /* shift column and row permutations by one since YSMP indices start at 1*/
  for ( i=0; i<N; i++ ) {
     R[i]++; IC[i]++; C[i]++;
  }

  /* malloc space for the factors */
  IL = (int *) PetscMalloc( (N+1)*sizeof(int) ); CHKPTRQ(IL);
  IU = (int *) PetscMalloc( (N+1)*sizeof(int) ); CHKPTRQ(IU);
 
  /* this factor 10 is arbitrary, usually much too big but sometimes too small */
  JUMAX = JLMAX = 10*IA[N];
  JL = (int *) PetscMalloc( JLMAX*sizeof(int) ); CHKPTRQ(JL);
  JU = (int *) PetscMalloc( JUMAX*sizeof(int) ); CHKPTRQ(JU);
  
  /* malloc work space */
  Q = (int *) PetscMalloc( (N+1)*sizeof(int) ); CHKPTRQ(Q);
  IM = (int *) PetscMalloc( N*sizeof(int) ); CHKPTRQ(IM);

  /* perform the symbolic factorization */
  nsf(&N, R, IC, IA,JA, IL,JL,&JLMAX, IU,JU,&JUMAX, Q, IM, &FLAG);
  if (FLAG) {
    SETERRQ2(1,0,"MatLUFactorSymbolic_SeqAIJ_YSMP:Error in YSMP symbolic factor N %d FLAG %d\n",N,FLAG);
  }
  PetscFree(Q); PetscFree(IM);

  /* generate the factored PETSc matrix */
  ierr = MatCreateSeqAIJ(A->comm,N,N,MAT_SKIP_ALLOCATION,PETSC_NULL,B); CHKERRQ(ierr);
  PLogObjectParent(*B,icol); 
  b = (Mat_SeqAIJ *) (*B)->data;
  PetscFree(b->imax); b->imax = 0;
  PetscFree(b->ilen);
  b->singlemalloc = PETSC_FALSE;

  b->row        = row;
  b->col        = col;
  b->solve_work = (Scalar *) PetscMalloc( N*sizeof(Scalar)); CHKPTRQ(b->solve_work);
  (*B)->factor  = FACTOR_LU;

  /* save the calculated symbolic factorization in its private structure */
  ysmp = PetscNew(Mat_SeqAIJ_YSMP); 
  ysmp->IL    = IL;
  ysmp->JL    = JL;
  ysmp->IU    = IU;
  ysmp->JU    = JU;
  ysmp->JLMAX = IL[N];
  ysmp->JUMAX = IU[N];
  ysmp->U     = (Scalar *) PetscMalloc( ysmp->JUMAX*sizeof(Scalar) ); CHKPTRQ(ysmp->U);
  ysmp->L     = (Scalar *) PetscMalloc( ysmp->JLMAX*sizeof(Scalar) ); CHKPTRQ(ysmp->L);
  ysmp->D     = (Scalar *) PetscMalloc( N*sizeof(Scalar) ); CHKPTRQ(ysmp->D);
  ysmp->ROW   = (Scalar *) PetscMalloc( N*sizeof(Scalar) ); CHKPTRQ(ysmp->ROW);
  ysmp->icol  = icol;
  ysmp->first = 0;
  ysmp->A     = A;
  b->spptr    = (void *) ysmp;  
  
  /* factored matrix should use YSMP solve */

  (*B)->ops->solve            = MatSolve_SeqAIJ_YSMP;
  (*B)->destroy              = MatDestroy_SeqAIJ_YSMP;
  return 0;
}
    
int MatLUFactorNumeric_SeqAIJ_YSMP(Mat AA,MatFactorInfo *info,Mat *BB)
{
  return 0;
}

/*
    Sets the factor and solve function pointers to use YSMP
*/
int SetUpForYsmp(Mat mat)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ*) mat->data;

  if (!aij->indexshift) SETERRQ(1,0,"Must use index shift for YSMP");
  mat->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_YSMP;
  mat->ops->lufactornumeric  = MatLUFactorNumeric_SeqAIJ_YSMP;
  mat->ops->solve            = MatSolve_SeqAIJ_YSMP;
  return 0;
}








