#define PETSCMAT_DLL

/* 
        Provides an interface to the LUSOL package of ....

*/
#include "src/mat/impls/aij/seq/aij.h"

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define LU1FAC   lu1fac_
#define LU6SOL   lu6sol_
#define M1PAGE   m1page_
#define M5SETX   m5setx_
#define M6RDEL   m6rdel_
#elif !defined(PETSC_HAVE_FORTRAN_CAPS)
#define LU1FAC   lu1fac
#define LU6SOL   lu6sol
#define M1PAGE   m1page
#define M5SETX   m5setx
#define M6RDEL   m6rdel
#endif

EXTERN_C_BEGIN
/*
    Dummy symbols that the MINOS files mi25bfac.f and mi15blas.f may require
*/
void PETSC_STDCALL M1PAGE() {
  ;
}
void PETSC_STDCALL M5SETX() {
  ;
}

void PETSC_STDCALL M6RDEL() {
  ;
}

extern void PETSC_STDCALL LU1FAC (int *m, int *n, int *nnz, int *size, int *luparm,
                        double *parmlu, double *data, int *indc, int *indr,
                        int *rowperm, int *colperm, int *collen, int *rowlen,
                        int *colstart, int *rowstart, int *rploc, int *cploc,
                        int *rpinv, int *cpinv, double *w, int *inform);

extern void PETSC_STDCALL LU6SOL (int *mode, int *m, int *n, double *rhs, double *x,
                        int *size, int *luparm, double *parmlu, double *data,
                        int *indc, int *indr, int *rowperm, int *colperm,
                        int *collen, int *rowlen, int *colstart, int *rowstart,
                        int *inform);
EXTERN_C_END

EXTERN PetscErrorCode MatDuplicate_LUSOL(Mat,MatDuplicateOption,Mat*);

typedef struct  {
  double *data;
  int *indc;
  int *indr;

  int *ip;
  int *iq;
  int *lenc;
  int *lenr;
  int *locc;
  int *locr;
  int *iploc;
  int *iqloc;
  int *ipinv;
  int *iqinv;
  double *mnsw;
  double *mnsv;

  double elbowroom;
  double luroom;		/* Extra space allocated when factor fails   */
  double parmlu[30];		/* Input/output to LUSOL                     */

  int n;			/* Number of rows/columns in matrix          */
  int nz;			/* Number of nonzeros                        */
  int nnz;			/* Number of nonzeros allocated for factors  */
  int luparm[30];		/* Input/output to LUSOL                     */

  PetscErrorCode (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);
  PetscErrorCode (*MatLUFactorSymbolic)(Mat,IS,IS,MatFactorInfo*,Mat*);
  PetscErrorCode (*MatDestroy)(Mat);
  PetscTruth CleanUpLUSOL;

} Mat_LUSOL;

/*  LUSOL input/Output Parameters (Description uses C-style indexes
 *
 *  Input parameters                                        Typical value
 *
 *  luparm(0) = nout     File number for printed messages.         6
 *  luparm(1) = lprint   Print level.                              0
 *                    < 0 suppresses output.
 *                    = 0 gives error messages.
 *                    = 1 gives debug output from some of the
 *                        other routines in LUSOL.
 *                   >= 2 gives the pivot row and column and the
 *                        no. of rows and columns involved at
 *                        each elimination step in lu1fac.
 *  luparm(2) = maxcol   lu1fac: maximum number of columns         5
 *                        searched allowed in a Markowitz-type
 *                        search for the next pivot element.
 *                        For some of the factorization, the
 *                        number of rows searched is
 *                        maxrow = maxcol - 1.
 *
 *
 *  Output parameters
 *
 *  luparm(9) = inform   Return code from last call to any LU routine.
 *  luparm(10) = nsing    No. of singularities marked in the
 *                        output array w(*).
 *  luparm(11) = jsing    Column index of last singularity.
 *  luparm(12) = minlen   Minimum recommended value for  lena.
 *  luparm(13) = maxlen   ?
 *  luparm(14) = nupdat   No. of updates performed by the lu8 routines.
 *  luparm(15) = nrank    No. of nonempty rows of U.
 *  luparm(16) = ndens1   No. of columns remaining when the density of
 *                        the matrix being factorized reached dens1.
 *  luparm(17) = ndens2   No. of columns remaining when the density of
 *                        the matrix being factorized reached dens2.
 *  luparm(18) = jumin    The column index associated with dumin.
 *  luparm(19) = numl0    No. of columns in initial  L.
 *  luparm(20) = lenl0    Size of initial  L  (no. of nonzeros).
 *  luparm(21) = lenu0    Size of initial  U.
 *  luparm(22) = lenl     Size of current  L.
 *  luparm(23) = lenu     Size of current  U.
 *  luparm(24) = lrow     Length of row file.
 *  luparm(25) = ncp      No. of compressions of LU data structures.
 *  luparm(26) = mersum   lu1fac: sum of Markowitz merit counts.
 *  luparm(27) = nutri    lu1fac: triangular rows in U.
 *  luparm(28) = nltri    lu1fac: triangular rows in L.
 *  luparm(29) =
 *
 *
 *  Input parameters                                        Typical value
 *
 *  parmlu(0) = elmax1   Max multiplier allowed in  L           10.0
 *                        during factor.
 *  parmlu(1) = elmax2   Max multiplier allowed in  L           10.0
 *                        during updates.
 *  parmlu(2) = small    Absolute tolerance for             eps**0.8
 *                        treating reals as zero.     IBM double: 3.0d-13
 *  parmlu(3) = utol1    Absolute tol for flagging          eps**0.66667
 *                        small diagonals of U.       IBM double: 3.7d-11
 *  parmlu(4) = utol2    Relative tol for flagging          eps**0.66667
 *                        small diagonals of U.       IBM double: 3.7d-11
 *  parmlu(5) = uspace   Factor limiting waste space in  U.      3.0
 *                        In lu1fac, the row or column lists
 *                        are compressed if their length
 *                        exceeds uspace times the length of
 *                        either file after the last compression.
 *  parmlu(6) = dens1    The density at which the Markowitz      0.3
 *                        strategy should search maxcol columns
 *                        and no rows.
 *  parmlu(7) = dens2    the density at which the Markowitz      0.6
 *                        strategy should search only 1 column
 *                        or (preferably) use a dense LU for
 *                        all the remaining rows and columns.
 *
 *
 *  Output parameters
 *
 *  parmlu(9) = amax     Maximum element in  A.
 *  parmlu(10) = elmax    Maximum multiplier in current  L.
 *  parmlu(11) = umax     Maximum element in current  U.
 *  parmlu(12) = dumax    Maximum diagonal in  U.
 *  parmlu(13) = dumin    Minimum diagonal in  U.
 *  parmlu(14) =
 *  parmlu(15) =
 *  parmlu(16) =
 *  parmlu(17) =
 *  parmlu(18) =
 *  parmlu(19) = resid    lu6sol: residual after solve with U or U'.
 *  ...
 *  parmlu(29) =
 */

#define Factorization_Tolerance       1e-1
#define Factorization_Pivot_Tolerance pow(2.2204460492503131E-16, 2.0 / 3.0) 
#define Factorization_Small_Tolerance 1e-15 /* pow(DBL_EPSILON, 0.8) */

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_LUSOL_SeqAIJ"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_LUSOL_SeqAIJ(Mat A,const MatType type,MatReuse reuse,Mat *newmat) 
{
  /* This routine is only called to convert an unfactored PETSc-LUSOL matrix */
  /* to its base PETSc type, so we will ignore 'MatType type'. */
  PetscErrorCode ierr;
  Mat            B=*newmat;
  Mat_LUSOL      *lusol=(Mat_LUSOL *)A->spptr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  B->ops->duplicate        = lusol->MatDuplicate;
  B->ops->lufactorsymbolic = lusol->MatLUFactorSymbolic;
  B->ops->destroy          = lusol->MatDestroy;
  
  ierr = PetscFree(lusol);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaij_lusol_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_lusol_seqaij_C","",PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJ);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_LUSOL"
PetscErrorCode MatDestroy_LUSOL(Mat A) 
{
  PetscErrorCode ierr;
  Mat_LUSOL *lusol=(Mat_LUSOL *)A->spptr;

  PetscFunctionBegin;
  if (lusol->CleanUpLUSOL) {
    ierr = PetscFree(lusol->ip);CHKERRQ(ierr);
    ierr = PetscFree(lusol->iq);CHKERRQ(ierr);
    ierr = PetscFree(lusol->lenc);CHKERRQ(ierr);
    ierr = PetscFree(lusol->lenr);CHKERRQ(ierr);
    ierr = PetscFree(lusol->locc);CHKERRQ(ierr);
    ierr = PetscFree(lusol->locr);CHKERRQ(ierr);
    ierr = PetscFree(lusol->iploc);CHKERRQ(ierr);
    ierr = PetscFree(lusol->iqloc);CHKERRQ(ierr);
    ierr = PetscFree(lusol->ipinv);CHKERRQ(ierr);
    ierr = PetscFree(lusol->iqinv);CHKERRQ(ierr);
    ierr = PetscFree(lusol->mnsw);CHKERRQ(ierr);
    ierr = PetscFree(lusol->mnsv);CHKERRQ(ierr);
    
    ierr = PetscFree(lusol->indc);CHKERRQ(ierr);
  }

  ierr = MatConvert_LUSOL_SeqAIJ(A,MATSEQAIJ,MAT_REUSE_MATRIX,&A);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__  "MatSolve_LUSOL"
PetscErrorCode MatSolve_LUSOL(Mat A,Vec b,Vec x) 
{
  Mat_LUSOL *lusol=(Mat_LUSOL*)A->spptr;
  double    *bb,*xx;
  int       mode=5;
  PetscErrorCode ierr;
  int       i,m,n,nnz,status;

  PetscFunctionBegin;
  ierr = VecGetArray(x, &xx);CHKERRQ(ierr);
  ierr = VecGetArray(b, &bb);CHKERRQ(ierr);

  m = n = lusol->n;
  nnz = lusol->nnz;

  for (i = 0; i < m; i++)
    {
      lusol->mnsv[i] = bb[i];
    }

  LU6SOL(&mode, &m, &n, lusol->mnsv, xx, &nnz,
         lusol->luparm, lusol->parmlu, lusol->data, 
         lusol->indc, lusol->indr, lusol->ip, lusol->iq, 
         lusol->lenc, lusol->lenr, lusol->locc, lusol->locr, &status);

  if (status != 0)
    {
      SETERRQ(PETSC_ERR_ARG_SIZ,"solve failed"); 
    }

  ierr = VecRestoreArray(x, &xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(b, &bb);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_LUSOL"
PetscErrorCode MatLUFactorNumeric_LUSOL(Mat A,MatFactorInfo *info,Mat *F)
{
  Mat_SeqAIJ     *a;
  Mat_LUSOL      *lusol = (Mat_LUSOL*)(*F)->spptr;
  PetscErrorCode ierr;
  int            m, n, nz, nnz, status;
  int            i, rs, re;
  int            factorizations;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);CHKERRQ(ierr);
  a = (Mat_SeqAIJ *)A->data;

  if (m != lusol->n) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"factorization struct inconsistent");
  }

  factorizations = 0;
  do
    {
      /*******************************************************************/
      /* Check the workspace allocation.                                 */
      /*******************************************************************/

      nz = a->nz;
      nnz = PetscMax(lusol->nnz, (int)(lusol->elbowroom*nz));
      nnz = PetscMax(nnz, 5*n);

      if (nnz < lusol->luparm[12]){
        nnz = (int)(lusol->luroom * lusol->luparm[12]);
      } else if ((factorizations > 0) && (lusol->luroom < 6)){
        lusol->luroom += 0.1;
      }

      nnz = PetscMax(nnz, (int)(lusol->luroom*(lusol->luparm[22] + lusol->luparm[23])));

      if (nnz > lusol->nnz){
        ierr = PetscFree(lusol->indc);CHKERRQ(ierr);
        ierr        = PetscMalloc((sizeof(double)+2*sizeof(int))*nnz,&lusol->indc);CHKERRQ(ierr);
        lusol->indr = lusol->indc + nnz;
        lusol->data = (double *)(lusol->indr + nnz);
        lusol->nnz  = nnz;
      }

      /*******************************************************************/
      /* Fill in the data for the problem.      (1-based Fortran style)  */
      /*******************************************************************/

      nz = 0;
      for (i = 0; i < n; i++)
        {
          rs = a->i[i];
          re = a->i[i+1];

          while (rs < re)
            {
              if (a->a[rs] != 0.0)
                {
                  lusol->indc[nz] = i + 1;
                  lusol->indr[nz] = a->j[rs] + 1;
                  lusol->data[nz] = a->a[rs];
                  nz++;
                }
              rs++;
            }
        }

      /*******************************************************************/
      /* Do the factorization.                                           */
      /*******************************************************************/

      LU1FAC(&m, &n, &nz, &nnz, 
             lusol->luparm, lusol->parmlu, lusol->data,
             lusol->indc, lusol->indr, lusol->ip, lusol->iq,
             lusol->lenc, lusol->lenr, lusol->locc, lusol->locr,
             lusol->iploc, lusol->iqloc, lusol->ipinv,
             lusol->iqinv, lusol->mnsw, &status);
	  
      switch(status)
        {
        case 0:		/* factored */
          break;

        case 7:		/* insufficient memory */
          break;

        case 1:
        case -1:		/* singular */
          SETERRQ(PETSC_ERR_LIB,"Singular matrix"); 

        case 3:
        case 4:		/* error conditions */
          SETERRQ(PETSC_ERR_LIB,"matrix error"); 

        default:		/* unknown condition */
          SETERRQ(PETSC_ERR_LIB,"matrix unknown return code"); 
        }

      factorizations++;
    } while (status == 7);
  (*F)->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_LUSOL"
PetscErrorCode MatLUFactorSymbolic_LUSOL(Mat A, IS r, IS c,MatFactorInfo *info, Mat *F) {
  /************************************************************************/
  /* Input                                                                */
  /*     A  - matrix to factor                                            */
  /*     r  - row permutation (ignored)                                   */
  /*     c  - column permutation (ignored)                                */
  /*                                                                      */
  /* Output                                                               */
  /*     F  - matrix storing the factorization;                           */
  /************************************************************************/
  Mat       B;
  Mat_LUSOL *lusol;
  PetscErrorCode ierr;
  int        i, m, n, nz, nnz;

  PetscFunctionBegin;
	  
  /************************************************************************/
  /* Check the arguments.                                                 */
  /************************************************************************/

  ierr = MatGetSize(A, &m, &n);CHKERRQ(ierr);
  nz = ((Mat_SeqAIJ *)A->data)->nz;

  /************************************************************************/
  /* Create the factorization.                                            */
  /************************************************************************/

  ierr = MatCreate(A->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactornumeric = MatLUFactorNumeric_LUSOL;
  B->ops->solve           = MatSolve_LUSOL;
  B->factor               = FACTOR_LU;
  lusol                   = (Mat_LUSOL*)(B->spptr);

  /************************************************************************/
  /* Initialize parameters                                                */
  /************************************************************************/

  for (i = 0; i < 30; i++)
    {
      lusol->luparm[i] = 0;
      lusol->parmlu[i] = 0;
    }

  lusol->luparm[1] = -1;
  lusol->luparm[2] = 5;
  lusol->luparm[7] = 1;

  lusol->parmlu[0] = 1 / Factorization_Tolerance;
  lusol->parmlu[1] = 1 / Factorization_Tolerance;
  lusol->parmlu[2] = Factorization_Small_Tolerance;
  lusol->parmlu[3] = Factorization_Pivot_Tolerance;
  lusol->parmlu[4] = Factorization_Pivot_Tolerance;
  lusol->parmlu[5] = 3.0;
  lusol->parmlu[6] = 0.3;
  lusol->parmlu[7] = 0.6;

  /************************************************************************/
  /* Allocate the workspace needed by LUSOL.                              */
  /************************************************************************/

  lusol->elbowroom = PetscMax(lusol->elbowroom, info->fill);
  nnz = PetscMax((int)(lusol->elbowroom*nz), 5*n);
     
  lusol->n = n;
  lusol->nz = nz;
  lusol->nnz = nnz;
  lusol->luroom = 1.75;

  ierr = PetscMalloc(sizeof(int)*n,&lusol->ip);
  ierr = PetscMalloc(sizeof(int)*n,&lusol->iq);
  ierr = PetscMalloc(sizeof(int)*n,&lusol->lenc);
  ierr = PetscMalloc(sizeof(int)*n,&lusol->lenr);
  ierr = PetscMalloc(sizeof(int)*n,&lusol->locc);
  ierr = PetscMalloc(sizeof(int)*n,&lusol->locr);
  ierr = PetscMalloc(sizeof(int)*n,&lusol->iploc);
  ierr = PetscMalloc(sizeof(int)*n,&lusol->iqloc);
  ierr = PetscMalloc(sizeof(int)*n,&lusol->ipinv);
  ierr = PetscMalloc(sizeof(int)*n,&lusol->iqinv);
  ierr = PetscMalloc(sizeof(double)*n,&lusol->mnsw);
  ierr = PetscMalloc(sizeof(double)*n,&lusol->mnsv);

  ierr        = PetscMalloc((sizeof(double)+2*sizeof(int))*nnz,&lusol->indc);
  lusol->indr = lusol->indc + nnz;
  lusol->data = (double *)(lusol->indr + nnz);
  lusol->CleanUpLUSOL = PETSC_TRUE;
  *F = B;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_LUSOL"
PetscErrorCode PETSCMAT_DLLEXPORT MatConvert_SeqAIJ_LUSOL(Mat A,const MatType type,MatReuse reuse,Mat *newmat) 
{
  PetscErrorCode ierr;
  PetscInt       m, n;
  Mat_LUSOL      *lusol;
  Mat            B=*newmat;

  PetscFunctionBegin;
  ierr = MatGetSize(A, &m, &n);CHKERRQ(ierr);
  if (m != n) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square");
  }
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
		
  ierr                       = PetscNew(Mat_LUSOL,&lusol);CHKERRQ(ierr);
  lusol->MatDuplicate        = A->ops->duplicate;
  lusol->MatLUFactorSymbolic = A->ops->lufactorsymbolic;
  lusol->MatDestroy          = A->ops->destroy;
  lusol->CleanUpLUSOL        = PETSC_FALSE;

  B->spptr                   = (void*)lusol;
  B->ops->duplicate          = MatDuplicate_LUSOL;
  B->ops->lufactorsymbolic   = MatLUFactorSymbolic_LUSOL;
  B->ops->destroy            = MatDestroy_LUSOL;

  ierr = PetscLogInfo((0,"MatConvert_SeqAIJ_LUSOL:Using LUSOL for LU factorization and solves.\n"));CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_lusol_C",
                                           "MatConvert_SeqAIJ_LUSOL",MatConvert_SeqAIJ_LUSOL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_lusol_seqaij_C",
                                           "MatConvert_LUSOL_SeqAIJ",MatConvert_LUSOL_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,type);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_LUSOL"
PetscErrorCode MatDuplicate_LUSOL(Mat A, MatDuplicateOption op, Mat *M) {
  PetscErrorCode ierr;
  Mat_LUSOL *lu=(Mat_LUSOL *)A->spptr;
  PetscFunctionBegin;
  ierr = (*lu->MatDuplicate)(A,op,M);CHKERRQ(ierr);
  ierr = PetscMemcpy((*M)->spptr,lu,sizeof(Mat_LUSOL));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  MATLUSOL - MATLUSOL = "lusol" - A matrix type providing direct solvers (LU) for sequential matrices 
  via the external package LUSOL.

  If LUSOL is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes LUSOL solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATLUSOL).
  This matrix type is only supported for double precision real.

  This matrix inherits from MATSEQAIJ.  As a result, MatSeqAIJSetPreallocation is 
  supported for this matrix type.  MatConvert can be called for a fast inplace conversion
  to and from the MATSEQAIJ matrix type.

  Options Database Keys:
. -mat_type lusol - sets the matrix type to "lusol" during a call to MatSetFromOptions()

   Level: beginner

.seealso: PCLU
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_LUSOL"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_LUSOL(Mat A) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of SeqAIJ and LUSOL types */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATLUSOL);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_LUSOL(A,MATLUSOL,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
