/*$Id: lusol.c,v 1.11 2001/08/06 21:15:14 bsmith Exp $*/
/* 
        Provides an interface to the LUSOL package of ....

*/
#include "src/mat/impls/aij/seq/aij.h"

EXTERN int MatDestroy_SeqAIJ(Mat);

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
EXTERN_C_END

#if defined(PETSC_HAVE_LUSOL) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)

EXTERN_C_BEGIN
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

typedef struct 
{
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

} Mat_SeqAIJ_LUSOL;

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


#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ_LUSOL"
int MatDestroy_SeqAIJ_LUSOL(Mat A)
{
     Mat_SeqAIJ_LUSOL *lusol;
     int             ierr;

     PetscFunctionBegin;
     lusol = (Mat_SeqAIJ_LUSOL *)A->spptr;

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
     ierr = PetscFree(lusol);CHKERRQ(ierr);

     ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
     PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__  "MatSolve_SeqAIJ_LUSOL"
int MatSolve_SeqAIJ_LUSOL(Mat A,Vec b,Vec x)
{
     Mat_SeqAIJ_LUSOL *lusol = (Mat_SeqAIJ_LUSOL *)A->spptr;
     double *bb, *xx;
     int mode = 5;
     int i, m, n, nnz, status, ierr;

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
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJ_LUSOL"
int MatLUFactorNumeric_SeqAIJ_LUSOL(Mat A, Mat *F)
{
     Mat_SeqAIJ       *a;
     Mat_SeqAIJ_LUSOL *lusol = (Mat_SeqAIJ_LUSOL *)(*F)->spptr;
     int              m, n, nz, nnz, status;
     int              i, rs, re,ierr;
     int              factorizations;

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
	  /* Fill in the data for the problem.                               */
	  /*******************************************************************/

	  nz = 0;
	  if (a->indexshift)
	  {
	       for (i = 0; i < n; i++)
	       {
		    rs = a->i[i] - 1;
		    re = a->i[i+1] - 1;

		    while (rs < re)
		    {
		    	 if (a->a[rs] != 0.0)
			 {
			     lusol->indc[nz] = i + 1;
			     lusol->indr[nz] = a->j[rs];
			     lusol->data[nz] = a->a[rs];
			     nz++;
			 }
			 rs++;
		    }
	       }
	  } else
	  {
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
	       SETERRQ(1,"Singular matrix"); 

	  case 3:
	  case 4:		/* error conditions */
	       SETERRQ(1,"matrix error"); 

	  default:		/* unknown condition */
	       SETERRQ(1,"matrix unknown return code"); 
	  }

	  factorizations++;
     } while (status == 7);
     PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ_LUSOL"
int MatLUFactorSymbolic_SeqAIJ_LUSOL(Mat A, IS r, IS c,MatLUInfo *info, Mat *F)
{
     /************************************************************************/
     /* Input                                                                */
     /*     A  - matrix to factor                                            */
     /*     r  - row permutation (ignored)                                   */
     /*     c  - column permutation (ignored)                                */
     /*                                                                      */
     /* Output                                                               */
     /*     F  - matrix storing the factorization;                           */
     /************************************************************************/

     Mat_SeqAIJ_LUSOL *lusol;
     int              ierr,i, m, n, nz, nnz;

     PetscFunctionBegin;
	  
     /************************************************************************/
     /* Check the arguments.                                                 */
     /************************************************************************/

     ierr = MatGetSize(A, &m, &n);CHKERRQ(ierr);
     nz = ((Mat_SeqAIJ *)A->data)->nz;

     /************************************************************************/
     /* Create the factorization.                                            */
     /************************************************************************/

     ierr = MatCreateSeqAIJ(A->comm, m, n, 0, PETSC_NULL, F);CHKERRQ(ierr);

     (*F)->ops->destroy = MatDestroy_SeqAIJ_LUSOL;
     (*F)->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ_LUSOL;
     (*F)->ops->solve = MatSolve_SeqAIJ_LUSOL;
     (*F)->factor = FACTOR_LU;

     ierr = PetscNew(Mat_SeqAIJ_LUSOL,&lusol);CHKERRQ(ierr);
     (*F)->spptr = (void *)lusol;

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
     PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatUseLUSOL_SeqAIJ"
int MatUseLUSOL_SeqAIJ(Mat A)
{
  int        ierr, m, n;
  PetscTruth match;
     
  PetscFunctionBegin;
  ierr = MatGetSize(A, &m, &n);CHKERRQ(ierr);
  if (m != n) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be square");
  }
		
  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&match);CHKERRQ(ierr);      
  if (!match) {
    SETERRQ(PETSC_ERR_ARG_SIZ,"matrix must be Seq_AIJ");
  }
							    
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_LUSOL;
  PetscLogInfo(0,"Using LUSOL for SeqAIJ LU factorization and solves.");
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatUseLUSOL_SeqAIJ"
int MatUseLUSOL_SeqAIJ(Mat A)
{
     PetscFunctionBegin;
     PetscFunctionReturn(0);
}

#endif
