/*$Id: ramgpetsc.c,v 1.20 2001/08/24 16:04:09 bsmith Exp $*/

#include "src/mat/impls/aij/seq/aij.h"
#include "ramgfunc.h"
#include "petscfunc.h"
#include "petscsles.h"

/**************************************************************************/
/*                                                                        */
/*  PETSc - amg1r5 interface                                              */
/*  author: Domenico Lahaye (domenico.lahaye@cs.kuleuven.ac.be)           */ 
/*  May 2000                                                              */
/*  This interface allows to call amg1r5 as a shell preconditioner.       */
/*  amg1r5 is the algebraic multigrid code by John Ruge and Klaus         */
/*  Stueben [1,2].                                                        */ 
/*  [1] K. St\"{u}ben,"Algebraic Multigrid: An Introduction for Positive  */
/*      Definite Problems with Applications", Tech. Rep. 53, German       */
/*      National Research Center for Information Technology (GMD),        */
/*      Schloss Birlinhoven, D-53754 Sankt-Augustin, Germany, March 1999  */ 
/*  [2] J. Ruge and K. St\"{u}ben, "Algebraic Multigrid" in "Multigrid    */
/*      Methods" S. McCormick, Ed., vol. 3 of Frontiers in Applied        */
/*      Mathmatics, pp. 73--130, SIAM, Philadelphia, PA, 1987             */ 
/*                                                                        */
/**************************************************************************/

/**************************************************************************/
/*                                                                        */
/* Notes on the amg1r5 part of the interface                              */ 
/* The amg1r5 source code can be obtained from MGNET                      */
/* Information on the various options in the code can be found at the     */
/* beginning of the source code.                                          */ 
/*                                                                        */
/**************************************************************************/

/*.. Implementation notes ..*/
/*   i) Memory management: ramg1r5 is a Fortran 77 code and thus does not 
        have dymanic memory management. Sufficiently large real and integer 
        work arrays have to be allocated prior to calling the code. 
     ii) We allocate memory for the right-hand side and the approximation 
        in the set phase of the preconditioner, although this is not 
        strictly necesarry. The setup phase doesn't require any rhs or 
        approximation to the solution.                                   */

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RamgShellPCCreate"
/*.. RamgShellPCCreate - This routine creates a user-defined
     preconditioner context.

     Output Parameter:
     shell - user-defined preconditioner context..*/

int RamgShellPCCreate(RamgShellPC **shell)
{
   RamgShellPC *newctx;
   int         ierr;

   PetscFunctionBegin;
   ierr = PetscNew(RamgShellPC,&newctx); CHKERRQ(ierr);
   *shell = newctx; 
   PetscFunctionReturn(0); 
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RamgShellPCSetUp"
/*..RamgShellPCSetUp - This routine sets up a user-defined
    ramg preconditioner context.  

    Input Parameters:
    shell - user-defined preconditioner context
    pmat  - preconditioner matrix

    Output Parameter:
    shell - fully set up user-defined preconditioner context

    This routine calls the setup phase of RAMG..*/
 
int RamgShellPCSetUp(RamgShellPC *shell, Mat pmat)
{
   int                numnodes, numnonzero, nnz_count; 
   int                j, I, J, ncols_getrow, *cols_getrow,*diag; 
   PetscScalar        *vals_getrow;
   MatInfo            info;
   Mat_SeqAIJ         *aij = (Mat_SeqAIJ*)pmat->data;

   /*..RAMG variables..*/
   struct RAMG_PARAM  *ramg_param; 
   double           *u_approx, *rhs, *Asky; 
   int              *ia, *ja; 
   /*..RAMG names for number of unknowns and number of nonzeros..*/ 
   int              nnu, nna;
   /*..RAMG integer work array..*/
   int              *ig;  
   /*..RAMG input parameters..*/ 
   /*....Class 1 parameters....*/ 
   int              nda, ndia, ndja, ndu, ndf, ndig, matrix; 
   /*....Class 2 parameters....*/ 
   int              iswtch, iout, iprint; 
   /*....Class 3 parameters....*/ 
   int              levelx, ifirst, ncyc, madapt, nrd, nsolco, nru; 
   double           eps;
   /*....Class 4 parameters....*/ 
   int              nwt, ntr; 
   double           ecg1, ecg2, ewt2; 
   /*..RAMG output..*/
   int              ierr;

   PetscFunctionBegin;
   /*..Get size and number of unknowns of preconditioner matrix..*/ 
   ierr = MatGetSize(pmat, &numnodes, &numnodes); CHKERRQ(ierr);
   ierr = MatGetInfo(pmat,MAT_LOCAL,&info); CHKERRQ(ierr); 
   numnonzero = (int)info.nz_used;
   /*..Set number of unknowns and nonzeros in RAMG terminology..*/
   nnu    = numnodes; 
   nna    = numnonzero; 

   /*..Set RAMG Class 1 parameters..*/
   /*
         These are the sizes of all the arrays passed into RAMG
      They need to be large enough or RAMG will return telling how
      large they should be
   */
   nda    = 3*nna+5*nnu + 10;
   ndia   = (int)(2.5*nnu);
   ndja   = nda;
   ndu    = 5*nnu; 
   ndf    = ndu; 
   ndig   = 8*nnu; 

   /*..Allocate memory for RAMG variables..*/ 
   ierr = PetscMalloc(nda *sizeof(double),&Asky);CHKERRQ(ierr);
   ierr = PetscMalloc(ndia*sizeof(int),&ia);CHKERRQ(ierr); 
   ierr = PetscMalloc(ndja*sizeof(int),&ja);CHKERRQ(ierr); 
   ierr = PetscMalloc(ndu *sizeof(double),&u_approx);CHKERRQ(ierr); 
   ierr = PetscMalloc(ndf *sizeof(double),&rhs);CHKERRQ(ierr); 
   ierr = PetscMalloc(ndig*sizeof(int),&ig);CHKERRQ(ierr); 

   /*..Store PETSc matrix in compressed skyline format required by RAMG..*/ 
   nnz_count = 0;
   ierr = MatMarkDiagonal_SeqAIJ(pmat);CHKERRQ(ierr);
   diag = aij->diag;

   for (I=0;I<numnodes;I++){
     ia[I]           = nnz_count + 1; 

     /* put in diagonal entry first */
     ja[nnz_count]   = I + 1;
     Asky[nnz_count] = aij->a[diag[I]];
     nnz_count++;

     /* put in off diagonals */
     ncols_getrow = aij->i[I+1] - aij->i[I];
     vals_getrow  = aij->a + aij->i[I];
     cols_getrow  = aij->j + aij->i[I];
     for (j=0;j<ncols_getrow;j++){
       J = cols_getrow[j];
       if (J != I) {
         Asky[nnz_count] = vals_getrow[j];
         ja[nnz_count]   = J + 1; 
         nnz_count++; 
       }
     }
   }
   ia[numnodes] = nnz_count + 1; 

   /*..Allocate memory for RAMG parameters..*/
   ierr = PetscNew(struct RAMG_PARAM,&ramg_param);CHKERRQ(ierr);

   /*..Set RAMG parameters..*/
   RamgGetParam(pmat,ramg_param); 
   /*..Set remaining RAMG Class 1 parameters..*/     
   matrix = (*ramg_param).MATRIX;
   /*..Set RAMG Class 2 parameters..*/     
   iswtch = (*ramg_param).ISWTCH;
   iout   = (*ramg_param).IOUT; 
   iprint = (*ramg_param).IPRINT;
   /*..Set RAMG Class 3 parameters..*/
   levelx = (*ramg_param).LEVELX;
   ifirst = (*ramg_param).IFIRST; 
   ncyc   = (*ramg_param).NCYC;
   eps    = (*ramg_param).EPS; 
   madapt = (*ramg_param).MADAPT; 
   nrd    = (*ramg_param).NRD; 
   nsolco = (*ramg_param).NSOLCO; 
   nru    = (*ramg_param).NRU; 
   /*..Set RAMG Class 4 parameters..*/     
   ecg1   = (*ramg_param).ECG1;
   ecg2   = (*ramg_param).ECG2;
   ewt2   = (*ramg_param).EWT2;
   nwt    = (*ramg_param).NWT;
   ntr    = (*ramg_param).NTR;
   /*..Reset ncyc such that only setup is performed. This is done by setting 
     the last digit of ncyc (the number of cycles performed) equal to zero 
   ..*/ 
   ncyc   = 1030; 

   ierr = PetscLogInfo((PetscObject)pmat,"\n\n"); CHKERRQ(ierr);
   ierr = PetscLogInfo((PetscObject)pmat,"******************************************\n");CHKERRQ(ierr);
   ierr = PetscLogInfo((PetscObject)pmat,"*** RAMG Start Setup                   ***\n");CHKERRQ(ierr);
   ierr = PetscLogInfo((PetscObject)pmat,"******************************************\n");CHKERRQ(ierr);
   ierr = PetscLogInfo((PetscObject)pmat,"\n\n");CHKERRQ(ierr);

   /*..Call RAMG..*/  
   amg1r5_(Asky, ia, ja, u_approx, rhs, ig, &nda, &ndia, &ndja, &ndu, 
              &ndf, &ndig, &nnu, &matrix, &iswtch, &iout, &iprint, &levelx, 
              &ifirst, &ncyc, &eps, &madapt, &nrd, &nsolco, &nru, &ecg1, 
              &ecg2, &ewt2, &nwt, &ntr, &ierr);
   if (ierr) {
     if (ierr == 2 || ierr == 1) {
       (*PetscErrorPrintf)("Error from RAMG setup, not enough array work space provided\n");
       (*PetscErrorPrintf)("A provided %d\n",nda);
       (*PetscErrorPrintf)("JA provided %d\n",ndja);
       (*PetscErrorPrintf)("IA provided %d\n",ndia);
       (*PetscErrorPrintf)("U provided %d\n",ndu);
       (*PetscErrorPrintf)("F provided %d\n",ndf);
       (*PetscErrorPrintf)("IG provided %d\n",ndig);
     }
     if (ierr == -12) {
       (*PetscErrorPrintf)("Error from RAMG setup, could be matrix is symmetric but you have not\n");
       (*PetscErrorPrintf)("indicated it with MatSetOption(mat,MAT_SYMMETRIC); or -matload_symmetric\n");
     }
     if (ierr == 14) {
       (*PetscErrorPrintf)("Error from RAMG setup, diagonal element not positive\n");
     }
     SETERRQ1(PETSC_ERR_LIB,"Error in RAMG setup. Error number %d",ierr);
   }

   ierr = PetscLogInfo((PetscObject)pmat,"\n\n");CHKERRQ(ierr);  
   ierr = PetscLogInfo((PetscObject)pmat,"******************************************\n");CHKERRQ(ierr);
   ierr = PetscLogInfo((PetscObject)pmat,"*** RAMG End Setup                     ***\n");CHKERRQ(ierr);
   ierr = PetscLogInfo((PetscObject)pmat,"******************************************\n");CHKERRQ(ierr);
   ierr = PetscLogInfo((PetscObject)pmat,"\n\n");CHKERRQ(ierr); 
 
   /*..Store RAMG output in PETSc context..*/ 
   shell->A        = Asky; 
   shell->IA       = ia; 
   shell->JA       = ja; 
   shell->U_APPROX = u_approx; 
   shell->RHS      = rhs; 
   shell->IG       = ig; 
   shell->PARAM    = ramg_param; 

   /*..Save Class 1 parameters..*/ 
   (*ramg_param).NDA    = nda;
   (*ramg_param).NDIA   = ndia;
   (*ramg_param).NDJA   = ndja; 
   (*ramg_param).NDU    = ndu;
   (*ramg_param).NDF    = ndf;
   (*ramg_param).NDIG   = ndig; 
   (*ramg_param).MATRIX = matrix; 

   PetscFunctionReturn(0); 
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RamgShellPCApply"
/*..RamgShellPCApply - This routine applies the AMG code as preconditioner 

    Input Parameters:
    ctx - user-defined context, as set by RamgShellSetApply()
    x - input vector

    Output Parameter:
    y - preconditioned vector

   Notes:
   Note that the PCSHELL preconditioner passes a void pointer as the
   first input argument.  This can be cast to be the whatever the user
   has set (via PCSetShellApply()) the application-defined context to be.

   ..*/
/*..To apply AMG as a preconditioner we set:                        */
/*        i) rhs-vector equal to the residual                       */
/*       ii) start solution equal to zero                           */
/*  Implementation notes:                                           */
/*  For the residual (vector r) we take the values from the vector  */
/*  using VecGetArray. No explicit memory allocation for            */
/*  vals_getarray is thus  needed as VecGetArray takes care of it.  */
/*  The allocated memory is freed again using a call to             */
/*  VecRestoreArray. The values in vals_getarray are then copy to   */
/*  rhs of the AMG code using memcpy.                               */

int RamgShellPCApply(void *ctx, Vec r, Vec z)
{
   int               ierr, I, numnodes, *cols; 
   RamgShellPC       *shell = (RamgShellPC *) ctx; 
   double            *u_approx, *rhs, *Asky, *vals_getarray; 
   int               *ia, *ja; 
   struct RAMG_PARAM *ramg_param; 
   /*..RAMG integer work array..*/
   int               *ig;  
   /*..RAMG input parameters..*/ 
   int               nnu; 
   /*....Class 1 parameters....*/ 
   int               nda, ndia, ndja, ndu, ndf, ndig, matrix; 
   /*....Class 2 parameters....*/ 
   int               iswtch, iout, iprint; 
   /*....Class 3 parameters....*/ 
   int               levelx, ifirst, ncyc, madapt, nrd, nsolco, nru; 
   double            eps;
   /*....Class 4 parameters....*/ 
   int               nwt, ntr; 
   double            ecg1, ecg2, ewt2; 

   /*..Get numnodes as the size of the input vector r..*/
   PetscFunctionBegin;
   ierr = VecGetSize(r,&numnodes); CHKERRQ(ierr);
   nnu  = numnodes; 

   /*..Get values from context..*/
   Asky       = shell->A; 
   ia         = shell->IA; 
   ja         = shell->JA; 
   u_approx   = shell->U_APPROX;
   rhs        = shell->RHS; 
   ig         = shell->IG; 
   ramg_param = shell->PARAM; 

   /*..Set the rhs of the call to ramg equal to the residual..*/
   ierr = VecGetArray(r,&vals_getarray); CHKERRQ(ierr);

   /*..Set rhs of call to ramg..*/
   ierr = PetscMemcpy(rhs, vals_getarray, numnodes * sizeof(*rhs));CHKERRQ(ierr);
  
   /*..Set initial solution of call to ramg to zero..*/
   for (I=0;I<numnodes;I++){
       u_approx[I] = 0.;
   }

   /*..Set RAMG Class 1 parameters..*/     
   nda    = (*ramg_param).NDA; 
   ndia   = (*ramg_param).NDIA; 
   ndja   = (*ramg_param).NDJA; 
   ndu    = (*ramg_param).NDU; 
   ndf    = (*ramg_param).NDF; 
   ndig   = (*ramg_param).NDIG; 
   matrix = (*ramg_param).MATRIX; 
   /*..Set RAMG Class 2 parameters..*/     
   iswtch = (*ramg_param).ISWTCH;
   iout   = (*ramg_param).IOUT; 
   iprint = (*ramg_param).IPRINT;
   /*..Set RAMG Class 3 parameters..*/
   levelx = (*ramg_param).LEVELX;
   ifirst = (*ramg_param).IFIRST; 
   ncyc   = (*ramg_param).NCYC;
   eps    = (*ramg_param).EPS; 
   madapt = (*ramg_param).MADAPT; 
   nrd    = (*ramg_param).NRD; 
   nsolco = (*ramg_param).NSOLCO; 
   nru    = (*ramg_param).NRU; 
   /*..Set RAMG Class 4 parameters..*/     
   ecg1   = (*ramg_param).ECG1;
   ecg2   = (*ramg_param).ECG2;
   ewt2   = (*ramg_param).EWT2;
   nwt    = (*ramg_param).NWT;
   ntr    = (*ramg_param).NTR;

   /*..Redefine iswtch to bypass setup and first..*/ 
   iswtch = 2; 

   /*..Call RAMG..*/
   amg1r5_(Asky, ia, ja, u_approx, rhs, ig, &nda, &ndia, &ndja, &ndu, 
              &ndf, &ndig, &nnu, &matrix, &iswtch, &iout, &iprint, &levelx, 
              &ifirst, &ncyc, &eps, &madapt, &nrd, &nsolco, &nru, &ecg1, 
              &ecg2, &ewt2, &nwt, &ntr, &ierr);
   if (ierr) {
     if (ierr == -1) {
       (*PetscErrorPrintf)("Error from RAMG, not enough array work space provided in NDA\n");
       (*PetscErrorPrintf)("NDA provided %d\n",nda);
     }

     SETERRQ1(1,"Error from RAMG solve, number %d",ierr);
   }

   /*..Create auxilary vector..*/ 
   ierr = PetscMalloc(numnodes * sizeof(int),&cols);CHKERRQ(ierr);
   for (I=0;I<numnodes;I++)
       cols[I] = I; 

   /*..Store values computed by RAMG into the PETSc vector z..*/
   ierr = VecSetValues(z,numnodes,cols,u_approx,INSERT_VALUES);CHKERRQ(ierr);  

   /*..Restore PETSc rhs vector..*/
   ierr = VecRestoreArray(r, &vals_getarray); CHKERRQ(ierr);

   ierr = PetscFree(cols);CHKERRQ(ierr);
   
   PetscFunctionReturn(0); 
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "RamgShellPCDestroy"
/*..RamgShellPCDestroy - This routine destroys a user-defined
    preconditioner context.

    Input Parameter:
    shell - user-defined preconditioner context..*/

int RamgShellPCDestroy(RamgShellPC *shell)
{
  int ierr;

   PetscFunctionBegin;
  /*..Free PCShell context..*/
  ierr = PetscFree(shell->A);CHKERRQ(ierr);
  ierr = PetscFree(shell->IA);CHKERRQ(ierr);
  ierr = PetscFree(shell->JA);CHKERRQ(ierr);
  ierr = PetscFree(shell->U_APPROX);CHKERRQ(ierr);
  ierr = PetscFree(shell->RHS);CHKERRQ(ierr);
  ierr = PetscFree(shell->IG);CHKERRQ(ierr);   
  ierr = PetscFree(shell->PARAM);CHKERRQ(ierr);
  ierr = PetscFree(shell);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ " RamgGetParam"
int RamgGetParam(Mat A,struct RAMG_PARAM *ramg_param)
{
  int        ierr,cycles; 
  PetscTruth flg;

   PetscFunctionBegin;
  /*..Set default RAMG paramets..*/ 
  /*....Class 1 RAMG parameters....*/

  (*ramg_param).MATRIX    = 22;
  if (A->symmetric) {
    (*ramg_param).MATRIX -= 10;
  }

  /*....Class 2 RAMG parameters....*/
  (*ramg_param).ISWTCH    = 4;
  if (PetscLogPrintInfo) {
    (*ramg_param).IOUT    = 13;
  } else { /* no output by default */
    (*ramg_param).IOUT    = 1;
  }
  (*ramg_param).IPRINT    = 10606;
  /*....Class 3 RAMG parameters....*/
  (*ramg_param).LEVELX    = 0;
  (*ramg_param).IFIRST    = 10; 
  /*......note: in the AMG-PETSc interface the number of cycles is required 
          to equal on assure that in the PCApply routine AMG only performs 
          one cycle......*/ 
  (*ramg_param).NCYC      = 1031; 
  (*ramg_param).MADAPT    = 0;
  (*ramg_param).NRD       = 1234;
  (*ramg_param).NSOLCO    = 2; 
  (*ramg_param).NRU       = 1256; 
  (*ramg_param).EPS       = 1e-12; 
  /*....Class 4 RAMG parameters....*/
  (*ramg_param).NWT       = 2; 
  (*ramg_param).NTR       = 0; 
  (*ramg_param).ECG1      = 0.0;  
  (*ramg_param).ECG2      = 0.25; 
  (*ramg_param).EWT2      = 0.35;  

  /*..Overwrite default values by values specified at runtime..*/
  /*....Class 2 RAMG parameters....*/ 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-pc_ramg_iswtch",&(*ramg_param).ISWTCH,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-pc_ramg_iout",&(*ramg_param).IOUT,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-pc_ramg_cycles",&cycles,&flg);CHKERRQ(ierr);
  if (flg) {
    double scale = pow(10.0,((double)(1 + (int)(log10(1.e-12+(double)cycles)))));
    (*ramg_param).NCYC = (int)(103*scale + cycles);
    PetscLogInfo(0,"RAMG using %d for cycles (number after 103 is number of cycles)",(*ramg_param).NCYC);
  }  
  PetscFunctionReturn(0); 
}

/* -------------------------------------------------------------------------------------*/

#include "src/sles/pc/pcimpl.h"

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_RAMG"
static int PCSetUp_RAMG(PC pc)
{
  int        ierr;

  PetscFunctionBegin;
  ierr = RamgShellPCSetUp((RamgShellPC*)(pc->data),pc->pmat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_RAMG"
static int PCApply_RAMG(PC pc,Vec x,Vec y)
{
  int       ierr;

  PetscFunctionBegin;
  ierr = RamgShellPCApply(pc->data,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_RAMG"
static int PCDestroy_RAMG(PC pc)
{
  int       ierr;

  PetscFunctionBegin;
  ierr = RamgShellPCDestroy((RamgShellPC *)pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_RAMG"
int PCCreate_RAMG(PC pc)
{
  int       ierr;

  PetscFunctionBegin;
  ierr = RamgShellPCCreate((RamgShellPC **)&(pc->data));CHKERRQ(ierr);
  pc->ops->destroy = PCDestroy_RAMG;
  pc->ops->apply   = PCApply_RAMG;
  pc->ops->setup   = PCSetUp_RAMG;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
      The AMG code uses a silly timing routine. This captures it
*/
EXTERN_C_BEGIN
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define ctime_ CTIME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define ctime_ ctime
#endif
void ctime_(float *time)
{
  double ltime;
  PetscGetTime(&ltime);
  *time = (float) ltime;
}
EXTERN_C_END


