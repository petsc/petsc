#include "global.h"
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
/* Notes on PETSc part of this interface                                  */
/* Information on how to set up shell preconditioners in PETSc can be     */
/* in the PETSc documentation under preconditioners.                      */ 
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
#undef __FUNC__
#define __FUNC__ "RamgShellPCCreate"
/*.. RamgShellPCCreate - This routine creates a user-defined
     preconditioner context.

     Output Parameter:
     shell - user-defined preconditioner context..*/

int RamgShellPCCreate(RamgShellPC **shell)
{
   RamgShellPC *newctx = PetscNew(RamgShellPC); CHKPTRQ(newctx);
   *shell = newctx; 
   return 0; 
}

/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "RamgShellPCSetUp"
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
   int              numnodes, numnonzero, nnz_count; 
   int              j, I, J, ncols_getrow, *cols_getrow; 
   Scalar           *vals_getrow, rowentry; 
   MatInfo          info;
   /*..RAMG variables..*/
   RAMG_PARAM       *ramg_param; 
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

   /*..Get size and number of unknowns of preconditioner matrix..*/ 
   ierr = MatGetSize(pmat, &numnodes, &numnodes); CHKERRA(ierr);
   ierr = MatGetInfo(pmat,MAT_LOCAL,&info); CHKERRA(ierr); 
   numnonzero = int(info.nz_used);
   /*..Set number of unknowns and nonzeros in RAMG terminology..*/
   nnu    = numnodes; 
   nna    = numnonzero; 
   /*..Set RAMG Class 1 parameters..*/
   nda    = 3*nna+5*nnu;
   ndia   = (int)(2.2*nnu);
   ndja   = nda;
   ndu    = 5*nnu; 
   ndf    = ndu; 
   ndig   = 8*nnu; 

   /*..Allocate memory for RAMG variables..*/ 
   Asky     = (double *) PetscMalloc(nda  * sizeof(double)); CHKPTRQ(Asky); 
   ia       = (int*)     PetscMalloc(ndia * sizeof(int)); CHKPTRQ(ia); 
   ja       = (int*)     PetscMalloc(ndja * sizeof(int)); CHKPTRQ(ja); 
   u_approx = (double *) PetscMalloc(ndu  * sizeof(double)); 
                         CHKPTRQ(u_approx); 
   rhs      = (double *) PetscMalloc(ndf  * sizeof(double)); CHKPTRQ(rhs); 
   ig       = (int*)     PetscMalloc(ndig * sizeof(int)); CHKPTRQ(ig); 

   /*..Store PETSc matrix in compressed skyline format required by RAMG..*/ 
   nnz_count = 0;
   for (I=0;I<numnodes;I++){
     /*....Get row I of matrix....*/
     ierr = MatGetRow(pmat,I,&ncols_getrow,&cols_getrow,&vals_getrow); 
            CHKERRA(ierr); 
     ia[I] = nnz_count; 
     for (j=0;j<ncols_getrow;j++){
           J               = cols_getrow[j];
           rowentry        = vals_getrow[j]; 
           Asky[nnz_count] = rowentry; 
           ja[nnz_count]   = J; 
           nnz_count++; 
     }
   }
   ia[numnodes] = nnz_count; 

   makeskyline(numnodes,Asky,ja,ia);

   /*..Switch arrays ia and ja to Fortran conventions..*/ 
   for (j=0;j<=numnodes;j++)
       ia[j]++;
   for (j=0;j<numnonzero;j++)
       ja[j]++;

   /*..Allocate memory for RAMG parameters..*/
   ramg_param             = (RAMG_PARAM*) PetscMalloc(sizeof(RAMG_PARAM)); 
			    CHKPTRQ(ramg_param); ; 

   /*..Set RAMG parameters..*/
   RamgGetParam(ramg_param); 
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

   PetscPrintf(MPI_COMM_WORLD,"\n\n"); 
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"*** Start Setup                        ***\n");
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"\n\n"); 

   /*..Call RAMG..*/  
   symamg1r5_(Asky, ia, ja, u_approx, rhs, ig, &nda, &ndia, &ndja, &ndu, 
              &ndf, &ndig, &nnu, &matrix, &iswtch, &iout, &iprint, &levelx, 
              &ifirst, &ncyc, &eps, &madapt, &nrd, &nsolco, &nru, &ecg1, 
              &ecg2, &ewt2, &nwt, &ntr, &ierr); 

   PetscPrintf(MPI_COMM_WORLD,"\n\n");  
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"*** End Setup                          ***\n");
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"\n\n"); 
 
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

   return 0; 
}

/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "RamgShellPCApply"
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
   int              ierr, I, numnodes, *cols; 
   RamgShellPC      *shell = (RamgShellPC *) ctx; 
   double           *u_approx, *rhs, *Asky, *vals_getarray; 
   int              *ia, *ja; 
   RAMG_PARAM       *ramg_param; 
   /*..RAMG integer work array..*/
   int              *ig;  
   /*..RAMG input parameters..*/ 
   int              nnu; 
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

   /*..Get numnodes as the size of the input vector r..*/
   ierr = VecGetSize(r,&numnodes); CHKERRA(ierr);
   nnu = numnodes; 

   /*..Get values from context..*/
   Asky       = shell->A; 
   ia         = shell->IA; 
   ja         = shell->JA; 
   u_approx   = shell->U_APPROX;
   rhs        = shell->RHS; 
   ig         = shell->IG; 
   ramg_param = shell->PARAM; 

   /*..Set the rhs of the call to ramg equal to the residual..*/
   ierr = VecGetArray(r,&vals_getarray); CHKERRA(ierr);

   /*..Set rhs of call to ramg..*/
   memcpy(rhs, vals_getarray, numnodes * sizeof(*rhs)); 
  
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
   symamg1r5_(Asky, ia, ja, u_approx, rhs, ig, &nda, &ndia, &ndja, &ndu, 
              &ndf, &ndig, &nnu, &matrix, &iswtch, &iout, &iprint, &levelx, 
              &ifirst, &ncyc, &eps, &madapt, &nrd, &nsolco, &nru, &ecg1, 
              &ecg2, &ewt2, &nwt, &ntr, &ierr); 

   /*..Create auxilary vector..*/ 
   cols        = (int *) PetscMalloc(numnodes * sizeof(int) ); 
                 CHKPTRQ(cols);
   for (I=0;I<numnodes;I++)
       cols[I] = I; 

   /*..Store values computed by RAMG into the PETSc vector z..*/
   ierr = VecSetValues(z,numnodes,cols,u_approx,INSERT_VALUES); 
          CHKERRA(ierr);  

   /*..Restore PETSc rhs vector..*/
   ierr = VecRestoreArray(r, &vals_getarray); CHKERRA(ierr);

   PetscFree(cols); 
   
   return 0; 
}

/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "RamgShellPCDestroy"
/*..RamgShellPCDestroy - This routine destroys a user-defined
    preconditioner context.

    Input Parameter:
    shell - user-defined preconditioner context..*/

int RamgShellPCDestroy(RamgShellPC *shell)
{

  /*..Free PCShell context..*/
  PetscFree(shell->A); 
  PetscFree(shell->IA);
  PetscFree(shell->JA);
  PetscFree(shell->U_APPROX);
  PetscFree(shell->RHS);
  PetscFree(shell->IG);   
  PetscFree(shell->PARAM);
  PetscFree(shell);

  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ " RamgGetParam"
int RamgGetParam(RAMG_PARAM *ramg_param)
{
  int       ierr; 

  /*..Set default RAMG paramets..*/ 
  /*....Class 1 RAMG parameters....*/
  (*ramg_param).MATRIX    = 12;
  /*....Class 2 RAMG parameters....*/
  (*ramg_param).ISWTCH    = 4;
  (*ramg_param).IOUT      = 13;
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
  ierr = OptionsGetInt(PETSC_NULL,"-pc_ramg_iswtch",&(*ramg_param).ISWTCH,
                       PETSC_NULL);CHKERRA(ierr);

  ierr = OptionsGetInt(PETSC_NULL,"-pc_ramg_iout",&(*ramg_param).IOUT,
                       PETSC_NULL);CHKERRA(ierr);

  return 0; 
}
