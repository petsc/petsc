#include "global.h"
#include "samgfunc.h"
#include "petscfunc.h"
#include "petscsles.h"

/**************************************************************************/
/*                                                                        */
/*  PETSc - samg interface                                                */
/*  author: Domenico Lahaye (domenico.lahaye@cs.kuleuven.ac.be)           */ 
/*  May 2000                                                              */
/*  This interface allows to call samg as a shell preconditioner.         */
/*  samg is the new algebraic multigrid code by Klaus Stueben [1].        */ 
/*  [1] K. St\"{u}ben,"Algebraic Multigrid: An Introduction for Positive  */
/*      Definite Problems with Applications", Tech. Rep. 53, German       */
/*      National Research Center for Information Technology (GMD),        */
/*      Schloss Birlinhoven, D-53754 Sankt-Augustin, Germany, March 1999  */ 
/*                                                                        */
/**************************************************************************/

/**************************************************************************/
/*                                                                        */
/* Notes on PETSc part of this interface                                  */
/* Information on how to set up shell preconditioners in PETSc can be     */
/* in the PETSc documentation under preconditioners.                      */ 
/*                                                                        */
/**************************************************************************/

/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "SamgShellPCCreate"
/*.. SamgShellPCCreate - This routine creates a user-defined
     preconditioner context.

     Output Parameter:
     shell - user-defined preconditioner context..*/

int SamgShellPCCreate(SamgShellPC **shell)
{
   SamgShellPC *newctx = PetscNew(SamgShellPC); CHKPTRQ(newctx);
   *shell = newctx; 
   return 0; 
}

/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "SamgShellPCSetUp"
/*..SamgShellPCSetUp - This routine sets up a user-defined
    ramg preconditioner context.  

    Input Parameters:
    shell - user-defined preconditioner context
    pmat  - preconditioner matrix

    Output Parameter:
    shell - fully set up user-defined preconditioner context

    This routine calls the setup phase of RAMG..*/
 
int SamgShellPCSetUp(SamgShellPC *shell, Mat pmat)
{
   int              ierr, numnodes, numnonzero, nnz_count; 
   int              j, I, J, ncols_getrow, *cols_getrow; 
   Scalar           *vals_getrow, rowentry; 
   MatInfo          info;
   /*..SAMG variables..*/
   SAMG_PARAM       *samg_param; 
   double           *u_approx, *rhs, *Asky; 
   int              *ia, *ja; 
   /*....Class 0 input parameters..*/  
   int              matrix; 
   /*....Class 1 high-level control input parameters..*/  
   int              nsolve, ifirst, ncyc, iswtch;
   double           eps; 
   /*....Class 1 input parameters for dimensioning..*/
   double           a_cmplx, g_cmplx, p_cmplx, w_avrge; 
   /*....Class 1 input parameters controlling input/output..*/
   int              idump, iout;   
   double           chktol; 
   /*....Class 2 input parameters for the solution phase of SAMG (see above)
	 ....*/ 
   int              nrd, nrc, nru;
   /*....Class 3 general input parameters for the setup phase of SAMG....*/
   int              levelx, nptmn;    
   /*....Class 3 special input parameters for the setup phase of SAMG 
	(see above)....*/ 
   int              ncg, nwt, ntr; 
   double           ecg, ewt, etr; 
   /*..Number of levels that will be created..*/ 
   int              levels;  
   /*..Print intermediate results is desired..*/ 
   int              debug = 0; 

   /*..Get size and number of unknowns of preconditioner matrix..*/ 
   ierr = MatGetSize(pmat, &numnodes, &numnodes); CHKERRA(ierr);
   ierr = MatGetInfo(pmat,MAT_LOCAL,&info); CHKERRA(ierr); 
   numnonzero = int(info.nz_used);

   /*..Allocate memory for RAMG variables..*/ 
   Asky     = (double *) PetscMalloc(numnonzero  * sizeof(double)); 
              CHKPTRQ(Asky); 
   ia       = (int*)     PetscMalloc((numnodes+1)* sizeof(int));
              CHKPTRQ(ia); 
   ja       = (int*)     PetscMalloc(numnonzero * sizeof(int)); 
              CHKPTRQ(ja); 
   u_approx = (double *) PetscMalloc(numnodes * sizeof(double)); 
              CHKPTRQ(u_approx); 
   rhs      = (double *) PetscMalloc(numnodes * sizeof(double)); 
              CHKPTRQ(rhs); 

   /*..Store PETSc matrix in compressed skyline format required by SAMG..*/ 
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
 
   /*..Allocate memory for SAMG parameters..*/
   samg_param             = (SAMG_PARAM*) PetscMalloc(sizeof(SAMG_PARAM)); 
			    CHKPTRQ(samg_param); ; 
   /*..Set SAMG parameters..*/
   SamgGetParam(samg_param); 

   /*..Set SAMG Class 0 parameters..*/
   matrix  = (*samg_param).MATRIX;
   /*..Set SAMG Class 1 parameters..*/
   nsolve  = (*samg_param).NSOLVE; 
   ifirst  = (*samg_param).IFIRST; 
   eps     = (*samg_param).EPS; 
   ncyc    = (*samg_param).NCYC; 
   iswtch  = (*samg_param).ISWTCH; 
   a_cmplx = (*samg_param).A_CMPLX;
   g_cmplx = (*samg_param).G_CMPLX; 
   p_cmplx = (*samg_param).P_CMPLX; 
   w_avrge = (*samg_param).W_AVRGE; 
   chktol  = (*samg_param).CHKTOL; 
   idump   = (*samg_param).IDUMP; 
   iout    = (*samg_param).IOUT; 
   /*..Set SAMG Class 2 parameters..*/
   nrd     = (*samg_param).NRD; 
   nrc     = (*samg_param).NRC; 
   nru     = (*samg_param).NRU; 
   levelx  = (*samg_param).LEVELX; 
   nptmn   = (*samg_param).NPTMN; 
   /*..Set SAMG Class 3 parameters..*/
   ncg     = (*samg_param).NCG; 
   nwt     = (*samg_param).NWT; 
   ntr     = (*samg_param).NTR; 
   ecg     = (*samg_param).ECG; 
   ewt     = (*samg_param).EWT; 
   etr     = (*samg_param).ETR;
   /*..Reset ncyc such that only setup is performed. This is done by setting 
     the last digit of ncyc (the number of cycles performed) equal to zero 
   ..*/ 
   ncyc   = 1030; 

   /*..Switch arrays ia and ja to Fortran conventions..*/ 
   for (j=0;j<=numnodes;j++)
       ia[j]++;
   for (j=0;j<numnonzero;j++)
       ja[j]++;

   PetscPrintf(MPI_COMM_WORLD,"\n\n"); 
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"*** Start Setup new AMG code scal. mode***\n");
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"\n\n"); 

   /*..Call SAMG..*/  
   drvsamg_(&numnodes, Asky, ia, ja, u_approx, rhs, &matrix, &nsolve, &ifirst, 
              &eps, &ncyc, &iswtch, &a_cmplx, &g_cmplx, &p_cmplx, &w_avrge, 
              &chktol, &idump, &iout, &nrd, &nrc, &nru, &levelx, &nptmn, 
              &ncg, &nwt, &ntr, &ecg, &ewt, &etr,
              &levels, &debug); 

   PetscPrintf(MPI_COMM_WORLD,"\n\n");  
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"*** End Setup new AMG code scal. mode  ***\n");
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"\n\n"); 

   /*..Store RAMG output in PETSc context..*/
   shell->A        = Asky; 
   shell->IA       = ia; 
   shell->JA       = ja; 
   shell->PARAM    = samg_param; 
   (*shell).LEVELS = levels; 

   return 0; 
}

/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "SamgShellPCApply"
/*..SamgShellPCApply - This routine applies the AMG code as preconditioner 

    Input Parameters:
    ctx - user-defined context, as set by SamgShellSetApply()
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

int SamgShellPCApply(void *ctx, Vec r, Vec z)
{
   int              ierr, I, numnodes, *cols; 
   SamgShellPC      *shell = (SamgShellPC *) ctx; 
   double           *u_approx, *rhs, *Asky, *vals_getarray; 
   int              *ia, *ja; 
   SAMG_PARAM       *samg_param; 
   /*....Class 0 input parameters..*/  
   int              matrix; 
   /*....Class 1 high-level control input parameters..*/  
   int              nsolve, ifirst, ncyc, iswtch;
   double           eps; 
   /*....Class 1 input parameters for dimensioning..*/
   double           a_cmplx, g_cmplx, p_cmplx, w_avrge; 
   /*....Class 1 input parameters controlling input/output..*/
   int              idump, iout;   
   double           chktol; 
   /*....Class 2 input parameters for the solution phase of SAMG (see above)
	 ....*/ 
   int              nrd, nrc, nru;
   /*....Class 3 general input parameters for the setup phase of SAMG....*/
   int              levelx, nptmn;    
   /*....Class 3 special input parameters for the setup phase of SAMG 
	(see above)....*/ 
   int              ncg, nwt, ntr; 
   double           ecg, ewt, etr; 
   /*..Print intermediate results is desired..*/ 
   int              debug = 0; 
   /*..Number of levels that was be created during setup..*/ 
   int              levels;  

   /*..Get numnodes as the size of the input vector r..*/
   ierr = VecGetSize(r,&numnodes); CHKERRA(ierr);

   /*..Get values from context..*/
   Asky       = shell->A; 
   ia         = shell->IA; 
   ja         = shell->JA; 
   samg_param = shell->PARAM; 

   /*..Set the rhs of the call to ramg equal to the residual..*/
   ierr = VecGetArray(r,&vals_getarray); CHKERRA(ierr);

   /*..Allocate memory for rhs and initial solution of call to samg..*/
   rhs      = (double *) PetscMalloc( numnodes * sizeof(double) ); 
              CHKPTRQ(rhs);
   u_approx = (double *) PetscMalloc( numnodes * sizeof(double) ); 
              CHKPTRQ(u_approx);

   /*..Set rhs of call to ramg..*/
   memcpy(rhs, vals_getarray, numnodes * sizeof(*rhs)); 
  
   /*..Set initial solution of call to ramg to zero..*/
   for (I=0;I<numnodes;I++){
       u_approx[I] = 0.;
   }

   /*..Set SAMG Class 0 parameters..*/
   matrix  = (*samg_param).MATRIX;
   /*..Set SAMG Class 1 parameters..*/
   nsolve  = (*samg_param).NSOLVE; 
   ifirst  = (*samg_param).IFIRST; 
   eps     = (*samg_param).EPS; 
   ncyc    = (*samg_param).NCYC; 
   iswtch  = (*samg_param).ISWTCH; 
   a_cmplx = (*samg_param).A_CMPLX;
   g_cmplx = (*samg_param).G_CMPLX; 
   p_cmplx = (*samg_param).P_CMPLX; 
   w_avrge = (*samg_param).W_AVRGE; 
   chktol  = (*samg_param).CHKTOL; 
   idump   = (*samg_param).IDUMP; 
   iout    = (*samg_param).IOUT; 
   /*..Set SAMG Class 2 parameters..*/
   nrd     = (*samg_param).NRD; 
   nrc     = (*samg_param).NRC; 
   nru     = (*samg_param).NRU; 
   levelx  = (*samg_param).LEVELX; 
   nptmn   = (*samg_param).NPTMN; 
   /*..Set SAMG Class 3 parameters..*/
   ncg     = (*samg_param).NCG; 
   nwt     = (*samg_param).NWT; 
   ntr     = (*samg_param).NTR; 
   ecg     = (*samg_param).ECG; 
   ewt     = (*samg_param).EWT; 
   etr     = (*samg_param).ETR;
   /*..Redefine iswtch to bypass setup..*/ 
   iswtch = 210; 

   /*..Call SAMG..*/
   drvsamg_(&numnodes, Asky, ia, ja, u_approx, rhs, &matrix, &nsolve, &ifirst, 
              &eps, &ncyc, &iswtch, &a_cmplx, &g_cmplx, &p_cmplx, &w_avrge, 
              &chktol, &idump, &iout, &nrd, &nrc, &nru, &levelx, &nptmn, 
              &ncg, &nwt, &ntr, &ecg, &ewt, &etr,
              &levels, &debug);

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
   PetscFree(rhs); 
   PetscFree(u_approx); 

   return 0; 
}

/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "SamgShellPCDestroy"
/*..RamgShellPCDestroy - This routine destroys a user-defined
    preconditioner context.

    Input Parameter:
    shell - user-defined preconditioner context..*/

int SamgShellPCDestroy(SamgShellPC *shell)
{
  /*..Free memory allocated by samg..*/ 
  samg_cleanup_(); 
  /*..Free PCShell context..*/
  PetscFree(shell->A); 
  PetscFree(shell->IA);
  PetscFree(shell->JA);
  PetscFree(shell->PARAM);
  PetscFree(shell);

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "SamgGetParam"
/*..SamgGetParam - Gets SAMG parameters specified at runtime 
    OUTPUT: The parameters set in the SAMG_PARAM context
..*/   

int SamgGetParam(SAMG_PARAM *samg_param)
{
  int       ierr; 

  /*..Set default SAMG paramets..*/ 
  /*....Class 0 SAMG parameters....*/
  (*samg_param).MATRIX    = 12;
  /*....Class 1 SAMG parameters....*/
  (*samg_param).NSOLVE    = 3501; 
  (*samg_param).IFIRST    = 1;
  (*samg_param).EPS       = 1e-12;
  /*......note: in the AMG-PETSc interface the number of cycles is required 
          to equal on assure that in the PCApply routine AMG only performs 
          one cycle......*/ 
  (*samg_param).NCYC      = 1031;
  (*samg_param).ISWTCH    = 410;
  (*samg_param).A_CMPLX   = 2.5; 
  (*samg_param).G_CMPLX   = 1.9; 
  (*samg_param).P_CMPLX   = 1.9; 
  (*samg_param).W_AVRGE   = 2.5; 
  (*samg_param).CHKTOL    = 1e-8;
  (*samg_param).IDUMP     = 0; 
  (*samg_param).IOUT      = 2; 
  /*....Class 2 SAMG parameters....*/        
  (*samg_param).NRD       = 131; 
  (*samg_param).NRC       = 0; 
  (*samg_param).NRU       = -131; 
  (*samg_param).LEVELX    = 25; 
  (*samg_param).NPTMN     = 10; 
  /*....Class 3 SAMG parameters....*/        
  (*samg_param).NCG       = 1000; 
  (*samg_param).NWT       = 3000; 
  (*samg_param).NTR       = 2; 
  (*samg_param).ECG       = 0.25; 
  (*samg_param).EWT       = 0.5; 
  (*samg_param).ETR       = 12.2; 

  /*..Overwrite default values by values specified at runtime..*/
  /*....Class 2 SAMG parameters....*/ 
  ierr = OptionsGetInt(PETSC_NULL,"-pc_samg_iswtch",&(*samg_param).ISWTCH,
                       PETSC_NULL);CHKERRA(ierr);

  ierr = OptionsGetInt(PETSC_NULL,"-pc_samg_iout",&(*samg_param).IOUT,
                       PETSC_NULL);CHKERRA(ierr);

  /*....Class 3 SAMG parameters....*/ 
  ierr = OptionsGetInt(PETSC_NULL,"-pc_samg_levelx",&(*samg_param).LEVELX,
                       PETSC_NULL);CHKERRA(ierr);

  ierr = OptionsGetInt(PETSC_NULL,"-pc_samg_nptmn",&(*samg_param).NPTMN,
                       PETSC_NULL);CHKERRA(ierr);

  return 0; 
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "SamgGetGrid"
/*..SamgGetGrid - This routine gets an array of grids
    INPUT:  levels: number of levels created by SAMG 
            numnodes: number of nodes on finest grid 
            numnonzeros: number of nonzeros on coarsest grid 
    OUTPUT: grid  : array of grids                   ..*/   
int SamgGetGrid(int levels, int numnodes, int numnonzero, 
                GridCtx* grid, void* ctx)
{
   int      k; 
   int      ia_shift[MAX_LEVELS], ja_shift[MAX_LEVELS], nnu_cg, nna_cg;
   int      iw_shift[MAX_LEVELS], jw_shift[MAX_LEVELS], rows_weights, 
            nna_weights, dummy;
   int      ierr; 
   MatInfo  info;
   /*..Uncomment when debugging..*/ 
   //   int      size; 

   /*..Get coarse grid operators..*/ 
   /*....Initialize ia_shift, ja_shift, nnu_cg and nna_cg....*/ 
   ia_shift[1] = 1; 
   ja_shift[1] = 1; 
   nnu_cg = numnodes; 
   nna_cg = numnonzero; 

   for (k=2;k<=levels;k++){ /*....We do not get the finest level matrix....*/ 
       /*....Update ia_shift and ja_shift values with nna_cg and nnu_cg 
             from previous loop....*/ 
       ia_shift[k] = ia_shift[k-1] + nna_cg ; 
       ja_shift[k] = ja_shift[k-1] + nnu_cg ; 

       /*....Get coarse grid matrix on level k....*/ 
       ierr = SamgGetCoarseMat(k, ia_shift[k], ja_shift[k], &(grid[k].A), 
                               PETSC_NULL); 

       /*....Get size and number of nonzeros of coarse grid matrix on
             level k, i.e. get new nna_cg and nnu_cg values....*/ 
       ierr = MatGetSize(grid[k].A, &nnu_cg, &nnu_cg); CHKERRA(ierr);
       ierr = MatGetInfo(grid[k].A, MAT_LOCAL, &info); CHKERRA(ierr); 
       nna_cg = int(info.nz_used);
   }  

   /*..Get interpolation operators..*/ 
   /*....Initialize iw_shift, jw_shift and nna_weights....*/ 
   iw_shift[0] = 1; 
   jw_shift[0] = 1; 
   nna_weights = 0;
   rows_weights = numnodes;

   for (k=1;k<=levels-1;k++){/*....There's NO interpolation operator 
                                   associated to the coarsest level....*/ 
       /*....Update iw_shift with nna_weights value from 
             previous loop....*/ 
       iw_shift[k] = iw_shift[k-1] + nna_weights ; 
       /*....Update jw_shift with rows_weights value from 
             current loop....*/ 
       jw_shift[k] = jw_shift[k-1] + rows_weights ; 
         
       /*....Get interpolation from level k+1 to level k....*/
       ierr = SamgGetInterpolation(k, iw_shift[k], jw_shift[k],
                                   &(grid[k].Interp), PETSC_NULL) ; 

       /*....Get number of collumns and number of nonzeros of 
             interpolation associated to level k. NOTE: The 
             number of collums at this loop equals the number of 
             rows at the next loop...*/
       ierr = MatGetSize(grid[k].Interp, &dummy, &rows_weights); CHKERRA(ierr);
       ierr = MatGetInfo(grid[k].Interp, MAT_LOCAL, &info); CHKERRA(ierr); 
       nna_weights = int(info.nz_used);
   }

   //   ierr = MatGetSize(grid[2].A, &size, &size); CHKERRA(ierr);
 
   //   printf("Size of matrix on level 2 in MGPETSC.C is %4d\n",size); 

   return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "SamgCheckGalerkin"
/*..SamgCheckGalerkin - This routine offers a check on the correctness 
    of how SAMG interpolation and coarse grid operators are parsed to 
    PETSc. This routine computes I^H_h A^h I^h_H by PETSc matrix - matrix 
    multiplications, and compares this product with A^H..*/ 
 
int SamgCheckGalerkin(int levels, Mat A, GridCtx* grid, 
                      void* ctx)
{
   Mat     FineLevelMatrix, Restriction, HalfGalerkin, Galerkin, Diff; 
   double  normdiff; 
   int     ierr, k; 

   for (k=1;k<=levels-1;k++){ 
      if (k==1)
          FineLevelMatrix = A; 
          else
          FineLevelMatrix = grid[k].A; 
      /*....Compute A^h I^h_H....*/ 
      ierr = MatMatMult(FineLevelMatrix, grid[k].Interp, &HalfGalerkin); 
      /*....Get I^h_H....*/ 
      ierr = MatTranspose(grid[k].Interp,&Restriction);
      /*....Compute I^H_h A^h I^h_H....*/ 
      ierr = MatMatMult(Restriction, HalfGalerkin, &Galerkin);
      /*....Compute A^H - I^H_h A^h I^h_H....*/ 
      ierr = MatSubstract(grid[k+1].A, Galerkin, &Diff); 
     /*....Compute || A^H - I^H_h A^h I^h_H||_{\infty}....*/ 
      ierr = MatNorm(Diff,NORM_INFINITY,&normdiff); CHKERRA(ierr);

      printf("SamgCheckGalerkin :: || A^H - I^H_h A^h I^h_H||_{infty} on level %8d = %e\n", 
	      k+1, normdiff); 

      ierr = MatDestroy(Restriction); CHKERRA(ierr); 
      ierr = MatDestroy(HalfGalerkin); CHKERRA(ierr); 
      ierr = MatDestroy(Galerkin); CHKERRA(ierr); 
      ierr = MatDestroy(Diff); CHKERRA(ierr); 
   }
   return 0;
} 

