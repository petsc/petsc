#define PETSCKSP_DLL
#include "petscksp.h"
#include "src/mat/impls/aij/seq/aij.h"
#include "global.h"
#include "externc.h"
#include "samgfunc.h"
#include "petscfunc.h"

/*MC
     PCSAMG -   SAMG + PETSc interface                                                
                                                         
    This interface allows e.g. to call samg as a shell preconditioner.    
    samg is the new algebraic multigrid code by Klaus Stueben [1].         
    Reference for SAMG                                                     
    [1] K. St\"{u}ben,"Algebraic Multigrid: An Introduction for Positive  
        Definite Problems with Applications", in [2], pp. 413--532,        
        Academic Press, 2001.                                              
    [2] U. Trottenberg, C. Oosterlee and A. Sch\"{u}ller, "Multigrid      
        Methods", Academic Press, 2001.                                    
    [1] is also available as                                               
    [3] K. St\"{u}ben,"Algebraic Multigrid: An Introduction for Positive  
        Definite Problems with Applications", Tech. Rep. 53, German       
        National Research Center for Information Technology (GMD),        
        Schloss Birlinhoven, D-53754 Sankt-Augustin, Germany, March 1999   
    For more information on the SAMG-PETSc interface and examples of it's 
    use, see                                                              
    [4] D. Lahaye "Algebraic Multigrid for Time-Harmonic Magnetic Field    
        Computations", PhD Thesis, KU Leuven, Belgium, December 2001.      
        (www.cs.kuleuven.ac.be/~domenico)                                  
                                                                          
   Notes on PETSc part of this interface                                  
   Information on how to set up shell preconditioners in PETSc can be     
   in the PETSc documentation under preconditioners.                       

   This preconditioner has not been completely organized to match the PETSc style,
   see src/ksp/pc/impls/samgpetsc.c for the PC shell routines.

    SAMG is a commercial product available from Klaus Stueben.
 
   Level: developer

   Contributed by Domenico Lahaye (domenico.lahaye@cs.kuleuven.ac.be)   January 2001 
                                                                          
M*/

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SamgShellPCCreate"
/*.. SamgShellPCCreate - This routine creates a user-defined
     preconditioner context.

     Output Parameter:
     shell - user-defined preconditioner context..*/

PetscErrorCode PETSCKSP_DLLEXPORT SamgShellPCCreate(SamgShellPC **shell)
{
   SamgShellPC *newctx; 
   PetscErrorCode ierr; 

   ierr = PetscNew(SamgShellPC,&newctx);CHKERRQ(ierr);
   *shell = newctx; 
   return 0; 
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SamgShellPCSetUp"
/*..SamgShellPCSetUp - This routine sets up a user-defined
    ramg preconditioner context.  

    Input Parameters:
    shell - user-defined preconditioner context
    pmat  - preconditioner matrix

    Output Parameter:
    shell - fully set up user-defined preconditioner context

    This routine calls the setup phase of RAMG..*/
 
PetscErrorCode PETSCKSP_DLLEXPORT SamgShellPCSetUp(SamgShellPC *shell, Mat pmat)
{
   PetscErrorCode ierr;
   int  numnodes, numnonzero, nnz_count; 
   int              j, I, J, ncols_getrow, *cols_getrow, *diag; 
   PetscScalar      *vals_getrow; 
   MatInfo          info;
   Mat_SeqAIJ       *aij = (Mat_SeqAIJ*)pmat->data;
   /*..SAMG variables..*/
   SAMG_PARAM       *samg_param; 
   double           *u_approx, *rhs, *Asky; 
   int              *ia, *ja; 
   /*..Primary SAMG parameters..*/
   /*....System of equations....*/ 
   int      matrix; 
   /*....Start solution and stopping criterium....*/ 
   int      ifirst; 
   double   eps; 
   /*....Scalar and coupled system..*/ 
   int      nsys=1, ndiu=1, ndip=1;
   int      iscale[1], iu[1], ip[1];   
   /*....Approach and smoother....*/ 
   int      nsolve; 
   /*....Cycling process....*/ 
   int      ncyc; 
   /*....Repeated calls....*/ 
   int      iswtch; 
   /*....Initial dimensioning..*/
   double   a_cmplx, g_cmplx, p_cmplx, w_avrge; 
   /*....Class 1 input parameters controlling input/output..*/
   int      idump, iout;   
   double   chktol;
   /*....Output parameters....*/
   double   res_in, res_out; 
   int      ncyc_done;  
   /*..Numbers of levels created by SAMG..*/ 
   int      levels; 
   /*..Auxilary integer to set secondary parameters..*/ 
   int      intin; 

   /*..Get size and number of unknowns of preconditioner matrix..*/ 
   ierr = MatGetSize(pmat, &numnodes, &numnodes);CHKERRQ(ierr);
   ierr = MatGetInfo(pmat,MAT_LOCAL,&info);CHKERRQ(ierr); 
   numnonzero = int(info.nz_used);

   /*..Allocate memory for RAMG variables..*/ 
   ierr = PetscMalloc(numnonzero   * sizeof(double),&Asky);CHKERRQ(ierr);
   ierr = PetscMalloc((numnodes+1) * sizeof(int),&ia);CHKERRQ(ierr);
   ierr = PetscMalloc(numnonzero   * sizeof(int),&ja);CHKERRQ(ierr);
   ierr = PetscMalloc(numnodes     * sizeof(double),&u_approx);CHKERRQ(ierr);
   ierr = PetscMalloc(numnodes     * sizeof(double),&rhs);CHKERRQ(ierr);

   /*..Store PETSc matrix in compressed skyline format required by SAMG..*/ 
   nnz_count = 0;
   ierr = MatMarkDiagonal_SeqAIJ(pmat);
   diag = aij->diag;

   for (I=0;I<numnodes;I++){
     ia[I]           = nnz_count; 

     /*....put in diagonal entry first....*/
     ja[nnz_count]   = I;
     Asky[nnz_count] = aij->a[diag[I]];
     nnz_count++;

     /*....put in off-diagonals....*/
     ncols_getrow = aij->i[I+1] - aij->i[I];
     vals_getrow  = aij->a + aij->i[I];
     cols_getrow  = aij->j + aij->i[I];
     for (j=0;j<ncols_getrow;j++){
       J = cols_getrow[j];
       if (J != I) {
         Asky[nnz_count] = vals_getrow[j];
         ja[nnz_count]   = J; 
         nnz_count++; 
       }
     }
   }
   ia[numnodes] = nnz_count; 

   /*..Allocate memory for SAMG parameters..*/
   ierr = PetscNew(SAMG_PARAM,&samg_param);CHKERRQ(ierr);

   /*..Set SAMG parameters..*/
   SamgGetParam(samg_param); 

   /*..Set Primary parameters..*/
   matrix  = 12; 
   ifirst  = (*samg_param).IFIRST; 
   eps     = (*samg_param).EPS; 
   nsolve  = (*samg_param).NSOLVE; 
   ncyc    = (*samg_param).NCYC; 
   iswtch  = (*samg_param).ISWTCH; 
   a_cmplx = (*samg_param).A_CMPLX;
   g_cmplx = (*samg_param).G_CMPLX; 
   p_cmplx = (*samg_param).P_CMPLX; 
   w_avrge = (*samg_param).W_AVRGE; 
   chktol  = (*samg_param).CHKTOL; 
   idump   = (*samg_param).IDUMP; 
   iout    = (*samg_param).IOUT; 
   /*..Set secondary parameters..*/
   SAMG_set_levelx(&(samg_param->LEVELX)); 
   SAMG_set_nptmn(&(samg_param->NPTMN)); 
   SAMG_set_ecg(&(samg_param->ECG)); 
   SAMG_set_ewt(&(samg_param->EWT)); 
   SAMG_set_ncg(&(samg_param->NCG)); 
   SAMG_set_nwt(&(samg_param->NWT)); 
   SAMG_set_etr(&(samg_param->ETR)); 
   SAMG_set_ntr(&(samg_param->NTR)); 
   SAMG_set_nrd(&(samg_param->NRD)); 
   SAMG_set_nru(&(samg_param->NRU)); 
   SAMG_set_nrc(&(samg_param->NRC)); 
   intin = 6; SAMG_set_logio(&intin);
   intin = 1; SAMG_set_mode_debug(&intin);
   intin = 0; SAMG_set_iter_pre(&intin);
   intin = 0; SAMG_set_np_opt(&intin);
   intin = 0; SAMG_set_ncgrad_default(&intin);

   /*..Reset ncyc such that only setup is performed. This is done by setting 
     the last digit of ncyc (the number of cycles performed) equal to zero. 
     The first digits of ncyc (related to the solve phase) become 
     irrelevant.
   ..*/ 
   ncyc   = 1000; 

   /*..Switch arrays ia and ja to Fortran conventions..*/ 
   for (j=0;j<=numnodes;j++)
       ia[j]++;
   for (j=0;j<numnonzero;j++)
       ja[j]++;

   PetscPrintf(MPI_COMM_WORLD,"\n\n"); 
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"*** Start Setup SAMG code (scal. mode) ***\n");
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"\n\n"); 

   /*..Call SAMG..*/  
   SAMG(&numnodes, &numnonzero, &nsys,
	ia, ja, Asky, rhs, u_approx, iu, &ndiu, ip, &ndip, &matrix, 
	iscale, &res_in, &res_out, &ncyc_done, &ierr, &nsolve, 
	&ifirst, &eps, &ncyc, &iswtch, &a_cmplx, &g_cmplx, 
 	&p_cmplx, &w_avrge, &chktol, &idump, &iout);

   PetscPrintf(MPI_COMM_WORLD,"\n\n");  
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"*** End Setup SAMG code (scal. mode)   ***\n");
   PetscPrintf(MPI_COMM_WORLD,"******************************************\n");
   PetscPrintf(MPI_COMM_WORLD,"\n\n"); 

   /*..Get number of levels created..*/ 
   SAMGPETSC_get_levels(&levels); 

   /*..Store RAMG output in PETSc context..*/
   shell->A        = Asky; 
   shell->IA       = ia; 
   shell->JA       = ja; 
   shell->PARAM    = samg_param; 
   (*shell).LEVELS = levels; 

   return 0; 
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SamgShellPCApply"
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

PetscErrorCode PETSCKSP_DLLEXPORT SamgShellPCApply(void *ctx, Vec r, Vec z)
{
   PetscErrorCode ierr;
   int  I, numnodes, numnonzero, *cols; 
   SamgShellPC      *shell = (SamgShellPC *) ctx; 
   double           *u_approx, *rhs, *Asky, *vals_getarray; 
   int              *ia, *ja; 
   SAMG_PARAM       *samg_param; 
   /*..Primary SAMG parameters..*/
   /*....System of equations....*/ 
   int              matrix; 
   /*....Start solution and stopping criterium....*/ 
   int              ifirst; 
   double           eps; 
   /*....Scalar and coupled system..*/ 
   int              nsys=1, ndiu=1, ndip=1;
   int              iscale[1], iu[1], ip[1];   
   /*....Approach and smoother....*/ 
   int              nsolve; 
   /*....Cycling process....*/ 
   int              ncyc; 
   /*....Repeated calls....*/ 
   int              iswtch; 
   /*....Initial dimensioning..*/
   double           a_cmplx, g_cmplx, p_cmplx, w_avrge; 
   /*....Class 1 input parameters controlling input/output..*/
   int              idump, iout;   
   double           chktol;
   /*....Output parameters....*/
   double           res_in, res_out; 
   int              ncyc_done;  

   /*..Get values from context..*/
   Asky       = shell->A; 
   ia         = shell->IA; 
   ja         = shell->JA; 
   samg_param = shell->PARAM; 

   /*..Get numnodes and numnonzeros..*/ 
   /*....numnodes can be determined as the size of the input vector r....*/
   ierr = VecGetSize(r,&numnodes);CHKERRQ(ierr);
   /*....numnonzero is determined from the pointer ia....*/ 
   /*....Remember that ia following Fortran conventions....*/  
   numnonzero = ia[numnodes]-1; 

   /*..Set the rhs of the call to ramg equal to the residual..*/
   ierr = VecGetArray(r,&vals_getarray);CHKERRQ(ierr);

   /*..Allocate memory for rhs and initial solution of call to samg..*/
   ierr = PetscMalloc(numnodes     * sizeof(double),&u_approx);CHKERRQ(ierr);
   ierr = PetscMalloc(numnodes     * sizeof(double),&rhs);CHKERRQ(ierr);

   /*..Set rhs of call to ramg..*/
   memcpy(rhs, vals_getarray, numnodes * sizeof(*rhs)); 
  
   /*..Set initial solution of call to ramg to zero..*/
   for (I=0;I<numnodes;I++){
       u_approx[I] = 0.;
   }

   /*..Set Primary parameters..*/
   matrix  = (*samg_param).MATRIX; 
   ifirst  = (*samg_param).IFIRST; 
   eps     = (*samg_param).EPS; 
   nsolve  = (*samg_param).NSOLVE; 
   ncyc    = (*samg_param).NCYC; 
   iswtch  = (*samg_param).ISWTCH; 
   a_cmplx = (*samg_param).A_CMPLX;
   g_cmplx = (*samg_param).G_CMPLX; 
   p_cmplx = (*samg_param).P_CMPLX; 
   w_avrge = (*samg_param).W_AVRGE; 
   chktol  = (*samg_param).CHKTOL; 
   idump   = (*samg_param).IDUMP; 
   iout    = (*samg_param).IOUT; 

   /*..Redefine iswtch to bypass setup..*/ 
   /*....First digit of iswtch = 2: bypass setup and do not release memory
         upon return....*/ 
   /*....Second digit of iswtch = 1: memory extension switch....*/ 
   /*....Third and fourth digit of iswtch: n_default. If n_default = 0, 
     the user has to set secondary parameters....*/ 
   iswtch = 210;  

   /*..Call SAMG..*/
   SAMG(&numnodes, &numnonzero, &nsys,
	ia, ja, Asky, rhs, u_approx, iu, &ndiu, ip, &ndip, &matrix, 
	iscale, &res_in, &res_out, &ncyc_done, &ierr, &nsolve, 
	&ifirst, &eps, &ncyc, &iswtch, &a_cmplx, &g_cmplx, 
	&p_cmplx, &w_avrge, &chktol, &idump, &iout);

   /*..Create auxilary integer array..*/ 
   ierr = PetscMalloc(numnodes * sizeof(double),&cols);CHKERRQ(ierr);

   for (I=0;I<numnodes;I++)
       cols[I] = I; 

   /*..Store values computed by SAMG into the PETSc vector z..*/
   ierr = VecSetValues(z,numnodes,cols,u_approx,INSERT_VALUES); 
          CHKERRQ(ierr);  

   /*..Restore PETSc rhs vector..*/
   ierr = VecRestoreArray(r, &vals_getarray);CHKERRQ(ierr);

   PetscFree(cols); 
   PetscFree(rhs); 
   PetscFree(u_approx); 

   return 0; 
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SamgShellPCDestroy"
/*..RamgShellPCDestroy - This routine destroys a user-defined
    preconditioner context.

    Input Parameter:
    shell - user-defined preconditioner context..*/

PetscErrorCode PETSCKSP_DLLEXPORT SamgShellPCDestroy(SamgShellPC *shell)
{
  /*..Free memory allocated by samg..*/ 
  SAMG_cleanup(); 
  /*..Free PCShell context..*/
  PetscFree(shell->A); 
  PetscFree(shell->IA);
  PetscFree(shell->JA);
  PetscFree(shell->PARAM);
  PetscFree(shell);

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "SamgGetParam"
/*..SamgGetParam - Gets SAMG parameters specified at runtime 
    OUTPUT: The parameters set in the SAMG_PARAM context
..*/   

PetscErrorCode PETSCKSP_DLLEXPORT SamgGetParam(SAMG_PARAM *samg_param)
{
  PetscErrorCode ierr; 

  /*..Set default SAMG paramets..*/ 
  /*....Class 0 SAMG parameters....*/
  (*samg_param).MATRIX    = 12;
  /*....Primary SAMG parameters....*/
  /*......If ifirst=0, the vector u, as passed to SAMG, is taken as first 
          approximation......*/
  (*samg_param).IFIRST    = 0;
  (*samg_param).EPS       = 1e-12;
  /*......nsolve =1 denotes scalar approach......*/
  (*samg_param).NSOLVE    = 1; 
  /*......note: in the AMG-PETSc interface the number of cycles is required 
          to equal one to assure that in the PCApply routine AMG only performs 
          one cycle......*/ 
  (*samg_param).NCYC      = 1001;
  (*samg_param).ISWTCH    = 410;
  (*samg_param).A_CMPLX   = 2.5; 
  (*samg_param).G_CMPLX   = 1.9; 
  (*samg_param).P_CMPLX   = 1.9; 
  (*samg_param).W_AVRGE   = 2.5; 
  (*samg_param).CHKTOL    = 1e-8;
  (*samg_param).IDUMP     = 0; 
  (*samg_param).IOUT      = 2; 
  /*....Secundary SAMG parameters....*/ 
  (*samg_param).LEVELX    = 25; 
  (*samg_param).NPTMN     = 100;   
  (*samg_param).ECG       = 0.25; 
  (*samg_param).EWT       = 0.5;      
  (*samg_param).NCG       = 1000; 
  (*samg_param).NWT       = 3000; 
  (*samg_param).ETR       = 12.2; 
  (*samg_param).NTR       = 2; 
  (*samg_param).NRD       = 131; 
  (*samg_param).NRC       = 0; 
  (*samg_param).NRU       = -131; 

  /*..Overwrite default values by values specified at runtime..*/
  /*....Primary SAMG parameters....*/ 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-pc_samg_iswtch",&(*samg_param).ISWTCH,
                       PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-pc_samg_ncyc",&(*samg_param).NCYC,
                       PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-pc_samg_iout",&(*samg_param).IOUT,
                       PETSC_NULL);CHKERRQ(ierr);

 /*....Secundary SAMG parameters....*/ 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-pc_samg_levelx",&(*samg_param).LEVELX,
                       PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-pc_samg_nptmn",&(*samg_param).NPTMN,
                       PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-pc_samg_nrd",&(*samg_param).NRD,
                       PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-pc_samg_nrc",&(*samg_param).NRC,
                       PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-pc_samg_nru",&(*samg_param).NRU,
                       PETSC_NULL);CHKERRQ(ierr);

  return 0; 
}
