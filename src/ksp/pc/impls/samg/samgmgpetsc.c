#define PETSCKSP_DLL

#include "global.h"
#include "petscfunc.h"
#include "petscksp.h"
#include "petscmg.h"

static char help[] = "Does PETSc multigrid cycling using hierarchy build by SAMG\n\n";

PetscErrorCode samgmgpetsc(const int numnodes, double* Asky, int* ia, 
                int* ja, double* rhs, double* u_approx, 
                const OPTIONS *options)
{
   /*..Petsc variables..*/
   Vec      x, b;         /* approx solution and RHS */
   Mat      A;            /* linear system matrix */
   KSP     ksp;         /* linear solver context */
   PC       pc;           /* preconditioner context */
   PCType   pctype;       /* preconditioning technique */
   KSP      ksp;          /* KSP context */
   PetscErrorCode ierr;
   int  its;    /* Error messages and number of iterations */
   /*..Other variables for the PETSc interface..*/
   int      *nnz_per_row; /* integer vector to hold the number of nonzeros */
                          /* of each row. This vector will be used to      */
                          /* allocate memory for the matrix, and to store  */
                          /* elements in the matrix                        */
   int      *cols;        /* cols is a vector of collumn indices used in   */
                          /* assembling the PETSc rhs vector               */
   PetscScalar *sol_array;/* sol_array used to pass the PETSc solution     */ 
                          /* back to the calling program                   */ 
   /*..Variables used to customize the convergence criterium to            */
   /*  ||res|| / ||b|| < tol                                               */
   double      bnrm2; 
   CONVHIST    *convhist;
   /*..Context for the SAMG preconditioner..*/
   SamgShellPC *samg_ctx;
   /*..Variables to extract SAMG hierarchy..*/ 
   int         k, levels, numnonzero; 
   double      normdiff; 
   GridCtx     grid[MAX_LEVELS]; 
   char        pathfilename[80], basefilename[80];
   /*..Variables for intermediate levels..*/ 
   KSP         ksp_pre, ksp_post; 
   PC          pc_pre, pc_post; 
   Mat         FineLevelMatrix; 
   int         petsc_level, size; 
   /*..Variables for coarse grid solve..*/ 
   KSP        coarsegridksp;
   PC          coarsegridpc; 
   KSP         coarsegridksp; 
   int         coarsegrid_n; 
   double      coarsegrid_rnorm; 
   /*..Variables that determine behaviour of the code..*/
   static PetscErrorCode  debug = *(options->DEBUG); 
   /*..Other variables..*/
   int         I;   
   PetscTruth  flg, issamg, issamg_print; 
   /*..Variables for CPU timings..*/ 
   PetscLogDouble  v1,v2,t_setup, t_solve;

   /*..Executable statements..*/
   PetscInitialize( (int*) 0, (char ***) 0,(char *) 0, help);

   /*..Get start time of linear system setup..*/ 
   ierr = PetscGetTime(&v1);CHKERRQ(ierr); 

   ierr = PetscMalloc(numnodes * sizeof(int),&nnz_per_row);CHKERRQ(ierr);

   /*..The numbero f nonzeros entries in row I can be calculated as      
       ia[I+1] - 1 - ia[I] + 1 = ia[I+1] - ia[I]                         ..*/
   for (I=0;I<numnodes;I++)
       nnz_per_row[I] = ia[I+1] - ia[I]; 

   /*..Allocate (create) SeqAIJ matrix  for use within PETSc..*/
   ierr = MatCreate(PETSC_COMM_WORLD,numnodes,numnodes,numnodes,numnodes,&A);
   ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
   ierr = MatSeqAIJSetPreallocation(A,0,nnz_per_row);CHKERRQ(ierr);

  /*..Assemble matrix  for use within PETSc..*/
   for (I=0;I<numnodes;I++){
      ierr = MatSetValues(A, 
               1,              /* number of rows */
               &I,             /* pointer to global row number */
               nnz_per_row[I], /* number of collums = number of nonzero ... */
                               /* entries in row I                          */
               &(ja[ ia[I] ]), 
                              /* vector global column indices */
               (PetscScalar *) &(Asky[ ia[I] ]),
                              /* vector of coefficients */
	INSERT_VALUES);CHKERRQ(ierr);
   }

   ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
   if (debug) 
       ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

   /*..Create solution and rhs vector.  Note that we form  vector from 
     scratch and then duplicate as needed..*/
   ierr = VecCreate(PETSC_COMM_WORLD,&x);
   ierr = VecSetSizes(x,PETSC_DECIDE,numnodes);
   ierr = VecSetType(x,VECSEQ);CHKERRQ(ierr);
   ierr = VecDuplicate(x,&b);CHKERRQ(ierr); 

   ierr = PetscMalloc(numnodes * sizeof(int),&cols);CHKERRQ(ierr);
   for (I=0;I<numnodes;I++)
       cols[I] = I; 

   /*..Assemble the right-hand side vector for use within PETSc..*/
   ierr = VecSetValues(b,numnodes,cols,(PetscScalar*)rhs,INSERT_VALUES); 
          CHKERRQ(ierr);  
   ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
   ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
   if (debug){
      printf("[PETSc]:The right-hand side \n");
      ierr = VecView(b,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      printf("\n"); 
   }
   ierr = VecNorm(b,NORM_2,&bnrm2);CHKERRQ(ierr);

   /*..Assemble the start solution vector for use within PETSc..*/
   ierr = VecSetValues(x,numnodes,cols,(PetscScalar*)u_approx,INSERT_VALUES); 
          CHKERRQ(ierr);  
   ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
   ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

   ierr = VecNorm(b,NORM_2,&bnrm2);CHKERRQ(ierr);
   if (debug)  
       printf("[PETSc]:The right-hand side norm = %e \n",bnrm2); 

   /*..Create linear solver context..*/  
   ierr = KSPCreate(MPI_COMM_WORLD,&ksp);CHKERRQ(ierr);

   /*..Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix..*/
   ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

   /*..Extract pc type from context..*/
   ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);

   /*..Customize tolerances..*/ 
   ierr = KSPSetTolerances(ksp,1e-12,1e-14,PETSC_DEFAULT,
          PETSC_DEFAULT);CHKERRQ(ierr);

    /*..Create user defined context for the shell preconditioner..*/ 
   ierr = SamgShellPCCreate(&samg_ctx);CHKERRQ(ierr);

   /*..Do the setup for the SAMG precondioner..*/
   ierr = SamgShellPCSetUp(samg_ctx,A);CHKERRQ(ierr); 

   /*..Give user defined preconditioner a name..*/ 
   ierr = PCShellSetName(pc,"SAMG (Scalar mode)");
          CHKERRQ(ierr); 

   /*..Parse SAMG hierarchy to PETSc variables..*/  
   levels = samg_ctx->LEVELS; 
   numnonzero = ia[numnodes]; 
   ierr = SamgGetGrid(levels, numnodes, numnonzero, grid, PETSC_NULL);

   /*..Print coarser grid and interpolation operators to file..*/ 
   ierr = PetscOptionsHasName(PETSC_NULL,"-samg_print",&issamg_print); 
	  CHKERRQ(ierr);
   if (issamg_print){   
     for (k=2;k<=levels;k++){ 
	 sprintf(pathfilename,"./"); 
	 sprintf(basefilename,"Pcoarsemat.%02u",k); 
	 PrintMatrix(grid[k].A, pathfilename, basefilename); 
     }
     for (k=1;k<=levels-1;k++){ 
	 sprintf(basefilename,"Pinterpol.%02u%02",k, k-1); 
	 ierr = PrintMatrix(grid[k].Interp, pathfilename, basefilename); 
     }
   }
    
   /*..Perform check on parsing..*/ 
   ierr = PetscOptionsHasName(PETSC_NULL,"-samg_check",&issamg_print); 
	  CHKERRQ(ierr);
   if (issamg_print)
     ierr = SamgCheckGalerkin(levels, A, grid, PETSC_NULL); 
      
   /*..Set KSP solver type..*/ 
   ierr = KSPSetType(ksp,KSPRICHARDSON);CHKERRQ(ierr);  
   ierr = KSPSetMonitor(ksp,KSPDefaultMonitor,PETSC_NULL, PETSC_NULL); 
          CHKERRQ(ierr); 
   /*..Set MG preconditioner..*/
   ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
   ierr = PCMGSetLevels(pc,levels, PETSC_NULL);CHKERRQ(ierr);
   ierr = PCMGSetType(pc, MGMULTIPLICATIVE);CHKERRQ(ierr);
   ierr = PCMGSetCycles(pc, 1);CHKERRQ(ierr);
   ierr = PCMGSetNumberSmoothUp(pc,1);CHKERRQ(ierr);
   ierr = PCMGSetNumberSmoothDown(pc,1);CHKERRQ(ierr);

   /*....Set smoother, work vectors and residual calculation on each 
         level....*/ 
   for (k=1;k<=levels;k++){ 
       petsc_level = levels - k; 
       /*....Get pre-smoothing KSP context....*/ 
       ierr = PCMGGetSmootherDown(pc,petsc_level,&grid[k].ksp_pre); 
              CHKERRQ(ierr); 
       ierr = PCMGGetSmootherUp(pc,petsc_level,&grid[k].ksp_post); 
	      CHKERRQ(ierr); 
       if (k==1)
          FineLevelMatrix = A; 
          else
          FineLevelMatrix = grid[k].A; 
       ierr = MatGetSize(FineLevelMatrix, &size, &size);CHKERRQ(ierr);
       ierr = VecCreate(MPI_COMM_WORLD,&grid[k].x);CHKERRQ(ierr);
       ierr = VecSetSizes(grid[k].x,PETSC_DECIDE,size);
       ierr = VecSetType(grid[k].x,VECSEQ);CHKERRQ(ierr);
       ierr = VecDuplicate(grid[k].x,&grid[k].b);CHKERRQ(ierr); 
       ierr = VecDuplicate(grid[k].x,&grid[k].r);CHKERRQ(ierr); 

       /*....set ksp_pre context....*/ 
       ierr = KSPSetOperators(grid[k].ksp_pre, FineLevelMatrix, 
                               FineLevelMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRQ(ierr);
       ierr = KSPGetPC(ksp_pre_pre,&pc_pre);CHKERRQ(ierr);
       ierr = KSPSetType(grid[k].ksp_pre, KSPRICHARDSON);CHKERRQ(ierr);
       ierr = KSPSetTolerances(grid[k].ksp_pre, 1e-12, 1e-50, 1e7,1); 
              CHKERRQ(ierr); 
       ierr = PCSetType(pc_pre, PCSOR);CHKERRQ(ierr);   
       ierr = PCSORSetSymmetric(pc_pre,SOR_FORWARD_SWEEP);CHKERRQ(ierr); 

       /*....set ksp_post context....*/  
       ierr = KSPSetOperators(grid[k].ksp_post, FineLevelMatrix, 
                               FineLevelMatrix, DIFFERENT_NONZERO_PATTERN); 
              CHKERRQ(ierr);
       ierr = KSPGetPC(grid[k].ksp_post,&pc_post);CHKERRQ(ierr);
       ierr = KSPSetInitialGuessNonzero(grid[k].ksp_post, PETSC_TRUE);CHKERRQ(ierr);
       ierr = KSPSetType(grid[k].ksp_post, KSPRICHARDSON);CHKERRQ(ierr);
       ierr = KSPSetTolerances(grid[k].ksp_post, 1e-12, 1e-50, 1e7,1);
              CHKERRQ(ierr);  
       ierr = PCSetType(pc_post, PCSOR);CHKERRQ(ierr);   
       ierr = PCSORSetSymmetric(pc_post,SOR_BACKWARD_SWEEP);CHKERRQ(ierr);   

       ierr = PCMGSetX(pc,petsc_level,grid[k].x);CHKERRQ(ierr); 
       ierr = PCMGSetRhs(pc,petsc_level,grid[k].b);CHKERRQ(ierr); 
       ierr = PCMGSetR(pc,petsc_level,grid[k].r);CHKERRQ(ierr); 
       ierr = PCMGSetResidual(pc,petsc_level,PCMGDefaultResidual,FineLevelMatrix); 
              CHKERRQ(ierr);
   }

   /*....Create interpolation between the levels....*/   
   for (k=1;k<=levels-1;k++){
     petsc_level = levels - k; 
     ierr = PCMGSetInterpolate(pc,petsc_level,grid[k].Interp);CHKERRQ(ierr);
     ierr = PCMGSetRestriction(pc,petsc_level,grid[k].Interp);CHKERRQ(ierr);  
   }

   /*....Set coarse grid solver....*/ 
   ierr = PCMGGetCoarseSolve(pc,&coarsegridksp);CHKERRQ(ierr); 
   ierr = KSPSetFromOptions(coarsegridksp);CHKERRQ(ierr);
   ierr = KSPSetOperators(coarsegridksp, grid[levels].A, grid[levels].A, 
                           DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
   ierr = KSPGetPC(coarsegridksp,&coarsegridpc);CHKERRQ(ierr);
   ierr = KSPSetType(coarsegridksp, KSPPREONLY);CHKERRQ(ierr);
   ierr = PCSetType(coarsegridpc, PCLU);CHKERRQ(ierr); 

   /*..Allow above criterea to be overwritten..*/
   ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr); 

   /*..Indicate that we are going to use a non-zero initial solution..*/
   ierr = KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);CHKERRQ(ierr);

   /*..Get end time of linear system setup..*/ 
   ierr = PetscGetTime(&v2);CHKERRQ(ierr); 
   t_setup = v2 - v1;  

   /*..Get start time of linear solve..*/ 
   ierr = PetscGetTime(&v1);CHKERRQ(ierr); 

   /*..Solve linear system..*/
   ierr = KSPSolve(ksp,b,x);CHKERRQ(ierr);
   ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
 
   /*..Print number of iterations..*/ 
   PetscPrintf(PETSC_COMM_WORLD,"\n** Number of iterations done = %d \n",
               its);   

   /*..Get end time of linear solve..*/ 
   ierr = PetscGetTime(&v2);CHKERRQ(ierr); 
   t_solve = v2 - v1;  

   printf("\n[PETSc]:Time spend in setup = %e \n",t_setup);
   printf("[PETSc]:Time spend in solve = %e \n",t_solve);
   printf("[PETSc]:Total time = %e \n\n", t_setup + t_solve);
   
   /*..Copy PETSc solution back..*/
   ierr= VecGetArray(x, &sol_array);CHKERRQ(ierr);
   for (I=0;I<numnodes;I++){
     u_approx[I] = sol_array[I]; 
   }
   ierr = VecRestoreArray(x,&sol_array);

   if (debug){
      printf("[PETSc]:The solution \n");
      ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      printf("\n"); 
   }

   /*..Free work space..*/
   ierr = SamgShellPCDestroy(samg_ctx);CHKERRQ(ierr);
   ierr = VecDestroy(x);CHKERRQ(ierr);
   ierr = VecDestroy(b);CHKERRQ(ierr); 
   ierr = MatDestroy(A);CHKERRQ(ierr); 
   ierr = KSPDestroy(ksp);CHKERRQ(ierr);
   for (k=2;k<=levels;k++){ 
     ierr = MatDestroy(grid[k].A);CHKERRQ(ierr); 
   }
   for (k=1;k<=levels-1;k++){ 
     ierr = MatDestroy(grid[k].Interp);CHKERRQ(ierr); 
   }
   for (k=1;k<=levels-1;k++){ 
     ierr = VecDestroy(grid[k].b);CHKERRQ(ierr); 
     ierr = VecDestroy(grid[k].x);CHKERRQ(ierr); 
     ierr = VecDestroy(grid[k].r);CHKERRQ(ierr); 
   }

   PetscFinalize();   
   
   return 0; 
}
