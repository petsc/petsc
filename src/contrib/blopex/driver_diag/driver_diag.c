/* @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ */
/* @@@ BLOPEX (version 1.1) LGPL Version 2.1 or above.See www.gnu.org. */
/* @@@ Copyright 2010 BLOPEX team http://code.google.com/p/blopex/     */
/* @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ */

/* This is an example of creation of a matrix within Petsc and solution via lobpcg 
 * use to generate very large sparse diagonal matrix of size NxN via parameter -N xxxx */

static char help[] = "'Diagonal' test driver for 'abstract lobpcg' in PETSC\n\
Usage: mpirun -np <procs> driver_fiedler [-help] [all PETSc options]\n\
Special options:\n\
-n_eigs <integer>      Number of eigenvalues to calculate\n\
-tol <real number>     absolute tolerance for residuals\n\
-full_out              Produce more output\n\
-seed <integer>        seed for random number generator\n\
-itr <integer>         Maximal number of iterations\n\
-output_file <string>  Filename to write calculated eigenvectors.\n\
-shift <real number>   Apply shift to 'stiffness' matrix\n\
-N <integer>           Size of Matrix to create\n\
Example:\n\
mpirun -np 2 ./driver_diag -N 10000 -n_eigs 3 -tol 1e-6 -itr 20\n";

#include "petscksp.h"
#include <assert.h>
#include "fortran_matrix.h"
#include "fortran_interpreter.h"
#include "lobpcg.h"

#ifdef BLOPEX_DIR
#include "petsc-interface.h"
#else
#include "../src/contrib/blopex/petsc-interface/petsc-interface.h"
#endif

#include "interpreter.h"
#include "multivector.h"
#include "temp_multivector.h"

typedef struct
{
  KSP                      ksp;
  Mat                      A;
  mv_InterfaceInterpreter  ii;
} aux_data_struct;

void Precond_FnSingleVector(void * data, void * x, void * y)
{
      PetscErrorCode     ierr;

      ierr = KSPSolve(((aux_data_struct*)data)->ksp, (Vec)x, (Vec)y);
      assert(!ierr);
}

void Precond_FnMultiVector(void * data, void * x, void * y)
{
   ((aux_data_struct*)data)->ii.Eval(Precond_FnSingleVector, data, x, y);
}

void OperatorASingleVector (void * data, void * x, void * y)
{
   PetscErrorCode     ierr;

   ierr = MatMult(((aux_data_struct*)data)->A, (Vec)x, (Vec)y);
   assert(!ierr);
}

void OperatorAMultiVector(void * data, void * x, void * y)
{
   ((aux_data_struct*)data)->ii.Eval(OperatorASingleVector, data, x, y);
}


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
   Vec            u;
   Mat            A;

   PetscInt       N = 1000;
   PetscScalar    value;

   PetscErrorCode ierr;

   mv_MultiVectorPtr          eigenvectors;
   mv_TempMultiVector*        raw_eigenvectors;

   PetscScalar *              eigs;
   PetscScalar *              eigs_hist;
   double *                   resid;
   double *                   resid_hist;
   int                        iterations;
   PetscMPIInt                rank;
   int                        n_eigs = 3;
   int                        seed = 1;
   PetscInt                   i;
   int                        j;
   PetscLogDouble             t1,t2,elapsed_time;
   double                     tol=1e-06;
   PetscTruth                 full_output=PETSC_FALSE;
   KSP                        ksp;
   lobpcg_Tolerance           lobpcg_tol;
   int                        maxIt = 100;
   mv_InterfaceInterpreter    ii;
   lobpcg_BLASLAPACKFunctions blap_fn;
   aux_data_struct            aux_data;
   PetscViewer                fd;    /* viewer */

   PetscTruth                 option_present;
   PetscReal                  shift=0;
   PetscTruth                 shift_present;
   char                       output_filename[PETSC_MAX_PATH_LEN];
   PetscTruth                 output_filename_present;
   char                       tmp_str[PETSC_MAX_PATH_LEN];
   PetscInt                   tmp_int;

   PetscInitialize(&argc,&args,(char *)0,help);

   /* read command-line parameters */
   ierr = PetscOptionsGetInt(PETSC_NULL,"-n_eigs",&tmp_int,&option_present);CHKERRQ(ierr);
   if (option_present)
      n_eigs = tmp_int;
   ierr = PetscOptionsGetReal(PETSC_NULL,"-tol", &tol,PETSC_NULL); CHKERRQ(ierr);
   ierr = PetscOptionsHasName(PETSC_NULL,"-full_out",&full_output); CHKERRQ(ierr);
   ierr = PetscOptionsGetInt(PETSC_NULL,"-seed",&tmp_int,&option_present);CHKERRQ(ierr);
   if (option_present)
     seed = tmp_int;
   if (seed<1)
     seed=1;
   ierr = PetscOptionsGetInt(PETSC_NULL,"-itr",&tmp_int,&option_present);CHKERRQ(ierr);
   if (option_present)
     maxIt = tmp_int;
   ierr = PetscOptionsGetReal(PETSC_NULL,"-shift",&shift,&shift_present);
   ierr = PetscOptionsGetString(PETSC_NULL,"-output_file",output_filename,
            PETSC_MAX_PATH_LEN-1, &output_filename_present);
   ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&tmp_int,&option_present);CHKERRQ(ierr);
   if (option_present)
     N = tmp_int;


   /* Create matrix */
   ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,N,N,1,PETSC_NULL,0,PETSC_NULL,&A);
   CHKERRQ(ierr);
   for (i=0; i<N; i++) {
      value = i+1;
      ierr = MatSetValues(A,1,&i,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
      }
   ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
   ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

   /* printf("\nMatrix A:\n\n"); MatView(A,0); */


   /* apply shift to stiffness matrix if asked to do so */
   if (shift_present) {
      ierr = MatShift(A,shift);
      CHKERRQ(ierr);
      }

   /*
    Create parallel vectors.
     - We form 1 vector from scratch and then duplicate as needed.
   */

   MatGetVecs(A,&u,NULL);

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               Create the linear solver and set various options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 /* Here we START measuring time for preconditioner setup */
   ierr = PetscGetTime(&t1);CHKERRQ(ierr);

   /*
      Create linear solver context
   */
   ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);

   /*
      Set operators. Here the matrix that defines the linear system
      also serves as the preconditioning matrix.
   */
   ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

   /*
     Set runtime options, e.g.,
         -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
     These options will override those specified above as long as
     KSPSetFromOptions() is called _after_ any other customization
     routines.
   */
   ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

   /* probably this call actually builds the preconditioner */
   ierr = KSPSetUp(ksp);CHKERRQ(ierr);

   /* Here we STOP measuring time for preconditioner setup */
   ierr = PetscGetTime(&t2);CHKERRQ(ierr);
   elapsed_time=t2-t1;

   PetscPrintf(PETSC_COMM_WORLD,"Preconditioner setup, seconds: %f\n",elapsed_time);

   /* request memory for eig-vals */
   ierr = PetscMalloc(sizeof(PetscScalar)*n_eigs,&eigs); CHKERRQ(ierr);

   /* request memory for eig-vals history */
   ierr = PetscMalloc(sizeof(PetscScalar)*n_eigs*(maxIt+1),&eigs_hist); CHKERRQ(ierr);

   /* request memory for resid. norms */
   ierr = PetscMalloc(sizeof(double)*n_eigs,&resid); CHKERRQ(ierr);

   /* request memory for resid. norms hist. */
   ierr = PetscMalloc(sizeof(double)*n_eigs*(maxIt+1),&resid_hist); CHKERRQ(ierr);

   LOBPCG_InitRandomContext();

   MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

   PETSCSetupInterpreter( &ii );
   eigenvectors = mv_MultiVectorCreateFromSampleVector(&ii, n_eigs,u);
 
   for (i=0; i<seed; i++) /* this cycle is to imitate changing random seed */
      mv_MultiVectorSetRandom (eigenvectors, 1234);

   lobpcg_tol.absolute = tol;
   lobpcg_tol.relative = 1e-50;

   #ifdef PETSC_USE_COMPLEX
      blap_fn.zpotrf = PETSC_zpotrf_interface;
      blap_fn.zhegv = PETSC_zsygv_interface;
   #else
      blap_fn.dpotrf = PETSC_dpotrf_interface;
      blap_fn.dsygv = PETSC_dsygv_interface;
   #endif

   aux_data.A = A;
   aux_data.ksp = ksp;
   aux_data.ii = ii;

   /* Here we START measuring time for solution process */
   ierr = PetscGetTime(&t1);CHKERRQ(ierr);

#ifdef PETSC_USE_COMPLEX
   lobpcg_solve_complex
   (
      eigenvectors,
      &aux_data,
      OperatorAMultiVector,
      NULL,
      NULL,
      &aux_data,
      Precond_FnMultiVector,
      NULL,   /* no constraints */
      blap_fn,
      lobpcg_tol,
      maxIt,
      !rank,
      &iterations,
      (komplex *) eigs,
      (komplex *) eigs_hist,
      n_eigs,
      resid,
      resid_hist,
      n_eigs
   );
#else
   lobpcg_solve_double
   (
      eigenvectors,
      &aux_data,
      OperatorAMultiVector,
      NULL,
      NULL,
      &aux_data,
      Precond_FnMultiVector,
      NULL,  /* no constraints */
      blap_fn,
      lobpcg_tol,
      maxIt,
      !rank,
      &iterations,
        eigs,
      eigs_hist,
      n_eigs,
      resid,
      resid_hist,
      n_eigs
   );
   #endif

   /* Here we STOP measuring time for solution process */
   ierr = PetscGetTime(&t2);CHKERRQ(ierr);
   elapsed_time=t2-t1;

   PetscPrintf(PETSC_COMM_WORLD,"Solution process, seconds: %e\n",elapsed_time);

   /* shift eigenvalues back */
   if (shift_present)
   {
   for (i=0; i<n_eigs; i++)
      eigs[i]-=shift;
   }

   PetscPrintf(PETSC_COMM_WORLD,"Final eigenvalues:\n");
   for (i=0;i<n_eigs;i++)
     {
         ierr = PetscPrintf(PETSC_COMM_WORLD,"%e\n",PetscRealPart(eigs[i]));
         CHKERRQ(ierr);
     }

   if (full_output)
   {
     PetscPrintf(PETSC_COMM_WORLD,"Output from LOBPCG, eigenvalues history:\n");
     for (j=0; j<iterations+1; j++)
       for (i=0;i<n_eigs;i++)
       {
         ierr = PetscPrintf(PETSC_COMM_WORLD,"%e\n",PetscRealPart(*(eigs_hist+j*n_eigs+i)));
         CHKERRQ(ierr);
         }
     PetscPrintf(PETSC_COMM_WORLD,"Output from LOBPCG, residual norms:\n");
     for (i=0;i<n_eigs;i++)
       {
           ierr = PetscPrintf(PETSC_COMM_WORLD,"%e\n",resid[i]);
           CHKERRQ(ierr);
       }

     PetscPrintf(PETSC_COMM_WORLD,"Output from LOBPCG, residual norms history:\n");
     for (j=0; j<iterations+1; j++)
       for (i=0;i<n_eigs;i++)
       {
         ierr = PetscPrintf(PETSC_COMM_WORLD,"%e\n",*(resid_hist+j*n_eigs+i));
         CHKERRQ(ierr);
       }
   }

   /* write eigenvectors to disk, if told to do so */

   if (output_filename_present)
   {
      raw_eigenvectors = (mv_TempMultiVector*)mv_MultiVectorGetData (eigenvectors);
      for ( j = 0; j < n_eigs; j++ )
      {
        sprintf( tmp_str, "%s_%d.petsc", output_filename, j );
        PetscViewerBinaryOpen(PETSC_COMM_WORLD, tmp_str, FILE_MODE_WRITE, &fd);
        /* PetscViewerSetFormat(fd,PETSC_VIEWER_ASCII_MATLAB); */
        ierr = VecView((Vec)(raw_eigenvectors->vector)[j],fd); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);
      }

   }

   /*
      Free work space.  All PETSc objects should be destroyed when they
      are no longer needed.
   */
   ierr = VecDestroy(u);CHKERRQ(ierr);
   ierr = MatDestroy(A);CHKERRQ(ierr);
   ierr = KSPDestroy(ksp);CHKERRQ(ierr);

   LOBPCG_DestroyRandomContext();
   mv_MultiVectorDestroy(eigenvectors);
   /* mv_MultiVectorDestroy(constraints); */

   /* free memory used for eig-vals */
   ierr = PetscFree(eigs);
   CHKERRQ(ierr);
   ierr = PetscFree(eigs_hist);
   CHKERRQ(ierr);
   ierr = PetscFree(resid);
   CHKERRQ(ierr);
   ierr = PetscFree(resid_hist);
   CHKERRQ(ierr);

   ierr = PetscFinalize();CHKERRQ(ierr);
   return 0;
}
