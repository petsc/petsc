#define PCBOOMERAMG   "boomeramg"
#define PCEUCLID      "euclid"
#define PCSPBASICC    "spbasicc"

PetscErrorCode PETSCKSP_DLLEXPORT KSPLSQRGetArnorm( KSP ksp,
     PetscReal *arnorm, PetscReal *rhs_norm , PetscReal *anorm);
PetscErrorCode precond_lsqr_monitor(
                  KSP       solksp, /* Krylov Subspace method context */
                  PetscInt  iter,   /* Current iteration number */
                  PetscReal rnorm,  /* Current residual norm */
                  void      *ctx    /* Pointer to user defined context */
                  );

PetscErrorCode read_command_line( int argc, char **argv,
     int * order, int *maxit, char ** dirname, PetscScalar * droptol, 
     PetscScalar * epsdiag, PCType *pc_type);

PetscErrorCode precond_lsqr_converged(
			KSP       solksp, /* Krylov Subspace method context */
			PetscInt  iter,   /* Current iteration number */
			PetscReal rnorm,  /* Current residual norm */
			KSPConvergedReason *reason, /* duh... */
			void      *ctx    /* Pointer to user defined context */
		       );

