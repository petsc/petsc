#include "taosolver.h"
#include <math.h>  /*  For pow(), fabs(), log(), and exp()  */

#define MMAX 1000
#define NMAX 100
#define NFMAX 1500
#define RUNMAX 100
#define LMAX (RUNMAX+11)*(RUNMAX+NMAX)+NMAX*(3*NMAX+11)/2


char dfonames[][64] = {
    "Linear, full rank",
    "Linear, rank 1",
    "Linear, rank 1, zero row/cols",
    "Rosenbrock",
    "Helical Valley",
    "Powell singular",
    "Freudenstein and Roth",
    "Bard",
    "Kowalik and Osborne",
    "Meyer",
    "Watson",
    "Box 3-dimensional",
    "Jenrich and Sampson",
    "Brown and Dennis",
    "Chebyquad",
    "Brown almost-linear",
    "Osborne 1",
    "Osborne 2",
    "Bdqrtic",
    "Cube",
    "Mancino",
    "Heart 8"
};




typedef struct {
    PetscInt n;
    PetscInt m;
    PetscInt nprob; // problem number (1 == Linear, Full Rank, etc.)
    PetscInt nfev;
    PetscReal factor;
    PetscInt nrun; // run number (
    double delta;
    PetscReal fevals[NFMAX][RUNMAX];
} AppCtx;    

PetscErrorCode EvaluateFunction(TaoSolver, Vec, Vec, void *);
PetscErrorCode FormStartingPoint(Vec, AppCtx *);
void dfovec_(int *m, int *n, double *x, double *f, int *nprob);
void dfoxs_(int *n, double *x, int *nprob, double *factor);
void wallclock_(double *time);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
    AppCtx user;
    FILE *datfile;
    double wctime1, wctime2;
    double wctime;
    int nprob,n,m,nstart;
    TaoSolver tao;
    PetscErrorCode ierr;
    Vec X,F;

    PetscInitialize(&argc, &argv, 0,0);
    TaoInitialize(&argc, &argv,0,0);

    datfile = fopen("dfo.dat","r");
    user.nrun=0;
    fscanf(datfile,"%d %d %d %d\n",&nprob,&n,&m,&nstart);
    while (nprob != 0) {

	user.nprob = nprob;
	user.n = n;
	user.m = m;
	user.nrun++;
	
        user.factor = pow(10,nstart);
	
	ierr = VecCreateSeq(PETSC_COMM_SELF,n,&X); CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF,m,&F); CHKERRQ(ierr);
	ierr = FormStartingPoint(X,&user); CHKERRQ(ierr);
	printf(" Problem %2d: %30s\n",nprob,dfonames[nprob]);
	printf(" Number of components: %12d\n",user.n);
	printf(" Number of variables: %12d\n",user.m);

	wallclock_(&wctime1);
	ierr = TaoSolverCreate(PETSC_COMM_SELF,&tao); CHKERRQ(ierr);
	ierr = TaoSolverSetType(tao,"tao_pounders"); CHKERRQ(ierr);
	ierr = TaoSolverSetInitialVector(tao,X); CHKERRQ(ierr);
	ierr = TaoSolverSetSeparableObjectiveRoutine(tao,F,EvaluateFunction,
						     (void*)&user); CHKERRQ(ierr);
	ierr = TaoSolverSetFromOptions(tao); CHKERRQ(ierr);
	
	ierr = TaoSolverSolve(tao); CHKERRQ(ierr);
	ierr = TaoSolverView(tao,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);
	ierr = TaoSolverDestroy(&tao); CHKERRQ(ierr);
	
	ierr = VecDestroy(&X); CHKERRQ(ierr);
	ierr = VecDestroy(&F); CHKERRQ(ierr);
	wallclock_(&wctime2);
	wctime = wctime2 - wctime1;
	printf("time = %f\n",wctime);

	fscanf(datfile,"%d %d %d %d\n",&nprob,&n,&m,&nstart);
    }

    TaoFinalize();
    PetscFinalize();
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormStartingPoint"
PetscErrorCode FormStartingPoint(Vec X, AppCtx *ctx) {
    double *x;
    char str[32];
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = VecGetArray(X,&x); CHKERRQ(ierr);
    dfoxs_(&ctx->n,x,&ctx->nprob,&ctx->factor);
    ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
    ctx->nfev = 0;
    //ierr = VecNorm(X,NORM_INFINITY,&ctx->delta);
    //ctx->delta = PetscMax(ctx->delta,0.01);
    //snprintf(str,32,"%31f",ctx->delta);
    //ierr = PetscOptionsSetValue("-tao_pounders_delta",str); CHKERRQ(ierr);
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "EvaluateFunction"
PetscErrorCode EvaluateFunction(TaoSolver tao, Vec X, Vec F, void *vctx)
{
    AppCtx *ctx = (AppCtx*)vctx;
    double *x,*f,fnrm;
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = VecGetArray(X,&x); CHKERRQ(ierr);
    ierr = VecGetArray(F,&f); CHKERRQ(ierr);
    dfovec_(&ctx->m,&ctx->n,x,f,&ctx->nprob);
    ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&f); CHKERRQ(ierr);

    ierr = VecNorm(F,NORM_2,&fnrm); CHKERRQ(ierr);
    fnrm *= fnrm;
    if (PetscIsInfOrNanReal(fnrm)) fnrm=1.0e64;
    fnrm = PetscMin(fnrm,1.0e64);

    ctx->fevals[ctx->nfev][ctx->nrun] = fnrm;
    ctx->nfev++;
    PetscFunctionReturn(0);
}
