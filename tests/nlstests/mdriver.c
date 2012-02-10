#include "taosolver.h"

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
    PetscInt nprob; /* problem number (1 == Linear, Full Rank, etc.) */
    PetscInt nfev;
    PetscReal factor;
    PetscInt nrun; /* run number  */
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
    FILE *datfile,*parfile;
    double wctime1, wctime2;
    double wctime;
    double *x,*f;
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
	
        user.factor = PetscPowScalar(10,nstart);
	
	ierr = VecCreateSeq(PETSC_COMM_SELF,n,&X); CHKERRQ(ierr);
	ierr = VecCreateSeq(PETSC_COMM_SELF,m,&F); CHKERRQ(ierr);
	ierr = FormStartingPoint(X,&user); CHKERRQ(ierr);
	printf(" Problem %d: %30s\n",nprob,dfonames[nprob]);
	printf(" Number of components: %d\n",user.n);
	printf(" Number of variables: %d\n",user.m);

	wallclock_(&wctime1);
	
	parfile = fopen("parameters.m","w");
	fprintf(parfile,"m=%d;\n",user.m);
	fprintf(parfile,"n=%d;\n",user.n);
	fprintf(parfile,"X0=[\n");
	ierr = VecGetArray(X,&x); CHKERRQ(ierr);
	for (i=0;i<n;i++) {
	  fprintf(parfile,"%g20.12\n",x[i]);
	}
	fprintf(parfile,"];\n");
	fclose(parfile);

	system("matlab < runpounders.m");
	wallclock_(&wctime2);
	wctime = wctime2 - wctime1;
	printf("time = %g\n",wctime);

	fscanf(datfile,"%d %d %d %d\n",&nprob,&n,&m,&nstart);
    }
    fclose(datfile);
    TaoFinalize();
    PetscFinalize();
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormStartingPoint"
PetscErrorCode FormStartingPoint(Vec X, AppCtx *ctx) {
    double *x;
    PetscErrorCode ierr;
    PetscFunctionBegin;
    ierr = VecGetArray(X,&x); CHKERRQ(ierr);
    dfoxs_(&ctx->n,x,&ctx->nprob,&ctx->factor);
    ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
    ctx->nfev = 0;
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
