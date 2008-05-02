#include "taolinesearch.h"

typedef struct {
    int n;
    PetscScalar alpha;
} AppCtx;

PetscErrorCode MyFuncGrad(TaoLineSearch ls, Vec X, PetscScalar *f, Vec G, void *ctx);

int main(int argc, char *argv[])
{
    TaoLineSearch ls;
    TaoLineSearchTerminationReason reason;
    Vec x,g,s;
    PetscScalar f;
    AppCtx user;
    int info;
    
	
    info = PetscInitialize(&argc, &argv,0,0); CHKERRQ(info);

    user.n=2; user.alpha = 99.0;
    info = VecCreateSeq(PETSC_COMM_SELF,user.n,&x); CHKERRQ(info);
    info = VecDuplicate(x,&g); CHKERRQ(info);
    info = VecDuplicate(x,&s); CHKERRQ(info);

    info = VecSet(x,0.0); CHKERRQ(info);
    info = VecSet(s,0.0); CHKERRQ(info);
    f = 1.0;
    info = VecSetValue(g,0,-1.0, INSERT_VALUES); CHKERRQ(info);
    info = VecSetValue(g,1,0.0, INSERT_VALUES);CHKERRQ(info);
    info = VecAssemblyBegin(g); CHKERRQ(info);
    info = VecAssemblyEnd(g); CHKERRQ(info);
    info = VecAXPY(s,-1.0,g); CHKERRQ(info);

    info = TaoLineSearchCreate(PETSC_COMM_SELF,&ls); CHKERRQ(info);
    info = TaoLineSearchSetFromOptions(ls); CHKERRQ(info);
    info = TaoLineSearchSetObjectiveGradient(ls,MyFuncGrad,(void*)&user); 
    CHKERRQ(info);
    info = TaoLineSearchApply(ls,x,f,g,s); CHKERRQ(info);
    info = TaoLineSearchGetSolution(ls,x,&f,g,&reason); CHKERRQ(info);
    
    info = TaoLineSearchDestroy(ls); CHKERRQ(info);
    info = PetscFinalize();
    return(0);
}


PetscErrorCode MyFuncGrad(TaoLineSearch ls, Vec X, PetscScalar *f, Vec G, void *ctx)
{
    AppCtx *user = (AppCtx*)ctx;
    PetscErrorCode    info;
    PetscInt i,nn=user->n/2;
    double ff=0,t1,t2,alpha=user->alpha;
    PetscScalar *x,*g;

    /* Get pointers to vector data */
    info = VecGetArray(X,&x); CHKERRQ(info);
    info = VecGetArray(G,&g); CHKERRQ(info);

    /* Compute G(X) */
    for (i=0; i<nn; i++){
	t1 = x[2*i+1]-x[2*i]*x[2*i]; t2= 1-x[2*i];
	ff += alpha*t1*t1 + t2*t2;
	g[2*i] = -4*alpha*t1*x[2*i]-2.0*t2;
	g[2*i+1] = 2*alpha*t1;
    }

    /* Restore vectors */
    info = VecRestoreArray(X,&x); CHKERRQ(info);
    info = VecRestoreArray(G,&g); CHKERRQ(info);
    *f=ff;

    info = PetscLogFlops(nn*15); CHKERRQ(info);
    return 0;
}
