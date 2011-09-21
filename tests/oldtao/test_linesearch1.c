#include "tao.h"
#include "src/petsctao/vector/taovec_petsc.h"

typedef struct {
    int n;
    PetscReal alpha;
} AppCtx;

PetscErrorCode MyFuncGrad(TAO_APPLICATION myapp, Vec X, PetscReal *f, Vec G, void *ctx);

int main(int argc, char *argv[])
{ 
    TAO_SOLVER tao;
    TAO_APPLICATION app;
    Vec x,g,s,w;
    TaoVec *X, *G, *S, *W;
    PetscReal f,f_full,step;
    TaoInt flg;
    AppCtx user;
    int info;
    
	
    info = PetscInitialize(&argc, &argv,0,0); CHKERRQ(info);
    info = TaoInitialize(&argc, &argv, 0, 0); CHKERRQ(info);
    user.n=2; user.alpha = 99.0;
    info = VecCreate(PETSC_COMM_WORLD,&x); CHKERRQ(info);
    info = VecSetSizes(x,PETSC_DECIDE,2); CHKERRQ(info);
    info = VecSetFromOptions(x); CHKERRQ(info);

    info = VecDuplicate(x,&g); CHKERRQ(info);
    info = VecDuplicate(x,&s); CHKERRQ(info);
    info = VecDuplicate(x,&w); CHKERRQ(info);

    info = VecSet(x,0.0); CHKERRQ(info);
    info = VecSet(s,0.0); CHKERRQ(info);
    f = 1.0;
    info = VecSetValue(g,0,-1.0, INSERT_VALUES); CHKERRQ(info);
    info = VecSetValue(g,1,0.0, INSERT_VALUES);CHKERRQ(info);
    info = VecAssemblyBegin(g); CHKERRQ(info);
    info = VecAssemblyEnd(g); CHKERRQ(info);
    info = VecAXPY(s,-1.0,g); CHKERRQ(info);
    X = new TaoVecPetsc(x);
    G = new TaoVecPetsc(g);
    S = new TaoVecPetsc(s);
    W = new TaoVecPetsc(w);
    info = TaoCreate(PETSC_COMM_WORLD,"tao_lmvm",&tao); CHKERRQ(info);
    info = TaoApplicationCreate(PETSC_COMM_WORLD,&app); CHKERRQ(info);
    
    info = TaoAppSetObjectiveAndGradientRoutine(app,MyFuncGrad,(void*)&user); 
    CHKERRQ(info);
    info = TaoAppSetInitialSolutionVec(app,x); CHKERRQ(info);

    info = TaoCreateMoreThuenteLineSearch(tao,0.0001, 0.9); CHKERRQ(info);
//    info = TaoCreateUnitLineSearch(tao); CHKERRQ(info);
    info = TaoSetOptions(app,tao); CHKERRQ(info);
    info = TaoSetApplication(tao,app); CHKERRQ(info);
    info = TaoLineSearchSetUp(tao); CHKERRQ(info);
    step=1.0;
    info = TaoLineSearchApply(tao, X, G, S, W, &f, &f_full, &step, &flg);
    info = TaoLineSearchView(tao); CHKERRQ(info);
    info = PetscPrintf(PETSC_COMM_WORLD,"Status: %d\n",flg); CHKERRQ(info);
    info = PetscPrintf(PETSC_COMM_WORLD,"Step length: %G\n",step); CHKERRQ(info);
    info = PetscPrintf(PETSC_COMM_WORLD,"New Obj value: %G\n",f); CHKERRQ(info);
    info = PetscPrintf(PETSC_COMM_WORLD,"New gradient:\n"); CHKERRQ(info);
    info = G->View();

    info = TaoDestroy(tao); CHKERRQ(info);
    info = TaoAppDestroy(app); CHKERRQ(info);
    delete X;
    delete G;
    delete S;
    delete W;
    info = VecDestroy(&x); CHKERRQ(info);
    info = VecDestroy(&g); CHKERRQ(info);
    info = VecDestroy(&s); CHKERRQ(info);
    info = VecDestroy(&w); CHKERRQ(info);

    info = TaoFinalize();
    info = PetscFinalize();
    return(0);
}


PetscErrorCode MyFuncGrad(TAO_APPLICATION tao, Vec X, PetscReal *f, Vec G, void *ctx)
{
    AppCtx *user = (AppCtx*)ctx;
    PetscErrorCode    info;
    PetscInt i,nn=user->n/2;
    double ff=0,t1,t2,alpha=user->alpha;
    PetscReal *x,*g;

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
