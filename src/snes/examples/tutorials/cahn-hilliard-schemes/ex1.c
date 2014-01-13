
static char help[] = "Solves the Cahn-Hilliard equation u_t = (b(u)( -gamma u_xx + Psi'(u) )_x)_x in 1D.\n";

/*
 b(u) = (1-u^2)
 Psi(u) = (theta/2)*( (1+u)ln((1+u)/2) + (1-u)ln((1-u)/2) ) + p(u)
 where p(u) = (1-u^2)^2/4
 
 Instead of using chemical potential w = -gamma u_xx + Psi'(u), which leads to
 the system
 u_t = (b(u)w_x)_x
 w   = -gamma u_xx + Psi'(u),
 
 we consider
 
 u_t = (b(u)w_x)_x + theta u_xx
 w = -gamma u_xx + p'(u)
 */


#include "petscsnes.h"
#include "petscdmda.h"

typedef struct {
    PetscScalar u,w;
} Field;

typedef struct{
    PetscScalar gamma,theta;
    PetscReal   dt,T;
    Vec         xold, b_xold;       /* X^{n-1}, b(U^{n-1}) */
}AppCtx;

PetscErrorCode ComputeMobility(DM,Vec,Vec);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode SetInitialConditions(DM,Vec);
PetscErrorCode GetParams(AppCtx*);



#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
    PetscErrorCode ierr;
    Vec            x;         /* solution */
    SNES           snes;      /* nonlinear solver context */
    DM             da;
    AppCtx         user;      /* holds problem specific data */
    Mat            J;
    PetscScalar    t=0.0;
    
    PetscInitialize(&argc,&argv, (char*)0, help);
    
    /* Get physics and time parameters */
    ierr = GetParams(&user);CHKERRQ(ierr);
    ierr = DMDACreate1d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE,-8,2,1,PETSC_NULL,&da);CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,0,"u");CHKERRQ(ierr);
    ierr = DMDASetFieldName(da,1,"w");CHKERRQ(ierr);
    
    /* Create global vectors from DM  */
    ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da,&user.xold);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(da,&user.b_xold);CHKERRQ(ierr);
    
    /* Get Jacobian matrix structure from the da for the entire thing, da1 */
    ierr = DMSetMatType(da,MATAIJ);CHKERRQ(ierr);
    ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
    
    ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,da);CHKERRQ(ierr);
    ierr = SNESSetFunction(snes,PETSC_NULL,FormFunction,&user);CHKERRQ(ierr);
    ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
    
    ierr = SetInitialConditions(da,x);CHKERRQ(ierr);
    
    while(t<user.T) {
        ierr = VecCopy(x,user.xold);CHKERRQ(ierr);
        ierr = ComputeMobility(da,user.xold,user.b_xold);CHKERRQ(ierr);
        ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
        ierr = VecView(x,PETSC_VIEWER_DRAW_(PETSC_COMM_WORLD));CHKERRQ(ierr);
        PetscInt its;
        ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"SNESVI solver converged at t = %5.4g in %d iterations\n",t,its);CHKERRQ(ierr);
        t = t + user.dt;
    }
    
    ierr = VecDestroy(&user.xold);CHKERRQ(ierr);
    ierr = VecDestroy(&user.b_xold);CHKERRQ(ierr);
    ierr = SNESDestroy(&snes);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
    PetscFinalize();
    return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeMobility"
PetscErrorCode ComputeMobility(DM da,Vec X,Vec bX)
{
    PetscErrorCode ierr;
    Field          *x,*bx;
    PetscInt       i,xs,xm;
    
    ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,bX,&bx);CHKERRQ(ierr);
    /*
     Get local grid boundaries
     */
    ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    
    for (i=xs; i<xs+xm; i++) {
        bx[i].w = 0.0;
        if (x[i].u<-1.0 || x[i].u>1.0) bx[i].u = 0.0;
        else bx[i].u = 1.0 - x[i].u*x[i].u;
    }
    ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,bX,&bx);CHKERRQ(ierr);
    PetscFunctionReturn(0);
    
    
}


#undef __FUNCT__
#define __FUNCT__ "SetInitialConditions"
PetscErrorCode SetInitialConditions(DM da,Vec Y)
{
    PetscErrorCode ierr;
    PetscInt       i,xs,xm,Mx;
    Field          *y;
    PetscReal      hx,x;
    
    PetscFunctionBegin;
    ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
    
    hx     = 1.0/(PetscReal)(Mx-1);
    
    ierr = DMDAVecGetArray(da,Y,&y);CHKERRQ(ierr);
    
    /*
     Get local grid boundaries
     */
    ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    
    for (i=xs; i<xs+xm; i++) {
        x = i*hx;
        y[i].w = 0.0;
        if (x < 1.0/3.0 - .05) y[i].u = 1.0;
        else if (PetscAbs(x - 1.0/3.0) <= .05) y[i].u = 20.0*(1.0/3.0 - x);
        else if (PetscAbs(x - .82) < .05) y[i].u = -20.0*PetscAbs(x - .82);
        else y[i].u = -1.0;
    }
    
    /*
     Restore vectors
     */
    ierr = DMDAVecRestoreArray(da,Y,&y);CHKERRQ(ierr);
    PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void* ctx)
{
    PetscErrorCode ierr;
    DM             da;
    PetscInt       Mx,N,xs,xm,i;
    PetscReal      hx;
    PetscScalar    temp_u1,temp_u2,temp_u3,temp_w1,temp_w2,temp_w3,b_cl,b_cc,b_cr;
    Field          *x,*f,*xold,*b_xold;
    Vec            localX;
    AppCtx         *user=(AppCtx*)ctx;
    PetscScalar    gamma = user->gamma, theta = user->theta;
    PetscReal      dt = user->dt;
    
    PetscFunctionBegin;
    ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
    ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
    ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,&N,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
    
    hx     = 1.0/(PetscReal)(Mx-1);
    
    /*
     Scatter ghost points to local vector,using the 2-step process
     DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
     ` */
    ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    
    /*
     Get pointers to vector data
     */
    ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,F,&f);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,user->xold,&xold);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,user->b_xold,&b_xold);CHKERRQ(ierr);
    
    /*
     Get local grid boundaries
     */
    ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    
    /*
     Compute function over the locally owned part of the grid
     */
    
    
    if (!xs) {
        b_cc = (b_xold[0].u + b_xold[1].u)*hx/2.0;
        b_cr = -1.0*b_cc;
        
        temp_u1 = gamma*(x[0].u-x[1].u)/hx + (hx/2)*x[0].u*x[0].u*x[0].u;
        temp_u2 = - (hx/2)*x[0].w;
        temp_u3 =  - (hx/2)*xold[0].u;
        temp_w1 = hx*x[0].u/(2*dt) + theta*(x[0].u-x[1].u)/hx;
        temp_w2 = b_cc*x[0].u + b_cr*x[1].u;
        temp_w3 = - hx*xold[0].u/(2*dt);
        f[0].u  = temp_u1 + temp_u2 + temp_u3;
        f[0].w  = temp_w1 + temp_w2 + temp_w3;
        xs++;
        xm--;
    }
    if (xs+xm == Mx) {
        b_cc = (b_xold[Mx-2].u + b_xold[Mx-1].u)*hx/2;
        b_cl = -1.0*b_cc;
        temp_u1 = gamma*(x[Mx-1].u-x[Mx-2].u)/hx + (hx/2)*x[Mx-1].u*x[Mx-1].u*x[Mx-1].u;
        temp_u2 = - (hx/2)*x[Mx-1].w;
        temp_u3 = - (hx/2)*xold[Mx-1].u;
        temp_w1 = hx*x[Mx-1].u/(2*dt) + theta*(x[Mx-1].u-x[Mx-2].u)/hx;
        temp_w2 = b_cl*x[Mx-2].u + b_cc*x[Mx-1].u;
        temp_w3 = - hx*xold[Mx-1].u/(2*dt);
        f[Mx-1].u = temp_u1 + temp_u2 + temp_u3;
        f[Mx-1].w  = temp_w1 + temp_w2 + temp_w3;
        xm--;
    }
    for (i=xs; i<xs+xm; i++) {
        /*
         Compute relevant entries of the weighted stiffness matrix
         */
        b_cl = - (b_xold[i-1].u + b_xold[i].u)*hx/2;
        b_cc = (b_xold[i-1].u + 2.0*b_xold[i].u + b_xold[i+1].u)*hx/2;
        b_cr = - (b_xold[i+1].u + b_xold[i].u)*hx/2;
        
        temp_u1 = gamma*(-x[i-1].u + 2.0*x[i].u -x[i+1].u)/hx + hx*x[i].u*x[i].u*x[i].u;
        temp_u2 = - hx*x[i].w;
        temp_u3 = - hx*xold[i].u;
        temp_w1 = hx*x[i].u/dt + theta*(-x[i-1].u + 2.0*x[i].u - x[i+1].u)/hx;
        temp_w2 = b_cl*x[i-1].u + b_cc*x[i].u + b_cr*x[i+1].u;
        temp_w3 = - hx*xold[i].u/dt;
        f[i].u  = temp_u1 + temp_u2 + temp_u3;
        f[i].w  = temp_w1 + temp_w2 + temp_w3;
    }
    
    /*
     Restore vectors
     */
    ierr = DMDAVecRestoreArray(da,localX,&x);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,F,&f);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,user->xold,&xold);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,user->b_xold,&b_xold);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
PetscErrorCode FormJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flg,void *ctx)
{
    PetscErrorCode ierr;
    DM             da;
    PetscInt       Mx,xs,xm,i;
    PetscReal      hx;
    MatStencil     stencil[6],stencil_boundary[4],rowstencil;
    PetscScalar    entries[6],entries_boundary[4];
    Field          *x,*b_xold;
    Vec            localX;
    AppCtx         *user=(AppCtx*)ctx;
    PetscScalar    gamma = user->gamma, theta = user->theta;
    PetscReal      dt = user->dt;
    
    PetscFunctionBegin;
    ierr = SNESGetDM(snes,&da);CHKERRQ(ierr);
    ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
    ierr = DMDAGetInfo(da,PETSC_IGNORE,&Mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);
    ierr = DMDAGetCorners(da,&xs,PETSC_NULL,PETSC_NULL,&xm,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
    
    
    hx = 1.0/(PetscReal)(Mx-1);
    
    /*
     Scatter ghost points to local vector,using the 2-step process
     DMGlobalToLocalBegin(),DMGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
     */
    ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
    /*
     Get pointers to vector data
     */
    ierr = DMDAVecGetArray(da,localX,&x);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,user->b_xold,&b_xold);CHKERRQ(ierr);
    
    stencil_boundary[0].k = stencil_boundary[0].j = 0;
    stencil_boundary[1].k = stencil_boundary[1].j = 0;
    stencil_boundary[2].k = stencil_boundary[2].j = 0;
    stencil_boundary[3].k = stencil_boundary[3].j = 0;
    
    stencil[0].k = stencil[0].j = 0;
    stencil[1].k = stencil[1].j = 0;
    stencil[2].k = stencil[2].j = 0;
    stencil[3].k = stencil[3].j = 0;
    stencil[4].k = stencil[4].j = 0;
    stencil[5].k = stencil[5].j = 0;
    rowstencil.k = rowstencil.j = 0;
    
    if (!xs) {
        
        stencil_boundary[0].i = 0; stencil_boundary[0].c = 0; entries_boundary[0] = gamma/hx + (hx/2.0)*3.0*x[0].u*x[0].u;
        stencil_boundary[1].i = 1; stencil_boundary[1].c = 0; entries_boundary[1] = - gamma/hx;
        stencil_boundary[2].i = 0; stencil_boundary[2].c = 1; entries_boundary[2] = - hx/2.0;
        stencil_boundary[3].i = 1; stencil_boundary[3].c = 1; entries_boundary[3] = 0.0;
        rowstencil.i = 0; rowstencil.c = 0;
        ierr = MatSetValuesStencil(*B,1,&rowstencil,4,stencil_boundary,entries_boundary,INSERT_VALUES);CHKERRQ(ierr);
        
        entries_boundary[0] = hx/(2.0*dt) + theta/hx;
        entries_boundary[1] = - theta/hx;
        entries_boundary[2] = (b_xold[0].u + b_xold[1].u)/(2.0*hx);
        entries_boundary[3] = - (b_xold[0].u + b_xold[1].u)/(2.0*hx);
        rowstencil.c = 1;
        ierr = MatSetValuesStencil(*B,1,&rowstencil,4,stencil_boundary,entries_boundary,INSERT_VALUES);CHKERRQ(ierr);
        
        xs++;
        xm--;
    }
    
    if (xs+xm == Mx) {
        stencil_boundary[0].i = Mx-2; stencil_boundary[0].c = 0; entries_boundary[0] = - gamma/hx;
        stencil_boundary[1].i = Mx-1; stencil_boundary[1].c = 0; entries_boundary[1] = gamma/hx + (hx/2.0)*3.0*x[Mx-1].u*x[Mx-1].u;
        stencil_boundary[2].i = Mx-2; stencil_boundary[2].c = 1; entries_boundary[2] = 0.0;
        stencil_boundary[3].i = Mx-1; stencil_boundary[3].c = 1; entries_boundary[3] = - hx/2.0;
        rowstencil.i = Mx-1; rowstencil.c = 0;
        ierr = MatSetValuesStencil(*B,1,&rowstencil,4,stencil_boundary,entries_boundary,INSERT_VALUES);CHKERRQ(ierr);
        
        entries_boundary[0] = - theta/hx;
        entries_boundary[1] = hx/(2.0*dt) + theta/hx;
        entries_boundary[2] = - (b_xold[Mx-2].u + b_xold[Mx-1].u)/(2.0*hx);
        entries_boundary[3] = (b_xold[Mx-2].u + b_xold[Mx-1].u)/(2.0*hx);
        rowstencil.c = 1;
        ierr = MatSetValuesStencil(*B,1,&rowstencil,4,stencil_boundary,entries_boundary,INSERT_VALUES);CHKERRQ(ierr);
        
        xm--;
    }
    
    for (i=xs; i<xs+xm; i++) {
        stencil[0].i = i-1; stencil[0].c = 0; entries[0] = - gamma/hx;
        stencil[1].i = i; stencil[1].c = 0; entries[1] = gamma*2.0/hx + hx*3.0*x[i].u*x[i].u;
        stencil[2].i = i+1; stencil[2].c = 0; entries[2] = - gamma/hx;
        stencil[3].i = i-1; stencil[3].c = 1; entries[3] = 0.0;
        stencil[4].i = i; stencil[4].c = 1; entries[4] = - hx;
        stencil[5].i = i+1; stencil[5].c = 1; entries[5] = 0.0;
        rowstencil.i = i; rowstencil.c = 0;
        ierr = MatSetValuesStencil(*B,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
        
        entries[0] = - theta/hx;
        entries[1] = hx/dt + theta*2.0/hx;
        entries[2] = - theta/hx;
        entries[3] = - (b_xold[i-1].u + b_xold[i].u)/(2.0*hx);
        entries[4] = (b_xold[i-1].u + 2.0*b_xold[i].u + b_xold[i+1].u)/(2.0*hx);
        entries[5] = - (b_xold[i+1].u + b_xold[i].u)/(2.0*hx);
        rowstencil.c = 1;
        ierr = MatSetValuesStencil(*B,1,&rowstencil,6,stencil,entries,INSERT_VALUES);CHKERRQ(ierr);
    }
    
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (*J != *B){
        ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
        ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
    ierr = DMDAVecRestoreArray(da,localX,&x);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,user->b_xold,&b_xold);CHKERRQ(ierr); 
    ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetParams"
PetscErrorCode GetParams(AppCtx* user)
{
    PetscErrorCode ierr;
    PetscBool      flg;
    
    PetscFunctionBegin;
    
    /* Set default parameters */
    user->gamma = .001;
    user->dt = .01;;
    user->T = 10.0*user->dt;
    user->theta = .3;
    
    ierr = PetscOptionsGetReal(PETSC_NULL,"-dt",&user->dt,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-T",&user->T,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-gamma",&user->gamma,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsGetScalar(PETSC_NULL,"-theta",&user->theta,&flg);CHKERRQ(ierr);
    
    PetscFunctionReturn(0);
}

