
static char help[] = "Model multi-physics solver. Modified from ex19.c \n\\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/

/* ------------------------------------------------------------------------

    See ex19.c for discussion of the problem 

  ------------------------------------------------------------------------- */
#include "mp.h"

extern PetscErrorCode FormInitialGuessComp(DMMG,Vec);
extern PetscErrorCode FormFunctionComp(SNES,Vec,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg_comp; /* multilevel grid structure */
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its,dof,nlevels=1; 
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DA             da;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;

  ierr = PetscOptionsGetInt(PETSC_NULL,"-nlevels",&nlevels,PETSC_NULL);CHKERRQ(ierr);

  /* Problem parameters (velocity of lid, prandtl, and grashof numbers) */
  ierr = PetscOptionsGetReal(PETSC_NULL,"-lidvelocity",&user.lidvelocity,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-prandtl",&user.prandtl,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-grashof",&user.grashof,PETSC_NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create user context, set problem data, create vector data structures.
     Also, compute the initial guess.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup Physics 1: 
        - Lap(U) - Grad_y(Omega) = 0
	- Lap(V) + Grad_x(Omega) = 0
	- Lap(Omega) + Div([U*Omega,V*Omega]) - GR*Grad_x(T) = 0
        where T is given by the computed x.temp
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMGCreate(comm,nlevels,&user,&user.dmmg1);CHKERRQ(ierr);
  dof  = 3;
  ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMMGSetDM(user.dmmg1,(DM)da);CHKERRQ(ierr);
  ierr = DASetFieldName(da,0,"x-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(da,1,"y-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(da,2,"Omega");CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Setup Physics 2: 
        - Lap(T) + PR*Div([U*T,V*T]) = 0        
        where U and V are given by the computed x.u and x.v
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMGCreate(comm,nlevels,&user,&user.dmmg2);CHKERRQ(ierr);
  dof  = 1;
  ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,dof,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMMGSetDM(user.dmmg2,(DM)da);CHKERRQ(ierr);
  ierr = DASetFieldName(da,0,"temperature");CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create the DMComposite object to manage the two grids/physics. 
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCompositeCreate(comm,&user.pack);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.pack,(DM)DMMGGetDA(user.dmmg1));CHKERRQ(ierr);
  ierr = DMCompositeAddDM(user.pack,(DM)DMMGGetDA(user.dmmg2));CHKERRQ(ierr);

  /* Create the solver object and attach the grid/physics info */
  ierr = DMMGCreate(comm,nlevels,&user,&dmmg_comp);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg_comp,(DM)user.pack);CHKERRQ(ierr);
  ierr = DMMGSetISColoringType(dmmg_comp,IS_COLORING_GLOBAL);CHKERRQ(ierr);

  ierr = DMMGSetInitialGuess(dmmg_comp,FormInitialGuessComp);CHKERRQ(ierr);
  ierr = DMMGSetSNES(dmmg_comp,FormFunctionComp,0);CHKERRQ(ierr);

  da   = (DA) DMMGGetDM(dmmg_comp);
  ierr = DAGetInfo(da,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  user.lidvelocity = 1.0/(mx*my);
  user.prandtl     = 1.0;
  user.grashof     = 1.0;

  /* Solve the nonlinear system */
  ierr = DMMGSolve(dmmg_comp);CHKERRQ(ierr); 
  snes = DMMGGetSNES(dmmg_comp);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Composite Physics: Number of Newton iterations = %D\n\n", its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free spaces 
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCompositeDestroy(user.pack);CHKERRQ(ierr);
  ierr = DMMGDestroy(user.dmmg1);CHKERRQ(ierr);
  ierr = DMMGDestroy(user.dmmg2);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg_comp);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocal1"
PetscErrorCode FormInitialGuessLocal1(DALocalInfo *info,Field1 **x)
{
  PetscInt       i,j;
  PetscErrorCode ierr;

  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x[j][i].u     = 0.0;
      x[j][i].v     = 0.0;
      x[j][i].omega = 0.0;
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocal2"
PetscErrorCode FormInitialGuessLocal2(DALocalInfo *info,Field2 **x,AppCtx *user)
{
  PetscInt       i,j;
  PetscScalar    dx;

  dx  = 1.0/(info->mx-1);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      x[j][i].temp  = (user->grashof>0)*i*dx;  
    }
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessComp"
/* 
   FormInitialGuessComp - 
              Forms the initial guess for the composite model
              Unwraps the global solution vector and passes its local pieces into the user functions
 */
PetscErrorCode FormInitialGuessComp(DMMG dmmg,Vec X)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DMMG           *dmmg1 = user->dmmg1,*dmmg2=user->dmmg2;
  DMComposite    dm = (DMComposite)dmmg->dm;
  Vec            X1,X2;
  Field1         **x1;
  Field2         **x2;
  DALocalInfo    info1,info2;

  PetscFunctionBegin;
  /* Access the subvectors in X */
  ierr = DMCompositeGetAccess(dm,X,&X1,&X2);CHKERRQ(ierr);
  /* Access the arrays inside the subvectors of X */
  ierr = DAVecGetArray((DA)DMMGGetDM(dmmg1),X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)DMMGGetDM(dmmg2),X2,(void**)&x2);CHKERRQ(ierr);

  ierr = DAGetLocalInfo((DA)DMMGGetDM(dmmg1),&info1);CHKERRQ(ierr);
  ierr = DAGetLocalInfo((DA)DMMGGetDM(dmmg2),&info2);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocal1(&info1,x1);CHKERRQ(ierr);
  ierr = FormInitialGuessLocal2(&info2,x2,user);CHKERRQ(ierr);

  ierr = DAVecRestoreArray((DA)DMMGGetDM(dmmg1),X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray((DA)DMMGGetDM(dmmg2),X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,X,&X1,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal1"
/* 
      x2 contains given tempature field
*/
PetscErrorCode FormFunctionLocal1(DALocalInfo *info,Field1 **x,Field2 **x2,Field1 **f,void *ptr)
 {
  AppCtx         *user = (AppCtx*)ptr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  PetscFunctionBegin;
  grashof = user->grashof;  
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  dhx = (PetscReal)(info->mx-1);  dhy = (PetscReal)(info->my-1);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega + (x[j+1][i].u - x[j][i].u)*dhy;  
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info->my) {
    j = info->my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
        f[j][i].u     = x[j][i].u - lid;
        f[j][i].v     = x[j][i].v;
        f[j][i].omega = x[j][i].omega + (x[j][i].u - x[j-1][i].u)*dhy; 
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i = 0;
    xints = xints + 1;
    /* left edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i+1].v - x[j][i].v)*dhx; 
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */ 
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i].v - x[j][i-1].v)*dhx; 
    }
  }

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {      
      /* convective coefficients for upwinding */
	vx = x[j][i].u; avx = PetscAbsScalar(vx);
        vxp = .5*(vx+avx); vxm = .5*(vx-avx);
	vy = x[j][i].v; avy = PetscAbsScalar(vy);
        vyp = .5*(vy+avy); vym = .5*(vy-avy);

	/* U velocity */
        u          = x[j][i].u;
        uxx        = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
        uyy        = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
        f[j][i].u  = uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx;

	/* V velocity */
        u          = x[j][i].v;
        uxx        = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
        uyy        = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
        f[j][i].v  = uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy;

	/* Omega */
        u          = x[j][i].omega;
        uxx        = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
        uyy        = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;   
	f[j][i].omega = uxx + uyy + (vxp*(u - x[j][i-1].omega) + vxm*(x[j][i+1].omega - u))*hy + (vyp*(u - x[j-1][i].omega) + vym*(x[j+1][i].omega - u))*hx;
        f[j][i].omega += - .5 * grashof * (x2[j][i+1].temp - x2[j][i-1].temp) * hy;
    }
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal2"
/* 
      
     x1 contains given velocity field

*/
PetscErrorCode FormFunctionLocal2(DALocalInfo *info,Field1**x1,Field2 **x,Field2 **f,void *ptr)
 {
  AppCtx         *user = (AppCtx*)ptr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  PetscFunctionBegin;
  grashof = user->grashof;  
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  dhx = (PetscReal)(info->mx-1);  dhy = (PetscReal)(info->my-1);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].temp  = x[j][i].temp-x[j+1][i].temp;
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info->my) {
    j = info->my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].temp  = x[j][i].temp-x[j-1][i].temp;
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i = 0;
    xints = xints + 1;
    /* left edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].temp  = x[j][i].temp;
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */ 
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].temp  = x[j][i].temp - (PetscReal)(grashof>0);
    }
  }

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {
      /* convective coefficients for upwinding */
      vx = x1[j][i].u; vy = x1[j][i].v; 
      avx = PetscAbsScalar(vx); 
      vxp = .5*(vx+avx); vxm = .5*(vx-avx);
      avy = PetscAbsScalar(vy); 
      vyp = .5*(vy+avy); 
      vym = .5*(vy-avy);

      /* Temperature */
      u             = x[j][i].temp;
      uxx           = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
      uyy           = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
      f[j][i].temp  =  uxx + uyy  
		      + prandtl * ((vxp*(u - x[j][i-1].temp) + vxm*(x[j][i+1].temp - u)) * hy 
		      + (vyp*(u - x[j-1][i].temp) + vym*(x[j+1][i].temp - u)) * hx);
    }
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__
#define __FUNCT__ "FormFunctionComp"
/* 
   FormFunctionComp  - Unwraps the input vector and passes its local ghosted pieces into the user function
*/
PetscErrorCode FormFunctionComp(SNES snes,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  DMMG           dmmg = (DMMG)ctx;
  AppCtx         *user = (AppCtx*)dmmg->user;
  DMComposite    dm = (DMComposite)dmmg->dm;
  DALocalInfo    info1,info2;
  DA             da1,da2;
  Field1         **x1,**f1;
  Field2         **x2,**f2;
  Vec            X1,X2,F1,F2;

  PetscFunctionBegin;

  ierr = DMCompositeGetEntries(dm,&da1,&da2);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da2,&info2);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DMCompositeGetLocalVectors(dm,&X1,&X2);CHKERRQ(ierr);
  ierr = DMCompositeScatter(dm,X,X1,X2);CHKERRQ(ierr); 

  /* Access the arrays inside the subvectors of X */
  ierr = DAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);

  /* Access the subvectors in F. 
     These are not ghosted and directly access the memory locations in F */
  ierr = DMCompositeGetAccess(dm,F,&F1,&F2);CHKERRQ(ierr);

  /* Access the arrays inside the subvectors of F */  
  ierr = DAVecGetArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,F2,(void**)&f2);CHKERRQ(ierr);

  /* Evaluate local user provided function */    
  ierr = FormFunctionLocal1(&info1,x1,x2,f1,(void**)user);CHKERRQ(ierr);
  ierr = FormFunctionLocal2(&info2,x1,x2,f2,(void**)user);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,F2,(void**)&f2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,F,&F1,&F2);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(dm,&X1,&X2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

