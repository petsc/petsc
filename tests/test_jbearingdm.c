#include "petscdm.h"
#include "taosolver.h"
#include "taodm.h"

static char help[] ="Pressure distribution in a Journal Bearing. \n\
This example is based on the problem DPJB from the MINPACK-2 test suite.\n\
This pressure journal bearing problem is an example of elliptic variational\n\
problem defined over a two dimensional rectangle. By discretizing the domain \n\
into triangular elements, the pressure surrounding the journal bearing is\n\
defined as the minimum of a quadratic function whose variables are bounded\n\
below by zero. The command line options are:\n\
  -ecc <ecc>, where <ecc> = epsilon parameter\n\
  -b <b>, where <b> = half the upper limit in the 2nd coordinate direction\n\
  -mx <xg>, where <xg> = number of grid points in the 1st coordinate direction\n\
  -my <yg>, where <yg> = number of grid points in the 2nd coordinate direction\n\
  -nlevels <nlevels>, where <nlevels> = number of levels in multigrid\n\
  -byelement, if computation is made by functions on rectangular elements\n\
  -adic, if AD is used (AD is not used by default)\n\n";

#ifdef TAO_USE_ADIC
PetscErrorCode ad_JBearLocalFunction(PetscInt[2] ,DERIV_TYPE[4], DERIV_TYPE *, void*);
typedef struct {

  InactiveDouble      *wq, *wl;      /* vectors with the parameters w_q(x) and w_l(x) */
  InactiveDouble      hx, hy;        /* increment size in both directions */
  InactiveDouble      area;          /* area of the triangles */

} ADFGCtx;
#else
typedef PetscInt ADFGCtx;
#endif


typedef struct {
  PetscReal      ecc;           /* epsilon value */
  PetscReal      b;             /* 0.5 * upper limit for 2nd variable */
  Vec            B;
  PetscReal      *wq, *wl;      /* vectors with the parameters w_q(x) and w_l(x) */
  PetscReal      hx, hy;        /* increment size in both directions */
  PetscReal      area;          /* area of the triangles */

  PetscInt    mx, my;        /* discretization including boundaries */

  ADFGCtx     fgctx;         /* Used only when an ADIC generated gradient is used */

} AppCtx;


/* User-defined routines found in this file */
//static PetscErrorCode AppCtxInitialize(void *ptr);
static PetscErrorCode FormInitialGuess(TaoDM, Vec);
static PetscErrorCode FormBounds(TaoDM, Vec, Vec);
static PetscErrorCode FormFunctionGradient(TaoSolver, Vec, PetscScalar*, Vec, void*);
static PetscErrorCode FormHessian(TaoSolver, Vec, Mat*, Mat*, MatStructure*, void*);
#ifdef USELOCAL
static PetscErrorCode FormFunctionGradientLocal(DMDALocalInfo *info, PetscScalar **x, PetscScalar *f, PetscScalar **g, void *ctx);
static PetscErrorCode FormHessianLocal(DMDALocalInfo *info, PetscScalar **x, Mat H, void *ctx);
#endif
static PetscErrorCode ComputeB(TaoDM);
static PetscReal p(PetscReal xi, PetscReal ecc);

/*
static PetscErrorCode WholeJBearFunctionGradient(TAO_APPLICATION,DA,Vec,double *,Vec,void*);
static PetscErrorCode WholeJBearHessian(TAO_APPLICATION,DA,Vec,Mat,void*);
*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv) {
  PetscErrorCode       ierr;
  PetscInt             Nx,Ny;
  //PetscScalar          ff,gnorm;
  DM              dm;
  PetscBool       flg;
  AppCtx          user;                    /* user-defined work context */
  TaoDM           *taodm;                     /* TAO_SOLVER solver context */
  //TaoSolverTerminationReason reason;

    /* Initialize TAO */
  PetscInitialize(&argc, &argv, (char *)0, help);
  TaoInitialize(&argc, &argv, (char *)0, help);

  //nlevels=5;
  //ierr = PetscOptionsGetInt(PETSC_NULL,"-nlevels",&nlevels,&flg); CHKERRQ(ierr);

  /* Application specific parameters */
  user.ecc = 0.1;
  user.b = 10.0;
  user.mx = user.my = 11;
  ierr = PetscOptionsGetReal(TAO_NULL, "-ecc", &user.ecc, &flg); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(TAO_NULL, "-b", &user.b, &flg); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(TAO_NULL, "-mx", &user.mx, &flg); CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(TAO_NULL, "-my", &user.my, &flg); CHKERRQ(ierr);

  PetscPrintf(MPI_COMM_WORLD,"\n---- Journal Bearing Problem -----\n\n");

  /* Let PETSc determine the vector distribution */
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;

  ierr = TaoDMCreate(PETSC_COMM_WORLD,1,&user,&taodm); CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_NONPERIODIC,DMDA_STENCIL_BOX,user.mx,
                    user.my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,&dm); CHKERRQ(ierr);
  ierr = TaoDMSetDM(taodm,(DM)dm); CHKERRQ(ierr);
  ierr = DMDestroy(dm); CHKERRQ(ierr);

  //  ierr = TaoDMSetLocalObjectiveAndGradientRoutine(taodm,FormFunctionGradientLocal); CHKERRQ(ierr);
  //  ierr = TaoDMSetLocalHessianRoutine(taodm,FormHessianLocal); CHKERRQ(ierr);
  ierr = TaoDMSetObjectiveAndGradientRoutine(taodm,FormFunctionGradient); CHKERRQ(ierr);
  ierr = TaoDMSetHessianRoutine(taodm,FormHessian); CHKERRQ(ierr);
  ierr = TaoDMSetInitialGuessRoutine(taodm,FormInitialGuess); CHKERRQ(ierr);
  ierr = TaoDMSetVariableBoundsRoutine(taodm,FormBounds); CHKERRQ(ierr);
  ierr = TaoDMSetFromOptions(taodm); CHKERRQ(ierr);
  ierr = TaoDMSolve(taodm); CHKERRQ(ierr);
  
  ierr = TaoDMDestroy(taodm); CHKERRQ(ierr);
  ierr = TaoFinalize();
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}





#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
PetscErrorCode FormInitialGuess(TaoDM taodm, Vec X)
{
  PetscErrorCode ierr;
  AppCtx         *user;
  DM             dm;
  PetscInt    i, j, mx;
  PetscInt    xs, ys, xm, ym, xe, ye;
  PetscReal hx, val;
  PetscScalar **x;

  PetscFunctionBegin;
  ierr = TaoDMGetContext(taodm,(void**)&user);
  ierr = TaoDMGetDM(taodm,&dm);
  /* Get local mesh boundaries */
  ierr = DMDAGetInfo(dm, PETSC_IGNORE,&mx,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  hx = 2.0*4.0*atan(1.0)/((PetscReal)(mx-1));

  ierr = DMDAGetCorners(dm,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL); CHKERRQ(ierr);
  xe = xs+xm; ye = ys+ym;

  ierr = DMDAVecGetArray(dm, X, (void**)&x); CHKERRQ(ierr);
  /* Compute initial guess over locally owned part of mesh */
  for (j=ys; j<ye; j++) {  /*  for (j=0; j<my; j++) */
    for (i=xs; i<xe; i++) {  /*  for (i=0; i<mx; i++) */
      val = PetscMax(sin(((PetscReal)(i)+1.0)*hx),0.0);
      x[j][i] = val;
      x[j][i] = 0;
    }
  }
  ierr = DMDAVecRestoreArray(dm, X, (void**)&x); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormBounds"
PetscErrorCode FormBounds(TaoDM taodm, Vec XL, Vec XU)
{
  AppCtx *user;
  DM dm;
  PetscErrorCode ierr;
  PetscInt i, j, mx, my;
  PetscInt xs, xm, ys, ym;
  PetscScalar **xl, **xu;

  PetscFunctionBegin;  
  ierr = TaoDMGetContext(taodm,(void**)&user); CHKERRQ(ierr);
  ierr = TaoDMGetDM(taodm,&dm); CHKERRQ(ierr);
  mx = user->mx;
  my = user->my;

  ierr = DMDAVecGetArray(dm, XL, (void**)&xl); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm, XU, (void**)&xu); CHKERRQ(ierr);
  ierr = DMDAGetCorners(dm, &xs, &ys, TAO_NULL, &xm, &ym, TAO_NULL); CHKERRQ(ierr);

  for (j = ys; j < ys+ym; j++){
    for (i = xs; i < xs+xm; i++){
      xl[j][i] = 0.0;
      if (i == 0 || j == 0 || i == mx - 1 || j == my - 1) {
        xu[j][i] = 0.0;
      } else {
        xu[j][i] = TAO_INFINITY;
      }
    }
  }

  ierr = DMDAVecRestoreArray(dm, XL, (void**)&xl); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(dm, XU, (void**)&xu); CHKERRQ(ierr);

  PetscFunctionReturn(0);

} 
#ifdef USELOCAL
#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradientLocal"
static PetscErrorCode FormFunctionGradientLocal(DMDALocalInfo *info, PetscScalar **x, PetscScalar *f, PetscScalar **g, void *ctx)
{
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx*)ctx;

  PetscScalar avgWq, sqGrad, avgWV, fl, fu;
  PetscScalar hx, hy, area, aread3, *wq, *wl;
  PetscScalar dvdx, dvdy;
  PetscInt i;

  PetscFunctionBegin;
  hx = user->hx;
  hy = user->hy;
  area = user->area;
  aread3 = area / 3.0;
  wq = user->wq;
  wl = user->wl;
  i =0;// TODO coor[0]

  /* lower triangle contribution */
  dvdx = (x[0] - x[1]) / hx;
  dvdy = (x[0] - x[2]) / hy;
  sqGrad = dvdx * dvdx + dvdy * dvdy;
  avgWq = (2.0 * wq[i] + wq[i+1]) / 6.0;
  avgWV = (wl[i]*x[0] + wl[ib+1]*x[1] + wl[i]*x[2]) / 3.0;
  fl = avgWq * sqGrad - avgWV;

  dvdx = dvdx * hy * avgWq;
  dvdy = dvdy * hx * avgWq;
  g[0] = ( dvdx + dvdy ) - wl[i] * aread3;
  g[1] = ( -dvdx ) - wl[i+1] * aread3;
  g[2] = ( -dvdy ) - wl[i] * aread3;

  /* upper triangle contribution */
  dvdx = (x[3] - x[2]) / hx; 
  dvdy = (x[3] - x[1]) / hy;
  sqGrad = dvdx * dvdx + dvdy * dvdy;
  avgWq = (2.0 * wq[i+1] + wq[i]) / 6.0;
  avgWV = (wl[i+1]*x[1] + wl[i]*x[2] + wl[i+1]*x[3]) / 3.0;
  fu = avgWq * sqGrad - avgWV;

  dvdx = dvdx * hy * avgWq;
  dvdy = dvdy * hx * avgWq;
  g[1] += (-dvdy) - wl[i+1] * aread3;
  g[2] +=  (-dvdx) - wl[i] * aread3;
  g[3] = ( dvdx + dvdy ) - wl[i+1] * aread3;

  *f = area * (fl + fu);

  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormHessianLocal"

  
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx*)ptr;
  PetscInt i,j,k;
  PetscInt col[5],row,nx,ny;
  PetscReal one=1.0, two=2.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal hx,hy,hxhy,hxhx,hyhy;
  PetscReal xi,v[5];
  PetscReal ecc=user->ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal vmiddle, vup, vdown, vleft, vright;
  Mat hes=*H;
  PetscBool assembled;

  PetscFunctionBegin;
  nx=user->mx;
  ny=user->my;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  hxhy=hx*hy;
  hxhx=one/(hx*hx);
  hyhy=one/(hy*hy);

  *flg=SAME_NONZERO_PATTERN;
  /*
    Get local grid boundaries
  */
  
  ierr = MatAssembled(hes,&assembled); CHKERRQ(ierr);
  if (assembled){ierr = MatZeroEntries(hes);  CHKERRQ(ierr);}

  for (i=info->xs; i< info->xs+info->xm; i++){
    xi=(i+1)*hx;
    trule1=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi,ecc) ) / six; /* L(i,j) */
    trule2=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi,ecc) ) / six; /* U(i,j) */
    trule3=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi+hx,ecc) ) / six; /* U(i+1,j) */
    trule4=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi-hx,ecc) ) / six; /* L(i-1,j) */
    trule5=trule1; /* L(i,j-1) */
    trule6=trule2; /* U(i,j+1) */

    vdown=-(trule5+trule2)*hyhy;
    vleft=-hxhx*(trule2+trule4);
    vright= -hxhx*(trule1+trule3);
    vup=-hyhy*(trule1+trule6);
    vmiddle=(hxhx)*(trule1+trule2+trule3+trule4)+hyhy*(trule1+trule2+trule5+trule6);
    v[0]=0; v[1]=0; v[2]=0; v[3]=0; v[4]=0;

    for (j=info->ys; j<info->ys+info->ym; j++){
      row=(j-info->gys)*info->gxm + (i-info->gxs);
       
      k=0;
      if (j>info->gys){ 
	v[k]=vdown; col[k]=row - info->gxm; k++;
      }
       
      if (i>info->gxs){
	v[k]= vleft; col[k]=row - 1; k++;
      }

      v[k]= vmiddle; col[k]=row; k++;
       
      if (i+1 < info->gxs+info->gxm){
	v[k]= vright; col[k]=row+1; k++;
      }
       
      if (j+1 <info->gys+info->gym){
	v[k]= vup; col[k] = row+info->gxm; k++;
      }
      ierr = MatSetValuesLocal(hes,1,&row,k,col,v,INSERT_VALUES); CHKERRQ(ierr);
       
    }

  }

  /* 
     Assemble matrix, using the 2-step process:
     MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = MatAssemblyBegin(hes,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(hes,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /*
    Tell the matrix we will never add a new nonzero location to the
    matrix. If we do it will generate an error.
  */
  ierr = MatSetOption(hes,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE); CHKERRQ(ierr);
  ierr = MatSetOption(hes,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);

  ierr = PetscLogFlops(9*xm*ym+49*xm); CHKERRQ(ierr);
  ierr = MatNorm(hes,NORM_1,&hx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif

#undef __FUNCT__
#define __FUNCT__ "FormHessian"
/* 
   FormHessian computes the quadratic term in the quadratic objective function 
   Notice that the objective function in this problem is quadratic (therefore a constant
   hessian).  If using a nonquadratic solver, then you might want to reconsider this function
*/
PetscErrorCode FormHessian(TaoSolver tao,Vec X,Mat *H, Mat *Hpre, MatStructure *flg, void *ptr)
{
  TaoDM   taodm = (TaoDM)ptr;
  AppCtx* user;
  DM      dm;
  PetscErrorCode ierr;
  PetscInt i,j,k;
  PetscInt col[5],row,nx,ny,xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal one=1.0, two=2.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal hx,hy,hxhy,hxhx,hyhy;
  PetscReal xi,v[5];
  PetscReal ecc=user->ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal vmiddle, vup, vdown, vleft, vright;
  Mat hes=*H;
  PetscBool assembled;
  
  PetscFunctionBegin;
  ierr = TaoDMGetContext(taodm,(void**)&user); CHKERRQ(ierr);
  ierr = TaoDMGetDM(taodm,&dm); CHKERRQ(ierr);
  nx=user->mx;
  ny=user->my;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  hxhy=hx*hy;
  hxhx=one/(hx*hx);
  hyhy=one/(hy*hy);

  *flg=SAME_NONZERO_PATTERN;
  /*
    Get local grid boundaries
  */
  ierr = DMDAGetCorners(dm,&xs,&ys,TAO_NULL,&xm,&ym,TAO_NULL); CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dm,&gxs,&gys,TAO_NULL,&gxm,&gym,TAO_NULL); CHKERRQ(ierr);
  
  ierr = MatAssembled(hes,&assembled); CHKERRQ(ierr);
  if (assembled){ierr = MatZeroEntries(hes);  CHKERRQ(ierr);}

  for (i=xs; i< xs+xm; i++){
    xi=(i+1)*hx;
    trule1=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi,ecc) ) / six; /* L(i,j) */
    trule2=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi,ecc) ) / six; /* U(i,j) */
    trule3=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi+hx,ecc) ) / six; /* U(i+1,j) */
    trule4=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi-hx,ecc) ) / six; /* L(i-1,j) */
    trule5=trule1; /* L(i,j-1) */
    trule6=trule2; /* U(i,j+1) */

    vdown=-(trule5+trule2)*hyhy;
    vleft=-hxhx*(trule2+trule4);
    vright= -hxhx*(trule1+trule3);
    vup=-hyhy*(trule1+trule6);
    vmiddle=(hxhx)*(trule1+trule2+trule3+trule4)+hyhy*(trule1+trule2+trule5+trule6);
    v[0]=0; v[1]=0; v[2]=0; v[3]=0; v[4]=0;

    for (j=ys; j<ys+ym; j++){
      row=(j-gys)*gxm + (i-gxs);
       
      k=0;
      if (j>gys){ 
	v[k]=vdown; col[k]=row - gxm; k++;
      }
       
      if (i>gxs){
	v[k]= vleft; col[k]=row - 1; k++;
      }

      v[k]= vmiddle; col[k]=row; k++;
       
      if (i+1 < gxs+gxm){
	v[k]= vright; col[k]=row+1; k++;
      }
       
      if (j+1 <gys+gym){
	v[k]= vup; col[k] = row+gxm; k++;
      }
      ierr = MatSetValuesLocal(hes,1,&row,k,col,v,INSERT_VALUES); CHKERRQ(ierr);
       
    }

  }

  /* 
     Assemble matrix, using the 2-step process:
     MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = MatAssemblyBegin(hes,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(hes,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /*
    Tell the matrix we will never add a new nonzero location to the
    matrix. If we do it will generate an error.
  */
  ierr = MatSetOption(hes,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE); CHKERRQ(ierr);
  ierr = MatSetOption(hes,MAT_SYMMETRIC,PETSC_TRUE); CHKERRQ(ierr);

  ierr = PetscLogFlops(9*xm*ym+49*xm); CHKERRQ(ierr);
  ierr = MatNorm(hes,NORM_1,&hx); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
PetscErrorCode FormFunctionGradient(TaoSolver tao, Vec X, PetscReal *fcn,Vec G,void *ptr)
{
  TaoDM   taodm = (TaoDM)ptr;
  AppCtx* user;
  DM      dm;
  PetscErrorCode ierr;
  PetscInt i,j,k,kk;
  PetscInt col[5],row,nx,ny,xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal one=1.0, two=2.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal hx,hy,hxhy,hxhx,hyhy;
  PetscReal xi,v[5];
  PetscReal ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal vmiddle, vup, vdown, vleft, vright;
  PetscReal tt,f1,f2;
  PetscReal *x,*g,zero=0.0;
  Vec localX;

  PetscFunctionBegin;
  ierr = TaoDMGetContext(taodm,(void**)&user); CHKERRQ(ierr);
  ierr = TaoDMGetDM(taodm,&dm); CHKERRQ(ierr);
  ecc = user->ecc;
  nx=user->mx;
  ny=user->my;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  hxhy=hx*hy;
  hxhx=one/(hx*hx);
  hyhy=one/(hy*hy);

  ierr = DMGetLocalVector(dm,&localX);CHKERRQ(ierr);

  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,localX); CHKERRQ(ierr);

  ierr = VecSet(G, zero); CHKERRQ(ierr);
  /*
    Get local grid boundaries
  */
  ierr = DMDAGetCorners(dm,&xs,&ys,TAO_NULL,&xm,&ym,TAO_NULL); CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dm,&gxs,&gys,TAO_NULL,&gxm,&gym,TAO_NULL); CHKERRQ(ierr);
  
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);
  ierr = VecGetArray(G,&g); CHKERRQ(ierr);

  for (i=xs; i< xs+xm; i++){
    xi=(i+1)*hx;
    trule1=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi,ecc) ) / six; /* L(i,j) */
    trule2=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi,ecc) ) / six; /* U(i,j) */
    trule3=hxhy*( p(xi,ecc) + p(xi+hx,ecc) + p(xi+hx,ecc) ) / six; /* U(i+1,j) */
    trule4=hxhy*( p(xi,ecc) + p(xi-hx,ecc) + p(xi-hx,ecc) ) / six; /* L(i-1,j) */
    trule5=trule1; /* L(i,j-1) */
    trule6=trule2; /* U(i,j+1) */

    vdown=-(trule5+trule2)*hyhy;
    vleft=-hxhx*(trule2+trule4);
    vright= -hxhx*(trule1+trule3);
    vup=-hyhy*(trule1+trule6);
    vmiddle=(hxhx)*(trule1+trule2+trule3+trule4)+hyhy*(trule1+trule2+trule5+trule6);

    for (j=ys; j<ys+ym; j++){
      
      row=(j-gys)*gxm + (i-gxs);
       v[0]=0; v[1]=0; v[2]=0; v[3]=0; v[4]=0;
       
       k=0;
       if (j>gys){ 
	 v[k]=vdown; col[k]=row - gxm; k++;
       }
       
       if (i>gxs){
	 v[k]= vleft; col[k]=row - 1; k++;
       }

       v[k]= vmiddle; col[k]=row; k++;
       
       if (i+1 < gxs+gxm){
	 v[k]= vright; col[k]=row+1; k++;
       }
       
       if (j+1 <gys+gym){
	 v[k]= vup; col[k] = row+gxm; k++;
       }
       tt=0;
       for (kk=0;kk<k;kk++){
	 tt+=v[kk]*x[col[kk]];
       }
       row=(j-ys)*xm + (i-xs);
       g[row]=tt;

     }

  }

  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(G,&g); CHKERRQ(ierr);

  ierr = DMRestoreLocalVector(dm,&localX); CHKERRQ(ierr);

  ierr = VecDot(X,G,&f1); CHKERRQ(ierr);
  ierr = VecDuplicate(X,&user->B); CHKERRQ(ierr);
  ierr = ComputeB(taodm); CHKERRQ(ierr);
  ierr = VecDot(user->B,X,&f2); CHKERRQ(ierr);
  ierr = VecAXPY(G, one, user->B); CHKERRQ(ierr);
  ierr = VecDestroy(user->B); CHKERRQ(ierr);
  *fcn = f1/2.0 + f2;

  ierr = PetscLogFlops((91 + 10*ym) * xm); CHKERRQ(ierr);
  PetscFunctionReturn(0);

}


#undef __FUNCT__
#define __FUNCT__ "ComputeB"
PetscErrorCode ComputeB(TaoDM taodm)
{
  PetscErrorCode ierr;
  AppCtx* user;
  DM      dm;
  PetscInt i,j,k;
  PetscInt nx,ny,xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal two=2.0, pi=4.0*atan(1.0);
  PetscReal hx,hy,ehxhy;
  PetscReal temp,*b;

  ierr = TaoDMGetContext(taodm,(void**)&user); CHKERRQ(ierr);
  ierr = TaoDMGetDM(taodm,&dm); CHKERRQ(ierr);

  nx=user->mx;
  ny=user->my;
  hx=two*pi/(nx+1.0);
  hy=two*user->b/(ny+1.0);
  ehxhy = user->ecc*hx*hy;


  /*
     Get local grid boundaries
  */
  ierr = DMDAGetCorners(dm,&xs,&ys,TAO_NULL,&xm,&ym,TAO_NULL); CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(dm,&gxs,&gys,TAO_NULL,&gxm,&gym,TAO_NULL); CHKERRQ(ierr);
  

  /* Compute the linear term in the objective function */  
  ierr = VecGetArray(user->B,&b); CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++){
    temp=sin((i+1)*hx);
    for (j=ys; j<ys+ym; j++){
      k=xm*(j-ys)+(i-xs);
      b[k]=  - ehxhy*temp;
    }
  }
  ierr = VecRestoreArray(user->B,&b); CHKERRQ(ierr);
  ierr = PetscLogFlops(5*xm*ym+3*xm); CHKERRQ(ierr);

  return 0;
}

static PetscReal p(PetscReal xi, PetscReal ecc)
{ 
  PetscReal t=1.0+ecc*cos(xi); 
  return (t*t*t); 
}
