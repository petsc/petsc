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
  -taodm_nlevels <nlevels>, where <nlevels> = number of levels in multigrid\n	\
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
static PetscErrorCode FormFunctionGradientLocal(DMDALocalInfo *info, PetscScalar **x, PetscScalar *f, PetscScalar **g, void *ctx);
static PetscErrorCode FormHessianLocal(DMDALocalInfo *info, PetscScalar **x, Mat H, void *ctx);

static PetscErrorCode Monitor(TaoDM, PetscInt, void*); 
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

  ierr = TaoDMCreate(PETSC_COMM_WORLD,4,&user,&taodm); CHKERRQ(ierr);
  ierr = TaoDMSetSolverType(taodm,"tao_blmvm"); CHKERRQ(ierr);
  
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_NONPERIODIC,DMDA_STENCIL_BOX,user.mx,
                    user.my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,&dm); CHKERRQ(ierr);
  ierr = TaoDMSetDM(taodm,(DM)dm); CHKERRQ(ierr);
  ierr = DMDestroy(dm); CHKERRQ(ierr);
  ierr = TaoDMSetTolerances(taodm,0,0,0,0,0);


  ierr = TaoDMSetLocalObjectiveAndGradientRoutine(taodm,FormFunctionGradientLocal); CHKERRQ(ierr);
  ierr = TaoDMSetLocalHessianRoutine(taodm,FormHessianLocal); CHKERRQ(ierr);
  //ierr = TaoDMSetObjectiveAndGradientRoutine(taodm,FormFunctionGradient); CHKERRQ(ierr);
  //ierr = TaoDMSetHessianRoutine(taodm,FormHessian); CHKERRQ(ierr);
  ierr = TaoDMSetInitialGuessRoutine(taodm,FormInitialGuess); CHKERRQ(ierr);
  ierr = TaoDMSetVariableBoundsRoutine(taodm,FormBounds); CHKERRQ(ierr);
  ierr = TaoDMSetLevelMonitor(taodm,Monitor,PETSC_NULL); CHKERRQ(ierr);
  ierr = TaoDMSetFromOptions(taodm); CHKERRQ(ierr);
  ierr = TaoDMSolve(taodm); CHKERRQ(ierr);
  
  ierr = TaoDMDestroy(taodm); CHKERRQ(ierr);
  ierr = TaoFinalize();
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "Monitor"
PetscErrorCode Monitor(TaoDM taodm, PetscInt level, void *ctx) {
  PetscErrorCode ierr;
  AppCtx *user;
  DM dm;
  DMDALocalInfo dminfo;
  PetscFunctionBegin;
  ierr = TaoDMGetContext(taodm,(void**)&user);
  ierr = TaoDMGetDM(taodm,&dm);
  ierr = DMDAGetLocalInfo(dm,&dminfo);
  
  PetscPrintf(MPI_COMM_WORLD,"Grid: %d,    mx: %d     my: %d   \n",level,dminfo.mx,dminfo.my);
  PetscFunctionReturn(0);
  

}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
PetscErrorCode FormInitialGuess(TaoDM taodm, Vec X)
{
  PetscErrorCode ierr;
  AppCtx         *user;
  DM             dm;
  DMDALocalInfo  dminfo;
  PetscInt    i, j;
  PetscInt    xs, ys, xm, ym, xe, ye;
  PetscReal hx, val;
  PetscScalar **x;

  PetscFunctionBegin;
  ierr = TaoDMGetContext(taodm,(void**)&user);
  ierr = TaoDMGetDM(taodm,&dm);
  /* Get local mesh boundaries */
  ierr = DMDAGetLocalInfo(dm,&dminfo); CHKERRQ(ierr);
  xm = dminfo.xm;
  ym = dminfo.ym;
  xs = dminfo.xs;
  ys = dminfo.ys;

  hx = 2.0*4.0*atan(1.0)/((PetscReal)(dminfo.mx-1));
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
  DMDALocalInfo dminfo;
  PetscErrorCode ierr;
  PetscInt i, j, mx, my;
  PetscInt xs, xm, ys, ym;
  PetscScalar **xl, **xu;

  PetscFunctionBegin;  
  ierr = TaoDMGetContext(taodm,(void**)&user); CHKERRQ(ierr);
  ierr = TaoDMGetDM(taodm,&dm); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&dminfo); CHKERRQ(ierr);
  mx = dminfo.mx;
  my = dminfo.my;

  ierr = DMDAVecGetArray(dm, XL, (void**)&xl); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(dm, XU, (void**)&xu); CHKERRQ(ierr);
  xs = dminfo.xs; xm = dminfo.xm;
  ys = dminfo.ys; ym = dminfo.ym;

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

#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradientLocal"
static PetscErrorCode FormFunctionGradientLocal(DMDALocalInfo *dminfo, PetscScalar **x, PetscScalar *f, PetscScalar **g, void *ctx)
{
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx*)ctx;

  PetscScalar area, aread3;
  PetscInt i,j;
  PetscInt xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal hx,hy,dvdx,dvdy;
  PetscReal sqGrad, wq, wv, fl, fu;
  PetscReal xi,pi=4.0*atan(1.0);
  PetscReal ecc=user->ecc;
  PetscReal px,pxp,pxm,sinxi;
  PetscReal f1,f2;
  PetscReal elem[4], gelem[4];

  PetscFunctionBegin;
  *f = 0.0;
  hx = 2.0*pi/(dminfo->mx - 1.0);
  hy = 2.0*user->b/(dminfo->my - 1.0);
  area = 0.5*hx*hy;
  aread3 = area / 3.0;
  
  xm = dminfo->xm; gxm = dminfo->gxm;
  ym = dminfo->ym; gym = dminfo->gym;
  xs = dminfo->xs; gxs = dminfo->gxs;
  ys = dminfo->ys; gys = dminfo->gys;

  f1=0.0; f2=0.0;
  /* Initialize local area of g to zero */
  ierr = PetscMemzero((void*)&(g[dminfo->xs][dminfo->ys]),dminfo->xm*dminfo->ym*sizeof(PetscScalar)); CHKERRQ(ierr);
  for (i=xs; i< xs+xm-1; i++){
    xi=i*hx;
    sinxi = sin(xi);
    px = p(xi,ecc);
    pxp= p(xi+hx,ecc);
    pxm= p(xi-hx,ecc);
    
    for (j=ys; j<ys+ym-1; j++){
      elem[0] = x[j][i];
      elem[1] = x[j][i+1];
      elem[2] = x[j+1][i];
      elem[3] = x[j+1][i+1];

      /* Lower element */
      dvdx = (elem[0] - elem[1]) / hx;
      dvdy = (elem[0] - elem[2]) / hy;
      sqGrad= dvdx*dvdx + dvdy*dvdy;
      wq = (2.0*p(xi,ecc) + p(xi+hx,ecc)) / 6.0;
      wv = ecc*(sin(xi)*elem[0] + sin(xi+hx)*elem[1] + sin(xi)*elem[2]) / 3.0;
      fl = wq*sqGrad - wv;
      dvdx *= hy*wq;
      dvdy *= hx*wq;
      gelem[0] = (dvdx + dvdy) - ecc*sin(xi)*aread3;
      gelem[1] = -dvdx - ecc*sin(xi+hx)*aread3;
      gelem[2] = -dvdy - ecc*sin(xi)*aread3;

      /* Upper element */
      dvdx = (elem[3] - elem[2]) / hx;
      dvdy = (elem[3] - elem[1]) / hy;
      sqGrad = dvdx*dvdx + dvdy*dvdy;
      wq = (2.0*p(xi+hx,ecc) + p(xi,ecc)) / 6.0;
      wv = ecc*(sin(xi+hx)*elem[1] + sin(xi)*elem[2] + sin(xi+hx)*elem[3]) / 3.0;
      fu = wq*sqGrad - wv;

      dvdx *= hy*wq;
      dvdy *= hx*wq;
      gelem[1] += -dvdy - ecc*sin(xi+hx) * aread3;
      gelem[2] += -dvdx - ecc*sin(xi) * aread3;
      gelem[3] = dvdx + dvdy - ecc*sin(xi+hx) * aread3;
      
      g[j][i] += gelem[0];
      g[j][i+1] += gelem[1];
      g[j+1][i] += gelem[2];
      g[j+1][i+1] += gelem[3];
      *f  += (fl + fu);
    }
  }

  *f *= area;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormHessianLocal"
static PetscErrorCode FormHessianLocal(DMDALocalInfo *dminfo, PetscScalar **x, Mat hes, void *ptr)
{
  PetscErrorCode ierr;
  AppCtx *user = (AppCtx*)ptr;
  PetscInt i,j,k;
  PetscInt col[5],row;
  PetscReal one=1.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal hx,hy,hxhy,hxhx,hyhy;
  PetscReal xm,ym,xs,ys,gxm,gym,gxs,gys,area,aread3;
  PetscReal xi,v[5];
  PetscReal ecc=user->ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal vmiddle, vup, vdown, vleft, vright;

  PetscBool assembled;

  PetscFunctionBegin;
  hx = 2.0*pi/(dminfo->mx - 1.0);
  hy = 2.0*user->b/(dminfo->my - 1.0);
  area = 0.5*hx*hy;
  aread3 = area / 3.0;
  
  xm = dminfo->xm; gxm = dminfo->gxm;
  ym = dminfo->ym; gym = dminfo->gym;
  xs = dminfo->xs; gxs = dminfo->gxs;
  ys = dminfo->ys; gys = dminfo->gys;

  hxhy=hx*hy;
  hxhx=one/(hx*hx);
  hyhy=one/(hy*hy);

  /*
    Get local grid boundaries
  */
  
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
  PetscFunctionReturn(0);
}



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
  DMDALocalInfo dminfo;
  PetscErrorCode ierr;
  PetscInt i,j,k;
  PetscInt col[5],row,xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal one=1.0, two=2.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal hx,hy,hxhy,hxhx,hyhy;
  PetscReal xi,v[5];
  PetscReal ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal vmiddle, vup, vdown, vleft, vright;
  Mat hes=*H;
  PetscBool assembled;
  
  PetscFunctionBegin;
  ierr = TaoDMGetContext(taodm,(void**)&user); CHKERRQ(ierr);
  ierr = TaoDMGetDM(taodm,&dm); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&dminfo); CHKERRQ(ierr);
  ecc = user->ecc;
  hx=two*pi/(dminfo.mx+1.0);
  hy=two*user->b/(dminfo.my+1.0);
  hxhy=hx*hy;
  hxhx=one/(hx*hx);
  hyhy=one/(hy*hy);

  *flg=SAME_NONZERO_PATTERN;
  /*
    Get local grid boundaries
  */
  xm = dminfo.xm; gxm = dminfo.gxm;
  ym = dminfo.ym; gym = dminfo.gym;
  xs = dminfo.xs; gxs = dminfo.gxs;
  ys = dminfo.ys; gys = dminfo.gys;
  
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
  DMDALocalInfo dminfo;
  PetscErrorCode ierr;
  PetscInt i,j,k,kk;
  PetscInt col[5],row,xs,xm,gxs,gxm,ys,ym,gys,gym;
  PetscReal one=1.0, two=2.0, six=6.0,pi=4.0*atan(1.0);
  PetscReal hx,hy,hxhy,hxhx,hyhy;
  PetscReal xi,v[5];
  PetscReal ecc, trule1,trule2,trule3,trule4,trule5,trule6;
  PetscReal vmiddle, vup, vdown, vleft, vright;
  PetscReal px,pxp,pxm,sinxi;
  PetscReal tt,f1,f2;
  PetscReal *x,*g,zero=0.0;
  Vec localX;

  PetscFunctionBegin;
  ierr = TaoDMGetContext(taodm,(void**)&user); CHKERRQ(ierr);
  ierr = TaoDMGetDM(taodm,&dm); CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(dm,&dminfo); CHKERRQ(ierr);

  ecc = user->ecc;
  hx=two*pi/(dminfo.mx+1.0);
  hy=two*user->b/(dminfo.my+1.0);
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
  xm = dminfo.xm; gxm = dminfo.gxm;
  ym = dminfo.ym; gym = dminfo.gym;
  xs = dminfo.xs; gxs = dminfo.gxs;
  ys = dminfo.ys; gys = dminfo.gys;
  
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);
  ierr = VecGetArray(G,&g); CHKERRQ(ierr);
  f2=0.0;
  for (i=xs; i< xs+xm; i++){
    xi=(i+1)*hx;
    sinxi = sin(xi);
    px = p(xi,ecc);
    pxp= p(xi+hx,ecc);
    pxm= p(xi-hx,ecc);
    
    trule1=hxhy*( 2*px + pxp) / six; /* L(i,j) */
    trule2=hxhy*( 2*px + pxm) / six; /* U(i,j) */
    trule3=hxhy*( px + 2*pxp) / six; /* U(i+1,j) */
    trule4=hxhy*( px + 2*pxm) / six; /* L(i-1,j) */
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
       g[row]=tt - user->ecc*hxhy*sinxi; // B 
       f2+=user->ecc*hxhy*sinxi * x[row];
     }

  }


  ierr = VecDot(X,G,&f1); CHKERRQ(ierr);
  *fcn = f1/2.0 + f2;
  
  ierr = VecNorm(G,NORM_2,&f2); CHKERRQ(ierr);
  ierr = PetscLogFlops((91 + 10*ym) * xm); CHKERRQ(ierr); //TODO
  PetscFunctionReturn(0);

}


static PetscReal p(PetscReal xi, PetscReal ecc)
{ 
  PetscReal t=1.0+ecc*cos(xi); 
  return (t*t*t); 
}
