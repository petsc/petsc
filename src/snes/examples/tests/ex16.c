#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

static  char help[]=
"This example is an implementation of minimal surface area with \n\
a plate problem from the TAO package (examples/plate2.c) \n\
This example is based on a problem from the MINPACK-2 test suite.\n\
Given a rectangular 2-D domain, boundary values along the edges of \n\
the domain, and a plate represented by lower boundary conditions, \n\
the objective is to find the surface with the minimal area that \n\
satisfies the boundary conditions.\n\
The command line options are:\n\
  -bmx <bxg>, where <bxg> = number of grid points under plate in 1st direction\n\
  -bmy <byg>, where <byg> = number of grid points under plate in 2nd direction\n\
  -bheight <ht>, where <ht> = height of the plate\n\
  -start <st>, where <st> =0 for zero vector, <st> != 0 \n\
               for an average of the boundary conditions\n\n";

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobian() and
   FormFunction().
*/

typedef struct {
  DM          da;
  Vec         Bottom, Top, Left, Right;
  PetscScalar bheight;
  PetscInt    mx,my,bmx,bmy;
} AppCtx;


/* -------- User-defined Routines --------- */

PetscErrorCode MSA_BoundaryConditions(AppCtx*);
PetscErrorCode MSA_InitialPoint(AppCtx*, Vec);
PetscErrorCode MSA_Plate(Vec,Vec,void*);
PetscErrorCode FormGradient(SNES, Vec, Vec, void*);
PetscErrorCode FormJacobian(SNES, Vec, Mat, Mat, void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode ierr;              /* used to check for functions returning nonzeros */
  Vec            x,r;               /* solution and residual vectors */
  Vec            xl,xu;             /* Bounds on the variables */
  SNES           snes;              /* nonlinear solver context */
  Mat            J;                 /* Jacobian matrix */
  AppCtx         user;              /* user-defined work context */
  PetscBool      flg;

  /* Initialize PETSc */
  PetscInitialize(&argc, &argv, (char*)0, help);

#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This example does not work for scalar type complex\n");
#endif

  /* Create distributed array to manage the 2d grid */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,-10,-10,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&user.da);CHKERRQ(ierr);
  ierr = DMDAGetIerr(user.da,PETSC_IGNORE,&user.mx,&user.my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

  user.bheight = 0.1;
  ierr         = PetscOptionsGetScalar(NULL,NULL,"-bheight",&user.bheight,&flg);CHKERRQ(ierr);

  user.bmx = user.mx/2; user.bmy = user.my/2;
  ierr     = PetscOptionsGetInt(NULL,NULL,"-bmx",&user.bmx,&flg);CHKERRQ(ierr);
  ierr     = PetscOptionsGetInt(NULL,NULL,"-bmy",&user.bmy,&flg);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"\n---- Minimum Surface Area With Plate Problem -----\n");
  PetscPrintf(PETSC_COMM_WORLD,"mx:%d, my:%d, bmx:%d, bmy:%d, height:%4.2f\n",
              user.mx,user.my,user.bmx,user.bmy,user.bheight);

  /* Extract global vectors from DMDA; */
  ierr = DMCreateGlobalVector(user.da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &r);CHKERRQ(ierr);

  ierr = DMSetMatType(user.da,MATAIJ);CHKERRQ(ierr);
  ierr = DMCreateMatrix(user.da,&J);CHKERRQ(ierr);

  /* Create nonlinear solver context */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /*  Set function evaluation and Jacobian evaluation  routines */
  ierr = SNESSetDM(snes,user.da);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes,r,FormGradient,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,&user);CHKERRQ(ierr);

  /* Set the boundary conditions */
  ierr = MSA_BoundaryConditions(&user);CHKERRQ(ierr);

  /* Set initial solution guess */
  ierr = MSA_InitialPoint(&user, x);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* Set Bounds on variables */
  ierr = VecDuplicate(x, &xl);CHKERRQ(ierr);
  ierr = VecDuplicate(x, &xu);CHKERRQ(ierr);
  ierr = MSA_Plate(xl,xu,&user);CHKERRQ(ierr);

  ierr = SNESVISetVariableBounds(snes,xl,xu);CHKERRQ(ierr);

  /* Solve the application */
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-view_sol",&flg);CHKERRQ(ierr);
  if (flg) { ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); }

  /* Free memory */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&xl);CHKERRQ(ierr);
  ierr = VecDestroy(&xu);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);

  /* Free user-created data structures */
  ierr = DMDestroy(&user.da);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Bottom);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Top);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Left);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Right);CHKERRQ(ierr);

  ierr = PetscFinalize();

  return 0;
}

/* -------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormGradient"

/*  FormGradient - Evaluates gradient of f.

    Input Parameters:
.   snes  - the SNES context
.   X     - input vector
.   ptr   - optional user-defined context, as set by SNESSetFunction()

    Output Parameters:
.   G - vector containing the newly evaluated gradient
*/
PetscErrorCode FormGradient(SNES snes, Vec X, Vec G, void *ptr)
{
  AppCtx      *user = (AppCtx*) ptr;
  int         ierr;
  PetscInt    i,j;
  PetscInt    mx=user->mx, my=user->my;
  PetscScalar hx=1.0/(mx+1),hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy;
  PetscScalar f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscScalar df1dxc,df2dxc,df3dxc,df4dxc,df5dxc,df6dxc;
  PetscScalar **g, **x;
  PetscInt    xs,xm,ys,ym;
  Vec         localX;
  PetscScalar *top,*bottom,*left,*right;

  PetscFunctionBeginUser;
  /* Initialize vector to zero */
  ierr = VecSet(G,0.0);CHKERRQ(ierr);

  /* Get local vector */
  ierr = DMGetLocalVector(user->da,&localX);CHKERRQ(ierr);
  ierr = VecGetArray(user->Top,&top);CHKERRQ(ierr);
  ierr = VecGetArray(user->Bottom,&bottom);CHKERRQ(ierr);
  ierr = VecGetArray(user->Left,&left);CHKERRQ(ierr);
  ierr = VecGetArray(user->Right,&right);CHKERRQ(ierr);

  /* Get ghost points */
  ierr = DMGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  /* Get pointers to local vector data */
  ierr = DMDAVecGetArrayRead(user->da,localX, &x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,G, &g);CHKERRQ(ierr);

  ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  /* Compute function over the locally owned part of the mesh */
  for (j=ys; j < ys+ym; j++) {
    for (i=xs; i< xs+xm; i++) {

      xc = x[j][i];
      xlt=xrb=xl=xr=xb=xt=xc;

      if (i==0) { /* left side */
        xl  = left[j-ys+1];
        xlt = left[j-ys+2];
      } else xl = x[j][i-1];

      if (j==0) { /* bottom side */
        xb  = bottom[i-xs+1];
        xrb = bottom[i-xs+2];
      } else xb = x[j-1][i];

      if (i+1 == mx) { /* right side */
        xr  = right[j-ys+1];
        xrb = right[j-ys];
      } else xr = x[j][i+1];

      if (j+1==0+my) { /* top side */
        xt  = top[i-xs+1];
        xlt = top[i-xs];
      } else xt = x[j+1][i];

      if (i>0 && j+1<my) xlt = x[j+1][i-1]; /* left top side */
      if (j>0 && i+1<mx) xrb = x[j-1][i+1]; /* right bottom */

      d1 = (xc-xl);
      d2 = (xc-xr);
      d3 = (xc-xt);
      d4 = (xc-xb);
      d5 = (xr-xrb);
      d6 = (xrb-xb);
      d7 = (xlt-xl);
      d8 = (xt-xlt);

      df1dxc = d1*hydhx;
      df2dxc = (d1*hydhx + d4*hxdhy);
      df3dxc = d3*hxdhy;
      df4dxc = (d2*hydhx + d3*hxdhy);
      df5dxc = d2*hydhx;
      df6dxc = d4*hxdhy;

      d1 /= hx;
      d2 /= hx;
      d3 /= hy;
      d4 /= hy;
      d5 /= hy;
      d6 /= hx;
      d7 /= hy;
      d8 /= hx;

      f1 = PetscSqrtReal(1.0 + d1*d1 + d7*d7);
      f2 = PetscSqrtReal(1.0 + d1*d1 + d4*d4);
      f3 = PetscSqrtReal(1.0 + d3*d3 + d8*d8);
      f4 = PetscSqrtReal(1.0 + d3*d3 + d2*d2);
      f5 = PetscSqrtReal(1.0 + d2*d2 + d5*d5);
      f6 = PetscSqrtReal(1.0 + d4*d4 + d6*d6);

      df1dxc /= f1;
      df2dxc /= f2;
      df3dxc /= f3;
      df4dxc /= f4;
      df5dxc /= f5;
      df6dxc /= f6;

      g[j][i] = (df1dxc+df2dxc+df3dxc+df4dxc+df5dxc+df6dxc)/2.0;

    }
  }

  /* Restore vectors */
  ierr = DMDAVecRestoreArrayRead(user->da,localX, &x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(user->da,G, &g);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da,&localX);CHKERRQ(ierr);

  ierr = VecRestoreArray(user->Left,&left);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Top,&top);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Bottom,&bottom);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Right,&right);CHKERRQ(ierr);

  ierr = PetscLogFlops(67*mx*my);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
/*
   FormJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - SNES context
.  X    - input vector
.  ptr  - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
.  tH    - Jacobian matrix

*/
PetscErrorCode FormJacobian(SNES snes, Vec X, Mat H, Mat tHPre, void *ptr)
{
  AppCtx         *user = (AppCtx*) ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,k;
  PetscInt       mx=user->mx, my=user->my;
  MatStencil     row,col[7];
  PetscScalar    hx=1.0/(mx+1), hy=1.0/(my+1), hydhx=hy/hx, hxdhy=hx/hy;
  PetscScalar    f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscScalar    hl,hr,ht,hb,hc,htl,hbr;
  PetscScalar    **x, v[7];
  PetscBool      assembled;
  PetscInt       xs,xm,ys,ym;
  Vec            localX;
  PetscScalar    *top,*bottom,*left,*right;

  PetscFunctionBeginUser;
  /* Set various matrix options */
  ierr = MatAssembled(H,&assembled);CHKERRQ(ierr);
  if (assembled) {ierr = MatZeroEntries(H);CHKERRQ(ierr);}

  /* Get local vectors */
  ierr = DMGetLocalVector(user->da,&localX);CHKERRQ(ierr);
  ierr = VecGetArray(user->Top,&top);CHKERRQ(ierr);
  ierr = VecGetArray(user->Bottom,&bottom);CHKERRQ(ierr);
  ierr = VecGetArray(user->Left,&left);CHKERRQ(ierr);
  ierr = VecGetArray(user->Right,&right);CHKERRQ(ierr);

  /* Get ghost points */
  ierr = DMGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /* Get pointers to vector data */
  ierr = DMDAVecGetArrayRead(user->da,localX, &x);CHKERRQ(ierr);

  ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  /* Compute Jacobian over the locally owned part of the mesh */
  for (j=ys; j< ys+ym; j++) {
    for (i=xs; i< xs+xm; i++) {
      xc = x[j][i];
      xlt=xrb=xl=xr=xb=xt=xc;

      /* Left */
      if (i==0) {
        xl  = left[j+1];
        xlt = left[j+2];
      } else xl = x[j][i-1];

      /* Bottom */
      if (j==0) {
        xb  = bottom[i+1];
        xrb = bottom[i+2];
      } else xb = x[j-1][i];

      /* Right */
      if (i+1 == mx) {
        xr  = right[j+1];
        xrb = right[j];
      } else xr = x[j][i+1];

      /* Top */
      if (j+1==my) {
        xt  = top[i+1];
        xlt = top[i];
      } else xt = x[j+1][i];

      /* Top left */
      if (i>0 && j+1<my) xlt = x[j+1][i-1];

      /* Bottom right */
      if (j>0 && i+1<mx) xrb = x[j-1][i+1];

      d1 = (xc-xl)/hx;
      d2 = (xc-xr)/hx;
      d3 = (xc-xt)/hy;
      d4 = (xc-xb)/hy;
      d5 = (xrb-xr)/hy;
      d6 = (xrb-xb)/hx;
      d7 = (xlt-xl)/hy;
      d8 = (xlt-xt)/hx;

      f1 = PetscSqrtReal(1.0 + d1*d1 + d7*d7);
      f2 = PetscSqrtReal(1.0 + d1*d1 + d4*d4);
      f3 = PetscSqrtReal(1.0 + d3*d3 + d8*d8);
      f4 = PetscSqrtReal(1.0 + d3*d3 + d2*d2);
      f5 = PetscSqrtReal(1.0 + d2*d2 + d5*d5);
      f6 = PetscSqrtReal(1.0 + d4*d4 + d6*d6);


      hl = (-hydhx*(1.0+d7*d7)+d1*d7)/(f1*f1*f1)+
           (-hydhx*(1.0+d4*d4)+d1*d4)/(f2*f2*f2);
      hr = (-hydhx*(1.0+d5*d5)+d2*d5)/(f5*f5*f5)+
           (-hydhx*(1.0+d3*d3)+d2*d3)/(f4*f4*f4);
      ht = (-hxdhy*(1.0+d8*d8)+d3*d8)/(f3*f3*f3)+
           (-hxdhy*(1.0+d2*d2)+d2*d3)/(f4*f4*f4);
      hb = (-hxdhy*(1.0+d6*d6)+d4*d6)/(f6*f6*f6)+
           (-hxdhy*(1.0+d1*d1)+d1*d4)/(f2*f2*f2);

      hbr = -d2*d5/(f5*f5*f5) - d4*d6/(f6*f6*f6);
      htl = -d1*d7/(f1*f1*f1) - d3*d8/(f3*f3*f3);

      hc = hydhx*(1.0+d7*d7)/(f1*f1*f1) + hxdhy*(1.0+d8*d8)/(f3*f3*f3) +
           hydhx*(1.0+d5*d5)/(f5*f5*f5) + hxdhy*(1.0+d6*d6)/(f6*f6*f6) +
           (hxdhy*(1.0+d1*d1)+hydhx*(1.0+d4*d4)-2*d1*d4)/(f2*f2*f2) +
           (hxdhy*(1.0+d2*d2)+hydhx*(1.0+d3*d3)-2*d2*d3)/(f4*f4*f4);

      hl/=2.0; hr/=2.0; ht/=2.0; hb/=2.0; hbr/=2.0; htl/=2.0;  hc/=2.0;

      k     = 0;
      row.i = i;row.j = j;
      /* Bottom */
      if (j>0) {
        v[k]     = hb;
        col[k].i = i; col[k].j=j-1; k++;
      }

      /* Bottom right */
      if (j>0 && i < mx -1) {
        v[k]     = hbr;
        col[k].i = i+1; col[k].j = j-1; k++;
      }

      /* left */
      if (i>0) {
        v[k]     = hl;
        col[k].i = i-1; col[k].j = j; k++;
      }

      /* Centre */
      v[k]= hc; col[k].i= row.i; col[k].j = row.j; k++;

      /* Right */
      if (i < mx-1) {
        v[k]    = hr;
        col[k].i= i+1; col[k].j = j;k++;
      }

      /* Top left */
      if (i>0 && j < my-1) {
        v[k]     = htl;
        col[k].i = i-1;col[k].j = j+1; k++;
      }

      /* Top */
      if (j < my-1) {
        v[k]     = ht;
        col[k].i = i; col[k].j = j+1; k++;
      }

      ierr = MatSetValuesStencil(H,1,&row,k,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  ierr = VecRestoreArray(user->Left,&left);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Top,&top);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Bottom,&bottom);CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Right,&right);CHKERRQ(ierr);

  /* Assemble the matrix */
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(user->da,localX,&x);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->da,&localX);CHKERRQ(ierr);

  ierr = PetscLogFlops(199*mx*my);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MSA_BoundaryConditions"
/*
   MSA_BoundaryConditions -  Calculates the boundary conditions for
   the region.

   Input Parameter:
.  user - user-defined application context

   Output Parameter:
.  user - user-defined application context
*/
PetscErrorCode MSA_BoundaryConditions(AppCtx * user)
{
  PetscErrorCode ierr;
  PetscInt       i,j,k,limit=0,maxits=5;
  PetscInt       mx=user->mx,my=user->my;
  PetscInt       xs,ys,xm,ym;
  PetscInt       bsize=0, lsize=0, tsize=0, rsize=0;
  PetscScalar    one  =1.0, two=2.0, three=3.0, tol=1e-10;
  PetscScalar    fnorm,det,hx,hy,xt=0,yt=0;
  PetscScalar    u1,u2,nf1,nf2,njac11,njac12,njac21,njac22;
  PetscScalar    b=-0.5, t=0.5, l=-0.5, r=0.5;
  PetscScalar    *boundary;
  Vec            Bottom,Top,Right,Left;
  PetscScalar    scl=1.0;
  PetscBool      flg;

  PetscFunctionBeginUser;
  ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  bsize=xm+2; lsize=ym+2; rsize=ym+2; tsize=xm+2;

  ierr = VecCreateMPI(PETSC_COMM_WORLD,bsize,PETSC_DECIDE,&Bottom);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,tsize,PETSC_DECIDE,&Top);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,lsize,PETSC_DECIDE,&Left);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,rsize,PETSC_DECIDE,&Right);CHKERRQ(ierr);

  user->Top    = Top;
  user->Left   = Left;
  user->Bottom = Bottom;
  user->Right  = Right;

  hx= (r-l)/(mx+1); hy=(t-b)/(my+1);

  for (j=0; j<4; j++) {
    if (j==0) {
      yt    = b;
      xt    = l+hx*xs;
      limit = bsize;
      ierr  = VecGetArray(Bottom,&boundary);CHKERRQ(ierr);
    } else if (j==1) {
      yt   = t;
      xt   = l+hx*xs;
      limit= tsize;
      ierr = VecGetArray(Top,&boundary);CHKERRQ(ierr);
    } else if (j==2) {
      yt   = b+hy*ys;
      xt   = l;
      limit= lsize;
      ierr = VecGetArray(Left,&boundary);CHKERRQ(ierr);
    } else { /* if  (j==3) */
      yt   = b+hy*ys;
      xt   = r;
      limit= rsize;
      ierr = VecGetArray(Right,&boundary);CHKERRQ(ierr);
    }

    for (i=0; i<limit; i++) {
      u1=xt;
      u2=-yt;
      for (k=0; k<maxits; k++) {
        nf1   = u1 + u1*u2*u2 - u1*u1*u1/three-xt;
        nf2   = -u2 - u1*u1*u2 + u2*u2*u2/three-yt;
        fnorm = PetscSqrtReal(nf1*nf1+nf2*nf2);
        if (fnorm <= tol) break;
        njac11 = one+u2*u2-u1*u1;
        njac12 = two*u1*u2;
        njac21 = -two*u1*u2;
        njac22 = -one - u1*u1 + u2*u2;
        det    = njac11*njac22-njac21*njac12;
        u1     = u1-(njac22*nf1-njac12*nf2)/det;
        u2     = u2-(njac11*nf2-njac21*nf1)/det;
      }

      boundary[i]=u1*u1-u2*u2;
      if (j==0 || j==1) xt=xt+hx;
      else yt=yt+hy; /* if (j==2 || j==3) */
    }

    if (j==0) {
      ierr = VecRestoreArray(Bottom,&boundary);CHKERRQ(ierr);
    } else if (j==1) {
      ierr = VecRestoreArray(Top,&boundary);CHKERRQ(ierr);
    } else if (j==2) {
      ierr = VecRestoreArray(Left,&boundary);CHKERRQ(ierr);
    } else if (j==3) {
      ierr = VecRestoreArray(Right,&boundary);CHKERRQ(ierr);
    }

  }

  /* Scale the boundary if desired */

  ierr = PetscOptionsGetReal(NULL,NULL,"-bottom",&scl,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecScale(Bottom, scl);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetReal(NULL,NULL,"-top",&scl,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecScale(Top, scl);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetReal(NULL,NULL,"-right",&scl,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecScale(Right, scl);CHKERRQ(ierr);
  }

  ierr = PetscOptionsGetReal(NULL,NULL,"-left",&scl,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = VecScale(Left, scl);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "MSA_InitialPoint"
/*
   MSA_InitialPoint - Calculates the initial guess in one of three ways.

   Input Parameters:
.  user - user-defined application context
.  X - vector for initial guess

   Output Parameters:
.  X - newly computed initial guess
*/
PetscErrorCode MSA_InitialPoint(AppCtx * user, Vec X)
{
  PetscErrorCode ierr;
  PetscInt       start =-1,i,j;
  PetscScalar    zero  =0.0;
  PetscBool      flg;
  PetscScalar    *left,*right,*bottom,*top;

  PetscFunctionBeginUser;
  ierr = PetscOptionsGetInt(NULL,NULL,"-start",&start,&flg);CHKERRQ(ierr);

  if (flg && start==0) { /* The zero vector is reasonable */

    ierr = VecSet(X, zero);CHKERRQ(ierr);
    /* PLogIerr(user,"Min. Surface Area Problem: Start with 0 vector \n"); */


  } else { /* Take an average of the boundary conditions */
    PetscInt    mx=user->mx,my=user->my;
    PetscScalar **x;
    PetscInt    xs,xm,ys,ym;

    ierr = VecGetArray(user->Top,&top);CHKERRQ(ierr);
    ierr = VecGetArray(user->Bottom,&bottom);CHKERRQ(ierr);
    ierr = VecGetArray(user->Left,&left);CHKERRQ(ierr);
    ierr = VecGetArray(user->Right,&right);CHKERRQ(ierr);

    /* Get pointers to vector data */
    ierr = DMDAVecGetArray(user->da,X,&x);CHKERRQ(ierr);
    ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

    /* Perform local computations */
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i< xs+xm; i++) {
        x[j][i] = ((j+1)*bottom[i-xs+1]/my+(my-j+1)*top[i-xs+1]/(my+2)+
                   (i+1)*left[j-ys+1]/mx+(mx-i+1)*right[j-ys+1]/(mx+2))/2.0;
      }
    }

    /* Restore vectors */
    ierr = DMDAVecRestoreArray(user->da,X,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(user->Left,&left);CHKERRQ(ierr);
    ierr = VecRestoreArray(user->Top,&top);CHKERRQ(ierr);
    ierr = VecRestoreArray(user->Bottom,&bottom);CHKERRQ(ierr);
    ierr = VecRestoreArray(user->Right,&right);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MSA_Plate"
/*
   MSA_Plate -  Calculates an obstacle for surface to stretch over.
*/
PetscErrorCode MSA_Plate(Vec XL,Vec XU,void *ctx)
{
  AppCtx         *user=(AppCtx*)ctx;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscInt       xs,ys,xm,ym;
  PetscInt       mx=user->mx, my=user->my, bmy, bmx;
  PetscScalar    t1,t2,t3;
  PetscScalar    **xl;
  PetscScalar    lb=-PETSC_INFINITY, ub=PETSC_INFINITY;
  PetscBool      cylinder;

  user->bmy = PetscMax(0,user->bmy);user->bmy = PetscMin(my,user->bmy);
  user->bmx = PetscMax(0,user->bmx);user->bmx = PetscMin(mx,user->bmx);

  bmy=user->bmy, bmx=user->bmx;

  ierr = DMDAGetCorners(user->da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);
  ierr = VecSet(XL, lb);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(user->da,XL,&xl);CHKERRQ(ierr);
  ierr = VecSet(XU, ub);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-cylinder",&cylinder);CHKERRQ(ierr);
  /* Compute the optional lower box */
  if (cylinder) {
    for (i=xs; i< xs+xm; i++) {
      for (j=ys; j<ys+ym; j++) {
        t1=(2.0*i-mx)*bmy;
        t2=(2.0*j-my)*bmx;
        t3=bmx*bmx*bmy*bmy;
        if (t1*t1 + t2*t2 <= t3) xl[j][i] = user->bheight;
      }
    }
  } else {
    /* Compute the optional lower box */
    for (i=xs; i< xs+xm; i++) {
      for (j=ys; j<ys+ym; j++) {
        if (i>=(mx-bmx)/2 && i<mx-(mx-bmx)/2 &&
            j>=(my-bmy)/2 && j<my-(my-bmy)/2) {
          xl[j][i] = user->bheight;
        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(user->da,XL,&xl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
