/*$Id: ex20.c,v 1.6 2000/05/05 22:18:00 balay Exp bsmith $*/
/*
Laplacian in 3D. Modeled by the partial differential equation

   Laplacian u = 0,0 < x,y,z < 1,

with boundary conditions

   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.

   This uses multigrid to solve the linear system

   See ex18.c for a simpler example that does not use multigrid
*/

static char help[] = "Solves Laplacian in 3D using multigrid\n\
The command line options are:\n\
   -mx <xg>, where <xg> = number of grid points in the x-direction\n\
   -my <yg>, where <yg> = number of grid points in the y-direction\n\
   -mz <zg>, where <zg> = number of grid points in the z-direction\n\n";

#include "petscda.h"
#include "petscsles.h"
#include "petscmg.h"

#define MAX_LEVELS 12

typedef struct {
   DA         da;
   Vec        x,b,r;            /* global vectors */
   Mat        J;                /* Jacobian on grid */
   SLES       sles;
   Mat        R;                /* R and Rscale are not set on the coarsest grid */
} GridCtx;

#define MAX_LEVELS 12

typedef struct {
   GridCtx     grid[MAX_LEVELS];
   int         ratio;            /* ratio of grid lines between grid levels */
   int         nlevels;
} AppCtx;


extern int ComputeJacobian(DA,Mat);

int main(int argc,char **argv)
{
  MPI_Comm     comm;
  SLES         sles;
  PC           pc;
  Vec          x,b,r;
  Mat          J;
  DA           da;
  AppCtx       user;
  int          Nx = PETSC_DECIDE,Ny = PETSC_DECIDE,Nz = PETSC_DECIDE,i;
  int          ierr,N;
  int          its,m,mx,my,mz;
  double       norm;
  Scalar       one = 1.0,mone = -1.0;
  PetscTruth   no_output;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;

  ierr = OptionsHasName(PETSC_NULL,"-no_output",&no_output);CHKERRA(ierr);
  
  user.ratio      = 2;
  user.nlevels    = 2;
  mx = 5;
  my = 5;
  mz = 5; 
  ierr = OptionsGetInt(PETSC_NULL,"-ratio",&user.ratio,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-nlevels",&user.nlevels,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mx",&mx,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-my",&my,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mz",&mz,PETSC_NULL);CHKERRA(ierr);

  ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Nz",&Nz,PETSC_NULL);CHKERRA(ierr);

  ierr = SLESCreate(comm,&sles);CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCMG);CHKERRA(ierr);
  ierr = MGSetLevels(pc,user.nlevels);CHKERRA(ierr);
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);

  for (i=0; i<user.nlevels; i++) {
    ierr = DACreate3d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,mx,my,mz,Nx,Ny,Nz,1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&da);CHKERRA(ierr);
    ierr = DACreateGlobalVector(da,&x);CHKERRA(ierr);
    ierr = VecDuplicate(x,&b);CHKERRA(ierr);
    ierr = VecDuplicate(x,&r);CHKERRA(ierr);
    ierr = VecGetLocalSize(x,&m);CHKERRA(ierr);
    ierr = VecGetSize(x,&N);CHKERRA(ierr);
    ierr = MatCreateMPIAIJ(comm,m,m,N,N,7,PETSC_NULL,5,PETSC_NULL,&J);CHKERRA(ierr);
    ierr = ComputeJacobian(da,J);CHKERRA(ierr);

    ierr = MGGetSmoother(pc,i,&user.grid[i].sles);CHKERRA(ierr);
    ierr = SLESSetFromOptions(user.grid[i].sles);CHKERRA(ierr);
    ierr = SLESSetOperators(user.grid[i].sles,J,J,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
    ierr = MGSetX(pc,i,x);CHKERRA(ierr); 
    ierr = MGSetRhs(pc,i,b);CHKERRA(ierr); 
    ierr = MGSetR(pc,i,r);CHKERRA(ierr); 
    ierr = MGSetResidual(pc,i,MGDefaultResidual,J);CHKERRA(ierr);

    user.grid[i].da = da;
    user.grid[i].J  = J;
    user.grid[i].x  = x;
    user.grid[i].b  = b;
    user.grid[i].r  = r;

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Grid %d size %d by %d by %d\n",i,mx,my,mz);CHKERRA(ierr);
    mx = user.ratio*(mx-1)+1; 
    my = user.ratio*(my-1)+1;
    mz = user.ratio*(mz-1)+1;
  }

  /* Create interpolation between the levels */
  for (i=1; i<user.nlevels; i++) {
    ierr = DAGetInterpolation(user.grid[i-1].da,user.grid[i].da,&user.grid[i].R,PETSC_NULL);CHKERRA(ierr);
    ierr = MGSetInterpolate(pc,i,user.grid[i].R);CHKERRA(ierr); 
    ierr = MGSetRestriction(pc,i,user.grid[i].R);CHKERRA(ierr); 
  }
  
  ierr = VecSet(&one,b);CHKERRA(ierr);
  ierr = SLESSetOperators(sles,J,J,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);

  /* compute residual */
  ierr = MatMult(J,x,r);CHKERRA(ierr);
  ierr = VecAXPY(&mone,b,r);CHKERRA(ierr);
  ierr = VecNorm(r,NORM_2,&norm);CHKERRA(ierr);

  if (!no_output) {
    ierr = PetscPrintf(comm,"Residual norm %A iterations = %d,\n",norm,its);CHKERRA(ierr);
  }

  for (i=1; i<user.nlevels; i++) {
    ierr = MatDestroy(user.grid[i].R);CHKERRA(ierr);
  }
  for (i=0; i<user.nlevels; i++) {
    ierr = DADestroy(user.grid[i].da);CHKERRA(ierr);
    ierr = MatDestroy(user.grid[i].J);CHKERRA(ierr);
    ierr = VecDestroy(user.grid[i].x);CHKERRA(ierr);
    ierr = VecDestroy(user.grid[i].b);CHKERRA(ierr);
    ierr = VecDestroy(user.grid[i].r);CHKERRA(ierr);
  }
  ierr = SLESDestroy(sles);CHKERRA(ierr);

  
  PetscFinalize();

  return 0;
}

    
int ComputeJacobian(DA da,Mat jac)
{
  int    *ltog,ierr,i,j,k,mx,my,mz,xm,ym,zm,xs,ys,zs,Xm,Ym,Zm,Xs,Ys,Zs,row,nloc,col[7],base1,grow;
  Scalar two = 2.0,one = 1.0,v[7],Hx,Hy,Hz,HxHydHz,HyHzdHx,HxHzdHy;

  ierr = DAGetInfo(da,0,&mx,&my,&mz,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  Hx = one / (double)(mx-1); Hy = one / (double)(my-1); Hz = one / (double)(mz-1);
  HxHydHz = Hx*Hy/Hz; HxHzdHy = Hx*Hz/Hy; HyHzdHx = Hy*Hz/Hx;
  ierr = DAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(da,&nloc,&ltog);CHKERRQ(ierr);
  
  for (k=zs; k<zs+zm; k++){
    base1 = (k-Zs)*(Xm*Ym);
    for (j=ys; j<ys+ym; j++){
      row = base1 + (j-Ys)*Xm + xs - Xs - 1;
      for(i=xs; i<xs+xm; i++){
	row++;
	grow = ltog[row];
	if (i==0 || j==0 || k==0 || i==mx-1 || j==my-1 || k==mz-1){
	  ierr = MatSetValues(jac,1,&grow,1,&grow,&one,INSERT_VALUES);   CHKERRQ(ierr);
	  continue;
	}
	v[0] = -HxHydHz; col[0] = ltog[row - Xm*Ym];
	v[1] = -HxHzdHy; col[1] = ltog[row - Xm];
	v[2] = -HyHzdHx; col[2] = ltog[row - 1];
	v[3] = two*(HxHydHz + HxHzdHy + HyHzdHx); col[3]=grow;
	v[4] = -HyHzdHx; col[4] = ltog[row + 1];
	v[5] = -HxHzdHy; col[5] = ltog[row + Xm];
	v[6] = -HxHydHz; col[6] = ltog[row + Xm*Ym];
	ierr = MatSetValues(jac,1,&grow,7,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  return 0;
}
