/*$Id: ex22.c,v 1.1 2000/07/06 17:44:11 bsmith Exp bsmith $*/
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



extern int ComputeJacobian(DA,Mat);

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int  ierr,i,sw = 1,dof = 1,mx = 2,my = 2,mz = 2;
  DAMG *ctx;
  DA   cda; /* DA for the coarsest grid */
  SLES sles;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = OptionsGetInt(0,"-stencil_width",&sw,0);CHKERRQ(ierr);
  ierr = OptionsGetInt(0,"-dof",&dof,0);CHKERRQ(ierr);
  ierr = OptionsGetInt(0,"-mx",&mx,0);CHKERRQ(ierr);
  ierr = OptionsGetInt(0,"-my",&my,0);CHKERRQ(ierr);
  ierr = OptionsGetInt(0,"-mz",&mz,0);CHKERRQ(ierr);

  ierr = DAMGCreate(PETSC_COMM_WORLD,3,&ctx);CHKERRQ(ierr);

  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,mx,my,mz,PETSC_DECIDE,
                    PETSC_DECIDE,PETSC_DECIDE,sw,dof,0,0,0,&cda);CHKERRQ(ierr);  
  ierr = DASetFieldName(cda,0,"First field");CHKERRQ(ierr);
  ierr = DASetUniformCoordinates(cda,-1.0,1.0,-2.0,2.0,-3.0,3.0);CHKERRQ(ierr);
  ierr = DAMGSetCoarseDA(ctx,cda);CHKERRQ(ierr);
  ierr = DADestroy(cda);CHKERRQ(ierr);

  ierr = DAMGSetOperator(ctx,ComputeJacobian);CHKERRQ(ierr);

  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRQ(ierr);
  ierr = DAMGSetSLES(ctx,sles);CHKERRQ(ierr);

  ierr = DAMGView(ctx,VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DAMGDestroy(ctx);CHKERRQ(ierr);
  ierr = SLESDestroy(sles);CHKERRQ(ierr);
  PetscFinalize();

  return 0;
}

    
#undef __FUNC__
#define __FUNC__ "ComputeJacobian"
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
