
/*$Id: ex22.c,v 1.9 2000/09/28 14:46:50 bsmith Exp balay $*/
/*
Laplacian in 3D. Modeled by the partial differential equation

   Laplacian u = 0,0 < x,y,z < 1,

with boundary conditions

   u = 1 for x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.

   This uses multigrid to solve the linear system

   See ex18.c for a simpler example that does not use multigrid
*/

static char help[] = "Solves 3D Laplacian using multigrid.\n\
The command line options are:\n\
   -mx <xg>, where <xg> = number of grid points in the x-direction\n\
   -my <yg>, where <yg> = number of grid points in the y-direction\n\
   -mz <zg>, where <zg> = number of grid points in the z-direction\n\n";

#include "petscda.h"
#include "petscsles.h"
#include "petscmg.h"



extern int ComputeJacobian(DAMG,Mat);
extern int ComputeRHS(DAMG,Vec);

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int       ierr,sw = 1,dof = 1,mx = 2,my = 2,mz = 2,nlevels = 3;
  DAMG      *damg;
  Scalar    mone = -1.0;
  PetscReal norm;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = OptionsGetInt(0,"-stencil_width",&sw,0);CHKERRQ(ierr);
  ierr = OptionsGetInt(0,"-dof",&dof,0);CHKERRQ(ierr);

  ierr = OptionsGetInt(PETSC_NULL,"-mx",&mx,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-my",&my,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mz",&mz,PETSC_NULL);CHKERRA(ierr);

  ierr = DAMGCreate(PETSC_COMM_WORLD,nlevels,PETSC_NULL,&damg);CHKERRQ(ierr);

  ierr = DAMGSetGrid(damg,3,DA_NONPERIODIC,DA_STENCIL_STAR,mx,my,mz,sw,dof);CHKERRQ(ierr);  

  ierr = DAMGSetSLES(damg,ComputeRHS,ComputeJacobian);CHKERRQ(ierr);

  ierr = DAMGSolve(damg);CHKERRQ(ierr);

  ierr = MatMult(DAMGGetJ(damg),DAMGGetx(damg),DAMGGetr(damg));CHKERRQ(ierr);
  ierr = VecAXPY(&mone,DAMGGetb(damg),DAMGGetr(damg));CHKERRQ(ierr);
  ierr = VecNorm(DAMGGetr(damg),NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %g\n",norm);CHKERRQ(ierr);

  ierr = DAMGDestroy(damg);CHKERRQ(ierr);
  PetscFinalize();

  return 0;
}

#undef __FUNC__
#define __FUNC__ "ComputeRHS"
int ComputeRHS(DAMG damg,Vec b)
{
  int    ierr,mx,my,mz;
  Scalar h;

  PetscFunctionBegin;
  ierr = DAGetInfo(damg->da,0,&mx,&my,&mz,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h    = 1.0/((mx-1)*(my-1)*(mz-1));
  ierr = VecSet(&h,b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
    
#undef __FUNC__
#define __FUNC__ "ComputeJacobian"
int ComputeJacobian(DAMG damg,Mat jac)
{
  DA     da = damg->da;
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


