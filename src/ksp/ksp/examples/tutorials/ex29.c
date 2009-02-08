/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 2d
   Processors: n
T*/

/*
Added at the request of Marc Garbey.

Inhomogeneous Laplacian in 2D. Modeled by the partial differential equation

   div \rho grad u = f,  0 < x,y < 1,

with forcing function

   f = e^{-(1 - x)^2/\nu} e^{-(1 - y)^2/\nu}

with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

or pure Neumman boundary conditions

This uses multigrid to solve the linear system
*/

static char help[] = "Solves 2D inhomogeneous Laplacian using multigrid.\n\n";

#include "petscda.h"
#include "petscksp.h"
#include "petscmg.h"
#include "petscdmmg.h"

extern PetscErrorCode ComputeMatrix(DMMG,Mat,Mat);
extern PetscErrorCode ComputeRHS(DMMG,Vec);
extern PetscErrorCode VecView_VTK(Vec, const char [], const char []);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct {
  PetscScalar   rho;
  PetscScalar   nu;
  BCType        bcType;
} UserContext;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;
  DA             da;
  UserContext    user;
  PetscReal      norm;
  const char     *bcTypes[2] = {"dirichlet","neumann"};
  PetscErrorCode ierr;
  PetscInt       l,bc;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = DMMGCreate(PETSC_COMM_WORLD,3,PETSC_NULL,&dmmg);CHKERRQ(ierr);
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,-3,-3,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da);CHKERRQ(ierr);  
  ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  for (l = 0; l < DMMGGetLevels(dmmg); l++) {
    ierr = DMMGSetUser(dmmg,l,&user);CHKERRQ(ierr);
  }

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for the inhomogeneous Poisson equation", "DMMG");
    user.rho    = 1.0;
    ierr        = PetscOptionsScalar("-rho", "The conductivity", "ex29.c", user.rho, &user.rho, PETSC_NULL);CHKERRQ(ierr);
    user.nu     = 0.1;
    ierr        = PetscOptionsScalar("-nu", "The width of the Gaussian source", "ex29.c", user.nu, &user.nu, PETSC_NULL);CHKERRQ(ierr);
    bc          = (PetscInt)DIRICHLET;
    ierr        = PetscOptionsEList("-bc_type","Type of boundary condition","ex29.c",bcTypes,2,bcTypes[0],&bc,PETSC_NULL);CHKERRQ(ierr);
    user.bcType = (BCType)bc;
  ierr = PetscOptionsEnd();

  ierr = DMMGSetKSP(dmmg,ComputeRHS,ComputeMatrix);CHKERRQ(ierr);
  if (user.bcType == NEUMANN) {
    ierr = DMMGSetNullSpace(dmmg,PETSC_TRUE,0,PETSC_NULL);CHKERRQ(ierr);
  }

  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);

  ierr = MatMult(DMMGGetJ(dmmg),DMMGGetx(dmmg),DMMGGetr(dmmg));CHKERRQ(ierr);
  ierr = VecAXPY(DMMGGetr(dmmg),-1.0,DMMGGetRHS(dmmg));CHKERRQ(ierr);
  ierr = VecNorm(DMMGGetr(dmmg),NORM_2,&norm);CHKERRQ(ierr);
  /* ierr = PetscPrintf(PETSC_COMM_WORLD,"Residual norm %G\n",norm);CHKERRQ(ierr); */
  ierr = VecAssemblyBegin(DMMGGetx(dmmg));CHKERRQ(ierr);
  ierr = VecAssemblyEnd(DMMGGetx(dmmg));CHKERRQ(ierr);
  ierr = VecView_VTK(DMMGGetRHS(dmmg), "rhs.vtk", bcTypes[user.bcType]);CHKERRQ(ierr);
  ierr = VecView_VTK(DMMGGetx(dmmg), "solution.vtk", bcTypes[user.bcType]);CHKERRQ(ierr);

  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
PetscErrorCode ComputeRHS(DMMG dmmg, Vec b)
{
  DA             da = (DA)dmmg->dm;
  UserContext    *user = (UserContext *) dmmg->user;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscScalar    Hx,Hy;
  PetscScalar    **array;

  PetscFunctionBegin;
  ierr = DAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  Hx   = 1.0 / (PetscReal)(mx-1);
  Hy   = 1.0 / (PetscReal)(my-1);
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, b, &array);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      array[j][i] = PetscExpScalar(-((PetscReal)i*Hx)*((PetscReal)i*Hx)/user->nu)*PetscExpScalar(-((PetscReal)j*Hy)*((PetscReal)j*Hy)/user->nu)*Hx*Hy;
    }
  }
  ierr = DAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;

    ierr = KSPGetNullSpace(dmmg->ksp,&nullspace);CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b,PETSC_NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

    
#undef __FUNCT__
#define __FUNCT__ "ComputeRho"
PetscErrorCode ComputeRho(PetscInt i, PetscInt j, PetscInt mx, PetscInt my, PetscScalar centerRho, PetscScalar *rho)
{
  PetscFunctionBegin;
  if ((i > mx/3.0) && (i < 2.0*mx/3.0) && (j > my/3.0) && (j < 2.0*my/3.0)) {
    *rho = centerRho;
  } else {
    *rho = 1.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeMatrix"
PetscErrorCode ComputeMatrix(DMMG dmmg, Mat J,Mat jac)
{
  DA             da = (DA) dmmg->dm;
  UserContext    *user = (UserContext *) dmmg->user;
  PetscScalar    centerRho = user->rho;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xm,ym,xs,ys,num;
  PetscScalar    v[5],Hx,Hy,HydHx,HxdHy,rho;
  MatStencil     row, col[5];

  PetscFunctionBegin;
  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);  
  Hx    = 1.0 / (PetscReal)(mx-1);
  Hy    = 1.0 / (PetscReal)(my-1);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      row.i = i; row.j = j;
      ierr = ComputeRho(i, j, mx, my, centerRho, &rho);CHKERRQ(ierr);
      if (i==0 || j==0 || i==mx-1 || j==my-1) {
        if (user->bcType == DIRICHLET) {
           v[0] = 2.0*rho*(HxdHy + HydHx);
          ierr = MatSetValuesStencil(jac,1,&row,1,&row,v,INSERT_VALUES);CHKERRQ(ierr);
        } else if (user->bcType == NEUMANN) {
          num = 0;
          if (j!=0) {
            v[num] = -rho*HxdHy;              col[num].i = i;   col[num].j = j-1;
            num++;
          }
          if (i!=0) {
            v[num] = -rho*HydHx;              col[num].i = i-1; col[num].j = j;
            num++;
          }
          if (i!=mx-1) {
            v[num] = -rho*HydHx;              col[num].i = i+1; col[num].j = j;
            num++;
          }
          if (j!=my-1) {
            v[num] = -rho*HxdHy;              col[num].i = i;   col[num].j = j+1;
            num++;
          }
          v[num]   = (num/2.0)*rho*(HxdHy + HydHx); col[num].i = i;   col[num].j = j;
          num++;
          ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      } else {
        v[0] = -rho*HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -rho*HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*rho*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -rho*HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -rho*HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "VecView_VTK"
PetscErrorCode VecView_VTK(Vec x, const char filename[], const char bcName[])
{
  MPI_Comm           comm;
  DA                 da;
  Vec                coords;
  PetscViewer        viewer;
  PetscScalar       *array, *values;
  PetscInt           n, N, maxn, mx, my, dof;
  PetscInt           i, p;
  MPI_Status         status;
  PetscMPIInt        rank, size, tag, nn;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) x, &comm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(comm, filename, &viewer);CHKERRQ(ierr);

  ierr = VecGetSize(x, &N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x, &n);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) x, "DA", (PetscObject *) &da);CHKERRQ(ierr);
  if (!da) SETERRQ(PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");

  ierr = DAGetInfo(da, 0, &mx, &my, 0,0,0,0, &dof,0,0,0);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer, "# vtk DataFile Version 2.0\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Inhomogeneous Poisson Equation with %s boundary conditions\n", bcName);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "ASCII\n");CHKERRQ(ierr);
  /* get coordinates of nodes */
  ierr = DAGetCoordinates(da, &coords);CHKERRQ(ierr);
  if (!coords) {
    ierr = DASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);CHKERRQ(ierr);
    ierr = DAGetCoordinates(da, &coords);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "DATASET RECTILINEAR_GRID\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "DIMENSIONS %d %d %d\n", mx, my, 1);CHKERRQ(ierr);
  ierr = VecGetArray(coords, &array);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "X_COORDINATES %d double\n", mx);CHKERRQ(ierr);
  for(i = 0; i < mx; i++) {
    ierr = PetscViewerASCIIPrintf(viewer, "%G ", PetscRealPart(array[i*2]));CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Y_COORDINATES %d double\n", my);CHKERRQ(ierr);
  for(i = 0; i < my; i++) {
    ierr = PetscViewerASCIIPrintf(viewer, "%G ", PetscRealPart(array[i*mx*2+1]));CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Z_COORDINATES %d double\n", 1);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "%G\n", 0.0);CHKERRQ(ierr);
  ierr = VecRestoreArray(coords, &array);CHKERRQ(ierr);
  ierr = VecDestroy(coords);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "POINT_DATA %d\n", N);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "SCALARS scalars double %d\n", dof);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n");CHKERRQ(ierr);
  ierr = VecGetArray(x, &array);CHKERRQ(ierr);
  /* Determine maximum message to arrive */
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Reduce(&n, &maxn, 1, MPIU_INT, MPI_MAX, 0, comm);CHKERRQ(ierr);
  tag  = ((PetscObject) viewer)->tag;
  if (!rank) {
    ierr = PetscMalloc((maxn+1) * sizeof(PetscScalar), &values);CHKERRQ(ierr);
    for(i = 0; i < n; i++) {
      ierr = PetscViewerASCIIPrintf(viewer, "%G\n", PetscRealPart(array[i]));CHKERRQ(ierr);
    }
    for(p = 1; p < size; p++) {
      ierr = MPI_Recv(values, (PetscMPIInt) n, MPIU_SCALAR, p, tag, comm, &status);CHKERRQ(ierr);
      ierr = MPI_Get_count(&status, MPIU_SCALAR, &nn);CHKERRQ(ierr);        
      for(i = 0; i < nn; i++) {
        ierr = PetscViewerASCIIPrintf(viewer, "%G\n", PetscRealPart(array[i]));CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(values);CHKERRQ(ierr);
  } else {
    ierr = MPI_Send(array, n, MPIU_SCALAR, 0, tag, comm);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x, &array);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
