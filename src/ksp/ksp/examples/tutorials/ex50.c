/*   Concepts: DMMG/KSP solving a system of linear equations.
     Poisson equation in 2D: 

     div(grad p) = f,  0 < x,y < 1
     with
       forcing function f = -cos(m*pi*x)*cos(n*pi*y),
       Neuman boundary conditions
        dp/dx = 0 for x = 0, x = 1.
        dp/dy = 0 for y = 0, y = 1.

     Contributed by Michael Boghosian <boghmic@iit.edu>, 2008,
         based on petsc/src/ksp/ksp/examples/tutorials/ex29.c and ex32.c

     Example of Usage: 
          ./ex50 -mglevels 3 -ksp_monitor -M 3 -N 3 -ksp_view -da_view_draw -draw_pause -1 
          ./ex50 -M 100 -N 100 -mglevels 1 -mg_levels_0_pc_factor_levels <ilu_levels> -ksp_monitor -cmp_solu
          ./ex50 -M 100 -N 100 -mglevels 1 -mg_levels_0_pc_type lu -mg_levels_0_pc_factor_shift_type NONZERO -ksp_monitor -cmp_solu
          mpiexec -n 4 ./ex50 -M 3 -N 3 -ksp_monitor -ksp_view -mglevels 10 -log_summary
*/

static char help[] = "Solves 2D Poisson equation using multigrid.\n\n";

#include "petscda.h"
#include "petscksp.h"
#include "petscmg.h"
#include "petscdmmg.h"
#include "petscsys.h"
#include "petscvec.h"

extern PetscErrorCode ComputeJacobian(DMMG,Mat,Mat);
extern PetscErrorCode ComputeRHS(DMMG,Vec);
extern PetscErrorCode ComputeTrueSolution(DMMG *, Vec);
extern PetscErrorCode VecView_VTK(Vec, const char [], const char []);

typedef enum {DIRICHLET, NEUMANN} BCType;

typedef struct { 
  PetscScalar  uu, tt;
  BCType       bcType;
} UserContext;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;
  DA             da;
  UserContext    user;
  PetscInt       l, bc, mglevels, M, N, stages[3];
  PetscReal      norm;
  PetscErrorCode ierr;
  PetscMPIInt    rank,nproc;
  PetscBool      flg;
      
  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&nproc);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("DMMG Setup",&stages[0]);  CHKERRQ(ierr);
  ierr = PetscLogStageRegister("DMMG Solve",&stages[1]);  CHKERRQ(ierr);
      
  ierr = PetscLogStagePush(stages[0]);CHKERRQ(ierr);   /* Start DMMG Setup */
  
  /* SET VARIABLES: */
  mglevels = 1;  
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mglevels",&mglevels,PETSC_NULL);  CHKERRQ(ierr);
  M = 11;        /* number of grid points in x dir. on coarse grid */
  N = 11;        /* number of grid points in y dir. on coarse grid */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);  CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);  CHKERRQ(ierr);
  
  ierr = DMMGCreate(PETSC_COMM_WORLD,mglevels,PETSC_NULL,&dmmg); CHKERRQ(ierr);
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da); CHKERRQ(ierr);  
  ierr = DMMGSetDM(dmmg,(DM)da);
  ierr = DADestroy(da); CHKERRQ(ierr);
  
  /* Set user contex */
  user.uu = 1.0;
  user.tt = 1.0;
  bc   = (PetscInt)NEUMANN; // Use Neumann Boundary Conditions
  user.bcType = (BCType)bc;
  for (l = 0; l < DMMGGetLevels(dmmg); l++) {
    ierr = DMMGSetUser(dmmg,l,&user); CHKERRQ(ierr);
  }  
  
  ierr = DMMGSetKSP(dmmg,ComputeRHS,ComputeJacobian);  CHKERRQ(ierr);
  if (user.bcType == NEUMANN){
     ierr = DMMGSetNullSpace(dmmg,PETSC_TRUE,0,PETSC_NULL);  CHKERRQ(ierr);
  }
  ierr = PetscLogStagePop();  CHKERRQ(ierr); /* Finish DMMG Setup */

  /* DMMG SOLVE: */
  ierr = PetscLogStagePush(stages[1]);  CHKERRQ(ierr); /* Start DMMG Solve */
  ierr = DMMGSolve(dmmg);  CHKERRQ(ierr);
  ierr = PetscLogStagePop();  CHKERRQ(ierr); /* Finish DMMG Solve */

  /* Compare solution with the true solution p */
  ierr = PetscOptionsHasName(PETSC_NULL, "-cmp_solu", &flg);CHKERRQ(ierr);
  if (flg){
    Vec   x,p;
    if (mglevels != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"mglevels must equls 1 for comparison");
    if (nproc > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"num of proc must equls 1 for comparison");
    x = DMMGGetx(dmmg);
    ierr = VecDuplicate(x,&p);CHKERRQ(ierr);
    ierr = ComputeTrueSolution(dmmg,p);CHKERRQ(ierr);

    ierr = VecAXPY(p,-1.0,x);CHKERRQ(ierr);  /* p <- (p-x) */
    ierr = VecNorm(p, NORM_2, &norm);CHKERRQ(ierr);
    if (!rank){printf("| solu_compt - solu_true | = %g\n",norm);}
    ierr = VecDestroy(p);CHKERRQ(ierr);
  }
   
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

// COMPUTE RHS:--------------------------------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ComputeRHS" 
PetscErrorCode ComputeRHS(DMMG dmmg, Vec b)
{
  DA             da = (DA)dmmg->dm;
  UserContext    *user = (UserContext *) dmmg->user;
  PetscErrorCode ierr;
  PetscInt       i, j, M, N, xm ,ym ,xs, ys;
  PetscScalar    Hx, Hy, pi, uu, tt;
  PetscScalar    **array;

  PetscFunctionBegin;
  ierr = DAGetInfo(da, 0, &M, &N, 0,0,0,0,0,0,0,0); CHKERRQ(ierr);
  uu = user->uu; tt = user->tt;
  pi = 4*atan(1.0); 
  Hx   = 1.0/(PetscReal)(M);
  Hy   = 1.0/(PetscReal)(N);

  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0); CHKERRQ(ierr); // Fine grid
  //printf(" M N: %d %d; xm ym: %d %d; xs ys: %d %d\n",M,N,xm,ym,xs,ys);
  ierr = DAVecGetArray(da, b, &array); CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      array[j][i] = -PetscCosScalar(uu*pi*((PetscReal)i+0.5)*Hx)*cos(tt*pi*((PetscReal)j+0.5)*Hy)*Hx*Hy;
    }
  }
  ierr = DAVecRestoreArray(da, b, &array); CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

  /* force right hand side to be consistent for singular matrix */
  /* note this is really a hack, normally the model would provide you with a consistent right handside */
  if (user->bcType == NEUMANN) {
    MatNullSpace nullspace;
    ierr = KSPGetNullSpace(dmmg->ksp,&nullspace); CHKERRQ(ierr);
    ierr = MatNullSpaceRemove(nullspace,b,PETSC_NULL); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

// COMPUTE JACOBIAN:--------------------------------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian" 
PetscErrorCode ComputeJacobian(DMMG dmmg, Mat J, Mat jac)
{
  DA             da = (DA) dmmg->dm;
  UserContext    *user = (UserContext *) dmmg->user;
  PetscErrorCode ierr;
  PetscInt       i, j, M, N, xm, ym, xs, ys, num, numi, numj;
  PetscScalar    v[5], Hx, Hy, HydHx, HxdHy;
  MatStencil     row, col[5];

  PetscFunctionBegin;
  ierr = DAGetInfo(da,0,&M,&N,0,0,0,0,0,0,0,0); CHKERRQ(ierr);  
  Hx    = 1.0 / (PetscReal)(M);
  Hy    = 1.0 / (PetscReal)(N);
  HxdHy = Hx/Hy;
  HydHx = Hy/Hx;
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0); CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      row.i = i; row.j = j;
      
      if (i==0 || j==0 || i==M-1 || j==N-1) {
        if (user->bcType == DIRICHLET){
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Dirichlet boundary conditions not supported !\n");	  
        } else if (user->bcType == NEUMANN){
          num=0; numi=0; numj=0;
          if (j!=0) {
            v[num] = -HxdHy;              col[num].i = i;   col[num].j = j-1;
            num++; numj++;
          }
          if (i!=0) {
            v[num] = -HydHx;              col[num].i = i-1; col[num].j = j;
            num++; numi++;
          }
          if (i!=M-1) {
            v[num] = -HydHx;              col[num].i = i+1; col[num].j = j;
            num++; numi++;
          }
          if (j!=N-1) {
            v[num] = -HxdHy;              col[num].i = i;   col[num].j = j+1;
            num++; numj++;
          }
          v[num] = ( (PetscReal)(numj)*HxdHy + (PetscReal)(numi)*HydHx ); col[num].i = i;   col[num].j = j;
          num++;
          ierr = MatSetValuesStencil(jac,1,&row,num,col,v,INSERT_VALUES); CHKERRQ(ierr);
        }
      } else {
        v[0] = -HxdHy;              col[0].i = i;   col[0].j = j-1;
        v[1] = -HydHx;              col[1].i = i-1; col[1].j = j;
        v[2] = 2.0*(HxdHy + HydHx); col[2].i = i;   col[2].j = j;
        v[3] = -HydHx;              col[3].i = i+1; col[3].j = j;
        v[4] = -HxdHy;              col[4].i = i;   col[4].j = j+1;
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// COMPUTE TrueSolution:--------------------------------------------------------------
#undef __FUNCT__
#define __FUNCT__ "ComputeTrueSolution" 
PetscErrorCode ComputeTrueSolution(DMMG *dmmg, Vec b)
{
  DA             da = (DA)(*dmmg)->dm;
  UserContext    *user = (UserContext *) (*dmmg)->user;
  PetscErrorCode ierr;
  PetscInt       i, j, M, N, xm ,ym ,xs, ys;
  PetscScalar    Hx, Hy, pi, uu, tt, cc;
  PetscScalar    *array;

  PetscFunctionBegin;
  ierr = DAGetInfo(da, 0, &M, &N, 0,0,0,0,0,0,0,0); CHKERRQ(ierr); /* level_0 ! */
  //printf("ComputeTrueSolution - M N: %d %d;\n",M,N);

  uu = user->uu; tt = user->tt;
  pi = 4*atan(1.0); 
  cc = -1.0/( (uu*pi)*(uu*pi) + (tt*pi)*(tt*pi) );
  Hx   = 1.0/(PetscReal)(M);
  Hy   = 1.0/(PetscReal)(N);

  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0); CHKERRQ(ierr);
  ierr = VecGetArray(b, &array); CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      array[j*xm+i] = cos(uu*pi*((PetscReal)i+0.5)*Hx)*cos(tt*pi*((PetscReal)j+0.5)*Hy)*cc;
    }
  }
  ierr = VecRestoreArray(b, &array); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// VECVIEW_VTK:--------------------------------------------------------------
#undef __FUNCT__
#define __FUNCT__ "VecView_VTK" 
PetscErrorCode VecView_VTK(Vec x, const char filename[], const char bcName[])
{
  MPI_Comm           comm;
  DA                 da;
  Vec                coords;
  PetscViewer        viewer;
  PetscScalar        *array, *values;
  PetscInt           nn, NN, maxn, M, N;
  PetscInt           i, p, dof;
  MPI_Status         status;
  PetscMPIInt        rank, size, tag;
  PetscErrorCode     ierr;
  dof = 1;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) x, &comm); CHKERRQ(ierr);
  ierr = PetscViewerASCIIOpen(comm, filename, &viewer); CHKERRQ(ierr);

  ierr = VecGetSize(x, &NN); CHKERRQ(ierr);
  ierr = VecGetLocalSize(x, &nn); CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) x, "DA", (PetscObject *) &da); CHKERRQ(ierr);
  if (!da) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Vector not generated from a DA");

  ierr = DAGetInfo(da, 0, &M, &N, 0,0,0,0,&dof,0,0,0); CHKERRQ(ierr);

  ierr = PetscViewerASCIIPrintf(viewer, "# vtk DataFile Version 2.0\n"); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Inhomogeneous Poisson Equation with %s boundary conditions\n", bcName); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "ASCII\n"); CHKERRQ(ierr);
  // get coordinates of nodes 
  ierr = DAGetCoordinates(da, &coords); CHKERRQ(ierr);
  if (!coords) {
    ierr = DASetUniformCoordinates(da, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0); CHKERRQ(ierr);
    ierr = DAGetCoordinates(da, &coords); CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "DATASET RECTILINEAR_GRID\n"); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "DIMENSIONS %d %d %d\n", M, N, 1); CHKERRQ(ierr);
  ierr = VecGetArray(coords, &array);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "X_COORDINATES %d double\n", M); CHKERRQ(ierr);
  for(i = 0; i < M; i++) {
    ierr = PetscViewerASCIIPrintf(viewer, "%G ", PetscRealPart(array[i*2])); CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "\n"); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Y_COORDINATES %d double\n", N); CHKERRQ(ierr);
  for(i = 0; i < N; i++) {
    ierr = PetscViewerASCIIPrintf(viewer, "%G ", PetscRealPart(array[i*M*2+1])); CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "\n"); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Z_COORDINATES %d double\n", 1); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "%G\n", 0.0); CHKERRQ(ierr);
  ierr = VecRestoreArray(coords, &array); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "POINT_DATA %d\n", N); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "SCALARS scalars double %d\n", 1); CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "LOOKUP_TABLE default\n"); CHKERRQ(ierr);
  ierr = VecGetArray(x, &array); CHKERRQ(ierr);

  // Determine maximum message to arrive:
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size) ;CHKERRQ(ierr);
  ierr = MPI_Reduce(&nn, &maxn, 1, MPIU_INT, MPI_MAX, 0, comm); CHKERRQ(ierr);
  tag  = ((PetscObject) viewer)->tag;
  if (!rank) {
    ierr = PetscMalloc((maxn+1) * sizeof(PetscScalar), &values); CHKERRQ(ierr);
    for(i = 0; i < nn; i++) {
      ierr = PetscViewerASCIIPrintf(viewer, "%G\n", PetscRealPart(array[i])); CHKERRQ(ierr);
    }
    for(p = 1; p < size; p++) {
      ierr = MPI_Recv(values, (PetscMPIInt) nn, MPIU_SCALAR, p, tag, comm, &status); CHKERRQ(ierr);
      ierr = MPI_Get_count(&status, MPIU_SCALAR, &nn); CHKERRQ(ierr);        
      for(i = 0; i < nn; i++) {
        ierr = PetscViewerASCIIPrintf(viewer, "%G\n", PetscRealPart(array[i])); CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(values); CHKERRQ(ierr);
  } else {
    ierr = MPI_Send(array, nn, MPIU_SCALAR, 0, tag, comm); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x, &array); CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer); CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer); CHKERRQ(ierr);
      
  PetscFunctionReturn(0);
  }




