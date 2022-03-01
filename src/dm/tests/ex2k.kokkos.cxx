static char help[] = "Benchmarking various accessing methods of DMDA vectors on host\n\n";

/*
  On a machine with AMD EPYC-7452 CPUs, we got this data using one MPI rank and a serial-only Kokkos:
           Time (sec.), on Mar. 1, 2022
  ------------------------------------------
  n     PETSc          C          Kokkos
  ------------------------------------------
  32    4.6464E-05  4.7451E-05  1.6880E-04
  64    2.5654E-04  2.5164E-04  5.6780E-04
  128   1.9383E-03  1.8878E-03  4.7938E-03
  256   1.4450E-02  1.3619E-02  3.7778E-02
  512   1.1580E-01  1.1551E-01  2.8428E-01
  1024  1.4179E+00  1.3772E+00  2.2873E+00

  Overall, C is -2% ~ 5% faster than PETSc. But Kokkos is 1.6~3.6x slower than PETSc
*/

#include <petscdmda_kokkos.hpp>
#include <petscdm.h>
#include <petscdmda.h>

using namespace Kokkos;
using PetscScalarKokkosOffsetView3D      = Kokkos::Experimental::OffsetView<PetscScalar***,Kokkos::LayoutRight,Kokkos::HostSpace>;
using ConstPetscScalarKokkosOffsetView3D = Kokkos::Experimental::OffsetView<const PetscScalar***, Kokkos::LayoutRight,Kokkos::HostSpace>;

/* PETSc multi-dimensional array access */
static PetscErrorCode Update1(DM da,const PetscScalar ***__restrict__ x1, PetscScalar ***__restrict__ y1, PetscInt nwarm,PetscInt nloop,PetscLogDouble *avgTime)
{
  PetscErrorCode    ierr;
  PetscInt          it,i,j,k;
  PetscLogDouble    tstart=0.0,tend;
  PetscInt          xm,ym,zm,xs,ys,zs,gxm,gym,gzm,gxs,gys,gzs;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
  for (it=0; it<nwarm+nloop; it++) {
    if (it == nwarm) {ierr = PetscTime(&tstart);CHKERRQ(ierr);}
    for (k=zs; k<zs+zm; k++) {
      for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
          y1[k][j][i] = 6*x1[k][j][i] - x1[k-1][j][i] - x1[k][j-1][i] - x1[k][j][i-1]
                                      - x1[k+1][j][i] - x1[k][j+1][i] - x1[k][j][i+1];
        }
      }
    }
  }
  ierr    = PetscTime(&tend);CHKERRQ(ierr);
  *avgTime = (tend - tstart)/nloop;
  PetscFunctionReturn(ierr);
}

/* C multi-dimensional array access */
static PetscErrorCode Update2(DM da,const PetscScalar *__restrict__ x2, PetscScalar *__restrict__ y2, PetscInt nwarm,PetscInt nloop,PetscLogDouble *avgTime)
{
  PetscErrorCode    ierr;
  PetscInt          it,i,j,k;
  PetscLogDouble    tstart=0.0,tend;
  PetscInt          xm,ym,zm,xs,ys,zs,gxm,gym,gzm,gxs,gys,gzs;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
#define X2(k,j,i) x2[(k-gzs)*gym*gxm+(j-gys)*gxm+(i-gxs)]
#define Y2(k,j,i) y2[(k-zs)*ym*xm+(j-ys)*xm+(i-xs)]
  for (it=0; it<nwarm+nloop; it++) {
    if (it == nwarm) {ierr = PetscTime(&tstart);CHKERRQ(ierr);}
    for (k=zs; k<zs+zm; k++) {
      for (j=ys; j<ys+ym; j++) {
        for (i=xs; i<xs+xm; i++) {
          Y2(k,j,i) = 6*X2(k,j,i) - X2(k-1,j,i) - X2(k,j-1,i) - X2(k,j,i-1)
                                  - X2(k+1,j,i) - X2(k,j+1,i) - X2(k,j,i+1);
        }
      }
    }
  }
  ierr    = PetscTime(&tend);CHKERRQ(ierr);
  *avgTime = (tend - tstart)/nloop;
#undef X2
#undef Y2
  PetscFunctionReturn(ierr);
}

int main(int argc,char **argv)
{
  PetscErrorCode                       ierr;
  DM                                   da;
  PetscInt                             xm,ym,zm,xs,ys,zs,gxm,gym,gzm,gxs,gys,gzs;
  PetscInt                             dof = 1,sw = 1;
  DMBoundaryType                       bx = DM_BOUNDARY_PERIODIC,by = DM_BOUNDARY_PERIODIC,bz = DM_BOUNDARY_PERIODIC;
  DMDAStencilType                      st = DMDA_STENCIL_STAR;
  Vec                                  x,y; /* local/global vectors of the da */
  PetscRandom                          rctx;
  const PetscScalar                    ***x1;
  PetscScalar                          ***y1; /* arrays of g, ll */
  const PetscScalar                    *x2;
  PetscScalar                          *y2;
  ConstPetscScalarKokkosOffsetView3D   x3;
  PetscScalarKokkosOffsetView3D        y3;
  PetscLogDouble                       tstart = 0.0,tend = 0.0,avgTime = 0.0;
  PetscInt                             nwarm = 2, nloop = 10;
  PetscInt                             min = 32, max = 32*8; /* min and max sizes of the grids to sample */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rctx);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-min",&min,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-max",&max,NULL);CHKERRQ(ierr);

  for (PetscInt len=min; len<=max; len=len*2) {
    ierr = DMDACreate3d(PETSC_COMM_WORLD,bx,by,bz,st,len,len,len,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,sw,0,0,0,&da);CHKERRQ(ierr);
    ierr = DMSetFromOptions(da);CHKERRQ(ierr);
    ierr = DMSetUp(da);CHKERRQ(ierr);

    ierr = DMDAGetCorners(da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
    ierr = DMDAGetGhostCorners(da,&gxs,&gys,&gzs,&gxm,&gym,&gzm);CHKERRQ(ierr);
    ierr = DMCreateLocalVector(da,&x);CHKERRQ(ierr); /* Create local x and global y */
    ierr = DMCreateGlobalVector(da,&y);CHKERRQ(ierr);

    /* Access with petsc multi-dimensional arrays */
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayRead(da,x,&x1);CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da,y,&y1);CHKERRQ(ierr);
    ierr = Update1(da,x1,y1,nwarm,nloop,&avgTime);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayRead(da,x,&x1);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da,y,&y1);CHKERRQ(ierr);
    ierr = PetscTime(&tend);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%4d^3 -- PETSc average time  = %e\n",len,avgTime);CHKERRQ(ierr);

    /* Access with C multi-dimensional arrays */
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    ierr = VecGetArrayRead(x,&x2);CHKERRQ(ierr);
    ierr = VecGetArray(y,&y2);CHKERRQ(ierr);
    ierr = Update2(da,x2,y2,nwarm,nloop,&avgTime);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(x,&x2);CHKERRQ(ierr);
    ierr = VecRestoreArray(y,&y2);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%4d^3 -- C average time      = %e\n",len,avgTime);CHKERRQ(ierr);

    /* Access with Kokkos multi-dimensional OffsetViews */
    ierr = VecSet(y,0.0);CHKERRQ(ierr);
    ierr = VecSetRandom(x,rctx);CHKERRQ(ierr);
    ierr = DMDAVecGetKokkosOffsetView(da,x,&x3);CHKERRQ(ierr);
    ierr = DMDAVecGetKokkosOffsetView(da,y,&y3);CHKERRQ(ierr);

    for (PetscInt it=0; it<nwarm+nloop; it++) {
      if (it == nwarm) {ierr = PetscTime(&tstart);CHKERRQ(ierr);}
      Kokkos::parallel_for("stencil",MDRangePolicy<Kokkos::DefaultHostExecutionSpace,Rank<3,Iterate::Right,Iterate::Right>>({zs,ys,xs},{zs+zm,ys+ym,xs+xm}),
        KOKKOS_LAMBDA(PetscInt k,PetscInt j,PetscInt i) {
        y3(k,j,i) = 6*x3(k,j,i) - x3(k-1,j,i) - x3(k,j-1,i) - x3(k,j,i-1)
                                - x3(k+1,j,i) - x3(k,j+1,i) - x3(k,j,i+1);
      });
    }
    ierr = PetscTime(&tend);CHKERRQ(ierr);
    ierr = DMDAVecRestoreKokkosOffsetView(da,x,&x3);CHKERRQ(ierr);
    ierr = DMDAVecRestoreKokkosOffsetView(da,y,&y3);CHKERRQ(ierr);
    avgTime = (tend - tstart)/nloop;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"%4d^3 -- Kokkos average time = %e\n",len,avgTime);CHKERRQ(ierr);

    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = DMDestroy(&da);CHKERRQ(ierr);
  }
  ierr = PetscRandomDestroy(&rctx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  build:
    requires: kokkos_kernels

  test:
    suffix: 1
    requires: kokkos_kernels
    args: -min 32 -max 32 -dm_vec_type kokkos
    filter: grep -v "time"

TEST*/
