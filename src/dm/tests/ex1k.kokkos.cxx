static char help[] = "Tests DMDAVecGetKokkosOffsetView() and DMDAVecGetKokkosOffsetViewDOF() \n\n";

#include <petscdmda_kokkos.hpp>
#include <petscdm.h>
#include <petscdmda.h>

using namespace Kokkos;
using PetscScalarKokkosOffsetView2D      = Kokkos::Experimental::OffsetView<PetscScalar**,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space>;
using ConstPetscScalarKokkosOffsetView2D = Kokkos::Experimental::OffsetView<const PetscScalar**,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space>;

using PetscScalarKokkosOffsetView3D      = Kokkos::Experimental::OffsetView<PetscScalar***,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space>;
using ConstPetscScalarKokkosOffsetView3D = Kokkos::Experimental::OffsetView<const PetscScalar***,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space>;

/* can not define the type inside main, otherwise have this compilation error:
   error: A type local to a function ("Node") cannot be used in the type of a
   variable captured by an extended __device__ or __host__ __device__ lambda
*/
typedef struct {
  PetscScalar u,v,w;
} Node;

using NodeKokkosOffsetView2D      = Kokkos::Experimental::OffsetView<Node**,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space>;
using ConstNodeKokkosOffsetView2D = Kokkos::Experimental::OffsetView<const Node**,Kokkos::LayoutRight,Kokkos::DefaultExecutionSpace::memory_space>;

int main(int argc,char **argv)
{
  DM                  da;
  PetscInt            M = 5, N = 7,xm,ym,xs,ys;
  PetscInt            dof=1,sw = 1;
  DMBoundaryType      bx = DM_BOUNDARY_PERIODIC,by = DM_BOUNDARY_PERIODIC;
  DMDAStencilType     st = DMDA_STENCIL_STAR;
  PetscReal           nrm;
  Vec                 g,l,gg,ll; /* global/local vectors of the da */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* ===========================================================================
    Show how to manage a multi-component DMDA with DMDAVecGetKokkosOffsetViewDOF
   ============================================================================*/
  PetscScalar                        ***garray; /* arrays of g, ll */
  const PetscScalar                  ***larray;
  PetscScalarKokkosOffsetView3D      gview; /* views of gg, ll */
  ConstPetscScalarKokkosOffsetView3D lview;

  dof  = 2;
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,bx,by,st,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  PetscCall(DMCreateGlobalVector(da,&g));
  PetscCall(DMCreateLocalVector(da,&l));
  PetscCall(DMCreateGlobalVector(da,&gg));
  PetscCall(DMCreateLocalVector(da,&ll));

  /* Init g using array */
  PetscCall(DMDAVecGetArrayDOFWrite(da,g,&garray));
  for (PetscInt j=ys; j<ys+ym; j++) { /* run on host */
    for (PetscInt i=xs; i<xs+xm; i++) {
      for (PetscInt c=0; c<dof; c++) {
        garray[j][i][c] = 100*j + 10*(i+1) + c;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOFWrite(da,g,&garray));

  /* Init gg using view */
  PetscCall(DMDAVecGetKokkosOffsetViewDOFWrite(da,gg,&gview));
  Kokkos::parallel_for("init 1",MDRangePolicy<Rank<3,Iterate::Right,Iterate::Right>>({ys,xs,0},{ys+ym,xs+xm,dof}),
    KOKKOS_LAMBDA(PetscInt j,PetscInt i,PetscInt c) /* might run on device */
  {
    gview(j,i,c) = 100*j + 10*(i+1) + c;
  });
  PetscCall(DMDAVecRestoreKokkosOffsetViewDOFWrite(da,gg,&gview));

  /* Scatter g, gg to l, ll */
  PetscCall(DMGlobalToLocal(da,g,INSERT_VALUES,l));
  PetscCall(DMGlobalToLocal(da,gg,INSERT_VALUES,ll));

  /* Do stencil on g with values from l that contains ghosts */
  PetscCall(DMDAVecGetArrayDOFWrite(da,g,&garray));
  PetscCall(DMDAVecGetArrayDOFRead(da,l,&larray));
  for (PetscInt j=ys; j<ys+ym; j++) {
    for (PetscInt i=xs; i<xs+xm; i++) {
      for (PetscInt c=0; c<dof; c++) {
        garray[j][i][c] = (larray[j][i-1][c] + larray[j][i+1][c] + larray[j-1][i][c] + larray[j+1][i][c])/4.0;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOFWrite(da,g,&garray));
  PetscCall(DMDAVecRestoreArrayDOFRead(da,l,&larray));

  /* Do stencil on gg with values from ll that contains ghosts */
  PetscCall(DMDAVecGetKokkosOffsetViewDOFWrite(da,gg,&gview));
  PetscCall(DMDAVecGetKokkosOffsetViewDOF(da,ll,&lview));
  Kokkos::parallel_for("stencil 1",MDRangePolicy<Rank<3,Iterate::Right,Iterate::Right>>({ys,xs,0},{ys+ym,xs+xm,dof}),
    KOKKOS_LAMBDA(PetscInt j,PetscInt i,PetscInt c)
  {
    gview(j,i,c) = (lview(j,i-1,c) + lview(j,i+1,c) + lview(j-1,i,c) + lview(j+1,i,c))/4.0;
  });
  PetscCall(DMDAVecRestoreKokkosOffsetViewDOFWrite(da,gg,&gview));
  PetscCall(DMDAVecRestoreKokkosOffsetViewDOF(da,ll,&lview));

  /* gg should be equal to g */
  PetscCall(VecAXPY(g,-1.0,gg));
  PetscCall(VecNorm(g,NORM_2,&nrm));
  PetscCheck(nrm < PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"gg is not equal to g");

  PetscCall(DMDestroy(&da));
  PetscCall(VecDestroy(&l));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&ll));
  PetscCall(VecDestroy(&gg));

  /* =============================================================================
    Show how to manage a multi-component DMDA using DMDAVecGetKokkosOffsetView and
    a customized struct type
   ==============================================================================*/
  Node                               **garray2; /* Node arrays of g, l */
  const Node                         **larray2;
  PetscScalarKokkosOffsetView2D      gview2; /* PetscScalar views of gg, ll */
  ConstPetscScalarKokkosOffsetView2D lview2;
  NodeKokkosOffsetView2D             gnview; /* Node views of gg, ll */
  ConstNodeKokkosOffsetView2D        lnview;

  dof  = sizeof(Node)/sizeof(PetscScalar);
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,bx,by,st,M,N,PETSC_DECIDE,PETSC_DECIDE,sizeof(Node)/sizeof(PetscScalar),sw,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  PetscCall(DMCreateGlobalVector(da,&g));
  PetscCall(DMCreateLocalVector(da,&l));
  PetscCall(DMCreateGlobalVector(da,&gg));
  PetscCall(DMCreateLocalVector(da,&ll));

  /* Init g using array */
  PetscCall(DMDAVecGetArrayWrite(da,g,&garray2));
  for (PetscInt j=ys; j<ys+ym; j++) {
    for (PetscInt i=xs; i<xs+xm; i++) {
      garray2[j][i].u = 100*j + 10*(i+1) + 111;
      garray2[j][i].v = 100*j + 10*(i+1) + 222;
      garray2[j][i].w = 100*j + 10*(i+1) + 333;
    }
  }
  PetscCall(DMDAVecRestoreArrayWrite(da,g,&garray2));

  /* Init gg using view */
  PetscCall(DMDAVecGetKokkosOffsetViewWrite(da,gg,&gview2));
  gnview = NodeKokkosOffsetView2D(reinterpret_cast<Node*>(gview2.data()),{gview2.begin(0)/dof,gview2.begin(1)/dof}, {gview2.end(0)/dof,gview2.end(1)/dof});
  Kokkos::parallel_for("init 2",MDRangePolicy<Rank<2,Iterate::Right,Iterate::Right>>({ys,xs},{ys+ym,xs+xm}),
    KOKKOS_LAMBDA(PetscInt j,PetscInt i)
  {
    gnview(j,i).u = 100*j + 10*(i+1) + 111;
    gnview(j,i).v = 100*j + 10*(i+1) + 222;
    gnview(j,i).w = 100*j + 10*(i+1) + 333;
  });
  PetscCall(DMDAVecRestoreKokkosOffsetViewWrite(da,gg,&gview2));

  /* Scatter g, gg to l, ll */
  PetscCall(DMGlobalToLocal(da,g,INSERT_VALUES,l));
  PetscCall(DMGlobalToLocal(da,gg,INSERT_VALUES,ll));

  /* Do stencil on g with values from l that contains ghosts */
  PetscCall(DMDAVecGetArrayWrite(da,g,&garray2));
  PetscCall(DMDAVecGetArray(da,l,&larray2));
  for (PetscInt j=ys; j<ys+ym; j++) {
    for (PetscInt i=xs; i<xs+xm; i++) {
      garray2[j][i].u = (larray2[j][i-1].u + larray2[j][i+1].u + larray2[j-1][i].u + larray2[j+1][i].u)/4.0;
      garray2[j][i].v = (larray2[j][i-1].v + larray2[j][i+1].v + larray2[j-1][i].v + larray2[j+1][i].v)/4.0;
      garray2[j][i].w = (larray2[j][i-1].w + larray2[j][i+1].w + larray2[j-1][i].w + larray2[j+1][i].w)/4.0;
    }
  }
  PetscCall(DMDAVecRestoreArrayWrite(da,g,&garray2));
  PetscCall(DMDAVecRestoreArray(da,l,&larray2));

  /* Do stencil on gg with values from ll that contains ghosts */
  PetscCall(DMDAVecGetKokkosOffsetViewWrite(da,gg,&gview2)); /* write-only */
  PetscCall(DMDAVecGetKokkosOffsetView(da,ll,&lview2)); /* read-only */
  gnview = NodeKokkosOffsetView2D(reinterpret_cast<Node*>(gview2.data()),{gview2.begin(0)/dof,gview2.begin(1)/dof}, {gview2.end(0)/dof,gview2.end(1)/dof});
  lnview = ConstNodeKokkosOffsetView2D(reinterpret_cast<const Node*>(lview2.data()),{lview2.begin(0)/dof,lview2.begin(1)/dof}, {lview2.end(0)/dof,lview2.end(1)/dof});
  Kokkos::parallel_for("stencil 2",MDRangePolicy<Rank<2,Iterate::Right,Iterate::Right>>({ys,xs},{ys+ym,xs+xm}),
    KOKKOS_LAMBDA(PetscInt j,PetscInt i)
  {
    gnview(j,i).u = (lnview(j,i-1).u + lnview(j,i+1).u + lnview(j-1,i).u + lnview(j+1,i).u)/4.0;
    gnview(j,i).v = (lnview(j,i-1).v + lnview(j,i+1).v + lnview(j-1,i).v + lnview(j+1,i).v)/4.0;
    gnview(j,i).w = (lnview(j,i-1).w + lnview(j,i+1).w + lnview(j-1,i).w + lnview(j+1,i).w)/4.0;
  });
  PetscCall(DMDAVecRestoreKokkosOffsetViewWrite(da,gg,&gview2));
  PetscCall(DMDAVecRestoreKokkosOffsetView(da,ll,&lview2));

  /* gg should be equal to g */
  PetscCall(VecAXPY(g,-1.0,gg));
  PetscCall(VecNorm(g,NORM_2,&nrm));
  PetscCheck(nrm < PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"gg is not equal to g");

  PetscCall(DMDestroy(&da));
  PetscCall(VecDestroy(&l));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&ll));
  PetscCall(VecDestroy(&gg));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: kokkos_kernels

  test:
    suffix: 1
    nsize: 4
    requires: kokkos_kernels
    args: -dm_vec_type kokkos

TEST*/
