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
  PetscErrorCode      ierr;
  DM                  da;
  PetscInt            M = 5, N = 7,xm,ym,xs,ys;
  PetscInt            dof=1,sw = 1;
  DMBoundaryType      bx = DM_BOUNDARY_PERIODIC,by = DM_BOUNDARY_PERIODIC;
  DMDAStencilType     st = DMDA_STENCIL_STAR;
  PetscReal           nrm;
  Vec                 g,l,gg,ll; /* global/local vectors of the da */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* ===========================================================================
    Show how to manage a multi-component DMDA with DMDAVecGetKokkosOffsetViewDOF
   ============================================================================*/
  PetscScalar                        ***garray; /* arrays of g, ll */
  const PetscScalar                  ***larray;
  PetscScalarKokkosOffsetView3D      gview; /* views of gg, ll */
  ConstPetscScalarKokkosOffsetView3D lview;

  dof  = 2;
  ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,st,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,sw,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&g);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&l);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&gg);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&ll);CHKERRQ(ierr);

  /* Init g using array */
  ierr = DMDAVecGetArrayDOFWrite(da,g,&garray);CHKERRQ(ierr);
  for (PetscInt j=ys; j<ys+ym; j++) { /* run on host */
    for (PetscInt i=xs; i<xs+xm; i++) {
      for (PetscInt c=0; c<dof; c++) {
        garray[j][i][c] = 100*j + 10*(i+1) + c;
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOFWrite(da,g,&garray);CHKERRQ(ierr);

  /* Init gg using view */
  ierr = DMDAVecGetKokkosOffsetViewDOFWrite(da,gg,&gview);CHKERRQ(ierr);
  Kokkos::parallel_for("init 1",MDRangePolicy<Rank<3,Iterate::Right,Iterate::Right>>({ys,xs,0},{ys+ym,xs+xm,dof}),
    KOKKOS_LAMBDA(PetscInt j,PetscInt i,PetscInt c) /* might run on device */
  {
    gview(j,i,c) = 100*j + 10*(i+1) + c;
  });
  ierr = DMDAVecRestoreKokkosOffsetViewDOFWrite(da,gg,&gview);CHKERRQ(ierr);

  /* Scatter g, gg to l, ll */
  ierr = DMGlobalToLocal(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(da,gg,INSERT_VALUES,ll);CHKERRQ(ierr);

  /* Do stencil on g with values from l that contains ghosts */
  ierr = DMDAVecGetArrayDOFWrite(da,g,&garray);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(da,l,&larray);CHKERRQ(ierr);
  for (PetscInt j=ys; j<ys+ym; j++) {
    for (PetscInt i=xs; i<xs+xm; i++) {
      for (PetscInt c=0; c<dof; c++) {
        garray[j][i][c] = (larray[j][i-1][c] + larray[j][i+1][c] + larray[j-1][i][c] + larray[j+1][i][c])/4.0;
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOFWrite(da,g,&garray);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(da,l,&larray);CHKERRQ(ierr);

  /* Do stencil on gg with values from ll that contains ghosts */
  ierr = DMDAVecGetKokkosOffsetViewDOFWrite(da,gg,&gview);CHKERRQ(ierr);
  ierr = DMDAVecGetKokkosOffsetViewDOF(da,ll,&lview);CHKERRQ(ierr);
  Kokkos::parallel_for("stencil 1",MDRangePolicy<Rank<3,Iterate::Right,Iterate::Right>>({ys,xs,0},{ys+ym,xs+xm,dof}),
    KOKKOS_LAMBDA(PetscInt j,PetscInt i,PetscInt c)
  {
    gview(j,i,c) = (lview(j,i-1,c) + lview(j,i+1,c) + lview(j-1,i,c) + lview(j+1,i,c))/4.0;
  });
  ierr = DMDAVecRestoreKokkosOffsetViewDOFWrite(da,gg,&gview);CHKERRQ(ierr);
  ierr = DMDAVecRestoreKokkosOffsetViewDOF(da,ll,&lview);CHKERRQ(ierr);

  /* gg should be equal to g */
  ierr = VecAXPY(g,-1.0,gg);CHKERRQ(ierr);
  ierr = VecNorm(g,NORM_2,&nrm);CHKERRQ(ierr);
  PetscCheck(nrm < PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"gg is not equal to g");

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = VecDestroy(&l);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  ierr = VecDestroy(&ll);CHKERRQ(ierr);
  ierr = VecDestroy(&gg);CHKERRQ(ierr);

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
  ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,st,M,N,PETSC_DECIDE,PETSC_DECIDE,sizeof(Node)/sizeof(PetscScalar),sw,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&g);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&l);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&gg);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&ll);CHKERRQ(ierr);

  /* Init g using array */
  ierr = DMDAVecGetArrayWrite(da,g,&garray2);CHKERRQ(ierr);
  for (PetscInt j=ys; j<ys+ym; j++) {
    for (PetscInt i=xs; i<xs+xm; i++) {
      garray2[j][i].u = 100*j + 10*(i+1) + 111;
      garray2[j][i].v = 100*j + 10*(i+1) + 222;
      garray2[j][i].w = 100*j + 10*(i+1) + 333;
    }
  }
  ierr = DMDAVecRestoreArrayWrite(da,g,&garray2);CHKERRQ(ierr);

  /* Init gg using view */
  ierr   = DMDAVecGetKokkosOffsetViewWrite(da,gg,&gview2);CHKERRQ(ierr);
  gnview = NodeKokkosOffsetView2D(reinterpret_cast<Node*>(gview2.data()),{gview2.begin(0)/dof,gview2.begin(1)/dof}, {gview2.end(0)/dof,gview2.end(1)/dof});
  Kokkos::parallel_for("init 2",MDRangePolicy<Rank<2,Iterate::Right,Iterate::Right>>({ys,xs},{ys+ym,xs+xm}),
    KOKKOS_LAMBDA(PetscInt j,PetscInt i)
  {
    gnview(j,i).u = 100*j + 10*(i+1) + 111;
    gnview(j,i).v = 100*j + 10*(i+1) + 222;
    gnview(j,i).w = 100*j + 10*(i+1) + 333;
  });
  ierr = DMDAVecRestoreKokkosOffsetViewWrite(da,gg,&gview2);CHKERRQ(ierr);

  /* Scatter g, gg to l, ll */
  ierr = DMGlobalToLocal(da,g,INSERT_VALUES,l);CHKERRQ(ierr);
  ierr = DMGlobalToLocal(da,gg,INSERT_VALUES,ll);CHKERRQ(ierr);

  /* Do stencil on g with values from l that contains ghosts */
  ierr = DMDAVecGetArrayWrite(da,g,&garray2);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,l,&larray2);CHKERRQ(ierr);
  for (PetscInt j=ys; j<ys+ym; j++) {
    for (PetscInt i=xs; i<xs+xm; i++) {
      garray2[j][i].u = (larray2[j][i-1].u + larray2[j][i+1].u + larray2[j-1][i].u + larray2[j+1][i].u)/4.0;
      garray2[j][i].v = (larray2[j][i-1].v + larray2[j][i+1].v + larray2[j-1][i].v + larray2[j+1][i].v)/4.0;
      garray2[j][i].w = (larray2[j][i-1].w + larray2[j][i+1].w + larray2[j-1][i].w + larray2[j+1][i].w)/4.0;
    }
  }
  ierr = DMDAVecRestoreArrayWrite(da,g,&garray2);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,l,&larray2);CHKERRQ(ierr);

  /* Do stencil on gg with values from ll that contains ghosts */
  ierr   = DMDAVecGetKokkosOffsetViewWrite(da,gg,&gview2);CHKERRQ(ierr); /* write-only */
  ierr   = DMDAVecGetKokkosOffsetView(da,ll,&lview2);CHKERRQ(ierr); /* read-only */
  gnview = NodeKokkosOffsetView2D(reinterpret_cast<Node*>(gview2.data()),{gview2.begin(0)/dof,gview2.begin(1)/dof}, {gview2.end(0)/dof,gview2.end(1)/dof});
  lnview = ConstNodeKokkosOffsetView2D(reinterpret_cast<const Node*>(lview2.data()),{lview2.begin(0)/dof,lview2.begin(1)/dof}, {lview2.end(0)/dof,lview2.end(1)/dof});
  Kokkos::parallel_for("stencil 2",MDRangePolicy<Rank<2,Iterate::Right,Iterate::Right>>({ys,xs},{ys+ym,xs+xm}),
    KOKKOS_LAMBDA(PetscInt j,PetscInt i)
  {
    gnview(j,i).u = (lnview(j,i-1).u + lnview(j,i+1).u + lnview(j-1,i).u + lnview(j+1,i).u)/4.0;
    gnview(j,i).v = (lnview(j,i-1).v + lnview(j,i+1).v + lnview(j-1,i).v + lnview(j+1,i).v)/4.0;
    gnview(j,i).w = (lnview(j,i-1).w + lnview(j,i+1).w + lnview(j-1,i).w + lnview(j+1,i).w)/4.0;
  });
  ierr = DMDAVecRestoreKokkosOffsetViewWrite(da,gg,&gview2);CHKERRQ(ierr);
  ierr = DMDAVecRestoreKokkosOffsetView(da,ll,&lview2);CHKERRQ(ierr);

  /* gg should be equal to g */
  ierr = VecAXPY(g,-1.0,gg);CHKERRQ(ierr);
  ierr = VecNorm(g,NORM_2,&nrm);CHKERRQ(ierr);
  PetscCheck(nrm < PETSC_SMALL,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"gg is not equal to g");

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = VecDestroy(&l);CHKERRQ(ierr);
  ierr = VecDestroy(&g);CHKERRQ(ierr);
  ierr = VecDestroy(&ll);CHKERRQ(ierr);
  ierr = VecDestroy(&gg);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
