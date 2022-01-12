static char help[] = "Tests affine subspaces.\n\n";

#include <petscfe.h>
#include <petscdmplex.h>
#include <petscdmshell.h>

int main(int argc, char **argv)
{
  DM             dm;
  PetscFE        fe;
  PetscSpace     space;
  PetscDualSpace dualspace, dualsubspace;
  PetscInt       dim = 2, Nc = 3, cStart, cEnd;
  PetscBool      simplex = PETSC_TRUE;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm,"","Options for subspace test","none");CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The spatial dimension","ex5.c",dim,&dim,NULL,1,3);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Test simplex element","ex5.c",simplex,&simplex,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-num_comp", "Number of components in space","ex5.c",Nc,&Nc,NULL,1);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = DMShellCreate(comm,&dm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm,dim,Nc,simplex,NULL,PETSC_DEFAULT,&fe);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFESetName(fe, "solution");CHKERRQ(ierr);
  ierr = PetscFEGetBasisSpace(fe,&space);CHKERRQ(ierr);
  ierr = PetscSpaceGetNumComponents(space,&Nc);CHKERRQ(ierr);
  ierr = PetscFEGetDualSpace(fe,&dualspace);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetHeightSubspace(dualspace,1,&dualsubspace);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(dualspace,&dm);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  if (cEnd > cStart) {
    PetscInt coneSize;

    ierr = DMPlexGetConeSize(dm,cStart,&coneSize);CHKERRQ(ierr);
    if (coneSize) {
      PetscFE traceFE;
      const PetscInt *cone;
      PetscInt        point, nSub, nFull;
      PetscReal       xi0[3] = {-1., -1., -1.};
      PetscScalar     *outSub, *outFull;
      PetscReal       *testSub, *testFull;
      PetscTabulation Tsub, Tfull;
      PetscReal       J[9], detJ;
      PetscInt        i, j;
      PetscSection    sectionFull;
      Vec             vecFull;
      PetscScalar     *arrayFull, *arraySub;
      PetscReal       err;
      PetscRandom     rand;

      ierr = DMPlexGetCone(dm,cStart,&cone);CHKERRQ(ierr);
      point = cone[0];
      ierr = PetscFECreatePointTrace(fe,point,&traceFE);CHKERRQ(ierr);
      ierr = PetscFESetUp(traceFE);CHKERRQ(ierr);
      ierr = PetscFEViewFromOptions(traceFE,NULL,"-trace_fe_view");CHKERRQ(ierr);
      ierr = PetscMalloc4(dim - 1,&testSub,dim,&testFull,Nc,&outSub,Nc,&outFull);CHKERRQ(ierr);
      ierr = PetscRandomCreate(PETSC_COMM_SELF,&rand);CHKERRQ(ierr);
      ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
      ierr = PetscRandomSetInterval(rand,-1.,1.);CHKERRQ(ierr);
      /* create a random point in the trace domain */
      for (i = 0; i < dim - 1; i++) {
        ierr = PetscRandomGetValueReal(rand,&testSub[i]);CHKERRQ(ierr);
      }
      ierr = DMPlexComputeCellGeometryFEM(dm,point,NULL,testFull,J,NULL,&detJ);CHKERRQ(ierr);
      /* project it into the full domain */
      for (i = 0; i < dim; i++) {
        for (j = 0; j < dim - 1; j++) testFull[i] += J[i * dim + j] * (testSub[j] - xi0[j]);
      }
      /* create a random vector in the full domain */
      ierr = PetscFEGetDimension(fe,&nFull);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,nFull,&vecFull);CHKERRQ(ierr);
      ierr = VecGetArray(vecFull,&arrayFull);CHKERRQ(ierr);
      for (i = 0; i < nFull; i++) {
        ierr = PetscRandomGetValue(rand,&arrayFull[i]);CHKERRQ(ierr);
      }
      ierr = VecRestoreArray(vecFull,&arrayFull);CHKERRQ(ierr);
      /* create a vector on the trace domain */
      ierr = PetscFEGetDimension(traceFE,&nSub);CHKERRQ(ierr);
      /* get the subset of the original finite element space that is supported on the trace space */
      ierr = PetscDualSpaceGetSection(dualspace,&sectionFull);CHKERRQ(ierr);
      ierr = PetscSectionSetUp(sectionFull);CHKERRQ(ierr);
      /* get the trace degrees of freedom */
      ierr = PetscMalloc1(nSub,&arraySub);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(dm,sectionFull,vecFull,point,&nSub,&arraySub);CHKERRQ(ierr);
      /* get the tabulations */
      ierr = PetscFECreateTabulation(traceFE,1,1,testSub,0,&Tsub);CHKERRQ(ierr);
      ierr = PetscFECreateTabulation(fe,1,1,testFull,0,&Tfull);CHKERRQ(ierr);
      for (i = 0; i < Nc; i++) {
        outSub[i] = 0.0;
        for (j = 0; j < nSub; j++) {
          outSub[i] += Tsub->T[0][j * Nc + i] * arraySub[j];
        }
      }
      ierr = VecGetArray(vecFull,&arrayFull);CHKERRQ(ierr);
      err = 0.0;
      for (i = 0; i < Nc; i++) {
        PetscScalar diff;

        outFull[i] = 0.0;
        for (j = 0; j < nFull; j++) {
          outFull[i] += Tfull->T[0][j * Nc + i] * arrayFull[j];
        }
        diff = outFull[i] - outSub[i];
        err += PetscRealPart(PetscConj(diff) * diff);
      }
      err = PetscSqrtReal(err);
      if (err > PETSC_SMALL) {
        SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trace FE error %g",err);
      }
      ierr = VecRestoreArray(vecFull,&arrayFull);CHKERRQ(ierr);
      ierr = PetscTabulationDestroy(&Tfull);CHKERRQ(ierr);
      ierr = PetscTabulationDestroy(&Tsub);CHKERRQ(ierr);
      /* clean up */
      ierr = PetscFree(arraySub);CHKERRQ(ierr);
      ierr = VecDestroy(&vecFull);CHKERRQ(ierr);
      ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
      ierr = PetscFree4(testSub,testFull,outSub,outFull);CHKERRQ(ierr);
      ierr = PetscFEDestroy(&traceFE);CHKERRQ(ierr);
    }
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
    args: -petscspace_degree 1 -trace_fe_view
TEST*/
