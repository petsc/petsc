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
  CHKERRQ(PetscOptionsRangeInt("-dim", "The spatial dimension","ex5.c",dim,&dim,NULL,1,3));
  CHKERRQ(PetscOptionsBool("-simplex", "Test simplex element","ex5.c",simplex,&simplex,NULL));
  CHKERRQ(PetscOptionsBoundedInt("-num_comp", "Number of components in space","ex5.c",Nc,&Nc,NULL,1));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(DMShellCreate(comm,&dm));
  CHKERRQ(PetscFECreateDefault(comm,dim,Nc,simplex,NULL,PETSC_DEFAULT,&fe));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFESetName(fe, "solution"));
  CHKERRQ(PetscFEGetBasisSpace(fe,&space));
  CHKERRQ(PetscSpaceGetNumComponents(space,&Nc));
  CHKERRQ(PetscFEGetDualSpace(fe,&dualspace));
  CHKERRQ(PetscDualSpaceGetHeightSubspace(dualspace,1,&dualsubspace));
  CHKERRQ(PetscDualSpaceGetDM(dualspace,&dm));
  CHKERRQ(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
  if (cEnd > cStart) {
    PetscInt coneSize;

    CHKERRQ(DMPlexGetConeSize(dm,cStart,&coneSize));
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

      CHKERRQ(DMPlexGetCone(dm,cStart,&cone));
      point = cone[0];
      CHKERRQ(PetscFECreatePointTrace(fe,point,&traceFE));
      CHKERRQ(PetscFESetUp(traceFE));
      CHKERRQ(PetscFEViewFromOptions(traceFE,NULL,"-trace_fe_view"));
      CHKERRQ(PetscMalloc4(dim - 1,&testSub,dim,&testFull,Nc,&outSub,Nc,&outFull));
      CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rand));
      CHKERRQ(PetscRandomSetFromOptions(rand));
      CHKERRQ(PetscRandomSetInterval(rand,-1.,1.));
      /* create a random point in the trace domain */
      for (i = 0; i < dim - 1; i++) {
        CHKERRQ(PetscRandomGetValueReal(rand,&testSub[i]));
      }
      CHKERRQ(DMPlexComputeCellGeometryFEM(dm,point,NULL,testFull,J,NULL,&detJ));
      /* project it into the full domain */
      for (i = 0; i < dim; i++) {
        for (j = 0; j < dim - 1; j++) testFull[i] += J[i * dim + j] * (testSub[j] - xi0[j]);
      }
      /* create a random vector in the full domain */
      CHKERRQ(PetscFEGetDimension(fe,&nFull));
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,nFull,&vecFull));
      CHKERRQ(VecGetArray(vecFull,&arrayFull));
      for (i = 0; i < nFull; i++) {
        CHKERRQ(PetscRandomGetValue(rand,&arrayFull[i]));
      }
      CHKERRQ(VecRestoreArray(vecFull,&arrayFull));
      /* create a vector on the trace domain */
      CHKERRQ(PetscFEGetDimension(traceFE,&nSub));
      /* get the subset of the original finite element space that is supported on the trace space */
      CHKERRQ(PetscDualSpaceGetSection(dualspace,&sectionFull));
      CHKERRQ(PetscSectionSetUp(sectionFull));
      /* get the trace degrees of freedom */
      CHKERRQ(PetscMalloc1(nSub,&arraySub));
      CHKERRQ(DMPlexVecGetClosure(dm,sectionFull,vecFull,point,&nSub,&arraySub));
      /* get the tabulations */
      CHKERRQ(PetscFECreateTabulation(traceFE,1,1,testSub,0,&Tsub));
      CHKERRQ(PetscFECreateTabulation(fe,1,1,testFull,0,&Tfull));
      for (i = 0; i < Nc; i++) {
        outSub[i] = 0.0;
        for (j = 0; j < nSub; j++) {
          outSub[i] += Tsub->T[0][j * Nc + i] * arraySub[j];
        }
      }
      CHKERRQ(VecGetArray(vecFull,&arrayFull));
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
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Trace FE error %g",err);
      }
      CHKERRQ(VecRestoreArray(vecFull,&arrayFull));
      CHKERRQ(PetscTabulationDestroy(&Tfull));
      CHKERRQ(PetscTabulationDestroy(&Tsub));
      /* clean up */
      CHKERRQ(PetscFree(arraySub));
      CHKERRQ(VecDestroy(&vecFull));
      CHKERRQ(PetscRandomDestroy(&rand));
      CHKERRQ(PetscFree4(testSub,testFull,outSub,outFull));
      CHKERRQ(PetscFEDestroy(&traceFE));
    }
  }
  CHKERRQ(PetscFEDestroy(&fe));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  test:
    suffix: 0
    args: -petscspace_degree 1 -trace_fe_view
TEST*/
