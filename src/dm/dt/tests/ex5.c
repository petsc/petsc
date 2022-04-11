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

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  comm = PETSC_COMM_WORLD;
  PetscOptionsBegin(comm,"","Options for subspace test","none");
  PetscCall(PetscOptionsRangeInt("-dim", "The spatial dimension","ex5.c",dim,&dim,NULL,1,3));
  PetscCall(PetscOptionsBool("-simplex", "Test simplex element","ex5.c",simplex,&simplex,NULL));
  PetscCall(PetscOptionsBoundedInt("-num_comp", "Number of components in space","ex5.c",Nc,&Nc,NULL,1));
  PetscOptionsEnd();
  PetscCall(DMShellCreate(comm,&dm));
  PetscCall(PetscFECreateDefault(comm,dim,Nc,simplex,NULL,PETSC_DEFAULT,&fe));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFESetName(fe, "solution"));
  PetscCall(PetscFEGetBasisSpace(fe,&space));
  PetscCall(PetscSpaceGetNumComponents(space,&Nc));
  PetscCall(PetscFEGetDualSpace(fe,&dualspace));
  PetscCall(PetscDualSpaceGetHeightSubspace(dualspace,1,&dualsubspace));
  PetscCall(PetscDualSpaceGetDM(dualspace,&dm));
  PetscCall(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
  if (cEnd > cStart) {
    PetscInt coneSize;

    PetscCall(DMPlexGetConeSize(dm,cStart,&coneSize));
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

      PetscCall(DMPlexGetCone(dm,cStart,&cone));
      point = cone[0];
      PetscCall(PetscFECreatePointTrace(fe,point,&traceFE));
      PetscCall(PetscFESetUp(traceFE));
      PetscCall(PetscFEViewFromOptions(traceFE,NULL,"-trace_fe_view"));
      PetscCall(PetscMalloc4(dim - 1,&testSub,dim,&testFull,Nc,&outSub,Nc,&outFull));
      PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rand));
      PetscCall(PetscRandomSetFromOptions(rand));
      PetscCall(PetscRandomSetInterval(rand,-1.,1.));
      /* create a random point in the trace domain */
      for (i = 0; i < dim - 1; i++) {
        PetscCall(PetscRandomGetValueReal(rand,&testSub[i]));
      }
      PetscCall(DMPlexComputeCellGeometryFEM(dm,point,NULL,testFull,J,NULL,&detJ));
      /* project it into the full domain */
      for (i = 0; i < dim; i++) {
        for (j = 0; j < dim - 1; j++) testFull[i] += J[i * dim + j] * (testSub[j] - xi0[j]);
      }
      /* create a random vector in the full domain */
      PetscCall(PetscFEGetDimension(fe,&nFull));
      PetscCall(VecCreateSeq(PETSC_COMM_SELF,nFull,&vecFull));
      PetscCall(VecGetArray(vecFull,&arrayFull));
      for (i = 0; i < nFull; i++) {
        PetscCall(PetscRandomGetValue(rand,&arrayFull[i]));
      }
      PetscCall(VecRestoreArray(vecFull,&arrayFull));
      /* create a vector on the trace domain */
      PetscCall(PetscFEGetDimension(traceFE,&nSub));
      /* get the subset of the original finite element space that is supported on the trace space */
      PetscCall(PetscDualSpaceGetSection(dualspace,&sectionFull));
      PetscCall(PetscSectionSetUp(sectionFull));
      /* get the trace degrees of freedom */
      PetscCall(PetscMalloc1(nSub,&arraySub));
      PetscCall(DMPlexVecGetClosure(dm,sectionFull,vecFull,point,&nSub,&arraySub));
      /* get the tabulations */
      PetscCall(PetscFECreateTabulation(traceFE,1,1,testSub,0,&Tsub));
      PetscCall(PetscFECreateTabulation(fe,1,1,testFull,0,&Tfull));
      for (i = 0; i < Nc; i++) {
        outSub[i] = 0.0;
        for (j = 0; j < nSub; j++) {
          outSub[i] += Tsub->T[0][j * Nc + i] * arraySub[j];
        }
      }
      PetscCall(VecGetArray(vecFull,&arrayFull));
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
      PetscCall(VecRestoreArray(vecFull,&arrayFull));
      PetscCall(PetscTabulationDestroy(&Tfull));
      PetscCall(PetscTabulationDestroy(&Tsub));
      /* clean up */
      PetscCall(PetscFree(arraySub));
      PetscCall(VecDestroy(&vecFull));
      PetscCall(PetscRandomDestroy(&rand));
      PetscCall(PetscFree4(testSub,testFull,outSub,outFull));
      PetscCall(PetscFEDestroy(&traceFE));
    }
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: 0
    args: -petscspace_degree 1 -trace_fe_view
TEST*/
