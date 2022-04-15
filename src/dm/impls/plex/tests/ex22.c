
const char help[] = "Test DMPlexCoordinatesToReference().\n\n";

#include <petscdm.h>
#include <petscdmplex.h>

static PetscErrorCode testIdentity(DM dm, PetscBool dmIsSimplicial, PetscInt cell, PetscRandom randCtx, PetscInt numPoints, PetscReal tol)
{
  PetscInt       i, j, dimC, dimR;
  PetscReal      *preimage, *mapped, *inverted;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm,&dimR));
  PetscCall(DMGetCoordinateDim(dm,&dimC));

  PetscCall(DMGetWorkArray(dm,dimR * numPoints,MPIU_REAL,&preimage));
  PetscCall(DMGetWorkArray(dm,dimC * numPoints,MPIU_REAL,&mapped));
  PetscCall(DMGetWorkArray(dm,dimR * numPoints,MPIU_REAL,&inverted));

  for (i = 0; i < dimR * numPoints; i++) {
    PetscCall(PetscRandomGetValueReal(randCtx, &preimage[i]));
  }
  if (dmIsSimplicial && dimR > 1) {
    for (i = 0; i < numPoints; i++) {
      for (j = 0; j < dimR; j++) {
        PetscReal x = preimage[i * dimR + j];
        PetscReal y = preimage[i * dimR + ((j + 1) % dimR)];

        preimage[i * dimR + ((j + 1) % dimR)] = -1. + (y + 1.) * 0.5 * (1. - x);
      }
    }
  }

  PetscCall(DMPlexReferenceToCoordinates(dm,cell,numPoints,preimage,mapped));
  PetscCall(DMPlexCoordinatesToReference(dm,cell,numPoints,mapped,inverted));

  for (i = 0; i < numPoints; i++) {
    PetscReal max = 0.;

    for (j = 0; j < dimR; j++) {
      max = PetscMax(max,PetscAbsReal(preimage[i * dimR + j] - inverted[i * dimR + j]));
    }
    if (max > tol) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Bad inversion for cell %" PetscInt_FMT " with error %g (tol %g): (",cell,(double)max,(double)tol));
      for (j = 0; j < dimR; j++) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"%+f",(double) preimage[i * dimR + j]));
        if (j < dimR - 1) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF,","));
        }
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF,") --> ("));
      for (j = 0; j < dimC; j++) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"%+f",(double) mapped[i * dimC + j]));
        if (j < dimC - 1) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF,","));
        }
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF,") --> ("));
      for (j = 0; j < dimR; j++) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"%+f",(double) inverted[i * dimR + j]));
        if (j < dimR - 1) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF,","));
        }
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF,")\n"));
    } else {
      char   strBuf[BUFSIZ] = {'\0'};
      size_t offset = 0, count;

      PetscCall(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,"Good inversion for cell %" PetscInt_FMT ": (",&count,cell));
      offset += count;
      for (j = 0; j < dimR; j++) {
        PetscCall(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,"%+f",&count,(double) preimage[i * dimR + j]));
        offset += count;
        if (j < dimR - 1) {
          PetscCall(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,",",&count));
          offset += count;
        }
      }
      PetscCall(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,") --> (",&count));
      offset += count;
      for (j = 0; j < dimC; j++) {
        PetscCall(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,"%+f",&count,(double) mapped[i * dimC + j]));
        offset += count;
        if (j < dimC - 1) {
          PetscCall(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,",",&count));
          offset += count;
        }
      }
      PetscCall(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,") --> (",&count));
      offset += count;
      for (j = 0; j < dimR; j++) {
        PetscCall(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,"%+f",&count,(double) inverted[i * dimR + j]));
        offset += count;
        if (j < dimR - 1) {
          PetscCall(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,",",&count));
          offset += count;
        }
      }
      PetscCall(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,")\n",&count));
      PetscCall(PetscInfo(dm,"%s",strBuf));
    }
  }

  PetscCall(DMRestoreWorkArray(dm,dimR * numPoints,MPIU_REAL,&inverted));
  PetscCall(DMRestoreWorkArray(dm,dimC * numPoints,MPIU_REAL,&mapped));
  PetscCall(DMRestoreWorkArray(dm,dimR * numPoints,MPIU_REAL,&preimage));
  PetscFunctionReturn(0);
}

static PetscErrorCode identityEmbedding(PetscInt dim, PetscReal time, const PetscReal *x, PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt i;

  for (i = 0; i < dim; i++) {u[i] = x[i];};
  return 0;
}

int main(int argc, char **argv)
{
  PetscRandom    randCtx;
  PetscInt       dim, dimC, isSimplex, isFE, numTests = 10;
  PetscReal      perturb = 0.1, tol = 10. * PETSC_SMALL;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&randCtx));
  PetscCall(PetscRandomSetInterval(randCtx,-1.,1.));
  PetscCall(PetscRandomSetFromOptions(randCtx));
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex21",NULL);
  PetscCall(PetscOptionsReal("-vertex_perturbation","scale of random vertex distortion",NULL,perturb,&perturb,NULL));
  PetscCall(PetscOptionsBoundedInt("-num_test_points","number of points to test",NULL,numTests,&numTests,NULL,0));
  PetscOptionsEnd();
  for (dim = 1; dim <= 3; dim++) {
    for (dimC = dim; dimC <= PetscMin(3,dim + 1); dimC++) {
      for (isSimplex = 0; isSimplex < 2; isSimplex++) {
        for (isFE = 0; isFE < 2; isFE++) {
          DM           dm;
          Vec          coords;
          PetscScalar *coordArray;
          PetscReal    noise;
          PetscInt     i, n, order = 1;

          PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, isSimplex ? PETSC_TRUE : PETSC_FALSE), &dm));
          if (isFE) {
            DM             dmCoord;
            PetscSpace     sp;
            PetscFE        fe;
            Vec            localCoords;
            PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *) = {identityEmbedding};
            void           *ctxs[1] = {NULL};

            PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dm),dim,dim,isSimplex ? PETSC_TRUE : PETSC_FALSE,isSimplex ? NULL : "tensor_" ,PETSC_DEFAULT,&fe));
            PetscCall(PetscFEGetBasisSpace(fe,&sp));
            PetscCall(PetscSpaceGetDegree(sp,&order,NULL));
            PetscCall(DMSetField(dm,0,NULL,(PetscObject)fe));
            PetscCall(DMCreateDS(dm));
            PetscCall(DMCreateLocalVector(dm,&localCoords));
            PetscCall(DMProjectFunctionLocal(dm,0,funcs,ctxs,INSERT_VALUES,localCoords));
            PetscCall(VecSetDM(localCoords,NULL)); /* This is necessary to prevent a reference loop */
            PetscCall(DMClone(dm,&dmCoord));
            PetscCall(DMSetField(dmCoord,0,NULL,(PetscObject)fe));
            PetscCall(PetscFEDestroy(&fe));
            PetscCall(DMCreateDS(dmCoord));
            PetscCall(DMSetCoordinateDM(dm,dmCoord));
            PetscCall(DMDestroy(&dmCoord));
            PetscCall(DMSetCoordinatesLocal(dm,localCoords));
            PetscCall(VecDestroy(&localCoords));
          }
          PetscCall(PetscInfo(dm,"Testing %s%" PetscInt_FMT " %" PetscInt_FMT "D mesh embedded in %" PetscInt_FMT "D\n",isSimplex ? "P" : "Q" , order, dim, dimC));
          PetscCall(DMGetCoordinatesLocal(dm,&coords));
          PetscCall(VecGetLocalSize(coords,&n));
          if (dimC > dim) { /* reembed in higher dimension */
            PetscSection sec, newSec;
            PetscInt     pStart, pEnd, p, i, newN;
            Vec          newVec;
            DM           coordDM;
            PetscScalar  *newCoordArray;

            PetscCall(DMGetCoordinateSection(dm,&sec));
            PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)sec),&newSec));
            PetscCall(PetscSectionSetNumFields(newSec,1));
            PetscCall(PetscSectionGetChart(sec,&pStart,&pEnd));
            PetscCall(PetscSectionSetChart(newSec,pStart,pEnd));
            for (p = pStart; p < pEnd; p++) {
              PetscInt nDof;

              PetscCall(PetscSectionGetDof(sec,p,&nDof));
              PetscCheck(nDof % dim == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Coordinate section point %" PetscInt_FMT " has %" PetscInt_FMT " dofs != 0 mod %" PetscInt_FMT,p,nDof,dim);
              PetscCall(PetscSectionSetDof(newSec,p,(nDof/dim)*dimC));
              PetscCall(PetscSectionSetFieldDof(newSec,p,0,(nDof/dim)*dimC));
            }
            PetscCall(PetscSectionSetUp(newSec));
            PetscCall(PetscSectionGetStorageSize(newSec,&newN));
            PetscCall(VecCreateSeq(PETSC_COMM_SELF,newN,&newVec));
            PetscCall(VecGetArray(newVec,&newCoordArray));
            PetscCall(VecGetArray(coords,&coordArray));
            for (i = 0; i < n / dim; i++) {
              PetscInt j;

              for (j = 0; j < dim; j++) {
                newCoordArray[i * dimC + j] = coordArray[i * dim + j];
              }
              for (; j < dimC; j++) {
                newCoordArray[i * dimC + j] = 0.;
              }
            }
            PetscCall(VecRestoreArray(coords,&coordArray));
            PetscCall(VecRestoreArray(newVec,&newCoordArray));
            PetscCall(DMSetCoordinateDim(dm,dimC));
            PetscCall(DMSetCoordinateSection(dm,dimC,newSec));
            PetscCall(DMSetCoordinatesLocal(dm,newVec));
            PetscCall(VecDestroy(&newVec));
            PetscCall(PetscSectionDestroy(&newSec));
            PetscCall(DMGetCoordinatesLocal(dm,&coords));
            PetscCall(DMGetCoordinateDM(dm,&coordDM));
            if (isFE) {
              PetscFE fe;

              PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dm),dim,dimC,isSimplex ? PETSC_TRUE : PETSC_FALSE,isSimplex ? NULL : "tensor_",PETSC_DEFAULT,&fe));
              PetscCall(DMSetField(coordDM,0,NULL,(PetscObject)fe));
              PetscCall(PetscFEDestroy(&fe));
              PetscCall(DMCreateDS(coordDM));
            }
            PetscCall(DMSetCoordinateDim(coordDM,dimC));
            PetscCall(VecGetLocalSize(coords,&n));
          }
          PetscCall(VecGetArray(coords,&coordArray));
          for (i = 0; i < n; i++) {
            PetscCall(PetscRandomGetValueReal(randCtx,&noise));
            coordArray[i] += noise * perturb;
          }
          PetscCall(VecRestoreArray(coords,&coordArray));
          PetscCall(DMSetCoordinatesLocal(dm, coords));
          PetscCall(testIdentity(dm, isSimplex ? PETSC_TRUE : PETSC_FALSE, 0, randCtx, numTests, tol));
          PetscCall(DMDestroy(&dm));
        }
      }
    }
  }
  PetscCall(PetscRandomDestroy(&randCtx));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -petscspace_degree 2 -tensor_petscspace_degree 2

TEST*/
