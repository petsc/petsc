
const char help[] = "Test DMPlexCoordinatesToReference().\n\n";

#include <petscdm.h>
#include <petscdmplex.h>

static PetscErrorCode testIdentity(DM dm, PetscBool dmIsSimplicial, PetscInt cell, PetscRandom randCtx, PetscInt numPoints, PetscReal tol)
{
  PetscInt       i, j, dimC, dimR;
  PetscReal      *preimage, *mapped, *inverted;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm,&dimR));
  CHKERRQ(DMGetCoordinateDim(dm,&dimC));

  CHKERRQ(DMGetWorkArray(dm,dimR * numPoints,MPIU_REAL,&preimage));
  CHKERRQ(DMGetWorkArray(dm,dimC * numPoints,MPIU_REAL,&mapped));
  CHKERRQ(DMGetWorkArray(dm,dimR * numPoints,MPIU_REAL,&inverted));

  for (i = 0; i < dimR * numPoints; i++) {
    CHKERRQ(PetscRandomGetValueReal(randCtx, &preimage[i]));
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

  CHKERRQ(DMPlexReferenceToCoordinates(dm,cell,numPoints,preimage,mapped));
  CHKERRQ(DMPlexCoordinatesToReference(dm,cell,numPoints,mapped,inverted));

  for (i = 0; i < numPoints; i++) {
    PetscReal max = 0.;

    for (j = 0; j < dimR; j++) {
      max = PetscMax(max,PetscAbsReal(preimage[i * dimR + j] - inverted[i * dimR + j]));
    }
    if (max > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Bad inversion for cell %D with error %g (tol %g): (",cell,(double)max,(double)tol));
      for (j = 0; j < dimR; j++) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%+f",(double) preimage[i * dimR + j]));
        if (j < dimR - 1) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,","));
        }
      }
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,") --> ("));
      for (j = 0; j < dimC; j++) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%+f",(double) mapped[i * dimC + j]));
        if (j < dimC - 1) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,","));
        }
      }
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,") --> ("));
      for (j = 0; j < dimR; j++) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%+f",(double) inverted[i * dimR + j]));
        if (j < dimR - 1) {
          CHKERRQ(PetscPrintf(PETSC_COMM_SELF,","));
        }
      }
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,")\n"));
    } else {
      char   strBuf[BUFSIZ] = {'\0'};
      size_t offset = 0, count;

      CHKERRQ(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,"Good inversion for cell %D: (",&count,cell));
      offset += count;
      for (j = 0; j < dimR; j++) {
        CHKERRQ(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,"%+f",&count,(double) preimage[i * dimR + j]));
        offset += count;
        if (j < dimR - 1) {
          CHKERRQ(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,",",&count));
          offset += count;
        }
      }
      CHKERRQ(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,") --> (",&count));
      offset += count;
      for (j = 0; j < dimC; j++) {
        CHKERRQ(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,"%+f",&count,(double) mapped[i * dimC + j]));
        offset += count;
        if (j < dimC - 1) {
          CHKERRQ(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,",",&count));
          offset += count;
        }
      }
      CHKERRQ(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,") --> (",&count));
      offset += count;
      for (j = 0; j < dimR; j++) {
        CHKERRQ(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,"%+f",&count,(double) inverted[i * dimR + j]));
        offset += count;
        if (j < dimR - 1) {
          CHKERRQ(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,",",&count));
          offset += count;
        }
      }
      CHKERRQ(PetscSNPrintfCount(strBuf + offset,BUFSIZ-offset,")\n",&count));
      CHKERRQ(PetscInfo(dm,"%s",strBuf));
    }
  }

  CHKERRQ(DMRestoreWorkArray(dm,dimR * numPoints,MPIU_REAL,&inverted));
  CHKERRQ(DMRestoreWorkArray(dm,dimC * numPoints,MPIU_REAL,&mapped));
  CHKERRQ(DMRestoreWorkArray(dm,dimR * numPoints,MPIU_REAL,&preimage));
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
  PetscErrorCode ierr;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&randCtx));
  CHKERRQ(PetscRandomSetInterval(randCtx,-1.,1.));
  CHKERRQ(PetscRandomSetFromOptions(randCtx));
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex21",NULL);CHKERRQ(ierr);
  CHKERRQ(PetscOptionsReal("-vertex_perturbation","scale of random vertex distortion",NULL,perturb,&perturb,NULL));
  CHKERRQ(PetscOptionsBoundedInt("-num_test_points","number of points to test",NULL,numTests,&numTests,NULL,0));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  for (dim = 1; dim <= 3; dim++) {
    for (dimC = dim; dimC <= PetscMin(3,dim + 1); dimC++) {
      for (isSimplex = 0; isSimplex < 2; isSimplex++) {
        for (isFE = 0; isFE < 2; isFE++) {
          DM           dm;
          Vec          coords;
          PetscScalar *coordArray;
          PetscReal    noise;
          PetscInt     i, n, order = 1;

          CHKERRQ(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, isSimplex ? PETSC_TRUE : PETSC_FALSE), &dm));
          if (isFE) {
            DM             dmCoord;
            PetscSpace     sp;
            PetscFE        fe;
            Vec            localCoords;
            PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *) = {identityEmbedding};
            void           *ctxs[1] = {NULL};

            CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm),dim,dim,isSimplex ? PETSC_TRUE : PETSC_FALSE,isSimplex ? NULL : "tensor_" ,PETSC_DEFAULT,&fe));
            CHKERRQ(PetscFEGetBasisSpace(fe,&sp));
            CHKERRQ(PetscSpaceGetDegree(sp,&order,NULL));
            CHKERRQ(DMSetField(dm,0,NULL,(PetscObject)fe));
            CHKERRQ(DMCreateDS(dm));
            CHKERRQ(DMCreateLocalVector(dm,&localCoords));
            CHKERRQ(DMProjectFunctionLocal(dm,0,funcs,ctxs,INSERT_VALUES,localCoords));
            CHKERRQ(VecSetDM(localCoords,NULL)); /* This is necessary to prevent a reference loop */
            CHKERRQ(DMClone(dm,&dmCoord));
            CHKERRQ(DMSetField(dmCoord,0,NULL,(PetscObject)fe));
            CHKERRQ(PetscFEDestroy(&fe));
            CHKERRQ(DMCreateDS(dmCoord));
            CHKERRQ(DMSetCoordinateDM(dm,dmCoord));
            CHKERRQ(DMDestroy(&dmCoord));
            CHKERRQ(DMSetCoordinatesLocal(dm,localCoords));
            CHKERRQ(VecDestroy(&localCoords));
          }
          CHKERRQ(PetscInfo(dm,"Testing %s%D %DD mesh embedded in %DD\n",isSimplex ? "P" : "Q" , order, dim, dimC));
          CHKERRQ(DMGetCoordinatesLocal(dm,&coords));
          CHKERRQ(VecGetLocalSize(coords,&n));
          if (dimC > dim) { /* reembed in higher dimension */
            PetscSection sec, newSec;
            PetscInt     pStart, pEnd, p, i, newN;
            Vec          newVec;
            DM           coordDM;
            PetscScalar  *newCoordArray;

            CHKERRQ(DMGetCoordinateSection(dm,&sec));
            CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject)sec),&newSec));
            CHKERRQ(PetscSectionSetNumFields(newSec,1));
            CHKERRQ(PetscSectionGetChart(sec,&pStart,&pEnd));
            CHKERRQ(PetscSectionSetChart(newSec,pStart,pEnd));
            for (p = pStart; p < pEnd; p++) {
              PetscInt nDof;

              CHKERRQ(PetscSectionGetDof(sec,p,&nDof));
              PetscCheckFalse(nDof % dim,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Coordinate section point %D has %D dofs != 0 mod %D",p,nDof,dim);
              CHKERRQ(PetscSectionSetDof(newSec,p,(nDof/dim)*dimC));
              CHKERRQ(PetscSectionSetFieldDof(newSec,p,0,(nDof/dim)*dimC));
            }
            CHKERRQ(PetscSectionSetUp(newSec));
            CHKERRQ(PetscSectionGetStorageSize(newSec,&newN));
            CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,newN,&newVec));
            CHKERRQ(VecGetArray(newVec,&newCoordArray));
            CHKERRQ(VecGetArray(coords,&coordArray));
            for (i = 0; i < n / dim; i++) {
              PetscInt j;

              for (j = 0; j < dim; j++) {
                newCoordArray[i * dimC + j] = coordArray[i * dim + j];
              }
              for (; j < dimC; j++) {
                newCoordArray[i * dimC + j] = 0.;
              }
            }
            CHKERRQ(VecRestoreArray(coords,&coordArray));
            CHKERRQ(VecRestoreArray(newVec,&newCoordArray));
            CHKERRQ(DMSetCoordinateDim(dm,dimC));
            CHKERRQ(DMSetCoordinateSection(dm,dimC,newSec));
            CHKERRQ(DMSetCoordinatesLocal(dm,newVec));
            CHKERRQ(VecDestroy(&newVec));
            CHKERRQ(PetscSectionDestroy(&newSec));
            CHKERRQ(DMGetCoordinatesLocal(dm,&coords));
            CHKERRQ(DMGetCoordinateDM(dm,&coordDM));
            if (isFE) {
              PetscFE fe;

              CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm),dim,dimC,isSimplex ? PETSC_TRUE : PETSC_FALSE,isSimplex ? NULL : "tensor_",PETSC_DEFAULT,&fe));
              CHKERRQ(DMSetField(coordDM,0,NULL,(PetscObject)fe));
              CHKERRQ(PetscFEDestroy(&fe));
              CHKERRQ(DMCreateDS(coordDM));
            }
            CHKERRQ(DMSetCoordinateDim(coordDM,dimC));
            CHKERRQ(VecGetLocalSize(coords,&n));
          }
          CHKERRQ(VecGetArray(coords,&coordArray));
          for (i = 0; i < n; i++) {
            CHKERRQ(PetscRandomGetValueReal(randCtx,&noise));
            coordArray[i] += noise * perturb;
          }
          CHKERRQ(VecRestoreArray(coords,&coordArray));
          CHKERRQ(DMSetCoordinatesLocal(dm, coords));
          CHKERRQ(testIdentity(dm, isSimplex ? PETSC_TRUE : PETSC_FALSE, 0, randCtx, numTests, tol));
          CHKERRQ(DMDestroy(&dm));
        }
      }
    }
  }
  CHKERRQ(PetscRandomDestroy(&randCtx));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    args: -petscspace_degree 2 -tensor_petscspace_degree 2

TEST*/
