
const char help[] = "Test DMPlexCoordinatesToReference().\n\n";

#include <petscdm.h>
#include <petscdmplex.h>

static PetscErrorCode testIdentity(DM dm, PetscBool dmIsSimplicial, PetscInt cell, PetscRandom randCtx, PetscInt numPoints, PetscReal tol)
{
  PetscInt       i, j, dimC, dimR;
  PetscReal      *preimage, *mapped, *inverted;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm,&dimR);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm,&dimC);CHKERRQ(ierr);

  ierr = DMGetWorkArray(dm,dimR * numPoints,PETSC_REAL,&preimage);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm,dimC * numPoints,PETSC_REAL,&mapped);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm,dimR * numPoints,PETSC_REAL,&inverted);CHKERRQ(ierr);

  for (i = 0; i < dimR * numPoints; i++) {
    ierr = PetscRandomGetValueReal(randCtx, &preimage[i]);CHKERRQ(ierr);
  }
  if (dmIsSimplicial && dimR > 1) {
    for (i = 0; i < numPoints; i++) {
      for (j = 0; j < dimR; j++) {
        PetscReal x = preimage[i * dimR + j];
        PetscReal y = preimage[i * dimR + ((j + 1) % dimR)];

        preimage[i * dimR + ((j + 1) % dimR)] = -1. + (y + 1.) * 0.5 * (x - 1.);
      }
    }
  }

  ierr = DMPlexReferenceToCoordinates(dm,cell,numPoints,preimage,mapped);CHKERRQ(ierr);
  ierr = DMPlexCoordinatesToReference(dm,cell,numPoints,mapped,inverted);CHKERRQ(ierr);

  for (i = 0; i < numPoints; i++) {
    PetscReal max = 0.;

    for (j = 0; j < dimR; j++) {
      max = PetscMax(max,PetscAbsReal(preimage[i * dimR + j] - inverted[i * dimR + j]));
    }
    if (max > tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Bad inversion for cell %D with tolerance %f: (",cell,tol);CHKERRQ(ierr);
      for (j = 0; j < dimR; j++) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"%f",(double) preimage[i * dimR + j]);CHKERRQ(ierr);
        if (j < dimR - 1) {
          ierr = PetscPrintf(PETSC_COMM_SELF,",");CHKERRQ(ierr);
        }
      }
      ierr = PetscPrintf(PETSC_COMM_SELF,") --> (");CHKERRQ(ierr);
      for (j = 0; j < dimC; j++) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"%f",(double) mapped[i * dimC + j]);CHKERRQ(ierr);
        if (j < dimC - 1) {
          ierr = PetscPrintf(PETSC_COMM_SELF,",");CHKERRQ(ierr);
        }
      }
      ierr = PetscPrintf(PETSC_COMM_SELF,") --> (");CHKERRQ(ierr);
      for (j = 0; j < dimR; j++) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"%f",(double) inverted[i * dimR + j]);CHKERRQ(ierr);
        if (j < dimR - 1) {
          ierr = PetscPrintf(PETSC_COMM_SELF,",");CHKERRQ(ierr);
        }
      }
      ierr = PetscPrintf(PETSC_COMM_SELF,")\n");CHKERRQ(ierr);
    } else {
      char strBuf[BUFSIZ] = {'\0'};
      int  offset = 0;

      offset = snprintf(strBuf + offset,BUFSIZ-offset,"Good inversion for cell %d: (", (int) cell);
      for (j = 0; j < dimR; j++) {
        offset += snprintf(strBuf + offset,BUFSIZ-offset,"%f",(double) preimage[i * dimR + j]);
        if (j < dimR - 1) {
          offset += snprintf(strBuf + offset,BUFSIZ-offset,",");
        }
      }
      offset += snprintf(strBuf + offset,BUFSIZ-offset,") --> (");
      for (j = 0; j < dimC; j++) {
        offset += snprintf(strBuf + offset,BUFSIZ-offset,"%f",(double) mapped[i * dimC + j]);
        if (j < dimC - 1) {
          offset += snprintf(strBuf + offset,BUFSIZ-offset,",");
        }
      }
      offset += snprintf(strBuf + offset,BUFSIZ-offset,") --> (");
      for (j = 0; j < dimR; j++) {
        offset += snprintf(strBuf + offset,BUFSIZ-offset,"%f",(double) inverted[i * dimR + j]);
        if (j < dimR - 1) {
          offset += snprintf(strBuf + offset,BUFSIZ-offset,",");
        }
      }
      offset += snprintf(strBuf + offset,BUFSIZ-offset,")\n");
      ierr = PetscInfo1(dm,"%s",strBuf);CHKERRQ(ierr);
    }
  }

  ierr = DMRestoreWorkArray(dm,dimR * numPoints,PETSC_REAL,&inverted);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm,dimC * numPoints,PETSC_REAL,&mapped);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm,dimR * numPoints,PETSC_REAL,&preimage);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  PetscRandom    randCtx;
  PetscInt       dim, dimC, isSimplex, numTests = 10;
  PetscReal      perturb = 0.1, tol = PETSC_SMALL;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&randCtx);CHKERRQ(ierr);
  ierr = PetscRandomSetInterval(randCtx,-1.,1.);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(randCtx);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex21",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-vertex_perturbation","scale of random vertex distortion",NULL,perturb,&perturb,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_test_points","number of points to test",NULL,numTests,&numTests,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  for (dim = 1; dim <= 3; dim++) {
    for (dimC = dim; dimC <= PetscMin(3,dim + 1); dimC++) {
      for (isSimplex = 0; isSimplex < 2; isSimplex++) {
        DM           dm;
        Vec          coords;
        PetscScalar *coordArray;
        PetscReal    noise;
        PetscInt     i, n;

        ierr = DMPlexCreateReferenceCell(PETSC_COMM_SELF,dim,isSimplex ? PETSC_TRUE : PETSC_FALSE, &dm);CHKERRQ(ierr);
        ierr = PetscInfo3(dm,"Testing %s %DD mesh embedded in %DD\n",isSimplex ? "simplicial" : "tensor", dim, dimC);
        ierr = DMGetCoordinatesLocal(dm,&coords);CHKERRQ(ierr);
        ierr = VecGetLocalSize(coords,&n);CHKERRQ(ierr);
        if (dimC > dim) { /* reembed in higher dimension */
          PetscSection sec, newSec;
          PetscInt     pStart, pEnd, p, i, newN;
          Vec          newVec;
          DM           coordDM;
          PetscScalar  *newCoordArray;

          ierr = DMGetCoordinateSection(dm,&sec);CHKERRQ(ierr);
          ierr = PetscSectionCreate(PetscObjectComm((PetscObject)sec),&newSec);CHKERRQ(ierr);
          ierr = PetscSectionSetNumFields(newSec,1);CHKERRQ(ierr);
          ierr = PetscSectionGetChart(sec,&pStart,&pEnd);CHKERRQ(ierr);
          ierr = PetscSectionSetChart(newSec,pStart,pEnd);CHKERRQ(ierr);
          for (p = pStart; p < pEnd; p++) {
            PetscInt nDof;

            ierr = PetscSectionGetDof(sec,p,&nDof);CHKERRQ(ierr);
            if (nDof != dim) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Coordinate section point %D has %D dofs != dim %D",p,nDof,dim);
            ierr = PetscSectionSetDof(newSec,p,dimC);CHKERRQ(ierr);
            ierr = PetscSectionSetFieldDof(newSec,p,0,dimC);CHKERRQ(ierr);
          }
          ierr = PetscSectionSetUp(newSec);CHKERRQ(ierr);
          ierr = PetscSectionGetStorageSize(newSec,&newN);CHKERRQ(ierr);
          ierr = VecCreateSeq(PETSC_COMM_SELF,newN,&newVec);CHKERRQ(ierr);
          ierr = VecGetArray(newVec,&newCoordArray);CHKERRQ(ierr);
          ierr = VecGetArray(coords,&coordArray);CHKERRQ(ierr);
          for (i = 0; i < n / dim; i++) {
            PetscInt j;

            for (j = 0; j < dim; j++) {
              newCoordArray[i * dimC + j] = coordArray[i * dim + j];
            }
            for (; j < dimC; j++) {
              newCoordArray[i * dimC + j] = 0.;
            }
          }
          ierr = VecRestoreArray(coords,&coordArray);CHKERRQ(ierr);
          ierr = VecRestoreArray(newVec,&newCoordArray);CHKERRQ(ierr);
          ierr = DMSetCoordinateDim(dm,dimC);CHKERRQ(ierr);
          ierr = DMSetCoordinateSection(dm,dimC,newSec);CHKERRQ(ierr);
          ierr = DMSetCoordinatesLocal(dm,newVec);CHKERRQ(ierr);
          ierr = VecDestroy(&newVec);CHKERRQ(ierr);
          ierr = PetscSectionDestroy(&newSec);CHKERRQ(ierr);
          ierr = DMGetCoordinatesLocal(dm,&coords);CHKERRQ(ierr);
          ierr = DMGetCoordinateDM(dm,&coordDM);CHKERRQ(ierr);
          ierr = DMSetCoordinateDim(coordDM,dimC);CHKERRQ(ierr);
          ierr = VecGetLocalSize(coords,&n);CHKERRQ(ierr);
        }
        ierr = VecGetArray(coords,&coordArray);CHKERRQ(ierr);
        for (i = 0; i < n; i++) {
          ierr = PetscRandomGetValueReal(randCtx,&noise);CHKERRQ(ierr);
          coordArray[i] += noise * perturb;
        }
        ierr = VecRestoreArray(coords,&coordArray);CHKERRQ(ierr);
        ierr = DMSetCoordinatesLocal(dm, coords);CHKERRQ(ierr);
        ierr = testIdentity(dm, isSimplex ? PETSC_TRUE : PETSC_FALSE, 0, randCtx, numTests, tol);CHKERRQ(ierr);
        ierr = DMDestroy(&dm);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscRandomDestroy(&randCtx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
