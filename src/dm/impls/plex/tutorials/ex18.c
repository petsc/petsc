#include "petscsys.h"
static const char help[] = "Test of PETSc/CAD Shape Modification Technology";

#include <petscdmplexegads.h>
#include <petsc/private/hashmapi.h>

static PetscErrorCode surfArea(DM dm)
{
  DMLabel     bodyLabel, faceLabel;
  double      surfaceArea = 0., volume = 0.;
  PetscReal   vol, centroid[3], normal[3];
  PetscInt    dim, cStart, cEnd, fStart, fEnd;
  PetscInt    bodyID, faceID;
  MPI_Comm    comm;
  const char *name;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(PetscObjectGetName((PetscObject)dm, &name));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscPrintf(comm, "    dim = %" PetscInt_FMT "\n", dim));
  PetscCall(DMGetLabel(dm, "EGADS Body ID", &bodyLabel));
  PetscCall(DMGetLabel(dm, "EGADS Face ID", &faceLabel));
  PetscCall(DMGetCoordinatesLocalSetUp(dm));

  if (dim == 2) {
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    for (PetscInt ii = cStart; ii < cEnd; ++ii) {
      PetscCall(DMLabelGetValue(faceLabel, ii, &faceID));
      if (faceID >= 0) {
        PetscCall(DMPlexComputeCellGeometryFVM(dm, ii, &vol, centroid, normal));
        surfaceArea += vol;
      }
    }
  }

  if (dim == 3) {
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    for (PetscInt ii = fStart; ii < fEnd; ++ii) {
      PetscCall(DMLabelGetValue(faceLabel, ii, &faceID));
      if (faceID >= 0) {
        PetscCall(DMPlexComputeCellGeometryFVM(dm, ii, &vol, centroid, normal));
        surfaceArea += vol;
      }
    }

    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    for (PetscInt ii = cStart; ii < cEnd; ++ii) {
      PetscCall(DMLabelGetValue(bodyLabel, ii, &bodyID));
      if (bodyID >= 0) {
        PetscCall(DMPlexComputeCellGeometryFVM(dm, ii, &vol, centroid, normal));
        volume += vol;
      }
    }
  }

  if (dim == 2) {
    PetscCall(PetscPrintf(comm, "%s Surface Area = %.6e \n\n", name, (double)surfaceArea));
  } else if (dim == 3) {
    PetscCall(PetscPrintf(comm, "%s Volume = %.6e \n", name, (double)volume));
    PetscCall(PetscPrintf(comm, "%s Surface Area = %.6e \n\n", name, (double)surfaceArea));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  /* EGADSlite variables */
  PetscGeom model, *bodies, *fobjs;
  int       Nb, Nf, id;
  /* PETSc variables */
  DM             dm;
  MPI_Comm       comm;
  PetscContainer modelObj, cpHashTableObj, wHashTableObj, cpCoordDataLengthObj, wDataLengthObj;
  Vec            cntrlPtCoordsVec, cntrlPtWeightsVec;
  PetscScalar   *cpCoordData, *wData;
  PetscInt       cpCoordDataLength, wDataLength;
  PetscInt      *cpCoordDataLengthPtr, *wDataLengthPtr;
  PetscHMapI     cpHashTable, wHashTable;
  PetscInt       Nr = 2;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dm));

  PetscCall(PetscObjectSetName((PetscObject)dm, "Original Surface"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(surfArea(dm));

  // Expose Geometry Definition Data and Calculate Surface Gradients
  PetscCall(DMPlexGeomDataAndGrads(dm, PETSC_FALSE));

  // Get Attached EGADS model
  PetscCall(PetscObjectQuery((PetscObject)dm, "EGADS Model", (PetscObject *)&modelObj));
  if (!modelObj) PetscCall(PetscObjectQuery((PetscObject)dm, "EGADSlite Model", (PetscObject *)&modelObj));

  // Get attached EGADS model (pointer)
  PetscCall(PetscContainerGetPointer(modelObj, (void **)&model));

  // Look to see if DM has Container for Geometry Control Point Data
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Hash Table", (PetscObject *)&cpHashTableObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Coordinates", (PetscObject *)&cntrlPtCoordsVec));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Coordinate Data Length", (PetscObject *)&cpCoordDataLengthObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weights Hash Table", (PetscObject *)&wHashTableObj));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight Data", (PetscObject *)&cntrlPtWeightsVec));
  PetscCall(PetscObjectQuery((PetscObject)dm, "Control Point Weight Data Length", (PetscObject *)&wDataLengthObj));

  // Get attached EGADS model Control Point and Weights Hash Tables and Data Arrays (pointer)
  PetscCall(PetscContainerGetPointer(cpHashTableObj, (void **)&cpHashTable));
  PetscCall(VecGetArrayWrite(cntrlPtCoordsVec, &cpCoordData));
  PetscCall(PetscContainerGetPointer(cpCoordDataLengthObj, (void **)&cpCoordDataLengthPtr));
  PetscCall(PetscContainerGetPointer(wHashTableObj, (void **)&wHashTable));
  PetscCall(VecGetArrayWrite(cntrlPtWeightsVec, &wData));
  PetscCall(PetscContainerGetPointer(wDataLengthObj, (void **)&wDataLengthPtr));

  cpCoordDataLength = *cpCoordDataLengthPtr;
  wDataLength       = *wDataLengthPtr;
  PetscCheck(cpCoordDataLength && wDataLength, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Data sizes must be positive");

  // Get the number of bodies and body objects in the model
  PetscCall(DMPlexGetGeomModelBodies(dm, &bodies, &Nb));

  // Get all FACES of the Body
  PetscGeom body = bodies[0];
  PetscCall(DMPlexGetGeomModelBodyFaces(dm, body, &fobjs, &Nf));

  // Update Control Point and Weight definitions for each surface
  for (PetscInt jj = 0; jj < Nf; ++jj) {
    PetscGeom face         = fobjs[jj];
    PetscInt  numCntrlPnts = 0;

    PetscCall(DMPlexGetGeomID(dm, body, face, &id));
    PetscCall(DMPlexGetGeomFaceNumOfControlPoints(dm, face, &numCntrlPnts));

    // Update Control Points
    PetscHashIter CPiter, Witer;
    PetscBool     CPfound, Wfound;
    PetscInt      faceCPStartRow, faceWStartRow;

    PetscCall(PetscHMapIFind(cpHashTable, id, &CPiter, &CPfound));
    PetscCheck(CPfound, comm, PETSC_ERR_SUP, "FACE ID not found in Control Point Hash Table");
    PetscCall(PetscHMapIGet(cpHashTable, id, &faceCPStartRow));

    PetscCall(PetscHMapIFind(wHashTable, id, &Witer, &Wfound));
    PetscCheck(Wfound, comm, PETSC_ERR_SUP, "FACE ID not found in Control Point Weights Hash Table");
    PetscCall(PetscHMapIGet(wHashTable, id, &faceWStartRow));

    for (PetscInt ii = 0; ii < numCntrlPnts; ++ii) {
      if (ii == 4) {
        // Update Control Points - Change the location of the center control point of the faces
        // Note :: Modification geometry requires knowledge of how the geometry is defined.
        cpCoordData[faceCPStartRow + (3 * ii) + 0] = 2.0 * cpCoordData[faceCPStartRow + (3 * ii) + 0];
        cpCoordData[faceCPStartRow + (3 * ii) + 1] = 2.0 * cpCoordData[faceCPStartRow + (3 * ii) + 1];
        cpCoordData[faceCPStartRow + (3 * ii) + 2] = 2.0 * cpCoordData[faceCPStartRow + (3 * ii) + 2];
      } else {
        // Do Nothing
        // Note: Could use section the change location of other face control points.
      }
    }
  }
  PetscCall(DMPlexFreeGeomObject(dm, fobjs));

  // Modify Control Points of Geometry
  PetscCall(PetscObjectSetName((PetscObject)dm, "Modified Surface"));
  // TODO Wrap EG_saveModel() in a Plex viewer to manage file access
  PetscCall(DMPlexModifyGeomModel(dm, comm, cpCoordData, wData, PETSC_FALSE, PETSC_TRUE, "newModel.stp"));
  PetscCall(VecRestoreArrayWrite(cntrlPtCoordsVec, &cpCoordData));
  PetscCall(VecRestoreArrayWrite(cntrlPtWeightsVec, &wData));

  // Inflate Mesh to Geometry
  PetscCall(DMPlexInflateToGeomModel(dm, PETSC_FALSE));
  PetscCall(surfArea(dm));

  // Output .hdf5 file
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));

  for (PetscInt r = 0; r < Nr; ++r) {
    char name[PETSC_MAX_PATH_LEN];

    // Perform refinement on the Mesh attached to the new geometry
    PetscCall(DMSetFromOptions(dm));
    PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "Modified Surface refinement %" PetscInt_FMT, r));
    PetscCall(PetscObjectSetName((PetscObject)dm, name));
    PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
    PetscCall(surfArea(dm));
  }

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: egads

  # Use -dm_view hdf5:mesh_shapeMod_sphere.h5 to view the mesh
  test:
    suffix: sphere_shapeMod
    requires: datafilespath
    temporaries: newModel.stp
    args: -dm_plex_filename ${DATAFILESPATH}/meshes/cad/sphere_example.stp \
          -dm_refine -dm_plex_geom_print_model 1 -dm_plex_geom_shape_opt 1

TEST*/
