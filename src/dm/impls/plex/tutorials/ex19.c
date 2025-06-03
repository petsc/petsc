static const char help[] = "Test of PETSc CAD Shape Optimization & Mesh Modification Technology";

#include <petscdmplexegads.h>
#include <petsc/private/hashmapi.h>

typedef struct {
  char     filename[PETSC_MAX_PATH_LEN];
  PetscInt saMaxIter; // Maximum number of iterations of shape optimization loop
} AppCtx;

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

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->filename[0] = '\0';
  options->saMaxIter   = 200;

  PetscOptionsBegin(comm, "", "DMPlex w/CAD Options", "DMPlex w/CAD");
  PetscCall(PetscOptionsString("-filename", "The CAD/Geometry file", __FILE__, options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL));
  PetscCall(PetscOptionsBoundedInt("-sa_max_iter", "The maximum number of iterates for the shape optimization loop", __FILE__, options->saMaxIter, &options->saMaxIter, NULL, 0));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  /* EGADS/EGADSlite variables */
  PetscGeom model, *bodies, *fobjs;
  int       Nb, Nf, id;
  /* PETSc variables */
  DM          dmNozzle = NULL;
  MPI_Comm    comm;
  AppCtx      ctx;
  PetscScalar equivR     = 0.0;
  const char  baseName[] = "Nozzle_Mesh";

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &ctx));

  PetscCall(DMPlexCreateFromFile(comm, ctx.filename, "EGADS", PETSC_TRUE, &dmNozzle));
  PetscCall(PetscObjectSetName((PetscObject)dmNozzle, baseName));
  //PetscCall(DMCreate(PETSC_COMM_WORLD, &dmNozzle));
  //PetscCall(DMSetType(dmNozzle, DMPLEX));
  //PetscCall(DMPlexDistributeSetDefault(dmNozzle, PETSC_FALSE));
  //PetscCall(DMSetFromOptions(dmNozzle));

  // Get Common Viewer to store all Mesh Results
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         flg;
  PetscInt          num = 0;
  PetscViewerType   viewType;
  PetscViewerFormat viewFormat;
  PetscCall(PetscOptionsCreateViewer(PETSC_COMM_WORLD, NULL, NULL, "-dm_view_test", &viewer, &format, &flg));
  //PetscCall(PetscViewerPushFormat(viewer, format));
  if (flg) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  flg = TRUE \n"));
    PetscCall(PetscViewerGetType(viewer, &viewType));
    PetscCall(PetscViewerPushFormat(viewer, PETSC_VIEWER_HDF5_PETSC)); // PetscOptionsGetViewer returns &format as PETSC_VIEWER_DEFAULT need PETSC_VIEWER_HDF5_PETSC to save multiple DMPlexes in a single .h5 file.
    PetscCall(PetscViewerGetFormat(viewer, &viewFormat));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  viewer type = %s \n", viewType));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  viewer format = %d \n", viewFormat));
  }

  // Refines Surface Mesh per option -dm_refine
  PetscCall(DMSetFromOptions(dmNozzle));
  PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view"));
  //PetscCall(DMGetOutputSequenceNumber(dmNozzle, &num, NULL));
  //PetscCall(DMSetOutputSequenceNumber(dmNozzle, num, -1));
  num += 1;
  PetscCall(DMView(dmNozzle, viewer));
  PetscCall(surfArea(dmNozzle));

  for (PetscInt saloop = 0, seqNum = 0; saloop < ctx.saMaxIter; ++saloop) {
    PetscContainer modelObj;
    //PetscContainer gradSACPObj, gradSAWObj;
    PetscScalar *cpCoordData, *wData, *gradSACP, *gradSAW;
    PetscInt     cpCoordDataLength, wDataLength, maxNumEquiv;
    PetscHMapI   cpHashTable, wHashTable;
    Mat          cpEquiv;
    char         stpName[PETSC_MAX_PATH_LEN];
    char         meshName[PETSC_MAX_PATH_LEN];

    PetscCall(PetscSNPrintf(meshName, PETSC_MAX_PATH_LEN, "%s_Step_%" PetscInt_FMT, baseName, saloop));
    PetscCall(PetscObjectSetName((PetscObject)dmNozzle, meshName));
    // Save Step File of Updated Geometry at designated iterations
    if (saloop == 1 || saloop == 2 || saloop == 5 || saloop == 20 || saloop == 50 || saloop == 100 || saloop == 150 || saloop == 200 || saloop == 300 || saloop == 400 || saloop == 500) {
      PetscCall(PetscSNPrintf(stpName, sizeof(stpName), "newGeom_clean_%d.stp", saloop));
    }

    if (saloop == 0) PetscCall(DMSetFromOptions(dmNozzle));

    // Expose Geometry Definition Data and Calculate Surface Gradients
    PetscCall(DMPlexGeomDataAndGrads(dmNozzle, PETSC_FALSE));

    PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "EGADS Model", (PetscObject *)&modelObj));
    if (!modelObj) PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "EGADSlite Model", (PetscObject *)&modelObj));

    // Get attached EGADS model (pointer)
    PetscCall(PetscContainerGetPointer(modelObj, (void **)&model));

    // Look to see if DM has Container for Geometry Control Point Data
    //PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "Control Point Hash Table", (PetscObject *)&cpHashTableObj));
    //PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "Control Point Coordinates", (PetscObject *)&cpCoordDataObj));
    //PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "Control Point Coordinate Data Length", (PetscObject *)&cpCoordDataLengthObj));
    //PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "Control Point Weights Hash Table", (PetscObject *)&wHashTableObj));
    //PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "Control Point Weight Data", (PetscObject *)&wDataObj));
    //PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "Control Point Weight Data Length", (PetscObject *)&wDataLengthObj));
    ////PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "Surface Area Control Point Gradient", (PetscObject *)&gradSACPObj));
    ////PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "Surface Area Weights Gradient", (PetscObject *)&gradSAWObj));
    //PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "Control Point Equivalency Matrix", (PetscObject *)&cpEquivObj));
    //PetscCall(PetscObjectQuery((PetscObject)dmNozzle, "Maximum Number Control Point Equivalency", (PetscObject *)&maxNumRelateObj));

    // Get attached EGADS model Control Point and Weights Hash Tables and Data Arrays (pointer)
    //PetscCall(PetscContainerGetPointer(cpHashTableObj, (void **)&cpHashTable));
    //PetscCall(PetscContainerGetPointer(cpCoordDataObj, (void **)&cpCoordData));
    //PetscCall(PetscContainerGetPointer(cpCoordDataLengthObj, (void **)&cpCoordDataLengthPtr));
    //PetscCall(PetscContainerGetPointer(wHashTableObj, (void **)&wHashTable));
    //PetscCall(PetscContainerGetPointer(wDataObj, (void **)&wData));
    //PetscCall(PetscContainerGetPointer(wDataLengthObj, (void **)&wDataLengthPtr));
    ////PetscCall(PetscContainerGetPointer(gradSACPObj, (void **)&gradSACP));
    ////PetscCall(PetscContainerGetPointer(gradSAWObj, (void **)&gradSAW));
    //PetscCall(PetscContainerGetPointer(cpEquivObj, (void **)&cpEquiv));
    //PetscCall(PetscContainerGetPointer(maxNumRelateObj, (void **)&maxNumRelatePtr));

    // Trying out new Function
    PetscInt     cpArraySize, wArraySize;
    PetscHMapI   cpSurfGradHT;
    Mat          cpSurfGrad;
    PetscScalar *gradVolCP, *gradVolW;

    PetscCall(DMPlexGetGeomCntrlPntAndWeightData(dmNozzle, &cpHashTable, &cpCoordDataLength, &cpCoordData, &maxNumEquiv, &cpEquiv, &wHashTable, &wDataLength, &wData));
    PetscCall(DMPlexGetGeomGradData(dmNozzle, &cpSurfGradHT, &cpSurfGrad, &cpArraySize, &gradSACP, &gradVolCP, &wArraySize, &gradSAW, &gradVolW));

    // Get the number of bodies and body objects in the model
    PetscCall(DMPlexGetGeomModelBodies(dmNozzle, &bodies, &Nb));

    // Get all FACES of the Body
    PetscGeom body = bodies[0];
    PetscCall(DMPlexGetGeomModelBodyFaces(dmNozzle, body, &fobjs, &Nf));

    // Print out Surface Area and Volume using EGADS' Function
    PetscScalar  volume, surfaceArea;
    PetscInt     COGsize, IMCOGsize;
    PetscScalar *centerOfGravity, *inertiaMatrixCOG;
    PetscCall(DMPlexGetGeomBodyMassProperties(dmNozzle, body, &volume, &surfaceArea, &centerOfGravity, &COGsize, &inertiaMatrixCOG, &IMCOGsize));

    // Calculate Radius of Sphere for Equivalent Volume
    if (saloop == 0) equivR = PetscPowReal(volume * (3. / 4.) / PETSC_PI, 1. / 3.);

    // Get Size of Control Point Equivalency Matrix
    PetscInt numRows, numCols;
    PetscCall(MatGetSize(cpEquiv, &numRows, &numCols));

    // Get max number of relations
    PetscInt maxRelateNew = 0;
    for (PetscInt ii = 0; ii < numRows; ++ii) {
      PetscInt maxRelateNewTemp = 0;
      for (PetscInt jj = 0; jj < numCols; ++jj) {
        PetscScalar matValue;
        PetscCall(MatGetValue(cpEquiv, ii, jj, &matValue));

        if (matValue > 0.0) maxRelateNewTemp += 1;
      }
      if (maxRelateNewTemp > maxRelateNew) maxRelateNew = maxRelateNewTemp;
    }

    // Update Control Point and Weight definitions for each surface
    for (PetscInt jj = 0; jj < Nf; ++jj) {
      PetscGeom face         = fobjs[jj];
      PetscInt  numCntrlPnts = 0;
      PetscCall(DMPlexGetGeomID(dmNozzle, body, face, &id));
      PetscCall(DMPlexGetGeomFaceNumOfControlPoints(dmNozzle, face, &numCntrlPnts));

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
        PetscReal xold, yold, zold, rold, phi, theta;

        // Update Control Points - Original Code
        xold  = cpCoordData[faceCPStartRow + (3 * ii) + 0];
        yold  = cpCoordData[faceCPStartRow + (3 * ii) + 1];
        zold  = cpCoordData[faceCPStartRow + (3 * ii) + 2];
        rold  = PetscSqrtReal(xold * xold + yold * yold + zold * zold);
        phi   = PetscAtan2Real(yold, xold);
        theta = PetscAtan2Real(PetscSqrtReal(xold * xold + yold * yold), zold);

        // Account for Different Weights for Control Points on Degenerate Edges (multiple control points have same location and different weights
        // only use the largest weight
        PetscScalar maxWeight = 0.0;
        //PetscInt    wCntr = 0;
        for (PetscInt kk = faceWStartRow; kk < faceWStartRow + numCntrlPnts; ++kk) {
          PetscScalar matValue;
          PetscCall(MatGetValue(cpEquiv, faceWStartRow + ii, kk, &matValue));

          PetscScalar weight = 0.0;
          if (matValue > 0.0) {
            weight = wData[kk];

            if (weight > maxWeight) maxWeight = weight;
            //wCntr += 1;
          }
        }

        //Reduce to Constant R = 0.0254
        PetscScalar deltaR;
        PetscScalar localTargetR;
        localTargetR = equivR / maxWeight;
        deltaR       = rold - localTargetR;

        cpCoordData[faceCPStartRow + (3 * ii) + 0] += -0.05 * deltaR * PetscCosReal(phi) * PetscSinReal(theta);
        cpCoordData[faceCPStartRow + (3 * ii) + 1] += -0.05 * deltaR * PetscSinReal(phi) * PetscSinReal(theta);
        cpCoordData[faceCPStartRow + (3 * ii) + 2] += -0.05 * deltaR * PetscCosReal(theta);
      }
    }
    PetscCall(DMPlexFreeGeomObject(dmNozzle, fobjs));
    PetscBool writeFile = PETSC_FALSE;

    // Modify Control Points of Geometry
    PetscCall(DMPlexModifyGeomModel(dmNozzle, comm, cpCoordData, wData, PETSC_FALSE, writeFile, stpName));
    writeFile = PETSC_FALSE;

    // Get attached EGADS model (pointer)
    PetscGeom newmodel;
    PetscCall(PetscContainerGetPointer(modelObj, (void **)&newmodel));

    // Get the number of bodies and body objects in the model
    PetscCall(DMPlexGetGeomModelBodies(dmNozzle, &bodies, &Nb));

    // Get all FACES of the Body
    PetscGeom newbody = bodies[0];
    PetscCall(DMPlexGetGeomModelBodyFaces(dmNozzle, newbody, &fobjs, &Nf));

    // Save Step File of Updated Geometry at designated iterations
    if (saloop == 1 || saloop == 2 || saloop == 5 || saloop == 20 || saloop == 50 || saloop == 100 || saloop == 150 || saloop == 200 || saloop == 300 || saloop == 400 || saloop == 500) writeFile = PETSC_TRUE;

    // Modify Geometry and Inflate Mesh to New Geoemetry
    PetscCall(DMPlexModifyGeomModel(dmNozzle, comm, cpCoordData, wData, PETSC_FALSE, writeFile, stpName));
    PetscCall(DMPlexInflateToGeomModel(dmNozzle, PETSC_TRUE));

    // Periodically Refine and Write Mesh to hdf5 file
    if (saloop == 0) {
      PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view7"));
      //PetscCall(DMGetOutputSequenceNumber(dmNozzle, &num, NULL));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  num = %d \n", num));
      //PetscCall(DMGetOutputSequenceNumber(dmNozzle, NULL, &num));
      //PetscCall(DMSetOutputSequenceNumber(dmNozzle, num, saloop));
      num += 1;
      PetscCall(PetscObjectSetName((PetscObject)dmNozzle, "nozzle_meshes_1"));
      //PetscCall(PetscViewerFileSetMode(viewer, FILE_MODE_APPEND));
      PetscCall(DMView(dmNozzle, viewer));
    }
    if (saloop == 1 || saloop == 5 || saloop == 20 || saloop == 50 || saloop == 100 || saloop == 150 || saloop == 200 || saloop == 300 || saloop == 400 || saloop == 500) {
      PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view"));
      PetscCall(DMSetOutputSequenceNumber(dmNozzle, seqNum++, saloop));
      PetscCall(DMView(dmNozzle, viewer));
      if (saloop == 200 || saloop == 500) {
        PetscCall(DMSetFromOptions(dmNozzle));
        PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view"));
        PetscCall(DMSetOutputSequenceNumber(dmNozzle, seqNum++, saloop));
        PetscCall(DMView(dmNozzle, viewer));

        PetscCall(DMSetFromOptions(dmNozzle));
        PetscCall(DMViewFromOptions(dmNozzle, NULL, "-dm_view"));
        PetscCall(DMSetOutputSequenceNumber(dmNozzle, seqNum++, saloop));
        PetscCall(DMView(dmNozzle, viewer));
      }
    }
    PetscCall(DMPlexRestoreGeomBodyMassProperties(dmNozzle, body, &volume, &surfaceArea, &centerOfGravity, &COGsize, &inertiaMatrixCOG, &IMCOGsize));
    PetscCall(DMPlexRestoreGeomCntrlPntAndWeightData(dmNozzle, &cpHashTable, &cpCoordDataLength, &cpCoordData, &maxNumEquiv, &cpEquiv, &wHashTable, &wDataLength, &wData));
    PetscCall(DMPlexRestoreGeomGradData(dmNozzle, &cpSurfGradHT, &cpSurfGrad, &cpArraySize, &gradSACP, &gradVolCP, &wArraySize, &gradSAW, &gradVolW));
    PetscCall(DMPlexFreeGeomObject(dmNozzle, fobjs));
  }
  PetscCall(DMDestroy(&dmNozzle));
  PetscCall(PetscFinalize());
}

/*TEST
  build:
    requires: egads

  # To view mesh use -dm_plex_view_hdf5_storage_version 3.1.0 -dm_view hdf5:mesh_minSA_abstract.h5:hdf5_petsc:append
  test:
    suffix: minSA
    requires: datafilespath
    temporaries: newGeom_clean_1.stp newGeom_clean_2.stp newGeom_clean_5.stp
    args: -filename ${DATAFILESPATH}/meshes/cad/sphere_example.stp \
          -dm_refine 1 -dm_plex_geom_print_model 1 -dm_plex_geom_shape_opt 1 \
          -sa_max_iter 2

TEST*/
