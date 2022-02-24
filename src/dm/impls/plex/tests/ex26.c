static char help[] = "Test FEM layout with DM and ExodusII storage\n\n";

/*
  In order to see the vectors which are being tested, use

     -ua_vec_view -s_vec_view
*/

#include <petsc.h>
#include <exodusII.h>

#include <petsc/private/dmpleximpl.h>

int main(int argc, char **argv) {
  DM                dm, dmU, dmA, dmS, dmUA, dmUA2, *dmList;
  Vec               X, U, A, S, UA, UA2;
  IS                isU, isA, isS, isUA;
  PetscSection      section;
  const PetscInt    fieldU = 0;
  const PetscInt    fieldA = 2;
  const PetscInt    fieldS = 1;
  const PetscInt    fieldUA[2] = {0, 2};
  char              ifilename[PETSC_MAX_PATH_LEN], ofilename[PETSC_MAX_PATH_LEN];
  int               exoid = -1;
  IS                csIS;
  const PetscInt   *csID;
  PetscInt         *pStartDepth, *pEndDepth;
  PetscInt          order = 1;
  PetscInt          sdim, d, pStart, pEnd, p, numCS, set;
  PetscMPIInt       rank, size;
  PetscViewer       viewer;
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc, &argv,NULL, help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FEM Layout Options", "ex26");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-i", "Filename to read", "ex26", ifilename, ifilename, sizeof(ifilename), NULL));
  CHKERRQ(PetscOptionsString("-o", "Filename to write", "ex26", ofilename, ofilename, sizeof(ofilename), NULL));
  CHKERRQ(PetscOptionsBoundedInt("-order", "FEM polynomial order", "ex26", order, &order, NULL,1));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscCheckFalse((order > 2) || (order < 1),PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported polynomial order %D not in [1, 2]", order);

  /* Read the mesh from a file in any supported format */
  CHKERRQ(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, NULL, PETSC_TRUE, &dm));
  CHKERRQ(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMGetDimension(dm, &sdim));

  /* Create the exodus result file */
  {
    PetscInt      numstep = 3, step;
    char         *nodalVarName[4];
    char         *zonalVarName[6];
    int          *truthtable;
    PetscInt      numNodalVar, numZonalVar, i;

    /* enable exodus debugging information */
    ex_opts(EX_VERBOSE|EX_DEBUG);
    /* Create the exodus file */
    CHKERRQ(PetscViewerExodusIIOpen(PETSC_COMM_WORLD,ofilename,FILE_MODE_WRITE,&viewer));
    /* The long way would be */
    /*
      CHKERRQ(PetscViewerCreate(PETSC_COMM_WORLD,&viewer));
      CHKERRQ(PetscViewerSetType(viewer,PETSCVIEWEREXODUSII));
      CHKERRQ(PetscViewerFileSetMode(viewer,FILE_MODE_APPEND));
      CHKERRQ(PetscViewerFileSetName(viewer,ofilename));
    */
    /* set the mesh order */
    CHKERRQ(PetscViewerExodusIISetOrder(viewer,order));
    CHKERRQ(PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD));
    /*
      Notice how the exodus file is actually NOT open at this point (exoid is -1)
      Since we are overwritting the file (mode is FILE_MODE_WRITE), we are going to have to
      write the geometry (the DM), which can only be done on a brand new file.
    */

    /* Save the geometry to the file, erasing all previous content */
    CHKERRQ(DMView(dm,viewer));
    CHKERRQ(PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD));
    /*
      Note how the exodus file is now open
    */

    /* "Format" the exodus result file, i.e. allocate space for nodal and zonal variables */
    switch (sdim) {
    case 2:
      numNodalVar = 3;
      nodalVarName[0] = (char *) "U_x";
      nodalVarName[1] = (char *) "U_y";
      nodalVarName[2] = (char *) "Alpha";
      numZonalVar = 3;
      zonalVarName[0] = (char *) "Sigma_11";
      zonalVarName[1] = (char *) "Sigma_22";
      zonalVarName[2] = (char *) "Sigma_12";
      break;
    case 3:
      numNodalVar = 4;
      nodalVarName[0] = (char *) "U_x";
      nodalVarName[1] = (char *) "U_y";
      nodalVarName[2] = (char *) "U_z";
      nodalVarName[3] = (char *) "Alpha";
      numZonalVar = 6;
      zonalVarName[0] = (char *) "Sigma_11";
      zonalVarName[1] = (char *) "Sigma_22";
      zonalVarName[2] = (char *) "Sigma_33";
      zonalVarName[3] = (char *) "Sigma_23";
      zonalVarName[4] = (char *) "Sigma_13";
      zonalVarName[5] = (char *) "Sigma_12";
      break;
    default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No layout for dimension %D", sdim);
    }
    CHKERRQ(PetscViewerExodusIIGetId(viewer,&exoid));
    PetscStackCallStandard(ex_put_variable_param,exoid, EX_ELEM_BLOCK, numZonalVar);
    PetscStackCallStandard(ex_put_variable_names,exoid, EX_ELEM_BLOCK, numZonalVar, zonalVarName);
    PetscStackCallStandard(ex_put_variable_param,exoid, EX_NODAL, numNodalVar);
    PetscStackCallStandard(ex_put_variable_names,exoid, EX_NODAL, numNodalVar, nodalVarName);
    numCS = ex_inquire_int(exoid, EX_INQ_ELEM_BLK);

    /*
      An exodusII truth table specifies which fields are saved at which time step
      It speeds up I/O but reserving space for fieldsin the file ahead of time.
    */
    CHKERRQ(PetscMalloc1(numZonalVar * numCS, &truthtable));
    for (i = 0; i < numZonalVar * numCS; ++i) truthtable[i] = 1;
    PetscStackCallStandard(ex_put_truth_table,exoid, EX_ELEM_BLOCK, numCS, numZonalVar, truthtable);
    CHKERRQ(PetscFree(truthtable));

    /* Writing time step information in the file. Note that this is currently broken in the exodus library for netcdf4 (HDF5-based) files */
    for (step = 0; step < numstep; ++step) {
      PetscReal time = step;
      PetscStackCallStandard(ex_put_time,exoid, step+1, &time);
    }
  }

  /* Create the main section containing all fields */
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section));
  CHKERRQ(PetscSectionSetNumFields(section, 3));
  CHKERRQ(PetscSectionSetFieldName(section, fieldU, "U"));
  CHKERRQ(PetscSectionSetFieldName(section, fieldA, "Alpha"));
  CHKERRQ(PetscSectionSetFieldName(section, fieldS, "Sigma"));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(PetscSectionSetChart(section, pStart, pEnd));
  CHKERRQ(PetscMalloc2(sdim+1, &pStartDepth, sdim+1, &pEndDepth));
  for (d = 0; d <= sdim; ++d) CHKERRQ(DMPlexGetDepthStratum(dm, d, &pStartDepth[d], &pEndDepth[d]));
  /* Vector field U, Scalar field Alpha, Tensor field Sigma */
  CHKERRQ(PetscSectionSetFieldComponents(section, fieldU, sdim));
  CHKERRQ(PetscSectionSetFieldComponents(section, fieldA, 1));
  CHKERRQ(PetscSectionSetFieldComponents(section, fieldS, sdim*(sdim+1)/2));

  /* Going through cell sets then cells, and setting up storage for the sections */
  CHKERRQ(DMGetLabelSize(dm, "Cell Sets", &numCS));
  CHKERRQ(DMGetLabelIdIS(dm, "Cell Sets", &csIS));
  if (csIS) CHKERRQ(ISGetIndices(csIS, &csID));
  for (set = 0; set < numCS; set++) {
    IS                cellIS;
    const PetscInt   *cellID;
    PetscInt          numCells, cell, closureSize, *closureA = NULL;

    CHKERRQ(DMGetStratumSize(dm, "Cell Sets", csID[set], &numCells));
    CHKERRQ(DMGetStratumIS(dm, "Cell Sets", csID[set], &cellIS));
    if (numCells > 0) {
      /* dof layout ordered by increasing height in the DAG: cell, face, edge, vertex */
      PetscInt          dofUP1Tri[]  = {2, 0, 0};
      PetscInt          dofAP1Tri[]  = {1, 0, 0};
      PetscInt          dofUP2Tri[]  = {2, 2, 0};
      PetscInt          dofAP2Tri[]  = {1, 1, 0};
      PetscInt          dofUP1Quad[] = {2, 0, 0};
      PetscInt          dofAP1Quad[] = {1, 0, 0};
      PetscInt          dofUP2Quad[] = {2, 2, 2};
      PetscInt          dofAP2Quad[] = {1, 1, 1};
      PetscInt          dofS2D[]     = {0, 0, 3};
      PetscInt          dofUP1Tet[]  = {3, 0, 0, 0};
      PetscInt          dofAP1Tet[]  = {1, 0, 0, 0};
      PetscInt          dofUP2Tet[]  = {3, 3, 0, 0};
      PetscInt          dofAP2Tet[]  = {1, 1, 0, 0};
      PetscInt          dofUP1Hex[]  = {3, 0, 0, 0};
      PetscInt          dofAP1Hex[]  = {1, 0, 0, 0};
      PetscInt          dofUP2Hex[]  = {3, 3, 3, 3};
      PetscInt          dofAP2Hex[]  = {1, 1, 1, 1};
      PetscInt          dofS3D[]     = {0, 0, 0, 6};
      PetscInt         *dofU, *dofA, *dofS;

      switch (sdim) {
      case 2: dofS = dofS2D;break;
      case 3: dofS = dofS3D;break;
      default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No layout for dimension %D", sdim);
      }

      /* Identify cell type based on closure size only. This works for Tri/Tet/Quad/Hex meshes
         It will not be enough to identify more exotic elements like pyramid or prisms...  */
      CHKERRQ(ISGetIndices(cellIS, &cellID));
      CHKERRQ(DMPlexGetTransitiveClosure(dm, cellID[0], PETSC_TRUE, &closureSize, &closureA));
      switch (closureSize) {
        case 7: /* Tri */
        if (order == 1) {
          dofU = dofUP1Tri;
          dofA = dofAP1Tri;
        } else {
          dofU = dofUP2Tri;
          dofA = dofAP2Tri;
        }
        break;
        case 9: /* Quad */
        if (order == 1) {
          dofU = dofUP1Quad;
          dofA = dofAP1Quad;
        } else {
          dofU = dofUP2Quad;
          dofA = dofAP2Quad;
        }
        break;
        case 15: /* Tet */
        if (order == 1) {
          dofU = dofUP1Tet;
          dofA = dofAP1Tet;
        } else {
          dofU = dofUP2Tet;
          dofA = dofAP2Tet;
        }
        break;
        case 27: /* Hex */
        if (order == 1) {
          dofU = dofUP1Hex;
          dofA = dofAP1Hex;
        } else {
          dofU = dofUP2Hex;
          dofA = dofAP2Hex;
        }
        break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Unknown element with closure size %D", closureSize);
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cellID[0], PETSC_TRUE, &closureSize, &closureA));

      for (cell = 0; cell < numCells; cell++) {
        PetscInt *closure = NULL;

        CHKERRQ(DMPlexGetTransitiveClosure(dm, cellID[cell], PETSC_TRUE, &closureSize, &closure));
        for (p = 0; p < closureSize; ++p) {
          /* Find depth of p */
          for (d = 0; d <= sdim; ++d) {
            if ((closure[2*p] >= pStartDepth[d]) && (closure[2*p] < pEndDepth[d])) {
              CHKERRQ(PetscSectionSetDof(section, closure[2*p], dofU[d]+dofA[d]+dofS[d]));
              CHKERRQ(PetscSectionSetFieldDof(section, closure[2*p], fieldU, dofU[d]));
              CHKERRQ(PetscSectionSetFieldDof(section, closure[2*p], fieldA, dofA[d]));
              CHKERRQ(PetscSectionSetFieldDof(section, closure[2*p], fieldS, dofS[d]));
            }
          }
        }
        CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cellID[cell], PETSC_TRUE, &closureSize, &closure));
      }
      CHKERRQ(ISRestoreIndices(cellIS, &cellID));
      CHKERRQ(ISDestroy(&cellIS));
    }
  }
  if (csIS) CHKERRQ(ISRestoreIndices(csIS, &csID));
  CHKERRQ(ISDestroy(&csIS));
  CHKERRQ(PetscSectionSetUp(section));
  CHKERRQ(DMSetLocalSection(dm, section));
  CHKERRQ(PetscObjectViewFromOptions((PetscObject) section, NULL, "-dm_section_view"));
  CHKERRQ(PetscSectionDestroy(&section));

  {
    DM               pdm;
    PetscSF          migrationSF;
    PetscInt         ovlp = 0;
    PetscPartitioner part;

    CHKERRQ(DMSetUseNatural(dm,PETSC_TRUE));
    CHKERRQ(DMPlexGetPartitioner(dm,&part));
    CHKERRQ(PetscPartitionerSetFromOptions(part));
    CHKERRQ(DMPlexDistribute(dm,ovlp,&migrationSF,&pdm));
    if (pdm) {
      CHKERRQ(DMPlexSetMigrationSF(pdm,migrationSF));
      CHKERRQ(PetscSFDestroy(&migrationSF));
      CHKERRQ(DMDestroy(&dm));
      dm = pdm;
      CHKERRQ(DMViewFromOptions(dm,NULL,"-dm_view"));
    }
  }

  /* Get DM and IS for each field of dm */
  CHKERRQ(DMCreateSubDM(dm, 1, &fieldU, &isU,  &dmU));
  CHKERRQ(DMCreateSubDM(dm, 1, &fieldA, &isA,  &dmA));
  CHKERRQ(DMCreateSubDM(dm, 1, &fieldS, &isS,  &dmS));
  CHKERRQ(DMCreateSubDM(dm, 2, fieldUA, &isUA, &dmUA));

  CHKERRQ(PetscMalloc1(2,&dmList));
  dmList[0] = dmU;
  dmList[1] = dmA;
  /* We temporarily disable dmU->useNatural to test that we can reconstruct the
     NaturaltoGlobal SF from any of the dm in dms
  */
  dmU->useNatural = PETSC_FALSE;
  CHKERRQ(DMCreateSuperDM(dmList,2,NULL,&dmUA2));
  dmU->useNatural = PETSC_TRUE;
  CHKERRQ(PetscFree(dmList));

  CHKERRQ(DMGetGlobalVector(dm,   &X));
  CHKERRQ(DMGetGlobalVector(dmU,  &U));
  CHKERRQ(DMGetGlobalVector(dmA,  &A));
  CHKERRQ(DMGetGlobalVector(dmS,  &S));
  CHKERRQ(DMGetGlobalVector(dmUA, &UA));
  CHKERRQ(DMGetGlobalVector(dmUA2, &UA2));

  CHKERRQ(PetscObjectSetName((PetscObject) U,  "U"));
  CHKERRQ(PetscObjectSetName((PetscObject) A,  "Alpha"));
  CHKERRQ(PetscObjectSetName((PetscObject) S,  "Sigma"));
  CHKERRQ(PetscObjectSetName((PetscObject) UA, "UAlpha"));
  CHKERRQ(PetscObjectSetName((PetscObject) UA2, "UAlpha2"));
  CHKERRQ(VecSet(X, -111.));

  /* Setting u to [x,y,z]  and alpha to x^2+y^2+z^2 by writing in UAlpha then restricting to U and Alpha */
  {
    PetscSection sectionUA;
    Vec          UALoc;
    PetscSection coordSection;
    Vec          coord;
    PetscScalar *cval, *xyz;
    PetscInt     clSize, i, j;

    CHKERRQ(DMGetLocalSection(dmUA, &sectionUA));
    CHKERRQ(DMGetLocalVector(dmUA, &UALoc));
    CHKERRQ(VecGetArray(UALoc, &cval));
    CHKERRQ(DMGetCoordinateSection(dmUA, &coordSection));
    CHKERRQ(DMGetCoordinatesLocal(dmUA, &coord));
    CHKERRQ(DMPlexGetChart(dmUA, &pStart, &pEnd));

    for (p = pStart; p < pEnd; ++p) {
      PetscInt dofUA, offUA;

      CHKERRQ(PetscSectionGetDof(sectionUA, p, &dofUA));
      if (dofUA > 0) {
        xyz=NULL;
        CHKERRQ(PetscSectionGetOffset(sectionUA, p, &offUA));
        CHKERRQ(DMPlexVecGetClosure(dmUA, coordSection, coord, p, &clSize, &xyz));
        cval[offUA+sdim] = 0.;
        for (i = 0; i < sdim; ++i) {
          cval[offUA+i] = 0;
          for (j = 0; j < clSize/sdim; ++j) {
            cval[offUA+i] += xyz[j*sdim+i];
          }
          cval[offUA+i] = cval[offUA+i] * sdim / clSize;
          cval[offUA+sdim] += PetscSqr(cval[offUA+i]);
        }
        CHKERRQ(DMPlexVecRestoreClosure(dmUA, coordSection, coord, p, &clSize, &xyz));
      }
    }
    CHKERRQ(VecRestoreArray(UALoc, &cval));
    CHKERRQ(DMLocalToGlobalBegin(dmUA, UALoc, INSERT_VALUES, UA));
    CHKERRQ(DMLocalToGlobalEnd(dmUA, UALoc, INSERT_VALUES, UA));
    CHKERRQ(DMRestoreLocalVector(dmUA, &UALoc));

    /* Update X */
    CHKERRQ(VecISCopy(X, isUA, SCATTER_FORWARD, UA));
    CHKERRQ(VecViewFromOptions(UA, NULL, "-ua_vec_view"));

    /* Restrict to U and Alpha */
    CHKERRQ(VecISCopy(X, isU, SCATTER_REVERSE, U));
    CHKERRQ(VecISCopy(X, isA, SCATTER_REVERSE, A));

    /* restrict to UA2 */
    CHKERRQ(VecISCopy(X, isUA, SCATTER_REVERSE, UA2));
    CHKERRQ(VecViewFromOptions(UA2, NULL, "-ua2_vec_view"));
  }

  {
    Vec          tmpVec;
    PetscSection coordSection;
    Vec          coord;
    PetscReal    norm;
    PetscReal    time = 1.234;

    /* Writing nodal variables to ExodusII file */
    CHKERRQ(DMSetOutputSequenceNumber(dmU,0,time));
    CHKERRQ(DMSetOutputSequenceNumber(dmA,0,time));

    CHKERRQ(VecView(U, viewer));
    CHKERRQ(VecView(A, viewer));

    /* Saving U and Alpha in one shot.
       For this, we need to cheat and change the Vec's name
       Note that in the end we write variables one component at a time,
       so that there is no real values in doing this
    */

    CHKERRQ(DMSetOutputSequenceNumber(dmUA,1,time));
    CHKERRQ(DMGetGlobalVector(dmUA, &tmpVec));
    CHKERRQ(VecCopy(UA, tmpVec));
    CHKERRQ(PetscObjectSetName((PetscObject) tmpVec, "U"));
    CHKERRQ(VecView(tmpVec, viewer));
    /* Reading nodal variables in Exodus file */
    CHKERRQ(VecSet(tmpVec, -1000.0));
    CHKERRQ(VecLoad(tmpVec, viewer));
    CHKERRQ(VecAXPY(UA, -1.0, tmpVec));
    CHKERRQ(VecNorm(UA, NORM_INFINITY, &norm));
    PetscCheckFalse(norm > PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "UAlpha ||Vin - Vout|| = %g", (double) norm);
    CHKERRQ(DMRestoreGlobalVector(dmUA, &tmpVec));

    /* same thing with the UA2 Vec obtained from the superDM */
    CHKERRQ(DMGetGlobalVector(dmUA2, &tmpVec));
    CHKERRQ(VecCopy(UA2, tmpVec));
    CHKERRQ(PetscObjectSetName((PetscObject) tmpVec, "U"));
    CHKERRQ(DMSetOutputSequenceNumber(dmUA2,2,time));
    CHKERRQ(VecView(tmpVec, viewer));
    /* Reading nodal variables in Exodus file */
    CHKERRQ(VecSet(tmpVec, -1000.0));
    CHKERRQ(VecLoad(tmpVec,viewer));
    CHKERRQ(VecAXPY(UA2, -1.0, tmpVec));
    CHKERRQ(VecNorm(UA2, NORM_INFINITY, &norm));
    PetscCheckFalse(norm > PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "UAlpha2 ||Vin - Vout|| = %g", (double) norm);
    CHKERRQ(DMRestoreGlobalVector(dmUA2, &tmpVec));

    /* Building and saving Sigma
       We set sigma_0 = rank (to see partitioning)
              sigma_1 = cell set ID
              sigma_2 = x_coordinate of the cell center of mass
    */
    CHKERRQ(DMGetCoordinateSection(dmS, &coordSection));
    CHKERRQ(DMGetCoordinatesLocal(dmS, &coord));
    CHKERRQ(DMGetLabelIdIS(dmS, "Cell Sets", &csIS));
    CHKERRQ(DMGetLabelSize(dmS, "Cell Sets", &numCS));
    CHKERRQ(ISGetIndices(csIS, &csID));
    for (set = 0; set < numCS; ++set) {
      /* We know that all cells in a cell set have the same type, so we can dimension cval and xyz once for each cell set */
      IS              cellIS;
      const PetscInt *cellID;
      PetscInt        numCells, cell;
      PetscScalar    *cval = NULL, *xyz  = NULL;
      PetscInt        clSize, cdimCoord, c;

      CHKERRQ(DMGetStratumIS(dmS, "Cell Sets", csID[set], &cellIS));
      CHKERRQ(ISGetIndices(cellIS, &cellID));
      CHKERRQ(ISGetSize(cellIS, &numCells));
      for (cell = 0; cell < numCells; cell++) {
        CHKERRQ(DMPlexVecGetClosure(dmS, NULL, S, cellID[cell], &clSize, &cval));
        CHKERRQ(DMPlexVecGetClosure(dmS, coordSection, coord, cellID[cell], &cdimCoord, &xyz));
        cval[0] = rank;
        cval[1] = csID[set];
        cval[2] = 0.;
        for (c = 0; c < cdimCoord/sdim; c++) cval[2] += xyz[c*sdim];
        cval[2] = cval[2] * sdim / cdimCoord;
        CHKERRQ(DMPlexVecSetClosure(dmS, NULL, S, cellID[cell], cval, INSERT_ALL_VALUES));
      }
      CHKERRQ(DMPlexVecRestoreClosure(dmS, NULL, S, cellID[0], &clSize, &cval));
      CHKERRQ(DMPlexVecRestoreClosure(dmS, coordSection, coord, cellID[0], NULL, &xyz));
      CHKERRQ(ISRestoreIndices(cellIS, &cellID));
      CHKERRQ(ISDestroy(&cellIS));
    }
    CHKERRQ(ISRestoreIndices(csIS, &csID));
    CHKERRQ(ISDestroy(&csIS));
    CHKERRQ(VecViewFromOptions(S, NULL, "-s_vec_view"));

    /* Writing zonal variables in Exodus file */
    CHKERRQ(DMSetOutputSequenceNumber(dmS,0,time));
    CHKERRQ(VecView(S,viewer));

    /* Reading zonal variables in Exodus file */
    CHKERRQ(DMGetGlobalVector(dmS, &tmpVec));
    CHKERRQ(VecSet(tmpVec, -1000.0));
    CHKERRQ(PetscObjectSetName((PetscObject) tmpVec, "Sigma"));
    CHKERRQ(VecLoad(tmpVec,viewer));
    CHKERRQ(VecAXPY(S, -1.0, tmpVec));
    CHKERRQ(VecNorm(S, NORM_INFINITY, &norm));
    PetscCheckFalse(norm > PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Sigma ||Vin - Vout|| = %g", (double) norm);
    CHKERRQ(DMRestoreGlobalVector(dmS, &tmpVec));
  }
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(DMRestoreGlobalVector(dmUA2, &UA2));
  CHKERRQ(DMRestoreGlobalVector(dmUA, &UA));
  CHKERRQ(DMRestoreGlobalVector(dmS,  &S));
  CHKERRQ(DMRestoreGlobalVector(dmA,  &A));
  CHKERRQ(DMRestoreGlobalVector(dmU,  &U));
  CHKERRQ(DMRestoreGlobalVector(dm,   &X));
  CHKERRQ(DMDestroy(&dmU)); CHKERRQ(ISDestroy(&isU));
  CHKERRQ(DMDestroy(&dmA)); CHKERRQ(ISDestroy(&isA));
  CHKERRQ(DMDestroy(&dmS)); CHKERRQ(ISDestroy(&isS));
  CHKERRQ(DMDestroy(&dmUA));CHKERRQ(ISDestroy(&isUA));
  CHKERRQ(DMDestroy(&dmUA2));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFree2(pStartDepth, pEndDepth));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: exodusii pnetcdf !complex
  # 2D seq
  test:
    suffix: 0
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
    #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
  test:
    suffix: 1
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
  test:
    suffix: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
    #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
  test:
    suffix: 3
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
  test:
    suffix: 4
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
  test:
    suffix: 5
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2

  # 2D par
  test:
    suffix: 6
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
    #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
  test:
    suffix: 7
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
  test:
    suffix: 8
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
    #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: invalid dimension ID or name
  test:
    suffix: 9
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -o FourSquareT-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
  test:
    suffix: 10
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -o FourSquareQ-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
  test:
    # Something is now broken with parallel read/write for wahtever shape H is
    TODO: broken
    suffix: 11
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -o FourSquareH-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2

  #3d seq
  test:
    suffix: 12
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
  test:
    suffix: 13
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
    #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
  test:
    suffix: 14
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
  test:
    suffix: 15
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
    #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
  #3d par
  test:
    suffix: 16
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
  test:
    suffix: 17
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 1
    #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints
  test:
    suffix: 18
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -o FourBrickHex-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
  test:
    suffix: 19
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -o FourBrickTet-large_out.exo -dm_view -dm_section_view -petscpartitioner_type simple -order 2
    #TODO: bug in call to NetCDF failed to complete invalid type definition in file id 65536 NetCDF: One or more variable sizes violate format constraints

TEST*/
