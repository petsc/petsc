static char help[] = "Test FEM layout with DM and ExodusII storage\n\n";

/*
  In order to see the vectors which are being tested, use

     -ua_vec_view -s_vec_view
*/

#include <petsc.h>
#include <exodusII.h>

#include <petsc/private/dmpleximpl.h>

int main(int argc, char **argv) {
  DM                dm, pdm, dmU, dmA, dmS, dmUA, dmUA2, *dmList;
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv,NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FEM Layout Options", "ex26");
  PetscCall(PetscOptionsString("-i", "Filename to read", "ex26", ifilename, ifilename, sizeof(ifilename), NULL));
  PetscCall(PetscOptionsString("-o", "Filename to write", "ex26", ofilename, ofilename, sizeof(ofilename), NULL));
  PetscCall(PetscOptionsBoundedInt("-order", "FEM polynomial order", "ex26", order, &order, NULL,1));
  PetscOptionsEnd();
  PetscCheck((order >= 1) && (order <= 2),PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported polynomial order %" PetscInt_FMT " not in [1, 2]", order);

  /* Read the mesh from a file in any supported format */
  PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, NULL, PETSC_TRUE, &dm));
  PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMGetDimension(dm, &sdim));

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
    PetscCall(PetscViewerExodusIIOpen(PETSC_COMM_WORLD,ofilename,FILE_MODE_WRITE,&viewer));
    /* The long way would be */
    /*
      PetscCall(PetscViewerCreate(PETSC_COMM_WORLD,&viewer));
      PetscCall(PetscViewerSetType(viewer,PETSCVIEWEREXODUSII));
      PetscCall(PetscViewerFileSetMode(viewer,FILE_MODE_APPEND));
      PetscCall(PetscViewerFileSetName(viewer,ofilename));
    */
    /* set the mesh order */
    PetscCall(PetscViewerExodusIISetOrder(viewer,order));
    PetscCall(PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD));
    /*
      Notice how the exodus file is actually NOT open at this point (exoid is -1)
      Since we are overwritting the file (mode is FILE_MODE_WRITE), we are going to have to
      write the geometry (the DM), which can only be done on a brand new file.
    */

    /* Save the geometry to the file, erasing all previous content */
    PetscCall(DMView(dm,viewer));
    PetscCall(PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD));
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
    default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No layout for dimension %" PetscInt_FMT, sdim);
    }
    PetscCall(PetscViewerExodusIIGetId(viewer,&exoid));
    PetscCallExternal(ex_put_variable_param,exoid, EX_ELEM_BLOCK, numZonalVar);
    PetscCallExternal(ex_put_variable_names,exoid, EX_ELEM_BLOCK, numZonalVar, zonalVarName);
    PetscCallExternal(ex_put_variable_param,exoid, EX_NODAL, numNodalVar);
    PetscCallExternal(ex_put_variable_names,exoid, EX_NODAL, numNodalVar, nodalVarName);
    numCS = ex_inquire_int(exoid, EX_INQ_ELEM_BLK);

    /*
      An exodusII truth table specifies which fields are saved at which time step
      It speeds up I/O but reserving space for fieldsin the file ahead of time.
    */
    PetscCall(PetscMalloc1(numZonalVar * numCS, &truthtable));
    for (i = 0; i < numZonalVar * numCS; ++i) truthtable[i] = 1;
    PetscCallExternal(ex_put_truth_table,exoid, EX_ELEM_BLOCK, numCS, numZonalVar, truthtable);
    PetscCall(PetscFree(truthtable));

    /* Writing time step information in the file. Note that this is currently broken in the exodus library for netcdf4 (HDF5-based) files */
    for (step = 0; step < numstep; ++step) {
      PetscReal time = step;
      PetscCallExternal(ex_put_time,exoid, step+1, &time);
    }
  }

  /* Create the main section containing all fields */
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section));
  PetscCall(PetscSectionSetNumFields(section, 3));
  PetscCall(PetscSectionSetFieldName(section, fieldU, "U"));
  PetscCall(PetscSectionSetFieldName(section, fieldA, "Alpha"));
  PetscCall(PetscSectionSetFieldName(section, fieldS, "Sigma"));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionSetChart(section, pStart, pEnd));
  PetscCall(PetscMalloc2(sdim+1, &pStartDepth, sdim+1, &pEndDepth));
  for (d = 0; d <= sdim; ++d) {PetscCall(DMPlexGetDepthStratum(dm, d, &pStartDepth[d], &pEndDepth[d]));}
  /* Vector field U, Scalar field Alpha, Tensor field Sigma */
  PetscCall(PetscSectionSetFieldComponents(section, fieldU, sdim));
  PetscCall(PetscSectionSetFieldComponents(section, fieldA, 1));
  PetscCall(PetscSectionSetFieldComponents(section, fieldS, sdim*(sdim+1)/2));

  /* Going through cell sets then cells, and setting up storage for the sections */
  PetscCall(DMGetLabelSize(dm, "Cell Sets", &numCS));
  PetscCall(DMGetLabelIdIS(dm, "Cell Sets", &csIS));
  if (csIS) {PetscCall(ISGetIndices(csIS, &csID));}
  for (set = 0; set < numCS; set++) {
    IS                cellIS;
    const PetscInt   *cellID;
    PetscInt          numCells, cell, closureSize, *closureA = NULL;

    PetscCall(DMGetStratumSize(dm, "Cell Sets", csID[set], &numCells));
    PetscCall(DMGetStratumIS(dm, "Cell Sets", csID[set], &cellIS));
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
      default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No layout for dimension %" PetscInt_FMT, sdim);
      }

      /* Identify cell type based on closure size only. This works for Tri/Tet/Quad/Hex meshes
         It will not be enough to identify more exotic elements like pyramid or prisms...  */
      PetscCall(ISGetIndices(cellIS, &cellID));
      PetscCall(DMPlexGetTransitiveClosure(dm, cellID[0], PETSC_TRUE, &closureSize, &closureA));
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
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Unknown element with closure size %" PetscInt_FMT, closureSize);
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, cellID[0], PETSC_TRUE, &closureSize, &closureA));

      for (cell = 0; cell < numCells; cell++) {
        PetscInt *closure = NULL;

        PetscCall(DMPlexGetTransitiveClosure(dm, cellID[cell], PETSC_TRUE, &closureSize, &closure));
        for (p = 0; p < closureSize; ++p) {
          /* Find depth of p */
          for (d = 0; d <= sdim; ++d) {
            if ((closure[2*p] >= pStartDepth[d]) && (closure[2*p] < pEndDepth[d])) {
              PetscCall(PetscSectionSetDof(section, closure[2*p], dofU[d]+dofA[d]+dofS[d]));
              PetscCall(PetscSectionSetFieldDof(section, closure[2*p], fieldU, dofU[d]));
              PetscCall(PetscSectionSetFieldDof(section, closure[2*p], fieldA, dofA[d]));
              PetscCall(PetscSectionSetFieldDof(section, closure[2*p], fieldS, dofS[d]));
            }
          }
        }
        PetscCall(DMPlexRestoreTransitiveClosure(dm, cellID[cell], PETSC_TRUE, &closureSize, &closure));
      }
      PetscCall(ISRestoreIndices(cellIS, &cellID));
      PetscCall(ISDestroy(&cellIS));
    }
  }
  if (csIS) {PetscCall(ISRestoreIndices(csIS, &csID));}
  PetscCall(ISDestroy(&csIS));
  PetscCall(PetscSectionSetUp(section));
  PetscCall(DMSetLocalSection(dm, section));
  PetscCall(PetscObjectViewFromOptions((PetscObject) section, NULL, "-dm_section_view"));
  PetscCall(PetscSectionDestroy(&section));

  {
    PetscSF          migrationSF;
    PetscInt         ovlp = 0;
    PetscPartitioner part;

    PetscCall(DMSetUseNatural(dm,PETSC_TRUE));
    PetscCall(DMPlexGetPartitioner(dm,&part));
    PetscCall(PetscPartitionerSetFromOptions(part));
    PetscCall(DMPlexDistribute(dm,ovlp,&migrationSF,&pdm));
    if (!pdm) pdm = dm;
    /* Set the migrationSF is mandatory since useNatural = PETSC_TRUE */
    if (migrationSF) {
      PetscCall(DMPlexSetMigrationSF(pdm, migrationSF));
      PetscCall(PetscSFDestroy(&migrationSF));
    }
    PetscCall(DMViewFromOptions(pdm, NULL, "-dm_view"));
  }

  /* Get DM and IS for each field of dm */
  PetscCall(DMCreateSubDM(pdm, 1, &fieldU, &isU,  &dmU));
  PetscCall(DMCreateSubDM(pdm, 1, &fieldA, &isA,  &dmA));
  PetscCall(DMCreateSubDM(pdm, 1, &fieldS, &isS,  &dmS));
  PetscCall(DMCreateSubDM(pdm, 2, fieldUA, &isUA, &dmUA));

  PetscCall(PetscMalloc1(2,&dmList));
  dmList[0] = dmU;
  dmList[1] = dmA;

  PetscCall(DMCreateSuperDM(dmList,2,NULL,&dmUA2));
  PetscCall(PetscFree(dmList));

  PetscCall(DMGetGlobalVector(pdm,  &X));
  PetscCall(DMGetGlobalVector(dmU,  &U));
  PetscCall(DMGetGlobalVector(dmA,  &A));
  PetscCall(DMGetGlobalVector(dmS,  &S));
  PetscCall(DMGetGlobalVector(dmUA, &UA));
  PetscCall(DMGetGlobalVector(dmUA2, &UA2));

  PetscCall(PetscObjectSetName((PetscObject) U,  "U"));
  PetscCall(PetscObjectSetName((PetscObject) A,  "Alpha"));
  PetscCall(PetscObjectSetName((PetscObject) S,  "Sigma"));
  PetscCall(PetscObjectSetName((PetscObject) UA, "UAlpha"));
  PetscCall(PetscObjectSetName((PetscObject) UA2, "UAlpha2"));
  PetscCall(VecSet(X, -111.));

  /* Setting u to [x,y,z]  and alpha to x^2+y^2+z^2 by writing in UAlpha then restricting to U and Alpha */
  {
    PetscSection sectionUA;
    Vec          UALoc;
    PetscSection coordSection;
    Vec          coord;
    PetscScalar *cval, *xyz;
    PetscInt     clSize, i, j;

    PetscCall(DMGetLocalSection(dmUA, &sectionUA));
    PetscCall(DMGetLocalVector(dmUA, &UALoc));
    PetscCall(VecGetArray(UALoc, &cval));
    PetscCall(DMGetCoordinateSection(dmUA, &coordSection));
    PetscCall(DMGetCoordinatesLocal(dmUA, &coord));
    PetscCall(DMPlexGetChart(dmUA, &pStart, &pEnd));

    for (p = pStart; p < pEnd; ++p) {
      PetscInt dofUA, offUA;

      PetscCall(PetscSectionGetDof(sectionUA, p, &dofUA));
      if (dofUA > 0) {
        xyz=NULL;
        PetscCall(PetscSectionGetOffset(sectionUA, p, &offUA));
        PetscCall(DMPlexVecGetClosure(dmUA, coordSection, coord, p, &clSize, &xyz));
        cval[offUA+sdim] = 0.;
        for (i = 0; i < sdim; ++i) {
          cval[offUA+i] = 0;
          for (j = 0; j < clSize/sdim; ++j) {
            cval[offUA+i] += xyz[j*sdim+i];
          }
          cval[offUA+i] = cval[offUA+i] * sdim / clSize;
          cval[offUA+sdim] += PetscSqr(cval[offUA+i]);
        }
        PetscCall(DMPlexVecRestoreClosure(dmUA, coordSection, coord, p, &clSize, &xyz));
      }
    }
    PetscCall(VecRestoreArray(UALoc, &cval));
    PetscCall(DMLocalToGlobalBegin(dmUA, UALoc, INSERT_VALUES, UA));
    PetscCall(DMLocalToGlobalEnd(dmUA, UALoc, INSERT_VALUES, UA));
    PetscCall(DMRestoreLocalVector(dmUA, &UALoc));

    /* Update X */
    PetscCall(VecISCopy(X, isUA, SCATTER_FORWARD, UA));
    PetscCall(VecViewFromOptions(UA, NULL, "-ua_vec_view"));

    /* Restrict to U and Alpha */
    PetscCall(VecISCopy(X, isU, SCATTER_REVERSE, U));
    PetscCall(VecISCopy(X, isA, SCATTER_REVERSE, A));

    /* restrict to UA2 */
    PetscCall(VecISCopy(X, isUA, SCATTER_REVERSE, UA2));
    PetscCall(VecViewFromOptions(UA2, NULL, "-ua2_vec_view"));
  }

  {
    Vec          tmpVec;
    PetscSection coordSection;
    Vec          coord;
    PetscReal    norm;
    PetscReal    time = 1.234;

    /* Writing nodal variables to ExodusII file */
    PetscCall(DMSetOutputSequenceNumber(dmU,0,time));
    PetscCall(DMSetOutputSequenceNumber(dmA,0,time));

    PetscCall(VecView(U, viewer));
    PetscCall(VecView(A, viewer));

    /* Saving U and Alpha in one shot.
       For this, we need to cheat and change the Vec's name
       Note that in the end we write variables one component at a time,
       so that there is no real values in doing this
    */

    PetscCall(DMSetOutputSequenceNumber(dmUA,1,time));
    PetscCall(DMGetGlobalVector(dmUA, &tmpVec));
    PetscCall(VecCopy(UA, tmpVec));
    PetscCall(PetscObjectSetName((PetscObject) tmpVec, "U"));
    PetscCall(VecView(tmpVec, viewer));
    /* Reading nodal variables in Exodus file */
    PetscCall(VecSet(tmpVec, -1000.0));
    PetscCall(VecLoad(tmpVec, viewer));
    PetscCall(VecAXPY(UA, -1.0, tmpVec));
    PetscCall(VecNorm(UA, NORM_INFINITY, &norm));
    PetscCheck(norm <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "UAlpha ||Vin - Vout|| = %g", (double) norm);
    PetscCall(DMRestoreGlobalVector(dmUA, &tmpVec));

    /* same thing with the UA2 Vec obtained from the superDM */
    PetscCall(DMGetGlobalVector(dmUA2, &tmpVec));
    PetscCall(VecCopy(UA2, tmpVec));
    PetscCall(PetscObjectSetName((PetscObject) tmpVec, "U"));
    PetscCall(DMSetOutputSequenceNumber(dmUA2,2,time));
    PetscCall(VecView(tmpVec, viewer));
    /* Reading nodal variables in Exodus file */
    PetscCall(VecSet(tmpVec, -1000.0));
    PetscCall(VecLoad(tmpVec,viewer));
    PetscCall(VecAXPY(UA2, -1.0, tmpVec));
    PetscCall(VecNorm(UA2, NORM_INFINITY, &norm));
    PetscCheck(norm <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "UAlpha2 ||Vin - Vout|| = %g", (double) norm);
    PetscCall(DMRestoreGlobalVector(dmUA2, &tmpVec));

    /* Building and saving Sigma
       We set sigma_0 = rank (to see partitioning)
              sigma_1 = cell set ID
              sigma_2 = x_coordinate of the cell center of mass
    */
    PetscCall(DMGetCoordinateSection(dmS, &coordSection));
    PetscCall(DMGetCoordinatesLocal(dmS, &coord));
    PetscCall(DMGetLabelIdIS(dmS, "Cell Sets", &csIS));
    PetscCall(DMGetLabelSize(dmS, "Cell Sets", &numCS));
    PetscCall(ISGetIndices(csIS, &csID));
    for (set = 0; set < numCS; ++set) {
      /* We know that all cells in a cell set have the same type, so we can dimension cval and xyz once for each cell set */
      IS              cellIS;
      const PetscInt *cellID;
      PetscInt        numCells, cell;
      PetscScalar    *cval = NULL, *xyz  = NULL;
      PetscInt        clSize, cdimCoord, c;

      PetscCall(DMGetStratumIS(dmS, "Cell Sets", csID[set], &cellIS));
      PetscCall(ISGetIndices(cellIS, &cellID));
      PetscCall(ISGetSize(cellIS, &numCells));
      for (cell = 0; cell < numCells; cell++) {
        PetscCall(DMPlexVecGetClosure(dmS, NULL, S, cellID[cell], &clSize, &cval));
        PetscCall(DMPlexVecGetClosure(dmS, coordSection, coord, cellID[cell], &cdimCoord, &xyz));
        cval[0] = rank;
        cval[1] = csID[set];
        cval[2] = 0.;
        for (c = 0; c < cdimCoord/sdim; c++) cval[2] += xyz[c*sdim];
        cval[2] = cval[2] * sdim / cdimCoord;
        PetscCall(DMPlexVecSetClosure(dmS, NULL, S, cellID[cell], cval, INSERT_ALL_VALUES));
      }
      PetscCall(DMPlexVecRestoreClosure(dmS, NULL, S, cellID[0], &clSize, &cval));
      PetscCall(DMPlexVecRestoreClosure(dmS, coordSection, coord, cellID[0], NULL, &xyz));
      PetscCall(ISRestoreIndices(cellIS, &cellID));
      PetscCall(ISDestroy(&cellIS));
    }
    PetscCall(ISRestoreIndices(csIS, &csID));
    PetscCall(ISDestroy(&csIS));
    PetscCall(VecViewFromOptions(S, NULL, "-s_vec_view"));

    /* Writing zonal variables in Exodus file */
    PetscCall(DMSetOutputSequenceNumber(dmS,0,time));
    PetscCall(VecView(S,viewer));

    /* Reading zonal variables in Exodus file */
    PetscCall(DMGetGlobalVector(dmS, &tmpVec));
    PetscCall(VecSet(tmpVec, -1000.0));
    PetscCall(PetscObjectSetName((PetscObject) tmpVec, "Sigma"));
    PetscCall(VecLoad(tmpVec,viewer));
    PetscCall(VecAXPY(S, -1.0, tmpVec));
    PetscCall(VecNorm(S, NORM_INFINITY, &norm));
    PetscCheck(norm <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Sigma ||Vin - Vout|| = %g", (double) norm);
    PetscCall(DMRestoreGlobalVector(dmS, &tmpVec));
  }
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(DMRestoreGlobalVector(dmUA2, &UA2));
  PetscCall(DMRestoreGlobalVector(dmUA, &UA));
  PetscCall(DMRestoreGlobalVector(dmS,  &S));
  PetscCall(DMRestoreGlobalVector(dmA,  &A));
  PetscCall(DMRestoreGlobalVector(dmU,  &U));
  PetscCall(DMRestoreGlobalVector(pdm,   &X));
  PetscCall(DMDestroy(&dmU));PetscCall(ISDestroy(&isU));
  PetscCall(DMDestroy(&dmA));PetscCall(ISDestroy(&isA));
  PetscCall(DMDestroy(&dmS));PetscCall(ISDestroy(&isS));
  PetscCall(DMDestroy(&dmUA));PetscCall(ISDestroy(&isUA));
  PetscCall(DMDestroy(&dmUA2));
  PetscCall(DMDestroy(&pdm));
  if (size > 1) PetscCall(DMDestroy(&dm));
  PetscCall(PetscFree2(pStartDepth, pEndDepth));
  PetscCall(PetscFinalize());
  return 0;
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
