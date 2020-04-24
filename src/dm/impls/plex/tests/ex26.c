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
  PetscErrorCode    ierr;

  ierr = PetscInitialize(&argc, &argv,NULL, help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FEM Layout Options", "ex26");CHKERRQ(ierr);
  ierr = PetscOptionsString("-i", "Filename to read", "ex26", ifilename, ifilename, sizeof(ifilename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-o", "Filename to write", "ex26", ofilename, ofilename, sizeof(ofilename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-order", "FEM polynomial order", "ex26", order, &order, NULL,1);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if ((order > 2) || (order < 1)) SETERRQ1(PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported polynomial order %D not in [1, 2]", order);

  ex_opts(EX_VERBOSE+EX_DEBUG);
  ierr = DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, PETSC_TRUE, &dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);

  /* Create the main section containning all fields */
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &section);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(section, 3);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, fieldU, "U");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, fieldA, "Alpha");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, fieldS, "Sigma");CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section, pStart, pEnd);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &sdim);CHKERRQ(ierr);
  ierr = PetscMalloc2(sdim+1, &pStartDepth, sdim+1, &pEndDepth);CHKERRQ(ierr);
  for (d = 0; d <= sdim; ++d) {ierr = DMPlexGetDepthStratum(dm, d, &pStartDepth[d], &pEndDepth[d]);CHKERRQ(ierr);}
  /* Vector field U, Scalar field Alpha, Tensor field Sigma */
  ierr = PetscSectionSetFieldComponents(section, fieldU, sdim);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(section, fieldA, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(section, fieldS, sdim*(sdim+1)/2);CHKERRQ(ierr);

  /* Going through cell sets then cells, and setting up storage for the sections */
  ierr = DMGetLabelSize(dm, "Cell Sets", &numCS);
  ierr = DMGetLabelIdIS(dm, "Cell Sets", &csIS);CHKERRQ(ierr);
  if (csIS) {ierr = ISGetIndices(csIS, &csID);CHKERRQ(ierr);}
  for (set = 0; set < numCS; set++) {
    IS                cellIS;
    const PetscInt   *cellID;
    PetscInt          numCells, cell, closureSize, *closureA = NULL;

    ierr = DMGetStratumSize(dm, "Cell Sets", csID[set], &numCells);CHKERRQ(ierr);
    ierr = DMGetStratumIS(dm, "Cell Sets", csID[set], &cellIS);CHKERRQ(ierr);
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
      default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No layout for dimension %D", sdim);
      }

      /* Identify cell type based on closure size only. This works for Tri/Tet/Quad/Hex meshes
         It will not be enough to identify more exotic elements like pyramid or prisms...  */
      ierr = ISGetIndices(cellIS, &cellID);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(dm, cellID[0], PETSC_TRUE, &closureSize, &closureA);CHKERRQ(ierr);
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
        default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Unknown element with closure size %D\n", closureSize);
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, cellID[0], PETSC_TRUE, &closureSize, &closureA);CHKERRQ(ierr);

      for (cell = 0; cell < numCells; cell++) {
        PetscInt *closure = NULL;

        ierr = DMPlexGetTransitiveClosure(dm, cellID[cell], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
        for (p = 0; p < closureSize; ++p) {
          /* Find depth of p */
          for (d = 0; d <= sdim; ++d) {
            if ((closure[2*p] >= pStartDepth[d]) && (closure[2*p] < pEndDepth[d])) {
              ierr = PetscSectionSetDof(section, closure[2*p], dofU[d]+dofA[d]+dofS[d]);CHKERRQ(ierr);
              ierr = PetscSectionSetFieldDof(section, closure[2*p], fieldU, dofU[d]);CHKERRQ(ierr);
              ierr = PetscSectionSetFieldDof(section, closure[2*p], fieldA, dofA[d]);CHKERRQ(ierr);
              ierr = PetscSectionSetFieldDof(section, closure[2*p], fieldS, dofS[d]);CHKERRQ(ierr);
            }
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm, cellID[cell], PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      }
      ierr = ISRestoreIndices(cellIS, &cellID);CHKERRQ(ierr);
      ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
    }
  }
  if (csIS) {ierr = ISRestoreIndices(csIS, &csID);CHKERRQ(ierr);}
  ierr = ISDestroy(&csIS);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  ierr = DMSetLocalSection(dm, section);CHKERRQ(ierr);
  ierr = PetscObjectViewFromOptions((PetscObject) section, NULL, "-dm_section_view");CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);

  {
    /* TODO: Replace with ExodusII viewer */
    /* Create the exodus result file */
    PetscInt numstep = 3, step;
    char    *nodalVarName[4];
    char    *zonalVarName[6];
    int     *truthtable;
    PetscInt      numNodalVar, numZonalVar, i;
    int      CPU_word_size, IO_word_size, EXO_mode;

    ex_opts(EX_VERBOSE+EX_DEBUG);
    if (!rank) {
      CPU_word_size = sizeof(PetscReal);
      IO_word_size  = sizeof(PetscReal);
      EXO_mode      = EX_CLOBBER;
#if defined(PETSC_USE_64BIT_INDICES)
      EXO_mode += EX_ALL_INT64_API;
#endif
      exoid = ex_create(ofilename, EXO_mode, &CPU_word_size, &IO_word_size);
      if (exoid < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_UNEXPECTED, "Unable to open exodus file %\n", ofilename);
    }
    ierr = DMPlexView_ExodusII_Internal(dm, exoid, order);CHKERRQ(ierr);

    if (!rank) {
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
      default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_OUTOFRANGE, "No layout for dimension %D", sdim);
      }
      ierr = ex_put_variable_param(exoid, EX_ELEM_BLOCK, numZonalVar);CHKERRQ(ierr);
      ierr = ex_put_variable_names(exoid, EX_ELEM_BLOCK, numZonalVar, zonalVarName);CHKERRQ(ierr);
      ierr = ex_put_variable_param(exoid, EX_NODAL, numNodalVar);CHKERRQ(ierr);
      ierr = ex_put_variable_names(exoid, EX_NODAL, numNodalVar, nodalVarName);CHKERRQ(ierr);
      numCS = ex_inquire_int(exoid, EX_INQ_ELEM_BLK);
      ierr = PetscMalloc1(numZonalVar * numCS, &truthtable);CHKERRQ(ierr);
      for (i = 0; i < numZonalVar * numCS; ++i) truthtable[i] = 1;
      ierr = ex_put_truth_table(exoid, EX_ELEM_BLOCK, numCS, numZonalVar, truthtable);CHKERRQ(ierr);
      ierr = PetscFree(truthtable);CHKERRQ(ierr);
      /* Writing time step information in the file. Note that this is currently broken in the exodus library for netcdf4 (HDF5-based) files */
      for (step = 0; step < numstep; ++step) {
        PetscReal time = step;
        ierr = ex_put_time(exoid, step+1, &time);CHKERRQ(ierr);
      }
      ierr = ex_close(exoid);CHKERRQ(ierr);
    }
  }

  {
    DM               pdm;
    PetscSF          migrationSF;
    PetscInt         ovlp = 0;
    PetscPartitioner part;

    ierr = DMSetUseNatural(dm,PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMPlexGetPartitioner(dm,&part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(dm,ovlp,&migrationSF,&pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMPlexSetMigrationSF(pdm,migrationSF);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&migrationSF);CHKERRQ(ierr);
      ierr = DMDestroy(&dm);CHKERRQ(ierr);
      dm = pdm;
      ierr = DMViewFromOptions(dm,NULL,"-dm_view");CHKERRQ(ierr);
    }
  }

  {
    /* TODO Replace with ExodusII viewer */
    /* Reopen the exodus result file on all processors */
    MPI_Info mpi_info = MPI_INFO_NULL;
    int      CPU_word_size, IO_word_size, EXO_mode;
    float    EXO_version;

    EXO_mode      = EX_WRITE;
    CPU_word_size = sizeof(PetscReal);
    IO_word_size  = sizeof(PetscReal);
#if defined(PETSC_USE_64BIT_INDICES)
      EXO_mode += EX_ALL_INT64_API;
#endif
    exoid = ex_open_par(ofilename, EXO_mode, &CPU_word_size, &IO_word_size, &EXO_version, PetscObjectComm((PetscObject) dm), mpi_info);
  }

  /* Get DM and IS for each field of dm */
  ierr = DMCreateSubDM(dm, 1, &fieldU, &isU,  &dmU);CHKERRQ(ierr);
  ierr = DMCreateSubDM(dm, 1, &fieldA, &isA,  &dmA);CHKERRQ(ierr);
  ierr = DMCreateSubDM(dm, 1, &fieldS, &isS,  &dmS);CHKERRQ(ierr);
  ierr = DMCreateSubDM(dm, 2, fieldUA, &isUA, &dmUA);CHKERRQ(ierr);

  ierr = PetscMalloc1(2,&dmList);CHKERRQ(ierr);
  dmList[0] = dmU;
  dmList[1] = dmA;
  /* We temporarily disable dmU->useNatural to test that we can reconstruct the
     NaturaltoGlobal SF from any of the dm in dms
  */
  dmU->useNatural = PETSC_FALSE;
  ierr = DMCreateSuperDM(dmList,2,NULL,&dmUA2);CHKERRQ(ierr);
  dmU->useNatural = PETSC_TRUE;
  ierr = PetscFree(dmList);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dm,   &X);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmU,  &U);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmA,  &A);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmS,  &S);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmUA, &UA);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmUA2, &UA2);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) U,  "U");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) A,  "Alpha");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) S,  "Sigma");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) UA, "UAlpha");CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) UA2, "UAlpha2");CHKERRQ(ierr);
  ierr = VecSet(X, -111.);CHKERRQ(ierr);

  /* Setting u to [x,y,z]  and alpha to x^2+y^2+z^2 by writing in UAlpha then restricting to U and Alpha */
  {
    PetscSection sectionUA;
    Vec          UALoc;
    PetscSection coordSection;
    Vec          coord;
    PetscScalar *cval, *xyz;
    PetscInt     cdimCoord = 24;

    ierr = DMGetLocalSection(dmUA, &sectionUA);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dmUA, &UALoc);CHKERRQ(ierr);
    ierr = VecGetArray(UALoc, &cval);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dmUA, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dmUA, &coord);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dmUA, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscMalloc1(cdimCoord, &xyz);CHKERRQ(ierr);
    /* I know that UA has 0 or sdim+1 dof at each point, since both U and Alpha use the same discretization
     The maximum possible size for the closure of the coordinate section is 8*3 at the cell center */
    for (p = pStart; p < pEnd; ++p) {
      PetscInt dofUA, offUA;

      ierr = PetscSectionGetDof(sectionUA, p, &dofUA);CHKERRQ(ierr);
      if (dofUA > 0) {
        PetscInt clSize = cdimCoord, i, j;

        ierr = PetscSectionGetOffset(sectionUA, p, &offUA);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(dmUA, coordSection, coord, p, &clSize, &xyz);CHKERRQ(ierr);
        cval[offUA+sdim] = 0.;
        for (i = 0; i < sdim; ++i) {
          cval[offUA+i] = 0;
          for (j = 0; j < clSize/sdim; ++j) {
            cval[offUA+i] += xyz[j*sdim+i];
          }
          cval[offUA+i] = cval[offUA+i] * sdim / clSize;
          cval[offUA+sdim] += PetscSqr(cval[offUA+i]);
        }
      }
    }
    ierr = PetscFree(xyz);CHKERRQ(ierr);
    ierr = VecRestoreArray(UALoc, &cval);CHKERRQ(ierr);
    ierr = DMLocalToGlobalBegin(dmUA, UALoc, INSERT_VALUES, UA);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(dmUA, UALoc, INSERT_VALUES, UA);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmUA, &UALoc);CHKERRQ(ierr);
    /* Update X */
    ierr = VecISCopy(X, isUA, SCATTER_FORWARD, UA);CHKERRQ(ierr);
    ierr = VecViewFromOptions(UA, NULL, "-ua_vec_view");CHKERRQ(ierr);
    /* Restrict to U and Alpha */
    ierr = VecISCopy(X, isU, SCATTER_REVERSE, U);CHKERRQ(ierr);
    ierr = VecISCopy(X, isA, SCATTER_REVERSE, A);CHKERRQ(ierr);
    /* restrict to UA2 */
    ierr = VecISCopy(X, isUA, SCATTER_REVERSE, UA2);CHKERRQ(ierr);
    ierr = VecViewFromOptions(UA2, NULL, "-ua2_vec_view");CHKERRQ(ierr);
  }

  {
    Vec          tmpVec;
    PetscSection coordSection;
    Vec          coord;
    PetscReal    norm;

    /* Writing nodal variables to ExodusII file */
    ierr = VecViewPlex_ExodusII_Nodal_Internal(U, exoid, 1);CHKERRQ(ierr);
    ierr = VecViewPlex_ExodusII_Nodal_Internal(A, exoid, 1);CHKERRQ(ierr);
    /* Saving U and Alpha in one shot.
       For this, we need to cheat and change the Vec's name
       Note that in the end we write variables one component at a time, so that there is no real values in doing this */
    ierr = DMGetGlobalVector(dmUA, &tmpVec);CHKERRQ(ierr);
    ierr = VecCopy(UA, tmpVec);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) tmpVec, "U");CHKERRQ(ierr);
    ierr = VecViewPlex_ExodusII_Nodal_Internal(tmpVec, exoid, 2);CHKERRQ(ierr);
    /* Reading nodal variables in Exodus file */
    ierr = VecSet(tmpVec, -1000.0);CHKERRQ(ierr);
    ierr = VecLoadPlex_ExodusII_Nodal_Internal(tmpVec, exoid, 2);CHKERRQ(ierr);
    ierr = VecAXPY(UA, -1.0, tmpVec);CHKERRQ(ierr);
    ierr = VecNorm(UA, NORM_INFINITY, &norm);CHKERRQ(ierr);
    if (norm > PETSC_SQRT_MACHINE_EPSILON) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "UAlpha ||Vin - Vout|| = %g\n", (double) norm);
    ierr = DMRestoreGlobalVector(dmUA, &tmpVec);CHKERRQ(ierr);

    /* same thing with the UA2 Vec obtained from the superDM */
    ierr = DMGetGlobalVector(dmUA2, &tmpVec);CHKERRQ(ierr);
    ierr = VecCopy(UA2, tmpVec);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) tmpVec, "U");CHKERRQ(ierr);
    ierr = VecViewPlex_ExodusII_Nodal_Internal(tmpVec, exoid, 3);CHKERRQ(ierr);
    /* Reading nodal variables in Exodus file */
    ierr = VecSet(tmpVec, -1000.0);CHKERRQ(ierr);
    ierr = VecLoadPlex_ExodusII_Nodal_Internal(tmpVec, exoid, 3);CHKERRQ(ierr);
    ierr = VecAXPY(UA2, -1.0, tmpVec);CHKERRQ(ierr);
    ierr = VecNorm(UA2, NORM_INFINITY, &norm);CHKERRQ(ierr);
    if (norm > PETSC_SQRT_MACHINE_EPSILON) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "UAlpha2 ||Vin - Vout|| = %g\n", (double) norm);
    ierr = DMRestoreGlobalVector(dmUA2, &tmpVec);CHKERRQ(ierr);

    /* Building and saving Sigma
       We set sigma_0 = rank (to see partitioning)
              sigma_1 = cell set ID
              sigma_2 = x_coordinate of the cell center of mass */
    ierr = DMGetCoordinateSection(dmS, &coordSection);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dmS, &coord);CHKERRQ(ierr);
    ierr = DMGetLabelIdIS(dmS, "Cell Sets", &csIS);CHKERRQ(ierr);
    ierr = DMGetLabelSize(dmS, "Cell Sets", &numCS);
    ierr = ISGetIndices(csIS, &csID);CHKERRQ(ierr);
    for (set = 0; set < numCS; ++set) {
      /* We know that all cells in a cell set have the same type, so we can dimension cval and xyz once for each cell set */
      IS              cellIS;
      const PetscInt *cellID;
      PetscInt        numCells, cell;
      PetscScalar    *cval = NULL, *xyz  = NULL;
      PetscInt        clSize, cdimCoord, c;

      ierr = DMGetStratumIS(dmS, "Cell Sets", csID[set], &cellIS);CHKERRQ(ierr);
      ierr = ISGetIndices(cellIS, &cellID);CHKERRQ(ierr);
      ierr = ISGetSize(cellIS, &numCells);CHKERRQ(ierr);
      for (cell = 0; cell < numCells; cell++) {
        ierr = DMPlexVecGetClosure(dmS, NULL, S, cellID[cell], &clSize, &cval);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(dmS, coordSection, coord, cellID[cell], &cdimCoord, &xyz);CHKERRQ(ierr);
        cval[0] = rank;
        cval[1] = csID[set];
        cval[2] = 0.;
        for (c = 0; c < cdimCoord/sdim; c++) cval[2] += xyz[c*sdim];
        cval[2] = cval[2] * sdim / cdimCoord;
        ierr = DMPlexVecSetClosure(dmS, NULL, S, cellID[cell], cval, INSERT_ALL_VALUES);CHKERRQ(ierr);
      }
      ierr = DMPlexVecRestoreClosure(dmS, NULL, S, cellID[0], &clSize, &cval);CHKERRQ(ierr);
      ierr = DMPlexVecRestoreClosure(dmS, coordSection, coord, cellID[0], NULL, &xyz);CHKERRQ(ierr);
      ierr = ISRestoreIndices(cellIS, &cellID);CHKERRQ(ierr);
      ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(csIS, &csID);CHKERRQ(ierr);
    ierr = ISDestroy(&csIS);CHKERRQ(ierr);
    ierr = VecViewFromOptions(S, NULL, "-s_vec_view");CHKERRQ(ierr);
    /* Writing zonal variables in Exodus file */
    ierr = VecViewPlex_ExodusII_Zonal_Internal(S, exoid, 1);CHKERRQ(ierr);
    /* Reading zonal variables in Exodus file */
    ierr = DMGetGlobalVector(dmS, &tmpVec);CHKERRQ(ierr);
    ierr = VecSet(tmpVec, -1000.0);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) tmpVec, "Sigma");CHKERRQ(ierr);
    ierr = VecLoadPlex_ExodusII_Zonal_Internal(tmpVec, exoid, 1);CHKERRQ(ierr);
    ierr = VecAXPY(S, -1.0, tmpVec);CHKERRQ(ierr);
    ierr = VecNorm(S, NORM_INFINITY, &norm);
    if (norm > PETSC_SQRT_MACHINE_EPSILON) SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "Sigma ||Vin - Vout|| = %g\n", (double) norm);
    ierr = DMRestoreGlobalVector(dmS, &tmpVec);CHKERRQ(ierr);
  }
  ierr = ex_close(exoid);CHKERRQ(ierr);

  ierr = DMRestoreGlobalVector(dmUA2, &UA2);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmUA, &UA);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmS,  &S);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmA,  &A);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dmU,  &U);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm,   &X);CHKERRQ(ierr);
  ierr = DMDestroy(&dmU);CHKERRQ(ierr); ierr = ISDestroy(&isU);CHKERRQ(ierr);
  ierr = DMDestroy(&dmA);CHKERRQ(ierr); ierr = ISDestroy(&isA);CHKERRQ(ierr);
  ierr = DMDestroy(&dmS);CHKERRQ(ierr); ierr = ISDestroy(&isS);CHKERRQ(ierr);
  ierr = DMDestroy(&dmUA);CHKERRQ(ierr);ierr = ISDestroy(&isUA);CHKERRQ(ierr);
  ierr = DMDestroy(&dmUA2);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree2(pStartDepth, pEndDepth);CHKERRQ(ierr);
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
