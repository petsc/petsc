static char help[] = "Test FEM layout and GlobalToNaturalSF\n\n";

/*
  In order to see the vectors which are being tested, use

     -ua_vec_view -s_vec_view
*/

#include <petsc.h>
#include <exodusII.h>

#include <petsc/private/dmpleximpl.h>

int main(int argc, char **argv) {
  DM                dm, pdm, dmU, dmA, dmS, dmUA, dmUA2, *dmList;
  DM                seqdmU, seqdmA, seqdmS, seqdmUA, seqdmUA2, *seqdmList;
  Vec               X, U, A, S, UA, UA2;
  Vec               seqX, seqU, seqA, seqS, seqUA, seqUA2;
  IS                isU, isA, isS, isUA;
  IS                seqisU, seqisA, seqisS, seqisUA;
  PetscSection      section;
  const PetscInt    fieldU = 0;
  const PetscInt    fieldA = 2;
  const PetscInt    fieldS = 1;
  const PetscInt    fieldUA[2] = {0, 2};
  char              ifilename[PETSC_MAX_PATH_LEN];
  IS                csIS;
  const PetscInt   *csID;
  PetscInt         *pStartDepth, *pEndDepth;
  PetscInt          order = 1;
  PetscInt          sdim, d, pStart, pEnd, p, numCS, set;
  PetscMPIInt       rank, size;
  PetscSF           migrationSF;

  PetscCall(PetscInitialize(&argc, &argv,NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "FEM Layout Options", "ex64");
  PetscCall(PetscOptionsString("-i", "Filename to read", "ex64", ifilename, ifilename, sizeof(ifilename), NULL));
  PetscOptionsEnd();

  /* Read the mesh from a file in any supported format */
  PetscCall(DMPlexCreateFromFile(PETSC_COMM_WORLD, ifilename, "ex64_plex", PETSC_TRUE, &dm));
  PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMGetDimension(dm, &sdim));

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

  /* Get DM and IS for each field of dm */
  PetscCall(DMCreateSubDM(dm, 1, &fieldU, &seqisU,  &seqdmU));
  PetscCall(DMCreateSubDM(dm, 1, &fieldA, &seqisA,  &seqdmA));
  PetscCall(DMCreateSubDM(dm, 1, &fieldS, &seqisS,  &seqdmS));
  PetscCall(DMCreateSubDM(dm, 2, fieldUA, &seqisUA, &seqdmUA));

  PetscCall(PetscMalloc1(2,&seqdmList));
  seqdmList[0] = seqdmU;
  seqdmList[1] = seqdmA;

  PetscCall(DMCreateSuperDM(seqdmList,2,NULL,&seqdmUA2));
  PetscCall(PetscFree(seqdmList));

  PetscCall(DMGetGlobalVector(dm,  &seqX));
  PetscCall(DMGetGlobalVector(seqdmU,  &seqU));
  PetscCall(DMGetGlobalVector(seqdmA,  &seqA));
  PetscCall(DMGetGlobalVector(seqdmS,  &seqS));
  PetscCall(DMGetGlobalVector(seqdmUA, &seqUA));
  PetscCall(DMGetGlobalVector(seqdmUA2, &seqUA2));

  PetscCall(PetscObjectSetName((PetscObject) seqX,  "seqX"));
  PetscCall(PetscObjectSetName((PetscObject) seqU,  "seqU"));
  PetscCall(PetscObjectSetName((PetscObject) seqA,  "seqAlpha"));
  PetscCall(PetscObjectSetName((PetscObject) seqS,  "seqSigma"));
  PetscCall(PetscObjectSetName((PetscObject) seqUA, "seqUAlpha"));
  PetscCall(PetscObjectSetName((PetscObject) seqUA2, "seqUAlpha2"));
  PetscCall(VecSet(seqX, -111.));

  /* Setting u to [x,y,z]  and alpha to x^2+y^2+z^2 by writing in UAlpha then restricting to U and Alpha */
  {
    PetscSection sectionUA;
    Vec          UALoc;
    PetscSection coordSection;
    Vec          coord;
    PetscScalar *cval, *xyz;
    PetscInt     clSize, i, j;

    PetscCall(DMGetLocalSection(seqdmUA, &sectionUA));
    PetscCall(DMGetLocalVector(seqdmUA, &UALoc));
    PetscCall(VecGetArray(UALoc, &cval));
    PetscCall(DMGetCoordinateSection(seqdmUA, &coordSection));
    PetscCall(DMGetCoordinatesLocal(seqdmUA, &coord));
    PetscCall(DMPlexGetChart(seqdmUA, &pStart, &pEnd));

    for (p = pStart; p < pEnd; ++p) {
      PetscInt dofUA, offUA;

      PetscCall(PetscSectionGetDof(sectionUA, p, &dofUA));
      if (dofUA > 0) {
        xyz=NULL;
        PetscCall(PetscSectionGetOffset(sectionUA, p, &offUA));
        PetscCall(DMPlexVecGetClosure(seqdmUA, coordSection, coord, p, &clSize, &xyz));
        cval[offUA+sdim] = 0.;
        for (i = 0; i < sdim; ++i) {
          cval[offUA+i] = 0;
          for (j = 0; j < clSize/sdim; ++j) {
            cval[offUA+i] += xyz[j*sdim+i];
          }
          cval[offUA+i] = cval[offUA+i] * sdim / clSize;
          cval[offUA+sdim] += PetscSqr(cval[offUA+i]);
        }
        PetscCall(DMPlexVecRestoreClosure(seqdmUA, coordSection, coord, p, &clSize, &xyz));
      }
    }
    PetscCall(VecRestoreArray(UALoc, &cval));
    PetscCall(DMLocalToGlobalBegin(seqdmUA, UALoc, INSERT_VALUES, seqUA));
    PetscCall(DMLocalToGlobalEnd(seqdmUA, UALoc, INSERT_VALUES, seqUA));
    PetscCall(DMRestoreLocalVector(seqdmUA, &UALoc));

    /* Update X */
    PetscCall(VecISCopy(seqX, seqisUA, SCATTER_FORWARD, seqUA));
    PetscCall(VecViewFromOptions(seqUA, NULL, "-ua_vec_view"));

    /* Restrict to U and Alpha */
    PetscCall(VecISCopy(seqX, seqisU, SCATTER_REVERSE, seqU));
    PetscCall(VecISCopy(seqX, seqisA, SCATTER_REVERSE, seqA));

    /* restrict to UA2 */
    PetscCall(VecISCopy(seqX, seqisUA, SCATTER_REVERSE, seqUA2));
    PetscCall(VecViewFromOptions(seqUA2, NULL, "-ua2_vec_view"));
  }

  {
    PetscInt         ovlp = 0;
    PetscPartitioner part;

    PetscCall(DMSetUseNatural(dm,PETSC_TRUE));
    PetscCall(DMPlexGetPartitioner(dm,&part));
    PetscCall(PetscPartitionerSetFromOptions(part));
    PetscCall(DMPlexDistribute(dm,ovlp,&migrationSF,&pdm));
    if (!pdm) pdm = dm;
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

  PetscCall(PetscObjectSetName((PetscObject) X,  "X"));
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

  /* Verification */

  Vec Xnat, Unat, Anat, UAnat, Snat, UA2nat;
  PetscReal norm;
  PetscCall(DMGetGlobalVector(dm, &Xnat));
  PetscCall(DMGetGlobalVector(seqdmU, &Unat));
  PetscCall(DMGetGlobalVector(seqdmA, &Anat));
  PetscCall(DMGetGlobalVector(seqdmUA, &UAnat));
  PetscCall(DMGetGlobalVector(seqdmS, &Snat));
  PetscCall(DMGetGlobalVector(seqdmUA2, &UA2nat));

  PetscCall(DMPlexGlobalToNaturalBegin(pdm, X, Xnat));
  PetscCall(DMPlexGlobalToNaturalEnd(pdm, X, Xnat));
  PetscCall(VecAXPY(Xnat, -1.0, seqX));
  PetscCall(VecNorm(Xnat, NORM_INFINITY, &norm));
  PetscCheck(norm <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "X ||Vin - Vout|| = %g", (double) norm);

  PetscCall(DMPlexGlobalToNaturalBegin(dmU, U, Unat));
  PetscCall(DMPlexGlobalToNaturalEnd(dmU, U, Unat));
  PetscCall(VecAXPY(Unat, -1.0, seqU));
  PetscCall(VecNorm(Unat, NORM_INFINITY, &norm));
  PetscCheck(norm <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "U ||Vin - Vout|| = %g", (double) norm);

  PetscCall(DMPlexGlobalToNaturalBegin(dmA, A, Anat));
  PetscCall(DMPlexGlobalToNaturalEnd(dmA, A, Anat));
  PetscCall(VecAXPY(Anat, -1.0, seqA));
  PetscCall(VecNorm(Anat, NORM_INFINITY, &norm));
  PetscCheck(norm <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "A ||Vin - Vout|| = %g", (double) norm);

  PetscCall(DMPlexGlobalToNaturalBegin(dmUA, UA, UAnat));
  PetscCall(DMPlexGlobalToNaturalEnd(dmUA, UA, UAnat));
  PetscCall(VecAXPY(UAnat, -1.0, seqUA));
  PetscCall(VecNorm(UAnat, NORM_INFINITY, &norm));
  PetscCheck(norm <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "UA ||Vin - Vout|| = %g", (double) norm);

  PetscCall(DMPlexGlobalToNaturalBegin(dmS, S, Snat));
  PetscCall(DMPlexGlobalToNaturalEnd(dmS, S, Snat));
  PetscCall(VecAXPY(Snat, -1.0, seqS));
  PetscCall(VecNorm(Snat, NORM_INFINITY, &norm));
  PetscCheck(norm <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "S ||Vin - Vout|| = %g", (double) norm);

  PetscCall(DMPlexGlobalToNaturalBegin(dmUA2, UA2, UA2nat));
  PetscCall(DMPlexGlobalToNaturalEnd(dmUA2, UA2, UA2nat));
  PetscCall(VecAXPY(UA2nat, -1.0, seqUA2));
  PetscCall(VecNorm(UA2nat, NORM_INFINITY, &norm));
  PetscCheck(norm <= PETSC_SQRT_MACHINE_EPSILON,PetscObjectComm((PetscObject) dm), PETSC_ERR_PLIB, "UAlpha2 ||Vin - Vout|| = %g", (double) norm);

  PetscCall(DMRestoreGlobalVector(dm, &Xnat));
  PetscCall(DMRestoreGlobalVector(seqdmU, &Unat));
  PetscCall(DMRestoreGlobalVector(seqdmA, &Anat));
  PetscCall(DMRestoreGlobalVector(seqdmUA, &UAnat));
  PetscCall(DMRestoreGlobalVector(seqdmS, &Snat));
  PetscCall(DMRestoreGlobalVector(seqdmUA2, &UA2nat));

  PetscCall(DMRestoreGlobalVector(seqdmUA2, &seqUA2));
  PetscCall(DMRestoreGlobalVector(seqdmUA, &seqUA));
  PetscCall(DMRestoreGlobalVector(seqdmS,  &seqS));
  PetscCall(DMRestoreGlobalVector(seqdmA,  &seqA));
  PetscCall(DMRestoreGlobalVector(seqdmU,  &seqU));
  PetscCall(DMRestoreGlobalVector(dm,   &seqX));
  PetscCall(DMDestroy(&seqdmU));PetscCall(ISDestroy(&seqisU));
  PetscCall(DMDestroy(&seqdmA));PetscCall(ISDestroy(&seqisA));
  PetscCall(DMDestroy(&seqdmS));PetscCall(ISDestroy(&seqisS));
  PetscCall(DMDestroy(&seqdmUA));PetscCall(ISDestroy(&seqisUA));
  PetscCall(DMDestroy(&seqdmUA2));

  PetscCall(DMRestoreGlobalVector(dmUA2, &UA2));
  PetscCall(DMRestoreGlobalVector(dmUA, &UA));
  PetscCall(DMRestoreGlobalVector(dmS,  &S));
  PetscCall(DMRestoreGlobalVector(dmA,  &A));
  PetscCall(DMRestoreGlobalVector(dmU,  &U));
  PetscCall(DMRestoreGlobalVector(pdm,   &X));
  PetscCall(DMDestroy(&dmU)); PetscCall(ISDestroy(&isU));
  PetscCall(DMDestroy(&dmA)); PetscCall(ISDestroy(&isA));
  PetscCall(DMDestroy(&dmS)); PetscCall(ISDestroy(&isS));
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
    requires: !complex parmetis exodusii pnetcdf
  # 2D seq
  test:
    suffix: 0
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -dm_view -petscpartitioner_type parmetis
  test:
    suffix: 1
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -dm_view -petscpartitioner_type parmetis
  test:
    suffix: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -dm_view -petscpartitioner_type parmetis

  # 2D par
  test:
    suffix: 3
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareT-large.exo -dm_view -petscpartitioner_type parmetis
  test:
    suffix: 4
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareQ-large.exo -dm_view -petscpartitioner_type parmetis
  test:
    suffix: 5
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourSquareH-large.exo -dm_view -petscpartitioner_type parmetis

  #3d seq
  test:
    suffix: 6
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -dm_view -petscpartitioner_type parmetis
  test:
    suffix: 7
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -dm_view -petscpartitioner_type parmetis

  #3d par
  test:
    suffix: 8
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickHex-large.exo -dm_view -petscpartitioner_type parmetis
  test:
    suffix: 9
    nsize: 2
    args: -i ${wPETSC_DIR}/share/petsc/datafiles/meshes/FourBrickTet-large.exo -dm_view -petscpartitioner_type parmetis

TEST*/
