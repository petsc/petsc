#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/
#include <petsc/private/dmlabelimpl.h>
#include <petsc/private/sectionimpl.h>
#include <petsc/private/sfimpl.h>

static PetscErrorCode DMTransferMaterialParameters(DM dm, PetscSF sf, DM odm)
{
  Vec A;

  PetscFunctionBegin;
  /* TODO handle regions? */
  PetscCall(DMGetAuxiliaryVec(dm, NULL, 0, 0, &A));
  if (A) {
    DM           dmAux, ocdm, odmAux;
    Vec          oA, oAt;
    PetscSection sec, osec;

    PetscCall(VecGetDM(A, &dmAux));
    PetscCall(DMClone(odm, &odmAux));
    PetscCall(DMGetCoordinateDM(odm, &ocdm));
    PetscCall(DMSetCoordinateDM(odmAux, ocdm));
    PetscCall(DMCopyDisc(dmAux, odmAux));

    PetscCall(DMGetLocalSection(dmAux, &sec));
    if (sf) {
      PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)sec), &osec));
      PetscCall(VecCreate(PetscObjectComm((PetscObject)A), &oAt));
      PetscCall(DMPlexDistributeField(dmAux, sf, sec, A, osec, oAt));
    } else {
      PetscCall(PetscObjectReference((PetscObject)sec));
      osec = sec;
      PetscCall(PetscObjectReference((PetscObject)A));
      oAt = A;
    }
    PetscCall(DMSetLocalSection(odmAux, osec));
    PetscCall(PetscSectionDestroy(&osec));
    PetscCall(DMCreateLocalVector(odmAux, &oA));
    PetscCall(DMDestroy(&odmAux));
    PetscCall(VecCopy(oAt, oA));
    PetscCall(VecDestroy(&oAt));
    PetscCall(DMSetAuxiliaryVec(odm, NULL, 0, 0, oA));
    PetscCall(VecDestroy(&oA));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateDomainDecomposition_Plex(DM dm, PetscInt *nsub, char ***names, IS **innerises, IS **outerises, DM **dms)
{
  DM           odm;
  PetscSF      migrationSF = NULL, sectionSF;
  PetscSection sec, tsec, ogsec, olsec;
  PetscInt     n, mh, ddovl = 0, pStart, pEnd, ni, no, nl;
  PetscDS      ds;
  DMLabel      label;
  const char  *oname = "__internal_plex_dd_ovl_";
  IS           gi_is, li_is, go_is, gl_is, ll_is;
  IS           gis, lis;
  PetscInt     rst, ren, c, *gidxs, *lidxs, *tidxs;
  Vec          gvec;

  PetscFunctionBegin;
  n = 1;
  if (nsub) *nsub = n;
  if (names) PetscCall(PetscCalloc1(n, names));
  if (innerises) PetscCall(PetscCalloc1(n, innerises));
  if (outerises) PetscCall(PetscCalloc1(n, outerises));
  if (dms) PetscCall(PetscCalloc1(n, dms));

  PetscObjectOptionsBegin((PetscObject)dm);
  PetscCall(PetscOptionsBoundedInt("-dm_plex_dd_overlap", "The size of the overlap halo for the subdomains", "DMCreateDomainDecomposition", ddovl, &ddovl, NULL, 0));
  PetscOptionsEnd();

  PetscCall(DMViewFromOptions(dm, NULL, "-dm_plex_dd_dm_view"));
  PetscCall(DMPlexDistributeOverlap_Internal(dm, ddovl + 1, PETSC_COMM_SELF, oname, &migrationSF, &odm));
  if (!odm) PetscCall(DMClone(dm, &odm));
  if (migrationSF) PetscCall(PetscSFViewFromOptions(migrationSF, (PetscObject)dm, "-dm_plex_dd_sf_view"));

  PetscCall(DMPlexGetMaxProjectionHeight(dm, &mh));
  PetscCall(DMPlexSetMaxProjectionHeight(odm, mh));

  /* share discretization */
  /* TODO Labels for regions may need to updated,
     now it uses the original ones, not the ones for the odm.
     Not sure what to do here */
  /* PetscCall(DMCopyDisc(dm, odm)); */
  PetscCall(DMGetDS(odm, &ds));
  if (!ds) {
    PetscCall(DMGetLocalSection(dm, &sec));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)odm), &tsec));
    if (migrationSF) {
      PetscCall(PetscSFDistributeSection(migrationSF, sec, NULL, tsec));
    } else {
      PetscCall(PetscSectionCopy(sec, tsec));
    }
    PetscCall(DMSetLocalSection(dm, tsec));
    PetscCall(PetscSectionDestroy(&tsec));
  }
  /* TODO: what if these values changes? add to some DM hook? */
  PetscCall(DMTransferMaterialParameters(dm, migrationSF, odm));

  PetscCall(DMViewFromOptions(odm, (PetscObject)dm, "-dm_plex_dd_overlap_dm_view"));
#if 0
  {
    DM              seqdm;
    Vec             val;
    IS              is;
    PetscInt        vStart, vEnd;
    const PetscInt *vnum;
    char            name[256];
    PetscViewer     viewer;

    PetscCall(DMPlexDistributeOverlap_Internal(dm, 0, PETSC_COMM_SELF, "local_mesh", NULL, &seqdm));
    PetscCall(PetscSNPrintf(name, sizeof(name), "local_mesh_%d.vtu", PetscGlobalRank));
    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)seqdm), name, FILE_MODE_WRITE, &viewer));
    PetscCall(DMGetLabel(seqdm, "local_mesh", &label));
    PetscCall(DMPlexLabelComplete(seqdm, label));
    PetscCall(DMPlexCreateLabelField(seqdm, label, &val));
    PetscCall(VecView(val, viewer));
    PetscCall(VecDestroy(&val));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(PetscSNPrintf(name, sizeof(name), "asm_vertices_%d.vtu", PetscGlobalRank));
    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)seqdm), name, FILE_MODE_WRITE, &viewer));
    PetscCall(DMCreateLabel(seqdm, "asm_vertices"));
    PetscCall(DMGetLabel(seqdm, "asm_vertices", &label));
    PetscCall(DMPlexGetVertexNumbering(dm, &is));
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    PetscCall(ISGetIndices(is, &vnum));
    for (PetscInt v = 0; v < vEnd - vStart; v++) {
      if (vnum[v] < 0) continue;
      PetscCall(DMLabelSetValue(label, v + vStart, 1));
    }
    PetscCall(DMPlexCreateLabelField(seqdm, label, &val));
    PetscCall(VecView(val, viewer));
    PetscCall(VecDestroy(&val));
    PetscCall(ISRestoreIndices(is, &vnum));
    PetscCall(PetscViewerDestroy(&viewer));

    PetscCall(DMDestroy(&seqdm));
    PetscCall(PetscSNPrintf(name, sizeof(name), "ovl_mesh_%d.vtu", PetscGlobalRank));
    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)odm), name, FILE_MODE_WRITE, &viewer));
    PetscCall(DMView(odm, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
#endif

  /* propagate original global ordering to overlapping DM */
  PetscCall(DMGetSectionSF(dm, &sectionSF));
  PetscCall(DMGetLocalSection(dm, &sec));
  PetscCall(PetscSectionGetStorageSize(sec, &nl));
  PetscCall(DMGetGlobalVector(dm, &gvec));
  PetscCall(VecGetOwnershipRange(gvec, &rst, &ren));
  PetscCall(DMRestoreGlobalVector(dm, &gvec));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, ren - rst, rst, 1, &gi_is)); /* non-overlapping dofs */
  PetscCall(PetscMalloc1(nl, &lidxs));
  for (PetscInt i = 0; i < nl; i++) lidxs[i] = -1;
  PetscCall(ISGetIndices(gi_is, (const PetscInt **)&gidxs));
  PetscCall(PetscSFBcastBegin(sectionSF, MPIU_INT, gidxs, lidxs, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(sectionSF, MPIU_INT, gidxs, lidxs, MPI_REPLACE));
  PetscCall(ISRestoreIndices(gi_is, (const PetscInt **)&gidxs));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), nl, lidxs, PETSC_OWN_POINTER, &lis));
  if (migrationSF) {
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)odm), &tsec));
    PetscCall(DMPlexDistributeFieldIS(dm, migrationSF, sec, lis, tsec, &gis));
  } else {
    PetscCall(PetscObjectReference((PetscObject)lis));
    gis  = lis;
    tsec = NULL;
  }
  PetscCall(PetscSectionDestroy(&tsec));
  PetscCall(ISDestroy(&lis));
  PetscCall(PetscSFDestroy(&migrationSF));

  /* make dofs on the overlap boundary (not the global boundary) constrained */
  PetscCall(DMGetLabel(odm, oname, &label));
  if (label) {
    PetscCall(DMPlexLabelComplete(odm, label));
    PetscCall(DMGetLocalSection(odm, &tsec));
    PetscCall(PetscSectionGetChart(tsec, &pStart, &pEnd));
    PetscCall(DMLabelCreateIndex(label, pStart, pEnd));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)tsec), &sec));
    PetscCall(PetscSectionCopy_Internal(tsec, sec, label->bt));
    PetscCall(DMSetLocalSection(odm, sec));
    PetscCall(PetscSectionDestroy(&sec));
    PetscCall(DMRemoveLabel(odm, oname, NULL));
  } else { /* sequential case */
    PetscCall(DMGetLocalSection(dm, &sec));
    PetscCall(DMSetLocalSection(odm, sec));
  }

  /* Create index sets for dofs in the overlap dm */
  PetscCall(DMGetSectionSF(odm, &sectionSF));
  PetscCall(DMGetLocalSection(odm, &olsec));
  PetscCall(DMGetGlobalSection(odm, &ogsec));
  PetscCall(PetscSectionViewFromOptions(ogsec, (PetscObject)dm, "-dm_plex_dd_overlap_gsection_view"));
  PetscCall(PetscSectionViewFromOptions(olsec, (PetscObject)dm, "-dm_plex_dd_overlap_lsection_view"));
  ni = ren - rst;
  PetscCall(PetscSectionGetConstrainedStorageSize(ogsec, &no)); /* dofs in the overlap */
  PetscCall(PetscSectionGetStorageSize(olsec, &nl));            /* local dofs in the overlap */
  PetscCall(PetscMalloc1(no, &gidxs));
  PetscCall(ISGetIndices(gis, (const PetscInt **)&lidxs));
  PetscCall(PetscSFReduceBegin(sectionSF, MPIU_INT, lidxs, gidxs, MPI_REPLACE));
  PetscCall(PetscSFReduceEnd(sectionSF, MPIU_INT, lidxs, gidxs, MPI_REPLACE));
  PetscCall(ISRestoreIndices(gis, (const PetscInt **)&lidxs));

  /* non-overlapping dofs */
  PetscCall(PetscMalloc1(no, &lidxs));
  c = 0;
  for (PetscInt i = 0; i < no; i++) {
    if (gidxs[i] >= rst && gidxs[i] < ren) lidxs[c++] = i;
  }
  PetscCheck(c == ni, PETSC_COMM_SELF, PETSC_ERR_PLIB, "%" PetscInt_FMT " != %" PetscInt_FMT, c, ni);
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)odm), ni, lidxs, PETSC_OWN_POINTER, &li_is));

  /* global dofs in the overlap */
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), no, gidxs, PETSC_OWN_POINTER, &go_is));
  PetscCall(ISViewFromOptions(go_is, (PetscObject)dm, "-dm_plex_dd_overlap_gois_view"));
  /* PetscCall(ISCreateStride(PetscObjectComm((PetscObject)odm), no, 0, 1, &lo_is)); */

  /* local dofs of the overlapping subdomain (we actually need only dofs on the boundary of the subdomain) */
  PetscCall(PetscMalloc1(nl, &lidxs));
  PetscCall(PetscMalloc1(nl, &gidxs));
  PetscCall(ISGetIndices(gis, (const PetscInt **)&tidxs));
  c = 0;
  for (PetscInt i = 0; i < nl; i++) {
    if (tidxs[i] < 0) continue;
    lidxs[c] = i;
    gidxs[c] = tidxs[i];
    c++;
  }
  PetscCall(ISRestoreIndices(gis, (const PetscInt **)&tidxs));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)dm), c, gidxs, PETSC_OWN_POINTER, &gl_is));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)odm), c, lidxs, PETSC_OWN_POINTER, &ll_is));
  PetscCall(ISViewFromOptions(gl_is, (PetscObject)dm, "-dm_plex_dd_overlap_glis_view"));

  PetscCall(PetscObjectCompose((PetscObject)odm, "__Plex_DD_IS_gi", (PetscObject)gi_is));
  PetscCall(PetscObjectCompose((PetscObject)odm, "__Plex_DD_IS_li", (PetscObject)li_is));
  PetscCall(PetscObjectCompose((PetscObject)odm, "__Plex_DD_IS_go", (PetscObject)go_is));
  PetscCall(PetscObjectCompose((PetscObject)odm, "__Plex_DD_IS_gl", (PetscObject)gl_is));
  PetscCall(PetscObjectCompose((PetscObject)odm, "__Plex_DD_IS_ll", (PetscObject)ll_is));

  if (innerises) (*innerises)[0] = gi_is;
  else PetscCall(ISDestroy(&gi_is));
  if (outerises) (*outerises)[0] = go_is;
  else PetscCall(ISDestroy(&go_is));
  if (dms) (*dms)[0] = odm;
  else PetscCall(DMDestroy(&odm));
  PetscCall(ISDestroy(&li_is));
  PetscCall(ISDestroy(&gl_is));
  PetscCall(ISDestroy(&ll_is));
  PetscCall(ISDestroy(&gis));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCreateDomainDecompositionScatters_Plex(DM dm, PetscInt n, DM *subdms, VecScatter **iscat, VecScatter **oscat, VecScatter **lscat)
{
  Vec gvec, svec, lvec;
  IS  gi_is, li_is, go_is, gl_is, ll_is;

  PetscFunctionBegin;
  if (iscat) PetscCall(PetscMalloc1(n, iscat));
  if (oscat) PetscCall(PetscMalloc1(n, oscat));
  if (lscat) PetscCall(PetscMalloc1(n, lscat));

  PetscCall(DMGetGlobalVector(dm, &gvec));
  for (PetscInt i = 0; i < n; i++) {
    PetscCall(DMGetGlobalVector(subdms[i], &svec));
    PetscCall(DMGetLocalVector(subdms[i], &lvec));
    PetscCall(PetscObjectQuery((PetscObject)subdms[i], "__Plex_DD_IS_gi", (PetscObject *)&gi_is));
    PetscCall(PetscObjectQuery((PetscObject)subdms[i], "__Plex_DD_IS_li", (PetscObject *)&li_is));
    PetscCall(PetscObjectQuery((PetscObject)subdms[i], "__Plex_DD_IS_go", (PetscObject *)&go_is));
    PetscCall(PetscObjectQuery((PetscObject)subdms[i], "__Plex_DD_IS_gl", (PetscObject *)&gl_is));
    PetscCall(PetscObjectQuery((PetscObject)subdms[i], "__Plex_DD_IS_ll", (PetscObject *)&ll_is));
    PetscCheck(gi_is, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "SubDM not obtained from DMCreateDomainDecomposition");
    PetscCheck(li_is, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "SubDM not obtained from DMCreateDomainDecomposition");
    PetscCheck(go_is, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "SubDM not obtained from DMCreateDomainDecomposition");
    PetscCheck(gl_is, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "SubDM not obtained from DMCreateDomainDecomposition");
    PetscCheck(ll_is, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONGSTATE, "SubDM not obtained from DMCreateDomainDecomposition");
    if (iscat) PetscCall(VecScatterCreate(gvec, gi_is, svec, li_is, &(*iscat)[i]));
    if (oscat) PetscCall(VecScatterCreate(gvec, go_is, svec, NULL, &(*oscat)[i]));
    if (lscat) PetscCall(VecScatterCreate(gvec, gl_is, lvec, ll_is, &(*lscat)[i]));
    PetscCall(DMRestoreGlobalVector(subdms[i], &svec));
    PetscCall(DMRestoreLocalVector(subdms[i], &lvec));
  }
  PetscCall(DMRestoreGlobalVector(dm, &gvec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
     DMCreateNeumannOverlap_Plex - Generates an IS, an unassembled (Neumann) Mat, a setup function, and the corresponding context to be used by PCHPDDM.

   Input Parameter:
.     dm - preconditioner context

   Output Parameters:
+     ovl - index set of overlapping subdomains
.     J - unassembled (Neumann) local matrix
.     setup - function for generating the matrix
-     setup_ctx - context for setup

   Options Database Keys:
+   -dm_plex_view_neumann_original - view the DM without overlap
-   -dm_plex_view_neumann_overlap - view the DM with overlap as needed by PCHPDDM

   Level: advanced

.seealso: `DMCreate()`, `DM`, `MATIS`, `PCHPDDM`, `PCHPDDMSetAuxiliaryMat()`
*/
PetscErrorCode DMCreateNeumannOverlap_Plex(DM dm, IS *ovl, Mat *J, PetscErrorCode (**setup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void *), void **setup_ctx)
{
  DM                     odm;
  Mat                    pJ;
  PetscSF                sf = NULL;
  PetscSection           sec, osec;
  ISLocalToGlobalMapping l2g;
  const PetscInt        *idxs;
  PetscInt               n, mh;

  PetscFunctionBegin;
  *setup     = NULL;
  *setup_ctx = NULL;
  *ovl       = NULL;
  *J         = NULL;

  /* Overlapped mesh
     overlap is a little more generous, since it is not computed starting from the owned (Dirichlet) points, but from the locally owned cells */
  PetscCall(DMPlexDistributeOverlap(dm, 1, &sf, &odm));
  if (!odm) {
    PetscCall(PetscSFDestroy(&sf));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* share discretization */
  PetscCall(DMGetLocalSection(dm, &sec));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)sec), &osec));
  PetscCall(PetscSFDistributeSection(sf, sec, NULL, osec));
  /* what to do here? using both is fine? */
  PetscCall(DMSetLocalSection(odm, osec));
  PetscCall(DMCopyDisc(dm, odm));
  PetscCall(DMPlexGetMaxProjectionHeight(dm, &mh));
  PetscCall(DMPlexSetMaxProjectionHeight(odm, mh));
  PetscCall(PetscSectionDestroy(&osec));

  /* material parameters */
  /* TODO: what if these values changes? add to some DM hook? */
  PetscCall(DMTransferMaterialParameters(dm, sf, odm));
  PetscCall(PetscSFDestroy(&sf));

  PetscCall(DMViewFromOptions(dm, NULL, "-dm_plex_view_neumann_original"));
  PetscCall(PetscObjectSetName((PetscObject)odm, "OVL"));
  PetscCall(DMViewFromOptions(odm, NULL, "-dm_plex_view_neumann_overlap"));

  /* MATIS for the overlap region
     the HPDDM interface wants local matrices,
     we attach the global MATIS to the overlap IS,
     since we need it to do assembly */
  PetscCall(DMSetMatType(odm, MATIS));
  PetscCall(DMCreateMatrix(odm, &pJ));
  PetscCall(MatISGetLocalMat(pJ, J));
  PetscCall(PetscObjectReference((PetscObject)*J));

  /* overlap IS */
  PetscCall(MatISGetLocalToGlobalMapping(pJ, &l2g, NULL));
  PetscCall(ISLocalToGlobalMappingGetSize(l2g, &n));
  PetscCall(ISLocalToGlobalMappingGetIndices(l2g, &idxs));
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)odm), n, idxs, PETSC_COPY_VALUES, ovl));
  PetscCall(ISLocalToGlobalMappingRestoreIndices(l2g, &idxs));
  PetscCall(PetscObjectCompose((PetscObject)*ovl, "_DM_Overlap_HPDDM_MATIS", (PetscObject)pJ));
  PetscCall(DMDestroy(&odm));
  PetscCall(MatDestroy(&pJ));

  /* special purpose setup function (composed in DMPlexSetSNESLocalFEM) */
  PetscCall(PetscObjectQueryFunction((PetscObject)dm, "MatComputeNeumannOverlap_C", setup));
  if (*setup) PetscCall(PetscObjectCompose((PetscObject)*ovl, "_DM_Original_HPDDM", (PetscObject)dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
