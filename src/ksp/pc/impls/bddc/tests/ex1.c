static char help[] = "Test PCBDDCGraphCreateLocalSubdomainAdjacency() against the DMPlex facet graph.\n\n"
                     "Use -nfields to select the number of fields, with arrays -dofs_type and -p describing their layouts.\n"
                     "Use -bc_dir_dofs and -bc_neu_dofs to mark boundary dofs for each field, and -bc_label to select the boundary label.\n";

#include <petscdmplex.h>
#include <petscdt.h>
#include <petsc/private/pcbddcprivateimpl.h>
#include <petsc/private/pcbddcstructsimpl.h>

typedef enum {
  DOFS_H1,
  DOFS_HCURL,
  DOFS_HDIV
} DofType;

static const char *const DofTypes[] = {"H1", "Hcurl", "Hdiv", "DofType", "DOFS_", NULL};

typedef struct {
  PetscInt  nfields;
  DofType  *dofsTypes;
  PetscInt *p;
  PetscInt *bcDirDofs;
  PetscInt *bcNeuDofs;
  PetscBool bcDirSet;
  PetscBool bcNeuSet;
  char      bcLabel[PETSC_MAX_PATH_LEN];
  PetscBool view_plex_graph;
  PetscBool view_bddc_graph;
} TestOptions;

static PetscErrorCode ComputeNumDof(DM dm, DofType dofsType, PetscInt p, PetscInt numDof[], PetscInt *numComp)
{
  DMPolytopeType cellType, ct;
  PetscInt       dim, cStart, cEnd, formDegree;
  PetscBool      simplex;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(dim == 2 || dim == 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "Only dimensions 2 and 3 are supported, not dimension %" PetscInt_FMT, dim);
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCheck(cEnd > cStart, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The mesh has no cells");
  PetscCall(DMPlexGetCellType(dm, cStart, &cellType));
  simplex = (PetscBool)(DMPolytopeTypeGetNumVertices(cellType) == dim + 1);
  PetscCheck(cellType == DMPolytopeTypeSimpleShape(dim, simplex), PETSC_COMM_SELF, PETSC_ERR_SUP, "Only simplex and tensor-product cells are supported");
  for (PetscInt c = cStart + 1; c < cEnd; c++) {
    PetscCall(DMPlexGetCellType(dm, c, &ct));
    PetscCheck(ct == cellType, PETSC_COMM_SELF, PETSC_ERR_SUP, "Meshes with mixed cell types are not supported");
  }

  switch (dofsType) {
  case DOFS_H1:
    formDegree = 0;
    *numComp   = 1;
    break;
  case DOFS_HCURL:
    formDegree = 1;
    *numComp   = dim;
    break;
  case DOFS_HDIV:
    formDegree = dim - 1;
    *numComp   = dim;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Unknown dof type");
  }

  for (PetscInt d = 0; d <= dim; d++) {
    PetscInt value, forms;

    if (d < formDegree) {
      numDof[d] = 0;
      continue;
    }
    PetscCall(PetscDTBinomialInt(d, formDegree, &forms));
    if (simplex) {
      if (d > p + formDegree - 1) {
        numDof[d] = 0;
        continue;
      }
      PetscCall(PetscDTBinomialInt(p + formDegree - 1, d, &value));
      PetscCall(PetscIntMultError(value, forms, &numDof[d]));
    } else {
      value = forms;
      for (PetscInt i = 0; i < formDegree; i++) PetscCall(PetscIntMultError(value, p, &value));
      for (PetscInt i = formDegree; i < d; i++) PetscCall(PetscIntMultError(value, p - 1, &value));
      numDof[d] = value;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ProcessOptions(TestOptions *options)
{
  PetscInt  n;
  PetscBool set;

  PetscFunctionBeginUser;
  options->nfields = 1;
  PetscOptionsBegin(PETSC_COMM_SELF, "", "Section options", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-nfields", "Number of fields", "ex1.c", 1, &options->nfields, NULL, 1));
  PetscCall(PetscMalloc4(options->nfields, &options->dofsTypes, options->nfields, &options->p, options->nfields, &options->bcDirDofs, options->nfields, &options->bcNeuDofs));
  for (PetscInt f = 0; f < options->nfields; f++) {
    options->dofsTypes[f] = DOFS_H1;
    options->p[f]         = 1;
    options->bcDirDofs[f] = -1;
    options->bcNeuDofs[f] = -1;
  }
  n = options->nfields;
  PetscCall(PetscOptionsEnumArray("-dofs_type", "Finite element de Rham spaces", "ex1.c", DofTypes, (PetscEnum *)options->dofsTypes, &n, &set));
  PetscCheck(!set || n == options->nfields, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "-dofs_type requires exactly %" PetscInt_FMT " values, got %" PetscInt_FMT, options->nfields, n);
  n = options->nfields;
  PetscCall(PetscOptionsIntArray("-p", "Polynomial orders", "ex1.c", options->p, &n, &set));
  PetscCheck(!set || n == options->nfields, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "-p requires exactly %" PetscInt_FMT " values, got %" PetscInt_FMT, options->nfields, n);
  n = options->nfields;
  PetscCall(PetscOptionsIntArray("-bc_dir_dofs", "Dirichlet boundary label values for each field (-1 for the entire boundary)", "ex1.c", options->bcDirDofs, &n, &options->bcDirSet));
  PetscCheck(!options->bcDirSet || n == options->nfields, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "-bc_dir_dofs requires exactly %" PetscInt_FMT " values, got %" PetscInt_FMT, options->nfields, n);
  n = options->nfields;
  PetscCall(PetscOptionsIntArray("-bc_neu_dofs", "Neumann boundary label values for each field (-1 for the entire boundary)", "ex1.c", options->bcNeuDofs, &n, &options->bcNeuSet));
  PetscCheck(!options->bcNeuSet || n == options->nfields, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "-bc_neu_dofs requires exactly %" PetscInt_FMT " values, got %" PetscInt_FMT, options->nfields, n);
  PetscCall(PetscStrncpy(options->bcLabel, "Face Sets", sizeof(options->bcLabel)));
  PetscCall(PetscOptionsString("-bc_label", "Boundary label for nonnegative Dirichlet and Neumann values", "ex1.c", options->bcLabel, options->bcLabel, sizeof(options->bcLabel), NULL));
  options->view_plex_graph = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-view_plex_graph", "View DMPLEX connectivity graph", "ex1.c", options->view_plex_graph, &options->view_plex_graph, NULL));
  options->view_bddc_graph = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-view_bddc_graph", "View BDDC connectivity graph", "ex1.c", options->view_bddc_graph, &options->view_bddc_graph, NULL));
  PetscOptionsEnd();

  for (PetscInt f = 0; f < options->nfields; f++) {
    PetscCheck(options->p[f] >= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Polynomial order for field %" PetscInt_FMT " must be positive, got %" PetscInt_FMT, f, options->p[f]);
    PetscCheck(!options->bcDirSet || options->bcDirDofs[f] >= -1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Dirichlet boundary value for field %" PetscInt_FMT " must be at least -1, got %" PetscInt_FMT, f, options->bcDirDofs[f]);
    PetscCheck(!options->bcNeuSet || options->bcNeuDofs[f] >= -1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Neumann boundary value for field %" PetscInt_FMT " must be at least -1, got %" PetscInt_FMT, f, options->bcNeuDofs[f]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSection(DM dm, const TestOptions *options, PetscSection *section)
{
  PetscInt *numComp, *numDof;
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(PetscMalloc2(options->nfields, &numComp, options->nfields * (dim + 1), &numDof));
  for (PetscInt f = 0; f < options->nfields; f++) PetscCall(ComputeNumDof(dm, options->dofsTypes[f], options->p[f], &numDof[f * (dim + 1)], &numComp[f]));
  PetscCall(DMSetNumFields(dm, options->nfields));
  PetscCall(DMPlexCreateSection(dm, NULL, numComp, numDof, 0, NULL, NULL, NULL, NULL, section));
  PetscCall(PetscFree2(numComp, numDof));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateElementMapping(DM dm, PetscSection section, PetscInt nfields, PetscInt *numCells, PetscInt *numDofs, ISLocalToGlobalMapping *l2g, PetscInt **localSubs, IS *fieldIS[])
{
  PetscInt  *indices, *l2gIndices, *subs, *globalField, *fieldCounts, *fieldCursor;
  PetscInt **fieldIndices;
  PetscInt   cStart, cEnd, pStart, pEnd, c, n, nlocal = 0, off = 0;

  PetscFunctionBeginUser;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  for (c = cStart; c < cEnd; c++) {
    PetscCall(DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &n, &indices, NULL, NULL));
    nlocal += n;
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE, &n, &indices, NULL, NULL));
  }
  PetscCall(PetscSectionGetStorageSize(section, numDofs));
  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  PetscCall(PetscMalloc1(nlocal, &l2gIndices));
  PetscCall(PetscMalloc1(nlocal, &subs));
  PetscCall(PetscMalloc1(*numDofs, &globalField));
  for (PetscInt i = 0; i < *numDofs; i++) globalField[i] = -1;
  for (PetscInt p = pStart; p < pEnd; p++) {
    for (PetscInt f = 0; f < nfields; f++) {
      PetscInt fdof, foff;

      PetscCall(PetscSectionGetFieldDof(section, p, f, &fdof));
      PetscCall(PetscSectionGetFieldOffset(section, p, f, &foff));
      for (PetscInt i = 0; i < fdof; i++) {
        PetscCheck(foff + i >= 0 && foff + i < *numDofs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid field offset %" PetscInt_FMT " not in [0,%" PetscInt_FMT ")", foff + i, *numDofs);
        globalField[foff + i] = f;
      }
    }
  }
  for (c = cStart; c < cEnd; c++) {
    PetscCall(DMPlexGetClosureIndices(dm, section, section, c, PETSC_TRUE, &n, &indices, NULL, NULL));
    for (PetscInt i = 0; i < n; i++) {
      PetscCheck(indices[i] >= 0 && indices[i] < *numDofs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid section index %" PetscInt_FMT " not in [0,%" PetscInt_FMT ")", indices[i], *numDofs);
      PetscCheck(globalField[indices[i]] >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Section index %" PetscInt_FMT " does not belong to a field", indices[i]);
      l2gIndices[off] = indices[i];
      subs[off++]     = c - cStart;
    }
    PetscCall(DMPlexRestoreClosureIndices(dm, section, section, c, PETSC_TRUE, &n, &indices, NULL, NULL));
  }
  PetscCheck(off == nlocal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Mapped %" PetscInt_FMT " dofs, expected %" PetscInt_FMT, off, nlocal);
  PetscCall(PetscCalloc2(nfields, &fieldCounts, nfields, &fieldCursor));
  PetscCall(PetscMalloc1(nfields, &fieldIndices));
  PetscCall(PetscMalloc1(nfields, fieldIS));
  for (PetscInt i = 0; i < nlocal; i++) fieldCounts[globalField[l2gIndices[i]]]++;
  for (PetscInt f = 0; f < nfields; f++) PetscCall(PetscMalloc1(fieldCounts[f], &fieldIndices[f]));
  for (PetscInt i = 0; i < nlocal; i++) {
    const PetscInt f = globalField[l2gIndices[i]];

    fieldIndices[f][fieldCursor[f]++] = i;
  }
  for (PetscInt f = 0; f < nfields; f++) PetscCall(ISCreateGeneral(PETSC_COMM_SELF, fieldCounts[f], fieldIndices[f], PETSC_OWN_POINTER, &(*fieldIS)[f]));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF, 1, nlocal, l2gIndices, PETSC_OWN_POINTER, l2g));
  PetscCall(PetscFree(fieldIndices));
  PetscCall(PetscFree2(fieldCounts, fieldCursor));
  PetscCall(PetscFree(globalField));
  *numCells  = cEnd - cStart;
  *localSubs = subs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateBoundaryIS(DM dm, PetscSection section, PetscInt nfields, const PetscInt boundaryValues[], const char boundaryLabelName[], ISLocalToGlobalMapping l2g, const char optionName[], IS *boundaryIS)
{
  DMLabel         boundaryLabel;
  const PetscInt *l2gIndices;
  PetscInt       *indices;
  PetscInt        pStart, pEnd, numDofs, nlocal, n = 0;
  PetscBool      *marked;

  PetscFunctionBeginUser;
  PetscCall(DMGetLabel(dm, boundaryLabelName, &boundaryLabel));
  PetscCheck(boundaryLabel, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "%s requires the mesh label '%s'", optionName, boundaryLabelName);
  PetscCall(DMPlexLabelComplete(dm, boundaryLabel));
  for (PetscInt f = 0; f < nfields; f++) {
    if (boundaryValues[f] >= 0) {
      PetscInt labelSize;

      PetscCall(DMLabelGetStratumSize(boundaryLabel, boundaryValues[f], &labelSize));
      PetscCheck(labelSize, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "%s value %" PetscInt_FMT " for field %" PetscInt_FMT " is not present in '%s'", optionName, boundaryValues[f], f, boundaryLabelName);
    }
  }

  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  PetscCall(PetscSectionGetStorageSize(section, &numDofs));
  PetscCall(PetscCalloc1(numDofs, &marked));
  for (PetscInt p = pStart; p < pEnd; p++) {
    for (PetscInt f = 0; f < nfields; f++) {
      PetscInt  fdof, foff;
      PetscBool hasPoint;

      if (boundaryValues[f] == -1) PetscCall(DMLabelHasPoint(boundaryLabel, p, &hasPoint));
      else PetscCall(DMLabelStratumHasPoint(boundaryLabel, boundaryValues[f], p, &hasPoint));
      if (!hasPoint) continue;
      PetscCall(PetscSectionGetFieldDof(section, p, f, &fdof));
      PetscCall(PetscSectionGetFieldOffset(section, p, f, &foff));
      for (PetscInt i = 0; i < fdof; i++) marked[foff + i] = PETSC_TRUE;
    }
  }

  PetscCall(ISLocalToGlobalMappingGetSize(l2g, &nlocal));
  PetscCall(ISLocalToGlobalMappingGetIndices(l2g, &l2gIndices));
  for (PetscInt i = 0; i < nlocal; i++)
    if (marked[l2gIndices[i]]) n++;
  PetscCall(PetscMalloc1(n, &indices));
  for (PetscInt i = 0, j = 0; i < nlocal; i++)
    if (marked[l2gIndices[i]]) indices[j++] = i;
  PetscCall(ISLocalToGlobalMappingRestoreIndices(l2g, &l2gIndices));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, indices, PETSC_OWN_POINTER, boundaryIS));
  PetscCall(PetscFree(marked));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateReferenceGraph(DM dm, PetscInt *numCells, PetscInt **xadj, PetscInt **adjncy)
{
  PetscFunctionBeginUser;
  PetscCall(DMPlexCreatePartitionerGraph(dm, 0, numCells, xadj, adjncy, NULL));
  for (PetscInt c = 0; c < *numCells; c++) PetscCall(PetscSortInt((*xadj)[c + 1] - (*xadj)[c], *adjncy + (*xadj)[c]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CompareGraphs(PetscInt numCells, const PetscInt plexXadj[], const PetscInt plexAdjncy[], const PetscInt bddcXadj[], const PetscInt bddcAdjncy[])
{
  PetscFunctionBeginUser;
  for (PetscInt c = 0; c < numCells; c++) {
    const PetscInt plexDegree = plexXadj[c + 1] - plexXadj[c];
    const PetscInt bddcDegree = bddcXadj[c + 1] - bddcXadj[c];

    PetscCheck(plexDegree == bddcDegree, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %" PetscInt_FMT " has DMPlex degree %" PetscInt_FMT " and PCBDDCGraph degree %" PetscInt_FMT, c, plexDegree, bddcDegree);
    for (PetscInt i = 0; i < plexDegree; i++)
      PetscCheck(plexAdjncy[plexXadj[c] + i] == bddcAdjncy[bddcXadj[c] + i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cell %" PetscInt_FMT " adjacency %" PetscInt_FMT " is %" PetscInt_FMT " in DMPlex and %" PetscInt_FMT " in PCBDDCGraph", c, i, plexAdjncy[plexXadj[c] + i], bddcAdjncy[bddcXadj[c] + i]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PCBDDCGraph            graph;
  DM                     dm;
  TestOptions            options = {0};
  IS                    *fieldIS;
  IS                     dirichletIS = NULL, neumannIS = NULL;
  ISLocalToGlobalMapping l2g;
  PetscSection           section;
  PetscInt              *plexXadj, *plexAdjncy, *bddcXadj, *bddcAdjncy, *localSubs = NULL;
  PetscInt               dim, depth, numCells = 0, graphCells, bddcCells, numDofs;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(DMCreate(PETSC_COMM_SELF, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-mesh_view"));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCheck(depth == dim, PETSC_COMM_SELF, PETSC_ERR_SUP, "The mesh must be fully interpolated so that DMPlex facet connectivity is explicit");

  PetscCall(ProcessOptions(&options));
  PetscCall(CreateSection(dm, &options, &section));
  PetscCall(CreateElementMapping(dm, section, options.nfields, &numCells, &numDofs, &l2g, &localSubs, &fieldIS));
  if (options.bcDirSet) PetscCall(CreateBoundaryIS(dm, section, options.nfields, options.bcDirDofs, options.bcLabel, l2g, "-bc_dir_dofs", &dirichletIS));
  if (options.bcNeuSet) PetscCall(CreateBoundaryIS(dm, section, options.nfields, options.bcNeuDofs, options.bcLabel, l2g, "-bc_neu_dofs", &neumannIS));
  PetscCall(CreateReferenceGraph(dm, &graphCells, &plexXadj, &plexAdjncy));
  PetscCheck(graphCells == numCells, PETSC_COMM_SELF, PETSC_ERR_PLIB, "DMPlex graph has %" PetscInt_FMT " cells, element map has %" PetscInt_FMT, graphCells, numCells);

  PetscCall(PCBDDCGraphCreate(&graph));
  PetscCall(PCBDDCGraphInit(graph, l2g, numDofs, PETSC_INT_MAX));
  graph->n_local_subs = numCells;
  graph->local_subs   = localSubs;
  PetscCall(PCBDDCGraphSetUp(graph, 1, neumannIS, dirichletIS, options.nfields, fieldIS, NULL));
  PetscCall(PCBDDCGraphComputeConnectedComponents(graph));
  PetscCall(PCBDDCGraphCreateLocalSubdomainAdjacency(graph, &bddcCells, &bddcXadj, &bddcAdjncy));
  PetscCheck(bddcCells == numCells, PETSC_COMM_SELF, PETSC_ERR_PLIB, "PCBDDC graph has %" PetscInt_FMT " cells, element map has %" PetscInt_FMT, bddcCells, numCells);
  if (options.view_plex_graph) PetscCall(PetscIntCSRView(numCells, plexXadj, plexAdjncy, NULL));
  if (options.view_bddc_graph) PetscCall(PetscIntCSRView(bddcCells, bddcXadj, bddcAdjncy, NULL));
  PetscCall(CompareGraphs(numCells, plexXadj, plexAdjncy, bddcXadj, bddcAdjncy));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Graphs match\n"));

  PetscCall(PetscFree(bddcXadj));
  PetscCall(PetscFree(bddcAdjncy));
  PetscCall(PetscFree(plexXadj));
  PetscCall(PetscFree(plexAdjncy));
  PetscCall(PCBDDCGraphDestroy(&graph));
  for (PetscInt f = 0; f < options.nfields; f++) PetscCall(ISDestroy(&fieldIS[f]));
  PetscCall(PetscFree(fieldIS));
  PetscCall(ISDestroy(&dirichletIS));
  PetscCall(ISDestroy(&neumannIS));
  PetscCall(ISLocalToGlobalMappingDestroy(&l2g));
  PetscCall(PetscSectionDestroy(&section));
  PetscCall(PetscFree4(options.dofsTypes, options.p, options.bcDirDofs, options.bcNeuDofs));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: 1
    output_file: output/ex1.out
    args: -dm_plex_interpolate 1 -dm_plex_csr_alg graph

    test:
      requires: triangle
      suffix: h1_2d
      args: -dm_plex_dim 2 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3 -dofs_type H1 -p {{1 2 3}}

    test:
      requires: triangle
      suffix: hcurl_2d
      args: -dm_plex_dim 2 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3 -dofs_type Hcurl -p {{1 2 3}}

    test:
      requires: triangle
      suffix: hdiv_2d
      args: -dm_plex_dim 2 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3 -dofs_type Hdiv -p {{1 2 3}}

    test:
      requires: ctetgen
      suffix: h1_3d
      args: -dm_plex_dim 3 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3,3 -dofs_type H1 -p {{1 2 3}}

    test:
      requires: ctetgen
      suffix: hcurl_3d
      args: -dm_plex_dim 3 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3,3 -dofs_type Hcurl -p {{1 2 3}}

    test:
      requires: ctetgen
      suffix: hdiv_3d
      args: -dm_plex_dim 3 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3,3 -dofs_type Hdiv -p {{1 2 3}}

    test:
      requires: triangle
      suffix: multifield_2d
      args: -dm_plex_dim 2 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3 -nfields 3 -dofs_type H1,Hcurl,Hdiv -p 1,2,3

    test:
      requires: ctetgen
      suffix: multifield_3d
      args: -dm_plex_dim 3 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3,3 -nfields 3 -dofs_type H1,Hcurl,Hdiv -p 1,2,3

    test:
      requires: triangle
      suffix: dirichlet_all_2d
      args: -dm_plex_dim 2 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3 -dm_plex_box_label -dofs_type H1 -p {{1 2 3}} -bc_dir_dofs -1

    test:
      requires: ctetgen
      suffix: dirichlet_all_3d
      args: -dm_plex_dim 3 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3,3 -dm_plex_box_label -dofs_type H1 -p {{1 2 3}} -bc_dir_dofs -1

    test:
      requires: triangle
      suffix: boundary_fields_2d
      args: -dm_plex_dim 2 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3 -dm_plex_box_label -nfields 3 -dofs_type H1,Hcurl,Hdiv -p 1,2,3 -bc_dir_dofs -1,1,2 -bc_neu_dofs 3,-1,4

    test:
      requires: ctetgen
      suffix: boundary_fields_3d
      args: -dm_plex_dim 3 -dm_plex_simplex {{0 1}} -dm_plex_box_faces 3,3,3 -dm_plex_box_label -nfields 3 -dofs_type H1,Hcurl,Hdiv -p 1,2,3 -bc_dir_dofs -1,1,2 -bc_neu_dofs 4,-1,6

    test:
      requires: triangle
      suffix: boundary_label
      args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 3,3 -dm_plex_boundary_label boundary -bc_label boundary -bc_dir_dofs 1

TEST*/
