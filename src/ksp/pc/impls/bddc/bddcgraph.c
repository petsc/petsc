#include <petsc/private/petscimpl.h>
#include <petsc/private/pcbddcprivateimpl.h>
#include <petsc/private/pcbddcstructsimpl.h>
#include <petsc/private/hashmapi.h>
#include <petsc/private/pcbddcgraphhashmap.h>
#include <petscsf.h>

PetscErrorCode PCBDDCDestroyGraphCandidatesIS(void **ctx)
{
  PCBDDCGraphCandidates cand = (PCBDDCGraphCandidates)*ctx;

  PetscFunctionBegin;
  for (PetscInt i = 0; i < cand->nfc; i++) PetscCall(ISDestroy(&cand->Faces[i]));
  for (PetscInt i = 0; i < cand->nec; i++) PetscCall(ISDestroy(&cand->Edges[i]));
  PetscCall(PetscFree(cand->Faces));
  PetscCall(PetscFree(cand->Edges));
  PetscCall(ISDestroy(&cand->Vertices));
  PetscCall(PetscFree(cand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphGetDirichletDofsB(PCBDDCGraph graph, IS *dirdofs)
{
  PetscFunctionBegin;
  if (graph->dirdofsB) {
    PetscCall(PetscObjectReference((PetscObject)graph->dirdofsB));
  } else if (graph->has_dirichlet) {
    PetscInt  i, size;
    PetscInt *dirdofs_idxs;

    size = 0;
    for (i = 0; i < graph->nvtxs; i++) {
      if (graph->nodes[i].count > 1 && graph->nodes[i].special_dof == PCBDDCGRAPH_DIRICHLET_MARK) size++;
    }

    PetscCall(PetscMalloc1(size, &dirdofs_idxs));
    size = 0;
    for (i = 0; i < graph->nvtxs; i++) {
      if (graph->nodes[i].count > 1 && graph->nodes[i].special_dof == PCBDDCGRAPH_DIRICHLET_MARK) dirdofs_idxs[size++] = i;
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, size, dirdofs_idxs, PETSC_OWN_POINTER, &graph->dirdofsB));
    PetscCall(PetscObjectReference((PetscObject)graph->dirdofsB));
  }
  *dirdofs = graph->dirdofsB;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphGetDirichletDofs(PCBDDCGraph graph, IS *dirdofs)
{
  PetscFunctionBegin;
  if (graph->dirdofs) {
    PetscCall(PetscObjectReference((PetscObject)graph->dirdofs));
  } else if (graph->has_dirichlet) {
    PetscInt  i, size;
    PetscInt *dirdofs_idxs;

    size = 0;
    for (i = 0; i < graph->nvtxs; i++) {
      if (graph->nodes[i].special_dof == PCBDDCGRAPH_DIRICHLET_MARK) size++;
    }

    PetscCall(PetscMalloc1(size, &dirdofs_idxs));
    size = 0;
    for (i = 0; i < graph->nvtxs; i++) {
      if (graph->nodes[i].special_dof == PCBDDCGRAPH_DIRICHLET_MARK) dirdofs_idxs[size++] = i;
    }
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)graph->l2gmap), size, dirdofs_idxs, PETSC_OWN_POINTER, &graph->dirdofs));
    PetscCall(PetscObjectReference((PetscObject)graph->dirdofs));
  }
  *dirdofs = graph->dirdofs;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphASCIIView(PCBDDCGraph graph, PetscInt verbosity_level, PetscViewer viewer)
{
  PetscInt  i, j, tabs;
  PetscInt *queue_in_global_numbering;

  PetscFunctionBegin;
  if (!viewer) PetscCall(PetscViewerASCIIGetStdout(graph->seq_graph ? PETSC_COMM_SELF : PetscObjectComm((PetscObject)graph->l2gmap), &viewer));
  PetscCall(PetscViewerASCIIPushSynchronized(viewer));
  PetscCall(PetscViewerASCIIGetTab(viewer, &tabs));
  PetscCall(PetscViewerASCIIPrintf(viewer, "--------------------------------------------------\n"));
  PetscCall(PetscViewerFlush(viewer));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Local BDDC graph for subdomain %04d (seq %d)\n", PetscGlobalRank, graph->seq_graph));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Number of vertices %" PetscInt_FMT "\n", graph->nvtxs));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Number of local subdomains %" PetscInt_FMT "\n", graph->n_local_subs ? graph->n_local_subs : 1));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Custom minimal size %" PetscInt_FMT "\n", graph->custom_minimal_size));
  if (graph->maxcount != PETSC_INT_MAX) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Max count %" PetscInt_FMT "\n", graph->maxcount));
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Topological two dim? %s (set %s)\n", PetscBools[graph->twodim], PetscBools[graph->twodimset]));
  if (verbosity_level > 2) {
    for (i = 0; i < graph->nvtxs; i++) {
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "%" PetscInt_FMT ":\n", i));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "   which_dof: %" PetscInt_FMT "\n", graph->nodes[i].which_dof));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "   special_dof: %" PetscInt_FMT "\n", graph->nodes[i].special_dof));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "   shared by: %" PetscInt_FMT "\n", graph->nodes[i].count));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
      if (graph->nodes[i].count) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "     set of neighbours:"));
        for (j = 0; j < graph->nodes[i].count; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %" PetscInt_FMT, graph->nodes[i].neighbours_set[j]));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
      }
      PetscCall(PetscViewerASCIISetTab(viewer, tabs));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "   number of local groups: %" PetscInt_FMT "\n", graph->nodes[i].local_groups_count));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
      if (graph->nodes[i].local_groups_count) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "     groups:"));
        for (j = 0; j < graph->nodes[i].local_groups_count; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %" PetscInt_FMT, graph->nodes[i].local_groups[j]));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
      }
      PetscCall(PetscViewerASCIISetTab(viewer, tabs));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));

      if (verbosity_level > 3) {
        if (graph->xadj) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "   local adj list:"));
          PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
          for (j = graph->xadj[i]; j < graph->xadj[i + 1]; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %" PetscInt_FMT, graph->adjncy[j]));
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
          PetscCall(PetscViewerASCIISetTab(viewer, tabs));
          PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
        } else {
          PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "   no adj info\n"));
        }
      }
      if (graph->n_local_subs) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "   local sub id: %" PetscInt_FMT "\n", graph->local_subs[i]));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "   interface subset id: %" PetscInt_FMT "\n", graph->nodes[i].subset));
      if (graph->nodes[i].subset && graph->subset_ncc) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "   ncc for subset: %" PetscInt_FMT "\n", graph->subset_ncc[graph->nodes[i].subset - 1]));
    }
  }
  PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Total number of connected components %" PetscInt_FMT "\n", graph->ncc));
  PetscCall(PetscMalloc1(graph->cptr[graph->ncc], &queue_in_global_numbering));
  PetscCall(ISLocalToGlobalMappingApply(graph->l2gmap, graph->cptr[graph->ncc], graph->queue, queue_in_global_numbering));
  for (i = 0; i < graph->ncc; i++) {
    PetscInt  node_num = graph->queue[graph->cptr[i]];
    PetscBool printcc  = PETSC_FALSE;
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "  cc %" PetscInt_FMT " (size %" PetscInt_FMT ", fid %" PetscInt_FMT ", neighs:", i, graph->cptr[i + 1] - graph->cptr[i], graph->nodes[node_num].which_dof));
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
    for (j = 0; j < graph->nodes[node_num].count; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %" PetscInt_FMT, graph->nodes[node_num].neighbours_set[j]));
    if (verbosity_level > 1) {
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "):"));
      if (verbosity_level > 2 || graph->twodim || graph->nodes[node_num].count > 2 || (graph->nodes[node_num].count == 2 && graph->nodes[node_num].special_dof == PCBDDCGRAPH_NEUMANN_MARK)) printcc = PETSC_TRUE;
      if (printcc) {
        for (j = graph->cptr[i]; j < graph->cptr[i + 1]; j++) PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, " %" PetscInt_FMT " (%" PetscInt_FMT ")", graph->queue[j], queue_in_global_numbering[j]));
      }
    } else {
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, ")"));
    }
    PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "\n"));
    PetscCall(PetscViewerASCIISetTab(viewer, tabs));
    PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
  }
  PetscCall(PetscFree(queue_in_global_numbering));
  PetscCall(PetscViewerFlush(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphRestoreCandidatesIS(PCBDDCGraph graph, PetscInt *n_faces, IS *FacesIS[], PetscInt *n_edges, IS *EdgesIS[], IS *VerticesIS)
{
  PetscInt       i;
  PetscContainer gcand;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)graph->l2gmap, "_PCBDDCGraphCandidatesIS", (PetscObject *)&gcand));
  if (gcand) {
    if (n_faces) *n_faces = 0;
    if (n_edges) *n_edges = 0;
    if (FacesIS) *FacesIS = NULL;
    if (EdgesIS) *EdgesIS = NULL;
    if (VerticesIS) *VerticesIS = NULL;
  }
  if (n_faces) {
    if (FacesIS) {
      for (i = 0; i < *n_faces; i++) PetscCall(ISDestroy(&(*FacesIS)[i]));
      PetscCall(PetscFree(*FacesIS));
    }
    *n_faces = 0;
  }
  if (n_edges) {
    if (EdgesIS) {
      for (i = 0; i < *n_edges; i++) PetscCall(ISDestroy(&(*EdgesIS)[i]));
      PetscCall(PetscFree(*EdgesIS));
    }
    *n_edges = 0;
  }
  if (VerticesIS) PetscCall(ISDestroy(VerticesIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphGetCandidatesIS(PCBDDCGraph graph, PetscInt *n_faces, IS *FacesIS[], PetscInt *n_edges, IS *EdgesIS[], IS *VerticesIS)
{
  IS            *ISForFaces, *ISForEdges, ISForVertices;
  PetscInt       i, nfc, nec, nvc, *idx, *mark;
  PetscContainer gcand;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)graph->l2gmap, "_PCBDDCGraphCandidatesIS", (PetscObject *)&gcand));
  if (gcand) {
    PCBDDCGraphCandidates cand;

    PetscCall(PetscContainerGetPointer(gcand, (void **)&cand));
    if (n_faces) *n_faces = cand->nfc;
    if (FacesIS) *FacesIS = cand->Faces;
    if (n_edges) *n_edges = cand->nec;
    if (EdgesIS) *EdgesIS = cand->Edges;
    if (VerticesIS) *VerticesIS = cand->Vertices;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscCalloc1(graph->ncc, &mark));
  /* loop on ccs to evaluate number of faces, edges and vertices */
  nfc = 0;
  nec = 0;
  nvc = 0;
  for (i = 0; i < graph->ncc; i++) {
    PetscInt repdof = graph->queue[graph->cptr[i]];
    if (graph->cptr[i + 1] - graph->cptr[i] > graph->custom_minimal_size && graph->nodes[repdof].count <= graph->maxcount) {
      if (!graph->twodim && graph->nodes[repdof].count == 2 && graph->nodes[repdof].special_dof != PCBDDCGRAPH_NEUMANN_MARK) {
        nfc++;
        mark[i] = 2;
      } else {
        nec++;
        mark[i] = 1;
      }
    } else {
      nvc += graph->cptr[i + 1] - graph->cptr[i];
    }
  }

  /* allocate IS arrays for faces, edges. Vertices need a single index set. */
  if (FacesIS) PetscCall(PetscMalloc1(nfc, &ISForFaces));
  if (EdgesIS) PetscCall(PetscMalloc1(nec, &ISForEdges));
  if (VerticesIS) PetscCall(PetscMalloc1(nvc, &idx));

  /* loop on ccs to compute index sets for faces and edges */
  if (!graph->queue_sorted) {
    PetscInt *queue_global;

    PetscCall(PetscMalloc1(graph->cptr[graph->ncc], &queue_global));
    PetscCall(ISLocalToGlobalMappingApply(graph->l2gmap, graph->cptr[graph->ncc], graph->queue, queue_global));
    for (i = 0; i < graph->ncc; i++) PetscCall(PetscSortIntWithArray(graph->cptr[i + 1] - graph->cptr[i], &queue_global[graph->cptr[i]], &graph->queue[graph->cptr[i]]));
    PetscCall(PetscFree(queue_global));
    graph->queue_sorted = PETSC_TRUE;
  }
  nfc = 0;
  nec = 0;
  for (i = 0; i < graph->ncc; i++) {
    if (mark[i] == 2) {
      if (FacesIS) PetscCall(ISCreateGeneral(PETSC_COMM_SELF, graph->cptr[i + 1] - graph->cptr[i], &graph->queue[graph->cptr[i]], PETSC_USE_POINTER, &ISForFaces[nfc]));
      nfc++;
    } else if (mark[i] == 1) {
      if (EdgesIS) PetscCall(ISCreateGeneral(PETSC_COMM_SELF, graph->cptr[i + 1] - graph->cptr[i], &graph->queue[graph->cptr[i]], PETSC_USE_POINTER, &ISForEdges[nec]));
      nec++;
    }
  }

  /* index set for vertices */
  if (VerticesIS) {
    nvc = 0;
    for (i = 0; i < graph->ncc; i++) {
      if (!mark[i]) {
        PetscInt j;

        for (j = graph->cptr[i]; j < graph->cptr[i + 1]; j++) {
          idx[nvc] = graph->queue[j];
          nvc++;
        }
      }
    }
    /* sort vertex set (by local ordering) */
    PetscCall(PetscSortInt(nvc, idx));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nvc, idx, PETSC_OWN_POINTER, &ISForVertices));
  }
  PetscCall(PetscFree(mark));

  /* get back info */
  if (n_faces) *n_faces = nfc;
  if (FacesIS) *FacesIS = ISForFaces;
  if (n_edges) *n_edges = nec;
  if (EdgesIS) *EdgesIS = ISForEdges;
  if (VerticesIS) *VerticesIS = ISForVertices;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphComputeConnectedComponents(PCBDDCGraph graph)
{
  PetscBool adapt_interface;
  MPI_Comm  interface_comm;
  PetscBT   cornerp = NULL;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)graph->l2gmap, &interface_comm));
  /* compute connected components locally */
  PetscCall(PCBDDCGraphComputeConnectedComponentsLocal(graph));
  if (graph->seq_graph) PetscFunctionReturn(PETSC_SUCCESS);

  if (graph->active_coords && !graph->multi_element) { /* face based corner selection XXX multi_element */
    PetscBT    excluded;
    PetscReal *wdist;
    PetscInt   n_neigh, *neigh, *n_shared, **shared;
    PetscInt   maxc, ns;

    PetscCall(PetscBTCreate(graph->nvtxs, &cornerp));
    PetscCall(ISLocalToGlobalMappingGetInfo(graph->l2gmap, &n_neigh, &neigh, &n_shared, &shared));
    for (ns = 1, maxc = 0; ns < n_neigh; ns++) maxc = PetscMax(maxc, n_shared[ns]);
    PetscCall(PetscMalloc1(maxc * graph->cdim, &wdist));
    PetscCall(PetscBTCreate(maxc, &excluded));

    for (ns = 1; ns < n_neigh; ns++) { /* first proc is self */
      PetscReal *anchor, mdist;
      PetscInt   fst, j, k, d, cdim = graph->cdim, n = n_shared[ns];
      PetscInt   point1, point2, point3, point4;

      /* import coordinates on shared interface */
      PetscCall(PetscBTMemzero(n, excluded));
      for (j = 0, fst = -1, k = 0; j < n; j++) {
        PetscBool skip = PETSC_FALSE;
        for (d = 0; d < cdim; d++) {
          PetscReal c = graph->coords[shared[ns][j] * cdim + d];
          skip        = (PetscBool)(skip || c == PETSC_MAX_REAL);
          wdist[k++]  = c;
        }
        if (skip) PetscCall(PetscBTSet(excluded, j));
        else if (fst == -1) fst = j;
      }
      if (fst == -1) continue;

      /* the dofs are sorted by global numbering, so each rank starts from the same id
         and it will detect the same corners from the given set */

      /* find the farthest point from the starting one */
      anchor = wdist + fst * cdim;
      mdist  = -1.0;
      point1 = fst;
      for (j = fst; j < n; j++) {
        PetscReal dist = 0.0;

        if (PetscUnlikely(PetscBTLookup(excluded, j))) continue;
        for (d = 0; d < cdim; d++) dist += (wdist[j * cdim + d] - anchor[d]) * (wdist[j * cdim + d] - anchor[d]);
        if (dist > mdist) {
          mdist  = dist;
          point1 = j;
        }
      }

      /* find the farthest point from point1 */
      anchor = wdist + point1 * cdim;
      mdist  = -1.0;
      point2 = point1;
      for (j = fst; j < n; j++) {
        PetscReal dist = 0.0;

        if (PetscUnlikely(PetscBTLookup(excluded, j))) continue;
        for (d = 0; d < cdim; d++) dist += (wdist[j * cdim + d] - anchor[d]) * (wdist[j * cdim + d] - anchor[d]);
        if (dist > mdist) {
          mdist  = dist;
          point2 = j;
        }
      }

      /* find the third point maximizing the triangle area */
      point3 = point2;
      if (cdim > 2) {
        PetscReal a = 0.0;

        for (d = 0; d < cdim; d++) a += (wdist[point1 * cdim + d] - wdist[point2 * cdim + d]) * (wdist[point1 * cdim + d] - wdist[point2 * cdim + d]);
        a     = PetscSqrtReal(a);
        mdist = -1.0;
        for (j = fst; j < n; j++) {
          PetscReal area, b = 0.0, c = 0.0, s;

          if (PetscUnlikely(PetscBTLookup(excluded, j))) continue;
          for (d = 0; d < cdim; d++) {
            b += (wdist[point1 * cdim + d] - wdist[j * cdim + d]) * (wdist[point1 * cdim + d] - wdist[j * cdim + d]);
            c += (wdist[point2 * cdim + d] - wdist[j * cdim + d]) * (wdist[point2 * cdim + d] - wdist[j * cdim + d]);
          }
          b = PetscSqrtReal(b);
          c = PetscSqrtReal(c);
          s = 0.5 * (a + b + c);

          /* Heron's formula, area squared */
          area = s * (s - a) * (s - b) * (s - c);
          if (area > mdist) {
            mdist  = area;
            point3 = j;
          }
        }
      }

      /* find the farthest point from point3 different from point1 and point2 */
      anchor = wdist + point3 * cdim;
      mdist  = -1.0;
      point4 = point3;
      for (j = fst; j < n; j++) {
        PetscReal dist = 0.0;

        if (PetscUnlikely(PetscBTLookup(excluded, j)) || j == point1 || j == point2 || j == point3) continue;
        for (d = 0; d < cdim; d++) dist += (wdist[j * cdim + d] - anchor[d]) * (wdist[j * cdim + d] - anchor[d]);
        if (dist > mdist) {
          mdist  = dist;
          point4 = j;
        }
      }

      PetscCall(PetscBTSet(cornerp, shared[ns][point1]));
      PetscCall(PetscBTSet(cornerp, shared[ns][point2]));
      PetscCall(PetscBTSet(cornerp, shared[ns][point3]));
      PetscCall(PetscBTSet(cornerp, shared[ns][point4]));

      /* all dofs having the same coordinates will be primal */
      for (j = fst; j < n; j++) {
        PetscBool same[] = {PETSC_TRUE, PETSC_TRUE, PETSC_TRUE, PETSC_TRUE};

        if (PetscUnlikely(PetscBTLookup(excluded, j))) continue;
        for (d = 0; d < cdim; d++) {
          same[0] = (PetscBool)(same[0] && (PetscAbsReal(wdist[j * cdim + d] - wdist[point1 * cdim + d]) < PETSC_SMALL));
          same[1] = (PetscBool)(same[1] && (PetscAbsReal(wdist[j * cdim + d] - wdist[point2 * cdim + d]) < PETSC_SMALL));
          same[2] = (PetscBool)(same[2] && (PetscAbsReal(wdist[j * cdim + d] - wdist[point3 * cdim + d]) < PETSC_SMALL));
          same[3] = (PetscBool)(same[3] && (PetscAbsReal(wdist[j * cdim + d] - wdist[point4 * cdim + d]) < PETSC_SMALL));
        }
        if (same[0] || same[1] || same[2] || same[3]) PetscCall(PetscBTSet(cornerp, shared[ns][j]));
      }
    }
    PetscCall(PetscBTDestroy(&excluded));
    PetscCall(PetscFree(wdist));
    PetscCall(ISLocalToGlobalMappingRestoreInfo(graph->l2gmap, &n_neigh, &neigh, &n_shared, &shared));
  }

  /* Adapt connected components if needed */
  adapt_interface = (cornerp || graph->multi_element) ? PETSC_TRUE : PETSC_FALSE;
  for (PetscInt i = 0; i < graph->n_subsets && !adapt_interface; i++) {
    if (graph->subset_ncc[i] > 1) adapt_interface = PETSC_TRUE;
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &adapt_interface, 1, MPI_C_BOOL, MPI_LOR, interface_comm));
  if (adapt_interface) {
    PetscSF         msf;
    const PetscInt *n_ref_sharing;
    PetscInt       *labels, *rootlabels, *mrlabels;
    PetscInt        nr, nmr, nrs, ncc, cum_queue;

    PetscCall(PetscMalloc1(graph->nvtxs, &labels));
    PetscCall(PetscArrayzero(labels, graph->nvtxs));
    for (PetscInt i = 0, k = 0; i < graph->ncc; i++) {
      PetscInt s = 1;
      for (PetscInt j = graph->cptr[i]; j < graph->cptr[i + 1]; j++) {
        if (cornerp && PetscBTLookup(cornerp, graph->queue[j])) {
          labels[graph->queue[j]] = -(k + s + 1);
          s += 1;
        } else {
          labels[graph->queue[j]] = -(k + 1);
        }
      }
      k += s;
    }
    PetscCall(PetscSFGetGraph(graph->interface_ref_sf, &nr, NULL, NULL, NULL));
    PetscCall(PetscSFGetGraph(graph->interface_subset_sf, &nrs, NULL, NULL, NULL));
    PetscCall(PetscSFGetMultiSF(graph->interface_subset_sf, &msf));
    PetscCall(PetscSFGetGraph(msf, &nmr, NULL, NULL, NULL));
    PetscCall(PetscCalloc2(nmr, &mrlabels, nrs, &rootlabels));

    PetscCall(PetscSFComputeDegreeBegin(graph->interface_subset_sf, &n_ref_sharing));
    PetscCall(PetscSFComputeDegreeEnd(graph->interface_subset_sf, &n_ref_sharing));
    PetscCall(PetscSFGatherBegin(graph->interface_subset_sf, MPIU_INT, labels, mrlabels));
    PetscCall(PetscSFGatherEnd(graph->interface_subset_sf, MPIU_INT, labels, mrlabels));

    /* analyze contributions from processes
       The structure of mrlabels is suitable to find intersections of ccs.
       supposing the root subset has dimension 5 and leaves with labels:
         0: [4 4 7 4 7], (2 connected components)
         1: [3 2 2 3 2], (2 connected components)
         2: [1 1 6 5 6], (3 connected components)
       the multiroot data and the new labels corresponding to intersected connected components will be (column major)

                  4 4 7 4 7
       mrlabels   3 2 2 3 2
                  1 1 6 5 6
                  ---------
       rootlabels 0 1 2 3 2
    */
    for (PetscInt i = 0, rcumlabels = 0, mcumlabels = 0; i < nr; i++) {
      const PetscInt  subset_size    = graph->interface_ref_rsize[i];
      const PetscInt *n_sharing      = n_ref_sharing + rcumlabels;
      const PetscInt *mrbuffer       = mrlabels + mcumlabels;
      PetscInt       *rbuffer        = rootlabels + rcumlabels;
      PetscInt        subset_counter = 0;

      for (PetscInt j = 0; j < subset_size; j++) {
        if (!rbuffer[j]) { /* found a new cc  */
          const PetscInt *jlabels = mrbuffer + j * n_sharing[0];
          rbuffer[j]              = ++subset_counter;

          for (PetscInt k = j + 1; k < subset_size; k++) { /* check for other nodes in new cc */
            PetscBool       same_set = PETSC_TRUE;
            const PetscInt *klabels  = mrbuffer + k * n_sharing[0];

            for (PetscInt s = 0; s < n_sharing[0]; s++) {
              if (jlabels[s] != klabels[s]) {
                same_set = PETSC_FALSE;
                break;
              }
            }
            if (same_set) rbuffer[k] = subset_counter;
          }
        }
      }
      if (subset_size) {
        rcumlabels += subset_size;
        mcumlabels += n_sharing[0] * subset_size;
      }
    }

    /* Now communicate the intersected labels */
    PetscCall(PetscSFBcastBegin(graph->interface_subset_sf, MPIU_INT, rootlabels, labels, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(graph->interface_subset_sf, MPIU_INT, rootlabels, labels, MPI_REPLACE));
    PetscCall(PetscFree2(mrlabels, rootlabels));

    /* and adapt local connected components */
    PetscInt  *ocptr, *oqueue;
    PetscBool *touched;

    PetscCall(PetscMalloc3(graph->ncc + 1, &ocptr, graph->cptr[graph->ncc], &oqueue, graph->cptr[graph->ncc], &touched));
    PetscCall(PetscArraycpy(ocptr, graph->cptr, graph->ncc + 1));
    PetscCall(PetscArraycpy(oqueue, graph->queue, graph->cptr[graph->ncc]));
    PetscCall(PetscArrayzero(touched, graph->cptr[graph->ncc]));

    ncc       = 0;
    cum_queue = 0;
    for (PetscInt i = 0; i < graph->ncc; i++) {
      for (PetscInt j = ocptr[i]; j < ocptr[i + 1]; j++) {
        const PetscInt jlabel = labels[oqueue[j]];

        if (jlabel) {
          graph->cptr[ncc] = cum_queue;
          ncc++;
          for (PetscInt k = j; k < ocptr[i + 1]; k++) { /* check for other nodes in new cc */
            if (labels[oqueue[k]] == jlabel) {
              graph->queue[cum_queue++] = oqueue[k];
              labels[oqueue[k]]         = 0;
            }
          }
        }
      }
    }
    PetscCall(PetscFree3(ocptr, oqueue, touched));
    PetscCall(PetscFree(labels));
    graph->cptr[ncc]    = cum_queue;
    graph->queue_sorted = PETSC_FALSE;
    graph->ncc          = ncc;
  }
  PetscCall(PetscBTDestroy(&cornerp));

  /* Determine if we are in 2D or 3D */
  if (!graph->twodimset) {
    PetscBool twodim = PETSC_TRUE;
    for (PetscInt i = 0; i < graph->ncc; i++) {
      PetscInt repdof = graph->queue[graph->cptr[i]];
      PetscInt ccsize = graph->cptr[i + 1] - graph->cptr[i];
      if (graph->nodes[repdof].count > 2 && ccsize > graph->custom_minimal_size) {
        twodim = PETSC_FALSE;
        break;
      }
    }
    PetscCallMPI(MPIU_Allreduce(&twodim, &graph->twodim, 1, MPI_C_BOOL, MPI_LAND, PetscObjectComm((PetscObject)graph->l2gmap)));
    graph->twodimset = PETSC_TRUE;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PCBDDCGraphComputeCC_Private(PCBDDCGraph graph, PetscInt pid, PetscInt *PETSC_RESTRICT queue_tip, PetscInt n_prev, PetscInt *n_added)
{
  PetscInt i, j, n = 0;

  const PetscInt *PETSC_RESTRICT xadj        = graph->xadj;
  const PetscInt *PETSC_RESTRICT adjncy      = graph->adjncy;
  const PetscInt *PETSC_RESTRICT subset_idxs = graph->subset_idxs[pid - 1];
  const PetscInt *PETSC_RESTRICT local_subs  = graph->local_subs;
  const PetscInt                 subset_size = graph->subset_size[pid - 1];

  PCBDDCGraphNode *PETSC_RESTRICT nodes = graph->nodes;

  const PetscBool havecsr  = (PetscBool)(!!xadj);
  const PetscBool havesubs = (PetscBool)(!!graph->n_local_subs);

  PetscFunctionBegin;
  if (havecsr && !havesubs) {
    for (i = -n_prev; i < 0; i++) {
      const PetscInt start_dof = queue_tip[i];

      /* we assume that if a dof has a size 1 adjacency list and the corresponding entry is negative, it is connected to all dofs */
      if (xadj[start_dof + 1] - xadj[start_dof] == 1 && adjncy[xadj[start_dof]] < 0) {
        for (j = 0; j < subset_size; j++) { /* pid \in [1,graph->n_subsets] */
          const PetscInt dof = subset_idxs[j];

          if (!nodes[dof].touched && nodes[dof].subset == pid) {
            nodes[dof].touched = PETSC_TRUE;
            queue_tip[n]       = dof;
            n++;
          }
        }
      } else {
        for (j = xadj[start_dof]; j < xadj[start_dof + 1]; j++) {
          const PetscInt dof = adjncy[j];

          if (!nodes[dof].touched && nodes[dof].subset == pid) {
            nodes[dof].touched = PETSC_TRUE;
            queue_tip[n]       = dof;
            n++;
          }
        }
      }
    }
  } else if (havecsr && havesubs) {
    const PetscInt sid = local_subs[queue_tip[-n_prev]];

    for (i = -n_prev; i < 0; i++) {
      const PetscInt start_dof = queue_tip[i];

      /* we assume that if a dof has a size 1 adjacency list and the corresponding entry is negative, it is connected to all dofs belonging to the local sub */
      if (xadj[start_dof + 1] - xadj[start_dof] == 1 && adjncy[xadj[start_dof]] < 0) {
        for (j = 0; j < subset_size; j++) { /* pid \in [1,graph->n_subsets] */
          const PetscInt dof = subset_idxs[j];

          if (!nodes[dof].touched && nodes[dof].subset == pid && local_subs[dof] == sid) {
            nodes[dof].touched = PETSC_TRUE;
            queue_tip[n]       = dof;
            n++;
          }
        }
      } else {
        for (j = xadj[start_dof]; j < xadj[start_dof + 1]; j++) {
          const PetscInt dof = adjncy[j];

          if (!nodes[dof].touched && nodes[dof].subset == pid && local_subs[dof] == sid) {
            nodes[dof].touched = PETSC_TRUE;
            queue_tip[n]       = dof;
            n++;
          }
        }
      }
    }
  } else if (havesubs) { /* sub info only */
    const PetscInt sid = local_subs[queue_tip[-n_prev]];

    for (j = 0; j < subset_size; j++) { /* pid \in [1,graph->n_subsets] */
      const PetscInt dof = subset_idxs[j];

      if (!nodes[dof].touched && nodes[dof].subset == pid && local_subs[dof] == sid) {
        nodes[dof].touched = PETSC_TRUE;
        queue_tip[n]       = dof;
        n++;
      }
    }
  } else {
    for (j = 0; j < subset_size; j++) { /* pid \in [1,graph->n_subsets] */
      const PetscInt dof = subset_idxs[j];

      if (!nodes[dof].touched && nodes[dof].subset == pid) {
        nodes[dof].touched = PETSC_TRUE;
        queue_tip[n]       = dof;
        n++;
      }
    }
  }
  *n_added = n;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphComputeConnectedComponentsLocal(PCBDDCGraph graph)
{
  PetscInt ncc, cum_queue;

  PetscFunctionBegin;
  PetscCheck(graph->setupcalled, PetscObjectComm((PetscObject)graph->l2gmap), PETSC_ERR_ORDER, "PCBDDCGraphSetUp should be called first");
  /* quiet return if there isn't any local info */
  if (!graph->xadj && !graph->n_local_subs) PetscFunctionReturn(PETSC_SUCCESS);

  /* reset any previous search of connected components */
  for (PetscInt i = 0; i < graph->nvtxs; i++) graph->nodes[i].touched = PETSC_FALSE;
  if (!graph->seq_graph) {
    for (PetscInt i = 0; i < graph->nvtxs; i++) {
      if (graph->nodes[i].special_dof == PCBDDCGRAPH_DIRICHLET_MARK || graph->nodes[i].count < 2) graph->nodes[i].touched = PETSC_TRUE;
    }
  }

  /* begin search for connected components */
  cum_queue = 0;
  ncc       = 0;
  for (PetscInt n = 0; n < graph->n_subsets; n++) {
    const PetscInt *subset_idxs = graph->subset_idxs[n];
    const PetscInt  pid         = n + 1; /* partition labeled by 0 is discarded */

    PetscInt found = 0, prev = 0, first = 0, ncc_pid = 0;

    while (found != graph->subset_size[n]) {
      PetscInt added = 0;

      if (!prev) { /* search for new starting dof */
        while (graph->nodes[subset_idxs[first]].touched) first++;
        graph->nodes[subset_idxs[first]].touched = PETSC_TRUE;
        graph->queue[cum_queue]                  = subset_idxs[first];
        graph->cptr[ncc]                         = cum_queue;
        prev                                     = 1;
        cum_queue++;
        found++;
        ncc_pid++;
        ncc++;
      }
      PetscCall(PCBDDCGraphComputeCC_Private(graph, pid, graph->queue + cum_queue, prev, &added));
      if (!added) {
        graph->subset_ncc[n] = ncc_pid;
        graph->cptr[ncc]     = cum_queue;
      }
      prev = added;
      found += added;
      cum_queue += added;
      if (added && found == graph->subset_size[n]) {
        graph->subset_ncc[n] = ncc_pid;
        graph->cptr[ncc]     = cum_queue;
      }
    }
  }
  graph->ncc          = ncc;
  graph->queue_sorted = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphSetUp(PCBDDCGraph graph, PetscInt custom_minimal_size, IS neumann_is, IS dirichlet_is, PetscInt n_ISForDofs, IS ISForDofs[], IS custom_primal_vertices)
{
  IS                       subset;
  MPI_Comm                 comm;
  PetscHMapPCBDDCGraphNode subsetmaps;
  const PetscInt          *is_indices;
  PetscInt                *queue_global, *nodecount, **nodeneighs, *subset_sizes;
  PetscInt                 i, j, k, nodes_touched, is_size, nvtxs = graph->nvtxs;
  PetscMPIInt              size, rank;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(graph->l2gmap, custom_minimal_size, 2);
  if (neumann_is) {
    PetscValidHeaderSpecific(neumann_is, IS_CLASSID, 3);
    PetscCheckSameComm(graph->l2gmap, 1, neumann_is, 3);
  }
  graph->has_dirichlet = PETSC_FALSE;
  if (dirichlet_is) {
    PetscValidHeaderSpecific(dirichlet_is, IS_CLASSID, 4);
    PetscCheckSameComm(graph->l2gmap, 1, dirichlet_is, 4);
    graph->has_dirichlet = PETSC_TRUE;
  }
  PetscValidLogicalCollectiveInt(graph->l2gmap, n_ISForDofs, 5);
  for (i = 0; i < n_ISForDofs; i++) {
    PetscValidHeaderSpecific(ISForDofs[i], IS_CLASSID, 6);
    PetscCheckSameComm(graph->l2gmap, 1, ISForDofs[i], 6);
  }
  if (custom_primal_vertices) {
    PetscValidHeaderSpecific(custom_primal_vertices, IS_CLASSID, 7);
    PetscCheckSameComm(graph->l2gmap, 1, custom_primal_vertices, 7);
  }
  for (i = 0; i < nvtxs; i++) graph->nodes[i].touched = PETSC_FALSE;

  PetscCall(PetscObjectGetComm((PetscObject)graph->l2gmap, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  /* custom_minimal_size */
  graph->custom_minimal_size = custom_minimal_size;

  /* get node info from l2gmap */
  PetscCall(ISLocalToGlobalMappingGetNodeInfo(graph->l2gmap, NULL, &nodecount, &nodeneighs));

  /* Allocate space for storing the set of neighbours for each node */
  graph->multi_element = PETSC_FALSE;
  for (i = 0; i < nvtxs; i++) {
    graph->nodes[i].count = nodecount[i];
    if (!graph->seq_graph) {
      PetscCall(PetscMalloc1(nodecount[i], &graph->nodes[i].neighbours_set));
      PetscCall(PetscArraycpy(graph->nodes[i].neighbours_set, nodeneighs[i], nodecount[i]));

      if (!graph->multi_element) {
        PetscInt nself;
        for (j = 0, nself = 0; j < graph->nodes[i].count; j++)
          if (graph->nodes[i].neighbours_set[j] == rank) nself++;
        if (nself > 1) graph->multi_element = PETSC_TRUE;
      }
    } else {
      PetscCall(PetscCalloc1(nodecount[i], &graph->nodes[i].neighbours_set));
    }
  }
  PetscCall(ISLocalToGlobalMappingRestoreNodeInfo(graph->l2gmap, NULL, &nodecount, &nodeneighs));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &graph->multi_element, 1, MPI_C_BOOL, MPI_LOR, comm));

  /* compute local groups */
  if (graph->multi_element) {
    const PetscInt *idxs, *indegree;
    IS              is, lis;
    PetscLayout     layout;
    PetscSF         sf, multisf;
    PetscInt        n, nmulti, c, *multi_root_subs, *start;

    PetscCheck(!nvtxs || graph->local_subs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing local subdomain information");

    PetscCall(ISLocalToGlobalMappingGetIndices(graph->l2gmap, &idxs));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nvtxs, idxs, PETSC_USE_POINTER, &is));
    PetscCall(ISRenumber(is, NULL, &n, &lis));
    PetscCall(ISDestroy(&is));

    PetscCall(ISLocalToGlobalMappingRestoreIndices(graph->l2gmap, &idxs));
    PetscCall(ISGetIndices(lis, &idxs));
    PetscCall(PetscLayoutCreate(PETSC_COMM_SELF, &layout));
    PetscCall(PetscLayoutSetSize(layout, n));
    PetscCall(PetscSFCreate(PETSC_COMM_SELF, &sf));
    PetscCall(PetscSFSetGraphLayout(sf, layout, nvtxs, NULL, PETSC_OWN_POINTER, idxs));
    PetscCall(PetscLayoutDestroy(&layout));
    PetscCall(PetscSFGetMultiSF(sf, &multisf));
    PetscCall(PetscSFComputeDegreeBegin(sf, &indegree));
    PetscCall(PetscSFComputeDegreeEnd(sf, &indegree));
    PetscCall(PetscSFGetGraph(multisf, &nmulti, NULL, NULL, NULL));
    PetscCall(PetscMalloc2(nmulti, &multi_root_subs, n + 1, &start));
    start[0] = 0;
    for (i = 0; i < n; i++) start[i + 1] = start[i] + indegree[i];
    PetscCall(PetscSFGatherBegin(sf, MPIU_INT, graph->local_subs, multi_root_subs));
    PetscCall(PetscSFGatherEnd(sf, MPIU_INT, graph->local_subs, multi_root_subs));
    for (i = 0; i < nvtxs; i++) {
      PetscInt gid = idxs[i];

      graph->nodes[i].local_sub = graph->local_subs[i];
      for (j = 0, c = 0; j < graph->nodes[i].count; j++) {
        if (graph->nodes[i].neighbours_set[j] == rank) c++;
      }
      PetscCheck(c == indegree[idxs[i]], PETSC_COMM_SELF, PETSC_ERR_PLIB, "%" PetscInt_FMT " != %" PetscInt_FMT, c, indegree[idxs[i]]);
      PetscCall(PetscMalloc1(c, &graph->nodes[i].local_groups));
      for (j = 0; j < c; j++) graph->nodes[i].local_groups[j] = multi_root_subs[start[gid] + j];
      PetscCall(PetscSortInt(c, graph->nodes[i].local_groups));
      graph->nodes[i].local_groups_count = c;
    }
    PetscCall(PetscFree2(multi_root_subs, start));
    PetscCall(ISRestoreIndices(lis, &idxs));
    PetscCall(ISDestroy(&lis));
    PetscCall(PetscSFDestroy(&sf));
  }

  /*
     Get info for dofs splitting
     User can specify just a subset; an additional field is considered as a complementary field
  */
  for (i = 0, k = 0; i < n_ISForDofs; i++) {
    PetscInt bs;

    PetscCall(ISGetBlockSize(ISForDofs[i], &bs));
    k += bs;
  }
  for (i = 0; i < nvtxs; i++) graph->nodes[i].which_dof = k; /* by default a dof belongs to the complement set */
  for (i = 0, k = 0; i < n_ISForDofs; i++) {
    PetscInt bs;

    PetscCall(ISGetLocalSize(ISForDofs[i], &is_size));
    PetscCall(ISGetBlockSize(ISForDofs[i], &bs));
    PetscCall(ISGetIndices(ISForDofs[i], &is_indices));
    for (j = 0; j < is_size / bs; j++) {
      PetscInt b;

      for (b = 0; b < bs; b++) {
        PetscInt jj = bs * j + b;

        if (is_indices[jj] > -1 && is_indices[jj] < nvtxs) { /* out of bounds indices (if any) are skipped */
          graph->nodes[is_indices[jj]].which_dof = k + b;
        }
      }
    }
    PetscCall(ISRestoreIndices(ISForDofs[i], &is_indices));
    k += bs;
  }

  /* Take into account Neumann nodes */
  if (neumann_is) {
    PetscCall(ISGetLocalSize(neumann_is, &is_size));
    PetscCall(ISGetIndices(neumann_is, &is_indices));
    for (i = 0; i < is_size; i++) {
      if (is_indices[i] > -1 && is_indices[i] < nvtxs) { /* out of bounds indices (if any) are skipped */
        graph->nodes[is_indices[i]].special_dof = PCBDDCGRAPH_NEUMANN_MARK;
      }
    }
    PetscCall(ISRestoreIndices(neumann_is, &is_indices));
  }

  /* Take into account Dirichlet nodes (they overwrite any mark previously set) */
  if (dirichlet_is) {
    PetscCall(ISGetLocalSize(dirichlet_is, &is_size));
    PetscCall(ISGetIndices(dirichlet_is, &is_indices));
    for (i = 0; i < is_size; i++) {
      if (is_indices[i] > -1 && is_indices[i] < nvtxs) { /* out of bounds indices (if any) are skipped */
        if (!graph->seq_graph) {                         /* dirichlet nodes treated as internal */
          graph->nodes[is_indices[i]].touched = PETSC_TRUE;
          graph->nodes[is_indices[i]].subset  = 0;
        }
        graph->nodes[is_indices[i]].special_dof = PCBDDCGRAPH_DIRICHLET_MARK;
      }
    }
    PetscCall(ISRestoreIndices(dirichlet_is, &is_indices));
  }

  /* mark special nodes (if any) -> each will become a single dof equivalence class (i.e. point constraint for BDDC) */
  if (custom_primal_vertices) {
    PetscCall(ISGetLocalSize(custom_primal_vertices, &is_size));
    PetscCall(ISGetIndices(custom_primal_vertices, &is_indices));
    for (i = 0, j = 0; i < is_size; i++) {
      if (is_indices[i] > -1 && is_indices[i] < nvtxs && graph->nodes[is_indices[i]].special_dof != PCBDDCGRAPH_DIRICHLET_MARK) { /* out of bounds indices (if any) are skipped */
        graph->nodes[is_indices[i]].special_dof = PCBDDCGRAPH_SPECIAL_MARK - j;
        j++;
      }
    }
    PetscCall(ISRestoreIndices(custom_primal_vertices, &is_indices));
  }

  /* mark interior nodes as touched and belonging to partition number 0 */
  if (!graph->seq_graph) {
    for (i = 0; i < nvtxs; i++) {
      const PetscInt icount = graph->nodes[i].count;
      if (graph->nodes[i].count < 2) {
        graph->nodes[i].touched = PETSC_TRUE;
        graph->nodes[i].subset  = 0;
      } else {
        if (graph->multi_element) {
          graph->nodes[i].shared = PETSC_FALSE;
          for (k = 0; k < icount; k++)
            if (graph->nodes[i].neighbours_set[k] != rank) {
              graph->nodes[i].shared = PETSC_TRUE;
              break;
            }
        } else {
          graph->nodes[i].shared = PETSC_TRUE;
        }
      }
    }
  } else {
    for (i = 0; i < nvtxs; i++) graph->nodes[i].shared = PETSC_TRUE;
  }

  /* init graph structure and compute default subsets */
  nodes_touched = 0;
  for (i = 0; i < nvtxs; i++)
    if (graph->nodes[i].touched) nodes_touched++;

  /* allocated space for queues */
  if (graph->seq_graph) {
    PetscCall(PetscMalloc2(nvtxs + 1, &graph->cptr, nvtxs, &graph->queue));
  } else {
    PetscInt nused = nvtxs - nodes_touched;
    PetscCall(PetscMalloc2(nused + 1, &graph->cptr, nused, &graph->queue));
  }

  graph->ncc = 0;
  PetscCall(PetscHMapPCBDDCGraphNodeCreate(&subsetmaps));
  PetscCall(PetscCalloc1(nvtxs, &subset_sizes));
  for (i = 0; i < nvtxs; i++) {
    PetscHashIter iter;
    PetscBool     missing;
    PetscInt      subset;

    if (graph->nodes[i].touched) continue;
    graph->nodes[i].touched = PETSC_TRUE;
    PetscCall(PetscHMapPCBDDCGraphNodePut(subsetmaps, &graph->nodes[i], &iter, &missing));
    if (missing) {
      graph->ncc++;
      PetscCall(PetscHMapPCBDDCGraphNodeIterSet(subsetmaps, iter, graph->ncc));
      subset = graph->ncc;
    } else PetscCall(PetscHMapPCBDDCGraphNodeIterGet(subsetmaps, iter, &subset));
    subset_sizes[subset - 1] += 1;
    graph->nodes[i].subset = subset;
  }

  graph->cptr[0] = 0;
  for (i = 0; i < graph->ncc; i++) graph->cptr[i + 1] = graph->cptr[i] + subset_sizes[i];
  for (i = 0; i < graph->ncc; i++) subset_sizes[i] = 0;

  for (i = 0; i < nvtxs; i++) {
    const PetscInt subset = graph->nodes[i].subset - 1;
    if (subset < 0) continue;
    PetscCheck(subset_sizes[subset] + graph->cptr[subset] < graph->cptr[subset + 1], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error for subset %" PetscInt_FMT, subset);
    graph->queue[subset_sizes[subset] + graph->cptr[subset]] = i;
    subset_sizes[subset] += 1;
  }
  for (i = 0; i < graph->ncc; i++) PetscCheck(subset_sizes[i] + graph->cptr[i] == graph->cptr[i + 1], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error for subset %" PetscInt_FMT, i);

  PetscCall(PetscHMapPCBDDCGraphNodeDestroy(&subsetmaps));
  PetscCall(PetscFree(subset_sizes));

  /* set default number of subsets */
  graph->n_subsets = graph->ncc;
  PetscCall(PetscMalloc1(graph->n_subsets, &graph->subset_ncc));
  for (i = 0; i < graph->n_subsets; i++) graph->subset_ncc[i] = 1;

  PetscCall(PetscMalloc1(graph->ncc, &graph->subset_ref_node));
  PetscCall(PetscMalloc1(graph->cptr[graph->ncc], &queue_global));
  PetscCall(PetscMalloc2(graph->ncc, &graph->subset_size, graph->ncc, &graph->subset_idxs));
  if (graph->multi_element) PetscCall(PetscMalloc1(graph->ncc, &graph->gsubset_size));
  else graph->gsubset_size = graph->subset_size;
  PetscCall(ISLocalToGlobalMappingApply(graph->l2gmap, graph->cptr[graph->ncc], graph->queue, queue_global));

  PetscHMapI cnt_unique;

  PetscCall(PetscHMapICreate(&cnt_unique));
  for (j = 0; j < graph->ncc; j++) {
    PetscInt c = 0, ref_node = PETSC_INT_MAX;

    for (k = graph->cptr[j]; k < graph->cptr[j + 1]; k++) {
      ref_node = PetscMin(ref_node, queue_global[k]);
      if (graph->multi_element) {
        PetscBool     missing;
        PetscHashIter iter;

        PetscCall(PetscHMapIPut(cnt_unique, queue_global[k], &iter, &missing));
        if (missing) c++;
      }
    }
    graph->gsubset_size[j]    = c;
    graph->subset_size[j]     = graph->cptr[j + 1] - graph->cptr[j];
    graph->subset_ref_node[j] = ref_node;
    if (graph->multi_element) PetscCall(PetscHMapIClear(cnt_unique));
  }
  PetscCall(PetscHMapIDestroy(&cnt_unique));

  /* save information on subsets (needed when analyzing the connected components) */
  if (graph->ncc) {
    PetscCall(PetscMalloc1(graph->cptr[graph->ncc], &graph->subset_idxs[0]));
    PetscCall(PetscArrayzero(graph->subset_idxs[0], graph->cptr[graph->ncc]));
    for (j = 1; j < graph->ncc; j++) graph->subset_idxs[j] = graph->subset_idxs[j - 1] + graph->subset_size[j - 1];
    PetscCall(PetscArraycpy(graph->subset_idxs[0], graph->queue, graph->cptr[graph->ncc]));
  }

  /* check consistency and create SF to analyze components on the interface between subdomains */
  if (!graph->seq_graph) {
    PetscSF         msf;
    PetscLayout     map;
    const PetscInt *degree;
    PetscInt        nr, nmr, *rdata;
    PetscBool       valid = PETSC_TRUE;
    PetscInt        subset_N;
    IS              subset_n;
    const PetscInt *idxs;

    PetscCall(ISCreateGeneral(comm, graph->n_subsets, graph->subset_ref_node, PETSC_USE_POINTER, &subset));
    PetscCall(ISRenumber(subset, NULL, &subset_N, &subset_n));
    PetscCall(ISDestroy(&subset));

    PetscCall(PetscSFCreate(comm, &graph->interface_ref_sf));
    PetscCall(PetscLayoutCreateFromSizes(comm, PETSC_DECIDE, subset_N, 1, &map));
    PetscCall(ISGetIndices(subset_n, &idxs));
    PetscCall(PetscSFSetGraphLayout(graph->interface_ref_sf, map, graph->n_subsets, NULL, PETSC_OWN_POINTER, idxs));
    PetscCall(ISRestoreIndices(subset_n, &idxs));
    PetscCall(ISDestroy(&subset_n));
    PetscCall(PetscLayoutDestroy(&map));

    PetscCall(PetscSFComputeDegreeBegin(graph->interface_ref_sf, &degree));
    PetscCall(PetscSFComputeDegreeEnd(graph->interface_ref_sf, &degree));
    PetscCall(PetscSFGetMultiSF(graph->interface_ref_sf, &msf));
    PetscCall(PetscSFGetGraph(graph->interface_ref_sf, &nr, NULL, NULL, NULL));
    PetscCall(PetscSFGetGraph(msf, &nmr, NULL, NULL, NULL));
    PetscCall(PetscCalloc1(nmr, &rdata));
    PetscCall(PetscSFGatherBegin(graph->interface_ref_sf, MPIU_INT, graph->gsubset_size, rdata));
    PetscCall(PetscSFGatherEnd(graph->interface_ref_sf, MPIU_INT, graph->gsubset_size, rdata));
    for (PetscInt i = 0, c = 0; i < nr && valid; i++) {
      for (PetscInt j = 0; j < degree[i]; j++) {
        if (rdata[j + c] != rdata[c]) valid = PETSC_FALSE;
      }
      c += degree[i];
    }
    PetscCall(PetscFree(rdata));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &valid, 1, MPI_C_BOOL, MPI_LAND, comm));
    PetscCheck(valid, comm, PETSC_ERR_PLIB, "Initial local subsets are not consistent");

    /* Now create SF with each root extended to gsubset_size roots */
    PetscInt           mss = 0;
    const PetscSFNode *subs_remote;

    PetscCall(PetscSFGetGraph(graph->interface_ref_sf, NULL, NULL, NULL, &subs_remote));
    for (PetscInt i = 0; i < graph->n_subsets; i++) mss = PetscMax(graph->subset_size[i], mss);

    PetscInt nri, nli, *start_rsize, *cum_rsize;
    PetscCall(PetscCalloc1(graph->n_subsets + 1, &start_rsize));
    PetscCall(PetscCalloc1(nr, &graph->interface_ref_rsize));
    PetscCall(PetscMalloc1(nr + 1, &cum_rsize));
    PetscCall(PetscSFReduceBegin(graph->interface_ref_sf, MPIU_INT, graph->gsubset_size, graph->interface_ref_rsize, MPI_REPLACE));
    PetscCall(PetscSFReduceEnd(graph->interface_ref_sf, MPIU_INT, graph->gsubset_size, graph->interface_ref_rsize, MPI_REPLACE));

    nri          = 0;
    cum_rsize[0] = 0;
    for (PetscInt i = 0; i < nr; i++) {
      nri += graph->interface_ref_rsize[i];
      cum_rsize[i + 1] = cum_rsize[i] + graph->interface_ref_rsize[i];
    }
    nli = graph->cptr[graph->ncc];
    PetscCall(PetscSFBcastBegin(graph->interface_ref_sf, MPIU_INT, cum_rsize, start_rsize, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(graph->interface_ref_sf, MPIU_INT, cum_rsize, start_rsize, MPI_REPLACE));
    PetscCall(PetscFree(cum_rsize));

    PetscInt    *ilocal, *queue_global_uniq;
    PetscSFNode *iremote;
    PetscBool   *touched;

    PetscCall(PetscSFCreate(comm, &graph->interface_subset_sf));
    PetscCall(PetscMalloc1(nli, &ilocal));
    PetscCall(PetscMalloc1(nli, &iremote));
    PetscCall(PetscMalloc2(mss, &queue_global_uniq, mss, &touched));
    for (PetscInt i = 0, nli = 0; i < graph->n_subsets; i++) {
      const PetscMPIInt rr                = (PetscMPIInt)subs_remote[i].rank;
      const PetscInt    start             = start_rsize[i];
      const PetscInt    subset_size       = graph->subset_size[i];
      const PetscInt    gsubset_size      = graph->gsubset_size[i];
      const PetscInt   *subset_idxs       = graph->subset_idxs[i];
      const PetscInt   *lsub_queue_global = queue_global + graph->cptr[i];

      k = subset_size;
      PetscCall(PetscArrayzero(touched, subset_size));
      PetscCall(PetscArraycpy(queue_global_uniq, lsub_queue_global, subset_size));
      PetscCall(PetscSortRemoveDupsInt(&k, queue_global_uniq));
      PetscCheck(k == gsubset_size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid local subset %" PetscInt_FMT " size %" PetscInt_FMT " != %" PetscInt_FMT, i, k, gsubset_size);

      PetscInt t = 0, j = 0;
      while (t < subset_size) {
        while (j < subset_size && touched[j]) j++;
        PetscCheck(j < subset_size, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected %" PetscInt_FMT " >= %" PetscInt_FMT, j, subset_size);
        const PetscInt ls = graph->nodes[subset_idxs[j]].local_sub;

        for (k = j; k < subset_size; k++) {
          if (graph->nodes[subset_idxs[k]].local_sub == ls) {
            PetscInt ig;

            PetscCall(PetscFindInt(lsub_queue_global[k], gsubset_size, queue_global_uniq, &ig));
            ilocal[nli]        = subset_idxs[k];
            iremote[nli].rank  = rr;
            iremote[nli].index = start + ig;
            touched[k]         = PETSC_TRUE;
            nli++;
            t++;
          }
        }
      }
    }
    PetscCheck(nli == graph->cptr[graph->ncc], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid ilocal size %" PetscInt_FMT " != %" PetscInt_FMT, nli, graph->cptr[graph->ncc]);
    PetscCall(PetscSFSetGraph(graph->interface_subset_sf, nri, nli, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    PetscCall(PetscFree(start_rsize));
    PetscCall(PetscFree2(queue_global_uniq, touched));
  }
  PetscCall(PetscFree(queue_global));

  /* free workspace */
  graph->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphResetCoords(PCBDDCGraph graph)
{
  PetscFunctionBegin;
  if (!graph) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscFree(graph->coords));
  graph->cdim  = 0;
  graph->cnloc = 0;
  graph->cloc  = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphResetCSR(PCBDDCGraph graph)
{
  PetscFunctionBegin;
  if (!graph) PetscFunctionReturn(PETSC_SUCCESS);
  if (graph->freecsr) {
    PetscCall(PetscFree(graph->xadj));
    PetscCall(PetscFree(graph->adjncy));
  } else {
    graph->xadj   = NULL;
    graph->adjncy = NULL;
  }
  graph->freecsr   = PETSC_FALSE;
  graph->nvtxs_csr = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphReset(PCBDDCGraph graph)
{
  PetscFunctionBegin;
  if (!graph) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(ISLocalToGlobalMappingDestroy(&graph->l2gmap));
  PetscCall(PetscFree(graph->subset_ncc));
  PetscCall(PetscFree(graph->subset_ref_node));
  for (PetscInt i = 0; i < graph->nvtxs; i++) {
    PetscCall(PetscFree(graph->nodes[i].neighbours_set));
    PetscCall(PetscFree(graph->nodes[i].local_groups));
  }
  PetscCall(PetscFree(graph->nodes));
  PetscCall(PetscFree2(graph->cptr, graph->queue));
  if (graph->subset_idxs) PetscCall(PetscFree(graph->subset_idxs[0]));
  PetscCall(PetscFree2(graph->subset_size, graph->subset_idxs));
  if (graph->multi_element) PetscCall(PetscFree(graph->gsubset_size));
  PetscCall(PetscFree(graph->interface_ref_rsize));
  PetscCall(PetscSFDestroy(&graph->interface_subset_sf));
  PetscCall(PetscSFDestroy(&graph->interface_ref_sf));
  PetscCall(ISDestroy(&graph->dirdofs));
  PetscCall(ISDestroy(&graph->dirdofsB));
  if (graph->n_local_subs) PetscCall(PetscFree(graph->local_subs));
  graph->multi_element       = PETSC_FALSE;
  graph->has_dirichlet       = PETSC_FALSE;
  graph->twodimset           = PETSC_FALSE;
  graph->twodim              = PETSC_FALSE;
  graph->nvtxs               = 0;
  graph->nvtxs_global        = 0;
  graph->n_subsets           = 0;
  graph->custom_minimal_size = 1;
  graph->n_local_subs        = 0;
  graph->maxcount            = PETSC_INT_MAX;
  graph->seq_graph           = PETSC_FALSE;
  graph->setupcalled         = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphInit(PCBDDCGraph graph, ISLocalToGlobalMapping l2gmap, PetscInt N, PetscInt maxcount)
{
  PetscInt n;

  PetscFunctionBegin;
  PetscAssertPointer(graph, 1);
  PetscValidHeaderSpecific(l2gmap, IS_LTOGM_CLASSID, 2);
  PetscValidLogicalCollectiveInt(l2gmap, N, 3);
  PetscValidLogicalCollectiveInt(l2gmap, maxcount, 4);
  /* raise an error if already allocated */
  PetscCheck(!graph->nvtxs_global, PetscObjectComm((PetscObject)l2gmap), PETSC_ERR_PLIB, "BDDCGraph already initialized");
  /* set number of vertices */
  PetscCall(PetscObjectReference((PetscObject)l2gmap));
  graph->l2gmap = l2gmap;
  PetscCall(ISLocalToGlobalMappingGetSize(l2gmap, &n));
  graph->nvtxs        = n;
  graph->nvtxs_global = N;
  /* allocate used space */
  PetscCall(PetscCalloc1(graph->nvtxs, &graph->nodes));
  /* use -1 as a default value for which_dof array */
  for (n = 0; n < graph->nvtxs; n++) graph->nodes[n].which_dof = -1;

  /* zeroes workspace for values of ncc */
  graph->subset_ncc      = NULL;
  graph->subset_ref_node = NULL;
  /* maxcount for cc */
  graph->maxcount = maxcount;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphDestroy(PCBDDCGraph *graph)
{
  PetscFunctionBegin;
  PetscCall(PCBDDCGraphResetCSR(*graph));
  PetscCall(PCBDDCGraphResetCoords(*graph));
  PetscCall(PCBDDCGraphReset(*graph));
  PetscCall(PetscFree(*graph));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PCBDDCGraphCreate(PCBDDCGraph *graph)
{
  PCBDDCGraph new_graph;

  PetscFunctionBegin;
  PetscCall(PetscNew(&new_graph));
  new_graph->custom_minimal_size = 1;
  *graph                         = new_graph;
  PetscFunctionReturn(PETSC_SUCCESS);
}
