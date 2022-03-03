#include <petsc/private/petscimpl.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <../src/ksp/pc/impls/bddc/bddcstructs.h>

PetscErrorCode PCBDDCGraphGetDirichletDofsB(PCBDDCGraph graph, IS* dirdofs)
{
  PetscFunctionBegin;
  if (graph->dirdofsB) {
    CHKERRQ(PetscObjectReference((PetscObject)graph->dirdofsB));
  } else if (graph->has_dirichlet) {
    PetscInt i,size;
    PetscInt *dirdofs_idxs;

    size = 0;
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->count[i] && graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK) size++;
    }

    CHKERRQ(PetscMalloc1(size,&dirdofs_idxs));
    size = 0;
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->count[i] && graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK) dirdofs_idxs[size++] = i;
    }
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,size,dirdofs_idxs,PETSC_OWN_POINTER,&graph->dirdofsB));
    CHKERRQ(PetscObjectReference((PetscObject)graph->dirdofsB));
  }
  *dirdofs = graph->dirdofsB;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphGetDirichletDofs(PCBDDCGraph graph, IS* dirdofs)
{
  PetscFunctionBegin;
  if (graph->dirdofs) {
    CHKERRQ(PetscObjectReference((PetscObject)graph->dirdofs));
  } else if (graph->has_dirichlet) {
    PetscInt i,size;
    PetscInt *dirdofs_idxs;

    size = 0;
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK) size++;
    }

    CHKERRQ(PetscMalloc1(size,&dirdofs_idxs));
    size = 0;
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK) dirdofs_idxs[size++] = i;
    }
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)graph->l2gmap),size,dirdofs_idxs,PETSC_OWN_POINTER,&graph->dirdofs));
    CHKERRQ(PetscObjectReference((PetscObject)graph->dirdofs));
  }
  *dirdofs = graph->dirdofs;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphASCIIView(PCBDDCGraph graph, PetscInt verbosity_level, PetscViewer viewer)
{
  PetscInt       i,j,tabs;
  PetscInt*      queue_in_global_numbering;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerASCIIPushSynchronized(viewer));
  CHKERRQ(PetscViewerASCIIGetTab(viewer,&tabs));
  CHKERRQ(PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n"));
  CHKERRQ(PetscViewerFlush(viewer));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"Local BDDC graph for subdomain %04d\n",PetscGlobalRank));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"Number of vertices %d\n",graph->nvtxs));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"Custom minimal size %d\n",graph->custom_minimal_size));
  if (graph->maxcount != PETSC_MAX_INT) {
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"Max count %d\n",graph->maxcount));
  }
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"Topological two dim? %d (set %d)\n",graph->twodim,graph->twodimset));
  if (verbosity_level > 2) {
    for (i=0;i<graph->nvtxs;i++) {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"%d:\n",i));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"   which_dof: %d\n",graph->which_dof[i]));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"   special_dof: %d\n",graph->special_dof[i]));
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"   neighbours: %d\n",graph->count[i]));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
      if (graph->count[i]) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"     set of neighbours:"));
        for (j=0;j<graph->count[i];j++) {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer," %d",graph->neighbours_set[i][j]));
        }
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"\n"));
      }
      CHKERRQ(PetscViewerASCIISetTab(viewer,tabs));
      CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
      if (graph->mirrors) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"   mirrors: %d\n",graph->mirrors[i]));
        if (graph->mirrors[i]) {
          CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"     set of mirrors:"));
          for (j=0;j<graph->mirrors[i];j++) {
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer," %d",graph->mirrors_set[i][j]));
          }
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"\n"));
          CHKERRQ(PetscViewerASCIISetTab(viewer,tabs));
          CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
        }
      }
      if (verbosity_level > 3) {
        if (graph->xadj) {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"   local adj list:"));
          CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
          for (j=graph->xadj[i];j<graph->xadj[i+1];j++) {
            CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer," %d",graph->adjncy[j]));
          }
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"\n"));
          CHKERRQ(PetscViewerASCIISetTab(viewer,tabs));
          CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
        } else {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"   no adj info\n"));
        }
      }
      if (graph->n_local_subs) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"   local sub id: %d\n",graph->local_subs[i]));
      }
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"   interface subset id: %d\n",graph->subset[i]));
      if (graph->subset[i] && graph->subset_ncc) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"   ncc for subset: %d\n",graph->subset_ncc[graph->subset[i]-1]));
      }
    }
  }
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"Total number of connected components %d\n",graph->ncc));
  CHKERRQ(PetscMalloc1(graph->cptr[graph->ncc],&queue_in_global_numbering));
  CHKERRQ(ISLocalToGlobalMappingApply(graph->l2gmap,graph->cptr[graph->ncc],graph->queue,queue_in_global_numbering));
  for (i=0;i<graph->ncc;i++) {
    PetscInt node_num=graph->queue[graph->cptr[i]];
    PetscBool printcc = PETSC_FALSE;
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"  cc %d (size %d, fid %d, neighs:",i,graph->cptr[i+1]-graph->cptr[i],graph->which_dof[node_num]));
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    for (j=0;j<graph->count[node_num];j++) {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer," %d",graph->neighbours_set[node_num][j]));
    }
    if (verbosity_level > 1) {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"):"));
      if (verbosity_level > 2 || graph->twodim || graph->count[node_num] > 1 || (graph->count[node_num] == 1 && graph->special_dof[node_num] == PCBDDCGRAPH_NEUMANN_MARK)) {
        printcc = PETSC_TRUE;
      }
      if (printcc) {
        for (j=graph->cptr[i];j<graph->cptr[i+1];j++) {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer," %d (%d)",graph->queue[j],queue_in_global_numbering[j]));
        }
      }
    } else {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,")"));
    }
    CHKERRQ(PetscViewerASCIISynchronizedPrintf(viewer,"\n"));
    CHKERRQ(PetscViewerASCIISetTab(viewer,tabs));
    CHKERRQ(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  }
  CHKERRQ(PetscFree(queue_in_global_numbering));
  CHKERRQ(PetscViewerFlush(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphRestoreCandidatesIS(PCBDDCGraph graph, PetscInt *n_faces, IS *FacesIS[], PetscInt *n_edges, IS *EdgesIS[], IS *VerticesIS)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (n_faces) {
    if (FacesIS) {
      for (i=0;i<*n_faces;i++) {
        CHKERRQ(ISDestroy(&((*FacesIS)[i])));
      }
      CHKERRQ(PetscFree(*FacesIS));
    }
    *n_faces = 0;
  }
  if (n_edges) {
    if (EdgesIS) {
      for (i=0;i<*n_edges;i++) {
        CHKERRQ(ISDestroy(&((*EdgesIS)[i])));
      }
      CHKERRQ(PetscFree(*EdgesIS));
    }
    *n_edges = 0;
  }
  if (VerticesIS) {
    CHKERRQ(ISDestroy(VerticesIS));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphGetCandidatesIS(PCBDDCGraph graph, PetscInt *n_faces, IS *FacesIS[], PetscInt *n_edges, IS *EdgesIS[], IS *VerticesIS)
{
  IS             *ISForFaces,*ISForEdges,ISForVertices;
  PetscInt       i,nfc,nec,nvc,*idx,*mark;

  PetscFunctionBegin;
  CHKERRQ(PetscCalloc1(graph->ncc,&mark));
  /* loop on ccs to evalute number of faces, edges and vertices */
  nfc = 0;
  nec = 0;
  nvc = 0;
  for (i=0;i<graph->ncc;i++) {
    PetscInt repdof = graph->queue[graph->cptr[i]];
    if (graph->cptr[i+1]-graph->cptr[i] > graph->custom_minimal_size && graph->count[repdof] < graph->maxcount) {
      if (!graph->twodim && graph->count[repdof] == 1 && graph->special_dof[repdof] != PCBDDCGRAPH_NEUMANN_MARK) {
        nfc++;
        mark[i] = 2;
      } else {
        nec++;
        mark[i] = 1;
      }
    } else {
      nvc += graph->cptr[i+1]-graph->cptr[i];
    }
  }

  /* allocate IS arrays for faces, edges. Vertices need a single index set. */
  if (FacesIS) {
    CHKERRQ(PetscMalloc1(nfc,&ISForFaces));
  }
  if (EdgesIS) {
    CHKERRQ(PetscMalloc1(nec,&ISForEdges));
  }
  if (VerticesIS) {
    CHKERRQ(PetscMalloc1(nvc,&idx));
  }

  /* loop on ccs to compute index sets for faces and edges */
  if (!graph->queue_sorted) {
    PetscInt *queue_global;

    CHKERRQ(PetscMalloc1(graph->cptr[graph->ncc],&queue_global));
    CHKERRQ(ISLocalToGlobalMappingApply(graph->l2gmap,graph->cptr[graph->ncc],graph->queue,queue_global));
    for (i=0;i<graph->ncc;i++) {
      CHKERRQ(PetscSortIntWithArray(graph->cptr[i+1]-graph->cptr[i],&queue_global[graph->cptr[i]],&graph->queue[graph->cptr[i]]));
    }
    CHKERRQ(PetscFree(queue_global));
    graph->queue_sorted = PETSC_TRUE;
  }
  nfc = 0;
  nec = 0;
  for (i=0;i<graph->ncc;i++) {
    if (mark[i] == 2) {
      if (FacesIS) {
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],&graph->queue[graph->cptr[i]],PETSC_USE_POINTER,&ISForFaces[nfc]));
      }
      nfc++;
    } else if (mark[i] == 1) {
      if (EdgesIS) {
        CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],&graph->queue[graph->cptr[i]],PETSC_USE_POINTER,&ISForEdges[nec]));
      }
      nec++;
    }
  }

  /* index set for vertices */
  if (VerticesIS) {
    nvc = 0;
    for (i=0;i<graph->ncc;i++) {
      if (!mark[i]) {
        PetscInt j;

        for (j=graph->cptr[i];j<graph->cptr[i+1];j++) {
          idx[nvc]=graph->queue[j];
          nvc++;
        }
      }
    }
    /* sort vertex set (by local ordering) */
    CHKERRQ(PetscSortInt(nvc,idx));
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,nvc,idx,PETSC_OWN_POINTER,&ISForVertices));
  }
  CHKERRQ(PetscFree(mark));

  /* get back info */
  if (n_faces)       *n_faces = nfc;
  if (FacesIS)       *FacesIS = ISForFaces;
  if (n_edges)       *n_edges = nec;
  if (EdgesIS)       *EdgesIS = ISForEdges;
  if (VerticesIS) *VerticesIS = ISForVertices;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphComputeConnectedComponents(PCBDDCGraph graph)
{
  PetscBool      adapt_interface_reduced;
  MPI_Comm       interface_comm;
  PetscMPIInt    size;
  PetscInt       i;
  PetscBT        cornerp;

  PetscFunctionBegin;
  /* compute connected components locally */
  CHKERRQ(PetscObjectGetComm((PetscObject)(graph->l2gmap),&interface_comm));
  CHKERRQ(PCBDDCGraphComputeConnectedComponentsLocal(graph));

  cornerp = NULL;
  if (graph->active_coords) { /* face based corner selection */
    PetscBT   excluded;
    PetscReal *wdist;
    PetscInt  n_neigh,*neigh,*n_shared,**shared;
    PetscInt  maxc, ns;

    CHKERRQ(PetscBTCreate(graph->nvtxs,&cornerp));
    CHKERRQ(ISLocalToGlobalMappingGetInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared));
    for (ns = 1, maxc = 0; ns < n_neigh; ns++) maxc = PetscMax(maxc,n_shared[ns]);
    CHKERRQ(PetscMalloc1(maxc*graph->cdim,&wdist));
    CHKERRQ(PetscBTCreate(maxc,&excluded));

    for (ns = 1; ns < n_neigh; ns++) { /* first proc is self */
      PetscReal *anchor,mdist;
      PetscInt  fst,j,k,d,cdim = graph->cdim,n = n_shared[ns];
      PetscInt  point1,point2,point3;

      /* import coordinates on shared interface */
      CHKERRQ(PetscBTMemzero(n,excluded));
      for (j=0,fst=-1,k=0;j<n;j++) {
        PetscBool skip = PETSC_FALSE;
        for (d=0;d<cdim;d++) {
          PetscReal c = graph->coords[shared[ns][j]*cdim+d];
          skip = (PetscBool)(skip || c == PETSC_MAX_REAL);
          wdist[k++] = c;
        }
        if (skip) {
          CHKERRQ(PetscBTSet(excluded,j));
        } else if (fst == -1) fst = j;
      }
      if (fst == -1) continue;

      /* the dofs are sorted by global numbering, so each rank start from the same id and will detect the same corners from the given set */
      anchor = wdist + fst*cdim;

      /* find the farthest point from the starting one */
      mdist  = -1.0;
      point1 = fst;
      for (j=fst;j<n;j++) {
        PetscReal dist = 0.0;

        if (PetscUnlikely(PetscBTLookup(excluded,j))) continue;
        for (d=0;d<cdim;d++) dist += (wdist[j*cdim+d]-anchor[d])*(wdist[j*cdim+d]-anchor[d]);
        if (dist > mdist) { mdist = dist; point1 = j; }
      }

      /* find the farthest point from point1 */
      anchor = wdist + point1*cdim;
      mdist  = -1.0;
      point2 = point1;
      for (j=fst;j<n;j++) {
        PetscReal dist = 0.0;

        if (PetscUnlikely(PetscBTLookup(excluded,j))) continue;
        for (d=0;d<cdim;d++) dist += (wdist[j*cdim+d]-anchor[d])*(wdist[j*cdim+d]-anchor[d]);
        if (dist > mdist) { mdist = dist; point2 = j; }
      }

      /* find the third point maximizing the triangle area */
      point3 = point2;
      if (cdim > 2) {
        PetscReal a = 0.0;

        for (d=0;d<cdim;d++) a += (wdist[point1*cdim+d]-wdist[point2*cdim+d])*(wdist[point1*cdim+d]-wdist[point2*cdim+d]);
        a = PetscSqrtReal(a);
        mdist = -1.0;
        for (j=fst;j<n;j++) {
          PetscReal area,b = 0.0, c = 0.0,s;

          if (PetscUnlikely(PetscBTLookup(excluded,j))) continue;
          for (d=0;d<cdim;d++) {
            b += (wdist[point1*cdim+d]-wdist[j*cdim+d])*(wdist[point1*cdim+d]-wdist[j*cdim+d]);
            c += (wdist[point2*cdim+d]-wdist[j*cdim+d])*(wdist[point2*cdim+d]-wdist[j*cdim+d]);
          }
          b = PetscSqrtReal(b);
          c = PetscSqrtReal(c);
          s = 0.5*(a+b+c);

          /* Heron's formula, area squared */
          area = s*(s-a)*(s-b)*(s-c);
          if (area > mdist) { mdist = area; point3 = j; }
        }
      }

      CHKERRQ(PetscBTSet(cornerp,shared[ns][point1]));
      CHKERRQ(PetscBTSet(cornerp,shared[ns][point2]));
      CHKERRQ(PetscBTSet(cornerp,shared[ns][point3]));

      /* all dofs having the same coordinates will be primal */
      for (j=fst;j<n;j++) {
        PetscBool same[3] = {PETSC_TRUE,PETSC_TRUE,PETSC_TRUE};

        if (PetscUnlikely(PetscBTLookup(excluded,j))) continue;
        for (d=0;d<cdim;d++) {
          same[0] = (PetscBool)(same[0] && (PetscAbsReal(wdist[j*cdim + d]-wdist[point1*cdim+d]) < PETSC_SMALL));
          same[1] = (PetscBool)(same[1] && (PetscAbsReal(wdist[j*cdim + d]-wdist[point2*cdim+d]) < PETSC_SMALL));
          same[2] = (PetscBool)(same[2] && (PetscAbsReal(wdist[j*cdim + d]-wdist[point3*cdim+d]) < PETSC_SMALL));
        }
        if (same[0] || same[1] || same[2]) {
          CHKERRQ(PetscBTSet(cornerp,shared[ns][j]));
        }
      }
    }
    CHKERRQ(PetscBTDestroy(&excluded));
    CHKERRQ(PetscFree(wdist));
    CHKERRQ(ISLocalToGlobalMappingRestoreInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared));
  }

  /* check consistency of connected components among neighbouring subdomains -> it adapt them in case it is needed */
  CHKERRMPI(MPI_Comm_size(interface_comm,&size));
  adapt_interface_reduced = PETSC_FALSE;
  if (size > 1) {
    PetscInt i;
    PetscBool adapt_interface = cornerp ? PETSC_TRUE : PETSC_FALSE;
    for (i=0;i<graph->n_subsets && !adapt_interface;i++) {
      /* We are not sure that on a given subset of the local interface,
         with two connected components, the latters be the same among sharing subdomains */
      if (graph->subset_ncc[i] > 1) adapt_interface = PETSC_TRUE;
    }
    CHKERRMPI(MPIU_Allreduce(&adapt_interface,&adapt_interface_reduced,1,MPIU_BOOL,MPI_LOR,interface_comm));
  }

  if (graph->n_subsets && adapt_interface_reduced) {
    PetscBT     subset_cc_adapt;
    MPI_Request *send_requests,*recv_requests;
    PetscInt    *send_buffer,*recv_buffer;
    PetscInt    sum_requests,start_of_recv,start_of_send;
    PetscInt    *cum_recv_counts;
    PetscInt    *labels;
    PetscInt    ncc,cum_queue,mss,mns,j,k,s;
    PetscInt    **refine_buffer=NULL,*private_labels = NULL;
    PetscBool   *subset_has_corn,*recv_buffer_bool,*send_buffer_bool;

    CHKERRQ(PetscCalloc1(graph->n_subsets,&subset_has_corn));
    if (cornerp) {
      for (i=0;i<graph->n_subsets;i++) {
        for (j=0;j<graph->subset_size[i];j++) {
          if (PetscBTLookup(cornerp,graph->subset_idxs[i][j])) {
            subset_has_corn[i] = PETSC_TRUE;
            break;
          }
        }
      }
    }
    CHKERRQ(PetscMalloc1(graph->nvtxs,&labels));
    CHKERRQ(PetscArrayzero(labels,graph->nvtxs));
    for (i=0,k=0;i<graph->ncc;i++) {
      PetscInt s = 1;
      for (j=graph->cptr[i];j<graph->cptr[i+1];j++) {
        if (cornerp && PetscBTLookup(cornerp,graph->queue[j])) {
          labels[graph->queue[j]] = k+s;
          s += 1;
        } else {
          labels[graph->queue[j]] = k;
        }
      }
      k += s;
    }

    /* allocate some space */
    CHKERRQ(PetscMalloc1(graph->n_subsets+1,&cum_recv_counts));
    CHKERRQ(PetscArrayzero(cum_recv_counts,graph->n_subsets+1));

    /* first count how many neighbours per connected component I will receive from */
    cum_recv_counts[0] = 0;
    for (i=0;i<graph->n_subsets;i++) cum_recv_counts[i+1] = cum_recv_counts[i]+graph->count[graph->subset_idxs[i][0]];
    CHKERRQ(PetscMalloc1(graph->n_subsets,&send_buffer_bool));
    CHKERRQ(PetscMalloc1(cum_recv_counts[graph->n_subsets],&recv_buffer_bool));
    CHKERRQ(PetscMalloc2(cum_recv_counts[graph->n_subsets],&send_requests,cum_recv_counts[graph->n_subsets],&recv_requests));
    for (i=0;i<cum_recv_counts[graph->n_subsets];i++) {
      send_requests[i] = MPI_REQUEST_NULL;
      recv_requests[i] = MPI_REQUEST_NULL;
    }

    /* exchange with my neighbours the number of my connected components on the subset of interface */
    sum_requests = 0;
    for (i=0;i<graph->n_subsets;i++) {
      send_buffer_bool[i] = (PetscBool)(graph->subset_ncc[i] > 1 || subset_has_corn[i]);
    }
    for (i=0;i<graph->n_subsets;i++) {
      PetscMPIInt neigh,tag;
      PetscInt    count,*neighs;

      count  = graph->count[graph->subset_idxs[i][0]];
      neighs = graph->neighbours_set[graph->subset_idxs[i][0]];
      CHKERRQ(PetscMPIIntCast(2*graph->subset_ref_node[i],&tag));
      for (k=0;k<count;k++) {

        CHKERRQ(PetscMPIIntCast(neighs[k],&neigh));
        CHKERRMPI(MPI_Isend(send_buffer_bool + i,           1,MPIU_BOOL,neigh,tag,interface_comm,&send_requests[sum_requests]));
        CHKERRMPI(MPI_Irecv(recv_buffer_bool + sum_requests,1,MPIU_BOOL,neigh,tag,interface_comm,&recv_requests[sum_requests]));
        sum_requests++;
      }
    }
    CHKERRMPI(MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE));
    CHKERRMPI(MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE));

    /* determine the subsets I have to adapt (those having more than 1 cc) */
    CHKERRQ(PetscBTCreate(graph->n_subsets,&subset_cc_adapt));
    CHKERRQ(PetscBTMemzero(graph->n_subsets,subset_cc_adapt));
    for (i=0;i<graph->n_subsets;i++) {
      if (graph->subset_ncc[i] > 1 || subset_has_corn[i]) {
        CHKERRQ(PetscBTSet(subset_cc_adapt,i));
        continue;
      }
      for (j=cum_recv_counts[i];j<cum_recv_counts[i+1];j++) {
         if (recv_buffer_bool[j]) {
          CHKERRQ(PetscBTSet(subset_cc_adapt,i));
          break;
        }
      }
    }
    CHKERRQ(PetscFree(send_buffer_bool));
    CHKERRQ(PetscFree(recv_buffer_bool));
    CHKERRQ(PetscFree(subset_has_corn));

    /* determine send/recv buffers sizes */
    j = 0;
    mss = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (PetscBTLookup(subset_cc_adapt,i)) {
        j  += graph->subset_size[i];
        mss = PetscMax(graph->subset_size[i],mss);
      }
    }
    k = 0;
    mns = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (PetscBTLookup(subset_cc_adapt,i)) {
        k  += (cum_recv_counts[i+1]-cum_recv_counts[i])*graph->subset_size[i];
        mns = PetscMax(cum_recv_counts[i+1]-cum_recv_counts[i],mns);
      }
    }
    CHKERRQ(PetscMalloc2(j,&send_buffer,k,&recv_buffer));

    /* fill send buffer (order matters: subset_idxs ordered by global ordering) */
    j = 0;
    for (i=0;i<graph->n_subsets;i++)
      if (PetscBTLookup(subset_cc_adapt,i))
        for (k=0;k<graph->subset_size[i];k++)
          send_buffer[j++] = labels[graph->subset_idxs[i][k]];

    /* now exchange the data */
    start_of_recv = 0;
    start_of_send = 0;
    sum_requests  = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (PetscBTLookup(subset_cc_adapt,i)) {
        PetscMPIInt neigh,tag;
        PetscInt    size_of_send = graph->subset_size[i];

        j    = graph->subset_idxs[i][0];
        CHKERRQ(PetscMPIIntCast(2*graph->subset_ref_node[i]+1,&tag));
        for (k=0;k<graph->count[j];k++) {
          CHKERRQ(PetscMPIIntCast(graph->neighbours_set[j][k],&neigh));
          CHKERRMPI(MPI_Isend(&send_buffer[start_of_send],size_of_send,MPIU_INT,neigh,tag,interface_comm,&send_requests[sum_requests]));
          CHKERRMPI(MPI_Irecv(&recv_buffer[start_of_recv],size_of_send,MPIU_INT,neigh,tag,interface_comm,&recv_requests[sum_requests]));
          start_of_recv += size_of_send;
          sum_requests++;
        }
        start_of_send += size_of_send;
      }
    }
    CHKERRMPI(MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE));

    /* refine connected components */
    start_of_recv = 0;
    /* allocate some temporary space */
    if (mss) {
      CHKERRQ(PetscMalloc1(mss,&refine_buffer));
      CHKERRQ(PetscMalloc2(mss*(mns+1),&refine_buffer[0],mss,&private_labels));
    }
    ncc = 0;
    cum_queue = 0;
    graph->cptr[0] = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (PetscBTLookup(subset_cc_adapt,i)) {
        PetscInt subset_counter = 0;
        PetscInt sharingprocs = cum_recv_counts[i+1]-cum_recv_counts[i]+1; /* count myself */
        PetscInt buffer_size = graph->subset_size[i];

        /* compute pointers */
        for (j=1;j<buffer_size;j++) refine_buffer[j] = refine_buffer[j-1] + sharingprocs;
        /* analyze contributions from subdomains that share the i-th subset
           The structure of refine_buffer is suitable to find intersections of ccs among sharingprocs.
           supposing the current subset is shared by 3 processes and has dimension 5 with global dofs 0,1,2,3,4 (local 0,4,3,1,2)
           sharing procs connected components:
             neigh 0: [0 1 4], [2 3], labels [4,7]  (2 connected components)
             neigh 1: [0 1], [2 3 4], labels [3 2]  (2 connected components)
             neigh 2: [0 4], [1], [2 3], labels [1 5 6] (3 connected components)
           refine_buffer will be filled as:
             [ 4, 3, 1;
               4, 2, 1;
               7, 2, 6;
               4, 3, 5;
               7, 2, 6; ];
           The connected components in local ordering are [0], [1], [2 3], [4] */
        /* fill temp_buffer */
        for (k=0;k<buffer_size;k++) refine_buffer[k][0] = labels[graph->subset_idxs[i][k]];
        for (j=0;j<sharingprocs-1;j++) {
          for (k=0;k<buffer_size;k++) refine_buffer[k][j+1] = recv_buffer[start_of_recv+k];
          start_of_recv += buffer_size;
        }
        CHKERRQ(PetscArrayzero(private_labels,buffer_size));
        for (j=0;j<buffer_size;j++) {
          if (!private_labels[j]) { /* found a new cc  */
            PetscBool same_set;

            graph->cptr[ncc] = cum_queue;
            ncc++;
            subset_counter++;
            private_labels[j] = subset_counter;
            graph->queue[cum_queue++] = graph->subset_idxs[i][j];
            for (k=j+1;k<buffer_size;k++) { /* check for other nodes in new cc */
              same_set = PETSC_TRUE;
              for (s=0;s<sharingprocs;s++) {
                if (refine_buffer[j][s] != refine_buffer[k][s]) {
                  same_set = PETSC_FALSE;
                  break;
                }
              }
              if (same_set) {
                private_labels[k] = subset_counter;
                graph->queue[cum_queue++] = graph->subset_idxs[i][k];
              }
            }
          }
        }
        graph->cptr[ncc]     = cum_queue;
        graph->subset_ncc[i] = subset_counter;
        graph->queue_sorted  = PETSC_FALSE;
      } else { /* this subset does not need to be adapted */
        CHKERRQ(PetscArraycpy(graph->queue+cum_queue,graph->subset_idxs[i],graph->subset_size[i]));
        ncc++;
        cum_queue += graph->subset_size[i];
        graph->cptr[ncc] = cum_queue;
      }
    }
    graph->cptr[ncc] = cum_queue;
    graph->ncc       = ncc;
    if (mss) {
      CHKERRQ(PetscFree2(refine_buffer[0],private_labels));
      CHKERRQ(PetscFree(refine_buffer));
    }
    CHKERRQ(PetscFree(labels));
    CHKERRMPI(MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE));
    CHKERRQ(PetscFree2(send_requests,recv_requests));
    CHKERRQ(PetscFree2(send_buffer,recv_buffer));
    CHKERRQ(PetscFree(cum_recv_counts));
    CHKERRQ(PetscBTDestroy(&subset_cc_adapt));
  }
  CHKERRQ(PetscBTDestroy(&cornerp));

  /* Determine if we are in 2D or 3D */
  if (!graph->twodimset) {
    PetscBool twodim = PETSC_TRUE;
    for (i=0;i<graph->ncc;i++) {
      PetscInt repdof = graph->queue[graph->cptr[i]];
      PetscInt ccsize = graph->cptr[i+1]-graph->cptr[i];
      if (graph->count[repdof] > 1 && ccsize > graph->custom_minimal_size) {
        twodim = PETSC_FALSE;
        break;
      }
    }
    CHKERRMPI(MPIU_Allreduce(&twodim,&graph->twodim,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)graph->l2gmap)));
    graph->twodimset = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PCBDDCGraphComputeCC_Private(PCBDDCGraph graph,PetscInt pid,PetscInt* queue_tip,PetscInt n_prev,PetscInt* n_added)
{
  PetscInt       i,j,n;
  PetscInt       *xadj = graph->xadj,*adjncy = graph->adjncy;
  PetscBT        touched = graph->touched;
  PetscBool      havecsr = (PetscBool)(!!xadj);
  PetscBool      havesubs = (PetscBool)(!!graph->n_local_subs);

  PetscFunctionBegin;
  n = 0;
  if (havecsr && !havesubs) {
    for (i=-n_prev;i<0;i++) {
      PetscInt start_dof = queue_tip[i];
      /* we assume that if a dof has a size 1 adjacency list and the corresponding entry is negative, it is connected to all dofs */
      if (xadj[start_dof+1]-xadj[start_dof] == 1 && adjncy[xadj[start_dof]] < 0) {
        for (j=0;j<graph->subset_size[pid-1];j++) { /* pid \in [1,graph->n_subsets] */
          PetscInt dof = graph->subset_idxs[pid-1][j];
          if (!PetscBTLookup(touched,dof) && graph->subset[dof] == pid) {
            CHKERRQ(PetscBTSet(touched,dof));
            queue_tip[n] = dof;
            n++;
          }
        }
      } else {
        for (j=xadj[start_dof];j<xadj[start_dof+1];j++) {
          PetscInt dof = adjncy[j];
          if (!PetscBTLookup(touched,dof) && graph->subset[dof] == pid) {
            CHKERRQ(PetscBTSet(touched,dof));
            queue_tip[n] = dof;
            n++;
          }
        }
      }
    }
  } else if (havecsr && havesubs) {
    PetscInt sid = graph->local_subs[queue_tip[-n_prev]];
    for (i=-n_prev;i<0;i++) {
      PetscInt start_dof = queue_tip[i];
      /* we assume that if a dof has a size 1 adjacency list and the corresponding entry is negative, it is connected to all dofs belonging to the local sub */
      if (xadj[start_dof+1]-xadj[start_dof] == 1 && adjncy[xadj[start_dof]] < 0) {
        for (j=0;j<graph->subset_size[pid-1];j++) { /* pid \in [1,graph->n_subsets] */
          PetscInt dof = graph->subset_idxs[pid-1][j];
          if (!PetscBTLookup(touched,dof) && graph->subset[dof] == pid && graph->local_subs[dof] == sid) {
            CHKERRQ(PetscBTSet(touched,dof));
            queue_tip[n] = dof;
            n++;
          }
        }
      } else {
        for (j=xadj[start_dof];j<xadj[start_dof+1];j++) {
          PetscInt dof = adjncy[j];
          if (!PetscBTLookup(touched,dof) && graph->subset[dof] == pid && graph->local_subs[dof] == sid) {
            CHKERRQ(PetscBTSet(touched,dof));
            queue_tip[n] = dof;
            n++;
          }
        }
      }
    }
  } else if (havesubs) { /* sub info only */
    PetscInt sid = graph->local_subs[queue_tip[-n_prev]];
    for (j=0;j<graph->subset_size[pid-1];j++) { /* pid \in [1,graph->n_subsets] */
      PetscInt dof = graph->subset_idxs[pid-1][j];
      if (!PetscBTLookup(touched,dof) && graph->subset[dof] == pid && graph->local_subs[dof] == sid) {
        CHKERRQ(PetscBTSet(touched,dof));
        queue_tip[n] = dof;
        n++;
      }
    }
  } else {
    for (j=0;j<graph->subset_size[pid-1];j++) { /* pid \in [1,graph->n_subsets] */
      PetscInt dof = graph->subset_idxs[pid-1][j];
      if (!PetscBTLookup(touched,dof) && graph->subset[dof] == pid) {
        CHKERRQ(PetscBTSet(touched,dof));
        queue_tip[n] = dof;
        n++;
      }
    }
  }
  *n_added = n;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphComputeConnectedComponentsLocal(PCBDDCGraph graph)
{
  PetscInt       ncc,cum_queue,n;
  PetscMPIInt    commsize;

  PetscFunctionBegin;
  PetscCheck(graph->setupcalled,PetscObjectComm((PetscObject)graph->l2gmap),PETSC_ERR_ORDER,"PCBDDCGraphSetUp should be called first");
  /* quiet return if there isn't any local info */
  if (!graph->xadj && !graph->n_local_subs) {
    PetscFunctionReturn(0);
  }

  /* reset any previous search of connected components */
  CHKERRQ(PetscBTMemzero(graph->nvtxs,graph->touched));
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)graph->l2gmap),&commsize));
  if (commsize > graph->commsizelimit) {
    PetscInt i;
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK || !graph->count[i]) {
        CHKERRQ(PetscBTSet(graph->touched,i));
      }
    }
  }

  /* begin search for connected components */
  cum_queue = 0;
  ncc = 0;
  for (n=0;n<graph->n_subsets;n++) {
    PetscInt pid = n+1;  /* partition labeled by 0 is discarded */
    PetscInt found = 0,prev = 0,first = 0,ncc_pid = 0;
    while (found != graph->subset_size[n]) {
      PetscInt added = 0;
      if (!prev) { /* search for new starting dof */
        while (PetscBTLookup(graph->touched,graph->subset_idxs[n][first])) first++;
        CHKERRQ(PetscBTSet(graph->touched,graph->subset_idxs[n][first]));
        graph->queue[cum_queue] = graph->subset_idxs[n][first];
        graph->cptr[ncc] = cum_queue;
        prev = 1;
        cum_queue++;
        found++;
        ncc_pid++;
        ncc++;
      }
      CHKERRQ(PCBDDCGraphComputeCC_Private(graph,pid,graph->queue + cum_queue,prev,&added));
      if (!added) {
        graph->subset_ncc[n] = ncc_pid;
        graph->cptr[ncc] = cum_queue;
      }
      prev = added;
      found += added;
      cum_queue += added;
      if (added && found == graph->subset_size[n]) {
        graph->subset_ncc[n] = ncc_pid;
        graph->cptr[ncc] = cum_queue;
      }
    }
  }
  graph->ncc = ncc;
  graph->queue_sorted = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphSetUp(PCBDDCGraph graph, PetscInt custom_minimal_size, IS neumann_is, IS dirichlet_is, PetscInt n_ISForDofs, IS ISForDofs[], IS custom_primal_vertices)
{
  IS             subset,subset_n;
  MPI_Comm       comm;
  const PetscInt *is_indices;
  PetscInt       n_neigh,*neigh,*n_shared,**shared,*queue_global;
  PetscInt       i,j,k,s,total_counts,nodes_touched,is_size;
  PetscMPIInt    commsize;
  PetscBool      same_set,mirrors_found;

  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(graph->l2gmap,custom_minimal_size,2);
  if (neumann_is) {
    PetscValidHeaderSpecific(neumann_is,IS_CLASSID,3);
    PetscCheckSameComm(graph->l2gmap,1,neumann_is,3);
  }
  graph->has_dirichlet = PETSC_FALSE;
  if (dirichlet_is) {
    PetscValidHeaderSpecific(dirichlet_is,IS_CLASSID,4);
    PetscCheckSameComm(graph->l2gmap,1,dirichlet_is,4);
    graph->has_dirichlet = PETSC_TRUE;
  }
  PetscValidLogicalCollectiveInt(graph->l2gmap,n_ISForDofs,5);
  for (i=0;i<n_ISForDofs;i++) {
    PetscValidHeaderSpecific(ISForDofs[i],IS_CLASSID,6);
    PetscCheckSameComm(graph->l2gmap,1,ISForDofs[i],6);
  }
  if (custom_primal_vertices) {
    PetscValidHeaderSpecific(custom_primal_vertices,IS_CLASSID,7);
    PetscCheckSameComm(graph->l2gmap,1,custom_primal_vertices,7);
  }
  CHKERRQ(PetscObjectGetComm((PetscObject)(graph->l2gmap),&comm));
  CHKERRMPI(MPI_Comm_size(comm,&commsize));

  /* custom_minimal_size */
  graph->custom_minimal_size = custom_minimal_size;
  /* get info l2gmap and allocate work vectors  */
  CHKERRQ(ISLocalToGlobalMappingGetInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared));
  /* check if we have any local periodic nodes (periodic BCs) */
  mirrors_found = PETSC_FALSE;
  if (graph->nvtxs && n_neigh) {
    for (i=0; i<n_shared[0]; i++) graph->count[shared[0][i]] += 1;
    for (i=0; i<n_shared[0]; i++) {
      if (graph->count[shared[0][i]] > 1) {
        mirrors_found = PETSC_TRUE;
        break;
      }
    }
  }
  /* compute local mirrors (if any) */
  if (mirrors_found) {
    IS       to,from;
    PetscInt *local_indices,*global_indices;

    CHKERRQ(ISCreateStride(PETSC_COMM_SELF,graph->nvtxs,0,1,&to));
    CHKERRQ(ISLocalToGlobalMappingApplyIS(graph->l2gmap,to,&from));
    /* get arrays of local and global indices */
    CHKERRQ(PetscMalloc1(graph->nvtxs,&local_indices));
    CHKERRQ(ISGetIndices(to,(const PetscInt**)&is_indices));
    CHKERRQ(PetscArraycpy(local_indices,is_indices,graph->nvtxs));
    CHKERRQ(ISRestoreIndices(to,(const PetscInt**)&is_indices));
    CHKERRQ(PetscMalloc1(graph->nvtxs,&global_indices));
    CHKERRQ(ISGetIndices(from,(const PetscInt**)&is_indices));
    CHKERRQ(PetscArraycpy(global_indices,is_indices,graph->nvtxs));
    CHKERRQ(ISRestoreIndices(from,(const PetscInt**)&is_indices));
    /* allocate space for mirrors */
    CHKERRQ(PetscMalloc2(graph->nvtxs,&graph->mirrors,graph->nvtxs,&graph->mirrors_set));
    CHKERRQ(PetscArrayzero(graph->mirrors,graph->nvtxs));
    graph->mirrors_set[0] = NULL;

    k=0;
    for (i=0;i<n_shared[0];i++) {
      j=shared[0][i];
      if (graph->count[j] > 1) {
        graph->mirrors[j]++;
        k++;
      }
    }
    /* allocate space for set of mirrors */
    CHKERRQ(PetscMalloc1(k,&graph->mirrors_set[0]));
    for (i=1;i<graph->nvtxs;i++)
      graph->mirrors_set[i]=graph->mirrors_set[i-1]+graph->mirrors[i-1];

    /* fill arrays */
    CHKERRQ(PetscArrayzero(graph->mirrors,graph->nvtxs));
    for (j=0;j<n_shared[0];j++) {
      i=shared[0][j];
      if (graph->count[i] > 1)
        graph->mirrors_set[i][graph->mirrors[i]++]=global_indices[i];
    }
    CHKERRQ(PetscSortIntWithArray(graph->nvtxs,global_indices,local_indices));
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->mirrors[i] > 0) {
        CHKERRQ(PetscFindInt(graph->mirrors_set[i][0],graph->nvtxs,global_indices,&k));
        j = global_indices[k];
        while (k > 0 && global_indices[k-1] == j) k--;
        for (j=0;j<graph->mirrors[i];j++) {
          graph->mirrors_set[i][j]=local_indices[k+j];
        }
        CHKERRQ(PetscSortInt(graph->mirrors[i],graph->mirrors_set[i]));
      }
    }
    CHKERRQ(PetscFree(local_indices));
    CHKERRQ(PetscFree(global_indices));
    CHKERRQ(ISDestroy(&to));
    CHKERRQ(ISDestroy(&from));
  }
  CHKERRQ(PetscArrayzero(graph->count,graph->nvtxs));

  /* Count total number of neigh per node */
  k = 0;
  for (i=1;i<n_neigh;i++) {
    k += n_shared[i];
    for (j=0;j<n_shared[i];j++) {
      graph->count[shared[i][j]] += 1;
    }
  }
  /* Allocate space for storing the set of neighbours for each node */
  if (graph->nvtxs) {
    CHKERRQ(PetscMalloc1(k,&graph->neighbours_set[0]));
  }
  for (i=1;i<graph->nvtxs;i++) { /* dont count myself */
    graph->neighbours_set[i]=graph->neighbours_set[i-1]+graph->count[i-1];
  }
  /* Get information for sharing subdomains */
  CHKERRQ(PetscArrayzero(graph->count,graph->nvtxs));
  for (i=1;i<n_neigh;i++) { /* dont count myself */
    s = n_shared[i];
    for (j=0;j<s;j++) {
      k = shared[i][j];
      graph->neighbours_set[k][graph->count[k]] = neigh[i];
      graph->count[k] += 1;
    }
  }
  /* sort set of sharing subdomains */
  for (i=0;i<graph->nvtxs;i++) {
    CHKERRQ(PetscSortRemoveDupsInt(&graph->count[i],graph->neighbours_set[i]));
  }
  /* free memory allocated by ISLocalToGlobalMappingGetInfo */
  CHKERRQ(ISLocalToGlobalMappingRestoreInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared));

  /*
     Get info for dofs splitting
     User can specify just a subset; an additional field is considered as a complementary field
  */
  for (i=0,k=0;i<n_ISForDofs;i++) {
    PetscInt bs;

    CHKERRQ(ISGetBlockSize(ISForDofs[i],&bs));
    k   += bs;
  }
  for (i=0;i<graph->nvtxs;i++) graph->which_dof[i] = k; /* by default a dof belongs to the complement set */
  for (i=0,k=0;i<n_ISForDofs;i++) {
    PetscInt bs;

    CHKERRQ(ISGetLocalSize(ISForDofs[i],&is_size));
    CHKERRQ(ISGetBlockSize(ISForDofs[i],&bs));
    CHKERRQ(ISGetIndices(ISForDofs[i],(const PetscInt**)&is_indices));
    for (j=0;j<is_size/bs;j++) {
      PetscInt b;

      for (b=0;b<bs;b++) {
        PetscInt jj = bs*j + b;

        if (is_indices[jj] > -1 && is_indices[jj] < graph->nvtxs) { /* out of bounds indices (if any) are skipped */
          graph->which_dof[is_indices[jj]] = k+b;
        }
      }
    }
    CHKERRQ(ISRestoreIndices(ISForDofs[i],(const PetscInt**)&is_indices));
    k   += bs;
  }

  /* Take into account Neumann nodes */
  if (neumann_is) {
    CHKERRQ(ISGetLocalSize(neumann_is,&is_size));
    CHKERRQ(ISGetIndices(neumann_is,(const PetscInt**)&is_indices));
    for (i=0;i<is_size;i++) {
      if (is_indices[i] > -1 && is_indices[i] < graph->nvtxs) { /* out of bounds indices (if any) are skipped */
        graph->special_dof[is_indices[i]] = PCBDDCGRAPH_NEUMANN_MARK;
      }
    }
    CHKERRQ(ISRestoreIndices(neumann_is,(const PetscInt**)&is_indices));
  }
  /* Take into account Dirichlet nodes (they overwrite any neumann boundary mark previously set) */
  if (dirichlet_is) {
    CHKERRQ(ISGetLocalSize(dirichlet_is,&is_size));
    CHKERRQ(ISGetIndices(dirichlet_is,(const PetscInt**)&is_indices));
    for (i=0;i<is_size;i++) {
      if (is_indices[i] > -1 && is_indices[i] < graph->nvtxs) { /* out of bounds indices (if any) are skipped */
        if (commsize > graph->commsizelimit) { /* dirichlet nodes treated as internal */
          CHKERRQ(PetscBTSet(graph->touched,is_indices[i]));
          graph->subset[is_indices[i]] = 0;
        }
        graph->special_dof[is_indices[i]] = PCBDDCGRAPH_DIRICHLET_MARK;
      }
    }
    CHKERRQ(ISRestoreIndices(dirichlet_is,(const PetscInt**)&is_indices));
  }
  /* mark local periodic nodes (if any) and adapt CSR graph (if any) */
  if (graph->mirrors) {
    for (i=0;i<graph->nvtxs;i++)
      if (graph->mirrors[i])
        graph->special_dof[i] = PCBDDCGRAPH_LOCAL_PERIODIC_MARK;

    if (graph->xadj) {
      PetscInt *new_xadj,*new_adjncy;
      /* sort CSR graph */
      for (i=0;i<graph->nvtxs;i++) {
        CHKERRQ(PetscSortInt(graph->xadj[i+1]-graph->xadj[i],&graph->adjncy[graph->xadj[i]]));
      }
      /* adapt local CSR graph in case of local periodicity */
      k = 0;
      for (i=0;i<graph->nvtxs;i++)
        for (j=graph->xadj[i];j<graph->xadj[i+1];j++)
          k += graph->mirrors[graph->adjncy[j]];

      CHKERRQ(PetscMalloc1(graph->nvtxs+1,&new_xadj));
      CHKERRQ(PetscMalloc1(k+graph->xadj[graph->nvtxs],&new_adjncy));
      new_xadj[0] = 0;
      for (i=0;i<graph->nvtxs;i++) {
        k = graph->xadj[i+1]-graph->xadj[i];
        CHKERRQ(PetscArraycpy(&new_adjncy[new_xadj[i]],&graph->adjncy[graph->xadj[i]],k));
        new_xadj[i+1] = new_xadj[i]+k;
        for (j=graph->xadj[i];j<graph->xadj[i+1];j++) {
          k = graph->mirrors[graph->adjncy[j]];
          CHKERRQ(PetscArraycpy(&new_adjncy[new_xadj[i+1]],graph->mirrors_set[graph->adjncy[j]],k));
          new_xadj[i+1] += k;
        }
        k = new_xadj[i+1]-new_xadj[i];
        CHKERRQ(PetscSortRemoveDupsInt(&k,&new_adjncy[new_xadj[i]]));
        new_xadj[i+1] = new_xadj[i]+k;
      }
      /* set new CSR into graph */
      CHKERRQ(PetscFree(graph->xadj));
      CHKERRQ(PetscFree(graph->adjncy));
      graph->xadj = new_xadj;
      graph->adjncy = new_adjncy;
    }
  }

  /* mark special nodes (if any) -> each will become a single node equivalence class */
  if (custom_primal_vertices) {
    CHKERRQ(ISGetLocalSize(custom_primal_vertices,&is_size));
    CHKERRQ(ISGetIndices(custom_primal_vertices,(const PetscInt**)&is_indices));
    for (i=0,j=0;i<is_size;i++) {
      if (is_indices[i] > -1 && is_indices[i] < graph->nvtxs  && graph->special_dof[is_indices[i]] != PCBDDCGRAPH_DIRICHLET_MARK) { /* out of bounds indices (if any) are skipped */
        graph->special_dof[is_indices[i]] = PCBDDCGRAPH_SPECIAL_MARK-j;
        j++;
      }
    }
    CHKERRQ(ISRestoreIndices(custom_primal_vertices,(const PetscInt**)&is_indices));
  }

  /* mark interior nodes (if commsize > graph->commsizelimit) as touched and belonging to partition number 0 */
  if (commsize > graph->commsizelimit) {
    for (i=0;i<graph->nvtxs;i++) {
      if (!graph->count[i]) {
        CHKERRQ(PetscBTSet(graph->touched,i));
        graph->subset[i] = 0;
      }
    }
  }

  /* init graph structure and compute default subsets */
  nodes_touched = 0;
  for (i=0;i<graph->nvtxs;i++) {
    if (PetscBTLookup(graph->touched,i)) {
      nodes_touched++;
    }
  }
  i = 0;
  graph->ncc = 0;
  total_counts = 0;

  /* allocated space for queues */
  if (commsize == graph->commsizelimit) {
    CHKERRQ(PetscMalloc2(graph->nvtxs+1,&graph->cptr,graph->nvtxs,&graph->queue));
  } else {
    PetscInt nused = graph->nvtxs - nodes_touched;
    CHKERRQ(PetscMalloc2(nused+1,&graph->cptr,nused,&graph->queue));
  }

  while (nodes_touched<graph->nvtxs) {
    /*  find first untouched node in local ordering */
    while (PetscBTLookup(graph->touched,i)) i++;
    CHKERRQ(PetscBTSet(graph->touched,i));
    graph->subset[i] = graph->ncc+1;
    graph->cptr[graph->ncc] = total_counts;
    graph->queue[total_counts] = i;
    total_counts++;
    nodes_touched++;
    /* now find all other nodes having the same set of sharing subdomains */
    for (j=i+1;j<graph->nvtxs;j++) {
      /* check for same number of sharing subdomains, dof number and same special mark */
      if (!PetscBTLookup(graph->touched,j) && graph->count[i] == graph->count[j] && graph->which_dof[i] == graph->which_dof[j] && graph->special_dof[i] == graph->special_dof[j]) {
        /* check for same set of sharing subdomains */
        same_set = PETSC_TRUE;
        for (k=0;k<graph->count[j];k++) {
          if (graph->neighbours_set[i][k] != graph->neighbours_set[j][k]) {
            same_set = PETSC_FALSE;
          }
        }
        /* I have found a friend of mine */
        if (same_set) {
          CHKERRQ(PetscBTSet(graph->touched,j));
          graph->subset[j] = graph->ncc+1;
          nodes_touched++;
          graph->queue[total_counts] = j;
          total_counts++;
        }
      }
    }
    graph->ncc++;
  }
  /* set default number of subsets (at this point no info on csr and/or local_subs has been taken into account, so n_subsets = ncc */
  graph->n_subsets = graph->ncc;
  CHKERRQ(PetscMalloc1(graph->n_subsets,&graph->subset_ncc));
  for (i=0;i<graph->n_subsets;i++) {
    graph->subset_ncc[i] = 1;
  }
  /* final pointer */
  graph->cptr[graph->ncc] = total_counts;

  /* For consistency reasons (among neighbours), I need to sort (by global ordering) each connected component */
  /* Get a reference node (min index in global ordering) for each subset for tagging messages */
  CHKERRQ(PetscMalloc1(graph->ncc,&graph->subset_ref_node));
  CHKERRQ(PetscMalloc1(graph->cptr[graph->ncc],&queue_global));
  CHKERRQ(ISLocalToGlobalMappingApply(graph->l2gmap,graph->cptr[graph->ncc],graph->queue,queue_global));
  for (j=0;j<graph->ncc;j++) {
    CHKERRQ(PetscSortIntWithArray(graph->cptr[j+1]-graph->cptr[j],&queue_global[graph->cptr[j]],&graph->queue[graph->cptr[j]]));
    graph->subset_ref_node[j] = graph->queue[graph->cptr[j]];
  }
  CHKERRQ(PetscFree(queue_global));
  graph->queue_sorted = PETSC_TRUE;

  /* save information on subsets (needed when analyzing the connected components) */
  if (graph->ncc) {
    CHKERRQ(PetscMalloc2(graph->ncc,&graph->subset_size,graph->ncc,&graph->subset_idxs));
    CHKERRQ(PetscMalloc1(graph->cptr[graph->ncc],&graph->subset_idxs[0]));
    CHKERRQ(PetscArrayzero(graph->subset_idxs[0],graph->cptr[graph->ncc]));
    for (j=1;j<graph->ncc;j++) {
      graph->subset_size[j-1] = graph->cptr[j] - graph->cptr[j-1];
      graph->subset_idxs[j] = graph->subset_idxs[j-1] + graph->subset_size[j-1];
    }
    graph->subset_size[graph->ncc-1] = graph->cptr[graph->ncc] - graph->cptr[graph->ncc-1];
    CHKERRQ(PetscArraycpy(graph->subset_idxs[0],graph->queue,graph->cptr[graph->ncc]));
  }

  /* renumber reference nodes */
  CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)(graph->l2gmap)),graph->ncc,graph->subset_ref_node,PETSC_COPY_VALUES,&subset_n));
  CHKERRQ(ISLocalToGlobalMappingApplyIS(graph->l2gmap,subset_n,&subset));
  CHKERRQ(ISDestroy(&subset_n));
  CHKERRQ(ISRenumber(subset,NULL,NULL,&subset_n));
  CHKERRQ(ISDestroy(&subset));
  CHKERRQ(ISGetLocalSize(subset_n,&k));
  PetscCheckFalse(k != graph->ncc,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid size of new subset! %D != %D",k,graph->ncc);
  CHKERRQ(ISGetIndices(subset_n,&is_indices));
  CHKERRQ(PetscArraycpy(graph->subset_ref_node,is_indices,graph->ncc));
  CHKERRQ(ISRestoreIndices(subset_n,&is_indices));
  CHKERRQ(ISDestroy(&subset_n));

  /* free workspace */
  graph->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphResetCoords(PCBDDCGraph graph)
{
  PetscFunctionBegin;
  if (!graph) PetscFunctionReturn(0);
  CHKERRQ(PetscFree(graph->coords));
  graph->cdim  = 0;
  graph->cnloc = 0;
  graph->cloc  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphResetCSR(PCBDDCGraph graph)
{
  PetscFunctionBegin;
  if (!graph) PetscFunctionReturn(0);
  if (graph->freecsr) {
    CHKERRQ(PetscFree(graph->xadj));
    CHKERRQ(PetscFree(graph->adjncy));
  } else {
    graph->xadj = NULL;
    graph->adjncy = NULL;
  }
  graph->freecsr = PETSC_FALSE;
  graph->nvtxs_csr = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphReset(PCBDDCGraph graph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!graph) PetscFunctionReturn(0);
  CHKERRQ(ISLocalToGlobalMappingDestroy(&graph->l2gmap));
  CHKERRQ(PetscFree(graph->subset_ncc));
  CHKERRQ(PetscFree(graph->subset_ref_node));
  if (graph->nvtxs) {
    CHKERRQ(PetscFree(graph->neighbours_set[0]));
  }
  CHKERRQ(PetscBTDestroy(&graph->touched));
  ierr = PetscFree5(graph->count,
                    graph->neighbours_set,
                    graph->subset,
                    graph->which_dof,
                    graph->special_dof);CHKERRQ(ierr);
  CHKERRQ(PetscFree2(graph->cptr,graph->queue));
  if (graph->mirrors) {
    CHKERRQ(PetscFree(graph->mirrors_set[0]));
  }
  CHKERRQ(PetscFree2(graph->mirrors,graph->mirrors_set));
  if (graph->subset_idxs) {
    CHKERRQ(PetscFree(graph->subset_idxs[0]));
  }
  CHKERRQ(PetscFree2(graph->subset_size,graph->subset_idxs));
  CHKERRQ(ISDestroy(&graph->dirdofs));
  CHKERRQ(ISDestroy(&graph->dirdofsB));
  if (graph->n_local_subs) {
    CHKERRQ(PetscFree(graph->local_subs));
  }
  graph->has_dirichlet       = PETSC_FALSE;
  graph->twodimset           = PETSC_FALSE;
  graph->twodim              = PETSC_FALSE;
  graph->nvtxs               = 0;
  graph->nvtxs_global        = 0;
  graph->n_subsets           = 0;
  graph->custom_minimal_size = 1;
  graph->n_local_subs        = 0;
  graph->maxcount            = PETSC_MAX_INT;
  graph->setupcalled         = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphInit(PCBDDCGraph graph, ISLocalToGlobalMapping l2gmap, PetscInt N, PetscInt maxcount)
{
  PetscInt       n;

  PetscFunctionBegin;
  PetscValidPointer(graph,1);
  PetscValidHeaderSpecific(l2gmap,IS_LTOGM_CLASSID,2);
  PetscValidLogicalCollectiveInt(l2gmap,N,3);
  PetscValidLogicalCollectiveInt(l2gmap,maxcount,4);
  /* raise an error if already allocated */
  PetscCheck(!graph->nvtxs_global,PetscObjectComm((PetscObject)l2gmap),PETSC_ERR_PLIB,"BDDCGraph already initialized");
  /* set number of vertices */
  CHKERRQ(PetscObjectReference((PetscObject)l2gmap));
  graph->l2gmap = l2gmap;
  CHKERRQ(ISLocalToGlobalMappingGetSize(l2gmap,&n));
  graph->nvtxs = n;
  graph->nvtxs_global = N;
  /* allocate used space */
  CHKERRQ(PetscBTCreate(graph->nvtxs,&graph->touched));
  CHKERRQ(PetscMalloc5(graph->nvtxs,&graph->count,graph->nvtxs,&graph->neighbours_set,graph->nvtxs,&graph->subset,graph->nvtxs,&graph->which_dof,graph->nvtxs,&graph->special_dof));
  /* zeroes memory */
  CHKERRQ(PetscArrayzero(graph->count,graph->nvtxs));
  CHKERRQ(PetscArrayzero(graph->subset,graph->nvtxs));
  /* use -1 as a default value for which_dof array */
  for (n=0;n<graph->nvtxs;n++) graph->which_dof[n] = -1;
  CHKERRQ(PetscArrayzero(graph->special_dof,graph->nvtxs));
  /* zeroes first pointer to neighbour set */
  if (graph->nvtxs) {
    graph->neighbours_set[0] = NULL;
  }
  /* zeroes workspace for values of ncc */
  graph->subset_ncc = NULL;
  graph->subset_ref_node = NULL;
  /* maxcount for cc */
  graph->maxcount = maxcount;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphDestroy(PCBDDCGraph* graph)
{
  PetscFunctionBegin;
  CHKERRQ(PCBDDCGraphResetCSR(*graph));
  CHKERRQ(PCBDDCGraphResetCoords(*graph));
  CHKERRQ(PCBDDCGraphReset(*graph));
  CHKERRQ(PetscFree(*graph));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphCreate(PCBDDCGraph *graph)
{
  PCBDDCGraph    new_graph;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&new_graph));
  new_graph->custom_minimal_size = 1;
  new_graph->commsizelimit = 1;
  *graph = new_graph;
  PetscFunctionReturn(0);
}
