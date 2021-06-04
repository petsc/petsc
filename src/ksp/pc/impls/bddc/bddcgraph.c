#include <petsc/private/petscimpl.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <../src/ksp/pc/impls/bddc/bddcstructs.h>

PetscErrorCode PCBDDCGraphGetDirichletDofsB(PCBDDCGraph graph, IS* dirdofs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (graph->dirdofsB) {
    ierr = PetscObjectReference((PetscObject)graph->dirdofsB);CHKERRQ(ierr);
  } else if (graph->has_dirichlet) {
    PetscInt i,size;
    PetscInt *dirdofs_idxs;

    size = 0;
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->count[i] && graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK) size++;
    }

    ierr = PetscMalloc1(size,&dirdofs_idxs);CHKERRQ(ierr);
    size = 0;
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->count[i] && graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK) dirdofs_idxs[size++] = i;
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,size,dirdofs_idxs,PETSC_OWN_POINTER,&graph->dirdofsB);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)graph->dirdofsB);CHKERRQ(ierr);
  }
  *dirdofs = graph->dirdofsB;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphGetDirichletDofs(PCBDDCGraph graph, IS* dirdofs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (graph->dirdofs) {
    ierr = PetscObjectReference((PetscObject)graph->dirdofs);CHKERRQ(ierr);
  } else if (graph->has_dirichlet) {
    PetscInt i,size;
    PetscInt *dirdofs_idxs;

    size = 0;
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK) size++;
    }

    ierr = PetscMalloc1(size,&dirdofs_idxs);CHKERRQ(ierr);
    size = 0;
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK) dirdofs_idxs[size++] = i;
    }
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)graph->l2gmap),size,dirdofs_idxs,PETSC_OWN_POINTER,&graph->dirdofs);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)graph->dirdofs);CHKERRQ(ierr);
  }
  *dirdofs = graph->dirdofs;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphASCIIView(PCBDDCGraph graph, PetscInt verbosity_level, PetscViewer viewer)
{
  PetscInt       i,j,tabs;
  PetscInt*      queue_in_global_numbering;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIGetTab(viewer,&tabs);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"--------------------------------------------------\n");CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Local BDDC graph for subdomain %04d\n",PetscGlobalRank);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Number of vertices %d\n",graph->nvtxs);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Custom minimal size %d\n",graph->custom_minimal_size);CHKERRQ(ierr);
  if (graph->maxcount != PETSC_MAX_INT) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Max count %d\n",graph->maxcount);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Topological two dim? %d (set %d)\n",graph->twodim,graph->twodimset);CHKERRQ(ierr);
  if (verbosity_level > 2) {
    for (i=0;i<graph->nvtxs;i++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"%d:\n",i);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"   which_dof: %d\n",graph->which_dof[i]);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"   special_dof: %d\n",graph->special_dof[i]);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"   neighbours: %d\n",graph->count[i]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
      if (graph->count[i]) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"     set of neighbours:");CHKERRQ(ierr);
        for (j=0;j<graph->count[i];j++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d",graph->neighbours_set[i][j]);CHKERRQ(ierr);
        }
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISetTab(viewer,tabs);CHKERRQ(ierr);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
      if (graph->mirrors) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"   mirrors: %d\n",graph->mirrors[i]);CHKERRQ(ierr);
        if (graph->mirrors[i]) {
          ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"     set of mirrors:");CHKERRQ(ierr);
          for (j=0;j<graph->mirrors[i];j++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d",graph->mirrors_set[i][j]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIISetTab(viewer,tabs);CHKERRQ(ierr);
          ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
        }
      }
      if (verbosity_level > 3) {
        if (graph->xadj) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"   local adj list:");CHKERRQ(ierr);
          ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
          for (j=graph->xadj[i];j<graph->xadj[i+1];j++) {
            ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d",graph->adjncy[j]);CHKERRQ(ierr);
          }
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
          ierr = PetscViewerASCIISetTab(viewer,tabs);CHKERRQ(ierr);
          ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer,"   no adj info\n");CHKERRQ(ierr);
        }
      }
      if (graph->n_local_subs) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"   local sub id: %d\n",graph->local_subs[i]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"   interface subset id: %d\n",graph->subset[i]);CHKERRQ(ierr);
      if (graph->subset[i] && graph->subset_ncc) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"   ncc for subset: %d\n",graph->subset_ncc[graph->subset[i]-1]);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Total number of connected components %d\n",graph->ncc);CHKERRQ(ierr);
  ierr = PetscMalloc1(graph->cptr[graph->ncc],&queue_in_global_numbering);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(graph->l2gmap,graph->cptr[graph->ncc],graph->queue,queue_in_global_numbering);CHKERRQ(ierr);
  for (i=0;i<graph->ncc;i++) {
    PetscInt node_num=graph->queue[graph->cptr[i]];
    PetscBool printcc = PETSC_FALSE;
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  cc %d (size %d, fid %d, neighs:",i,graph->cptr[i+1]-graph->cptr[i],graph->which_dof[node_num]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (j=0;j<graph->count[node_num];j++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d",graph->neighbours_set[node_num][j]);CHKERRQ(ierr);
    }
    if (verbosity_level > 1) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"):");CHKERRQ(ierr);
      if (verbosity_level > 2 || graph->twodim || graph->count[node_num] > 1 || (graph->count[node_num] == 1 && graph->special_dof[node_num] == PCBDDCGRAPH_NEUMANN_MARK)) {
        printcc = PETSC_TRUE;
      }
      if (printcc) {
        for (j=graph->cptr[i];j<graph->cptr[i+1];j++) {
          ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d (%d)",graph->queue[j],queue_in_global_numbering[j]);CHKERRQ(ierr);
        }
      }
    } else {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,")");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISetTab(viewer,tabs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscFree(queue_in_global_numbering);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphRestoreCandidatesIS(PCBDDCGraph graph, PetscInt *n_faces, IS *FacesIS[], PetscInt *n_edges, IS *EdgesIS[], IS *VerticesIS)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (n_faces) {
    if (FacesIS) {
      for (i=0;i<*n_faces;i++) {
        ierr = ISDestroy(&((*FacesIS)[i]));CHKERRQ(ierr);
      }
      ierr = PetscFree(*FacesIS);CHKERRQ(ierr);
    }
    *n_faces = 0;
  }
  if (n_edges) {
    if (EdgesIS) {
      for (i=0;i<*n_edges;i++) {
        ierr = ISDestroy(&((*EdgesIS)[i]));CHKERRQ(ierr);
      }
      ierr = PetscFree(*EdgesIS);CHKERRQ(ierr);
    }
    *n_edges = 0;
  }
  if (VerticesIS) {
    ierr = ISDestroy(VerticesIS);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphGetCandidatesIS(PCBDDCGraph graph, PetscInt *n_faces, IS *FacesIS[], PetscInt *n_edges, IS *EdgesIS[], IS *VerticesIS)
{
  IS             *ISForFaces,*ISForEdges,ISForVertices;
  PetscInt       i,nfc,nec,nvc,*idx,*mark;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCalloc1(graph->ncc,&mark);CHKERRQ(ierr);
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
    ierr = PetscMalloc1(nfc,&ISForFaces);CHKERRQ(ierr);
  }
  if (EdgesIS) {
    ierr = PetscMalloc1(nec,&ISForEdges);CHKERRQ(ierr);
  }
  if (VerticesIS) {
    ierr = PetscMalloc1(nvc,&idx);CHKERRQ(ierr);
  }

  /* loop on ccs to compute index sets for faces and edges */
  if (!graph->queue_sorted) {
    PetscInt *queue_global;

    ierr = PetscMalloc1(graph->cptr[graph->ncc],&queue_global);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApply(graph->l2gmap,graph->cptr[graph->ncc],graph->queue,queue_global);CHKERRQ(ierr);
    for (i=0;i<graph->ncc;i++) {
      ierr = PetscSortIntWithArray(graph->cptr[i+1]-graph->cptr[i],&queue_global[graph->cptr[i]],&graph->queue[graph->cptr[i]]);CHKERRQ(ierr);
    }
    ierr = PetscFree(queue_global);CHKERRQ(ierr);
    graph->queue_sorted = PETSC_TRUE;
  }
  nfc = 0;
  nec = 0;
  for (i=0;i<graph->ncc;i++) {
    if (mark[i] == 2) {
      if (FacesIS) {
        ierr = ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],&graph->queue[graph->cptr[i]],PETSC_USE_POINTER,&ISForFaces[nfc]);CHKERRQ(ierr);
      }
      nfc++;
    } else if (mark[i] == 1) {
      if (EdgesIS) {
        ierr = ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],&graph->queue[graph->cptr[i]],PETSC_USE_POINTER,&ISForEdges[nec]);CHKERRQ(ierr);
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
    ierr = PetscSortInt(nvc,idx);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,nvc,idx,PETSC_OWN_POINTER,&ISForVertices);CHKERRQ(ierr);
  }
  ierr = PetscFree(mark);CHKERRQ(ierr);

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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* compute connected components locally */
  ierr = PetscObjectGetComm((PetscObject)(graph->l2gmap),&interface_comm);CHKERRQ(ierr);
  ierr = PCBDDCGraphComputeConnectedComponentsLocal(graph);CHKERRQ(ierr);

  cornerp = NULL;
  if (graph->active_coords) { /* face based corner selection */
    PetscBT   excluded;
    PetscReal *wdist;
    PetscInt  n_neigh,*neigh,*n_shared,**shared;
    PetscInt  maxc, ns;

    ierr = PetscBTCreate(graph->nvtxs,&cornerp);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingGetInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
    for (ns = 1, maxc = 0; ns < n_neigh; ns++) maxc = PetscMax(maxc,n_shared[ns]);
    ierr = PetscMalloc1(maxc*graph->cdim,&wdist);CHKERRQ(ierr);
    ierr = PetscBTCreate(maxc,&excluded);CHKERRQ(ierr);

    for (ns = 1; ns < n_neigh; ns++) { /* first proc is self */
      PetscReal *anchor,mdist;
      PetscInt  fst,j,k,d,cdim = graph->cdim,n = n_shared[ns];
      PetscInt  point1,point2,point3;

      /* import coordinates on shared interface */
      ierr = PetscBTMemzero(n,excluded);CHKERRQ(ierr);
      for (j=0,fst=-1,k=0;j<n;j++) {
        PetscBool skip = PETSC_FALSE;
        for (d=0;d<cdim;d++) {
          PetscReal c = graph->coords[shared[ns][j]*cdim+d];
          skip = (PetscBool)(skip || c == PETSC_MAX_REAL);
          wdist[k++] = c;
        }
        if (skip) {
          ierr = PetscBTSet(excluded,j);CHKERRQ(ierr);
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

      ierr = PetscBTSet(cornerp,shared[ns][point1]);CHKERRQ(ierr);
      ierr = PetscBTSet(cornerp,shared[ns][point2]);CHKERRQ(ierr);
      ierr = PetscBTSet(cornerp,shared[ns][point3]);CHKERRQ(ierr);

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
          ierr = PetscBTSet(cornerp,shared[ns][j]);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscBTDestroy(&excluded);CHKERRQ(ierr);
    ierr = PetscFree(wdist);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingRestoreInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
  }

  /* check consistency of connected components among neighbouring subdomains -> it adapt them in case it is needed */
  ierr = MPI_Comm_size(interface_comm,&size);CHKERRMPI(ierr);
  adapt_interface_reduced = PETSC_FALSE;
  if (size > 1) {
    PetscInt i;
    PetscBool adapt_interface = cornerp ? PETSC_TRUE : PETSC_FALSE;
    for (i=0;i<graph->n_subsets && !adapt_interface;i++) {
      /* We are not sure that on a given subset of the local interface,
         with two connected components, the latters be the same among sharing subdomains */
      if (graph->subset_ncc[i] > 1) adapt_interface = PETSC_TRUE;
    }
    ierr = MPIU_Allreduce(&adapt_interface,&adapt_interface_reduced,1,MPIU_BOOL,MPI_LOR,interface_comm);CHKERRMPI(ierr);
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

    ierr = PetscCalloc1(graph->n_subsets,&subset_has_corn);CHKERRQ(ierr);
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
    ierr = PetscMalloc1(graph->nvtxs,&labels);CHKERRQ(ierr);
    ierr = PetscArrayzero(labels,graph->nvtxs);CHKERRQ(ierr);
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
    ierr = PetscMalloc1(graph->n_subsets+1,&cum_recv_counts);CHKERRQ(ierr);
    ierr = PetscArrayzero(cum_recv_counts,graph->n_subsets+1);CHKERRQ(ierr);

    /* first count how many neighbours per connected component I will receive from */
    cum_recv_counts[0] = 0;
    for (i=0;i<graph->n_subsets;i++) cum_recv_counts[i+1] = cum_recv_counts[i]+graph->count[graph->subset_idxs[i][0]];
    ierr = PetscMalloc1(graph->n_subsets,&send_buffer_bool);CHKERRQ(ierr);
    ierr = PetscMalloc1(cum_recv_counts[graph->n_subsets],&recv_buffer_bool);CHKERRQ(ierr);
    ierr = PetscMalloc2(cum_recv_counts[graph->n_subsets],&send_requests,cum_recv_counts[graph->n_subsets],&recv_requests);CHKERRQ(ierr);
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
      ierr   = PetscMPIIntCast(2*graph->subset_ref_node[i],&tag);CHKERRQ(ierr);
      for (k=0;k<count;k++) {

        ierr = PetscMPIIntCast(neighs[k],&neigh);CHKERRQ(ierr);
        ierr = MPI_Isend(send_buffer_bool + i,           1,MPIU_BOOL,neigh,tag,interface_comm,&send_requests[sum_requests]);CHKERRMPI(ierr);
        ierr = MPI_Irecv(recv_buffer_bool + sum_requests,1,MPIU_BOOL,neigh,tag,interface_comm,&recv_requests[sum_requests]);CHKERRMPI(ierr);
        sum_requests++;
      }
    }
    ierr = MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);
    ierr = MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);

    /* determine the subsets I have to adapt (those having more than 1 cc) */
    ierr = PetscBTCreate(graph->n_subsets,&subset_cc_adapt);CHKERRQ(ierr);
    ierr = PetscBTMemzero(graph->n_subsets,subset_cc_adapt);CHKERRQ(ierr);
    for (i=0;i<graph->n_subsets;i++) {
      if (graph->subset_ncc[i] > 1 || subset_has_corn[i]) {
        ierr = PetscBTSet(subset_cc_adapt,i);CHKERRQ(ierr);
        continue;
      }
      for (j=cum_recv_counts[i];j<cum_recv_counts[i+1];j++) {
         if (recv_buffer_bool[j]) {
          ierr = PetscBTSet(subset_cc_adapt,i);CHKERRQ(ierr);
          break;
        }
      }
    }
    ierr = PetscFree(send_buffer_bool);CHKERRQ(ierr);
    ierr = PetscFree(recv_buffer_bool);CHKERRQ(ierr);
    ierr = PetscFree(subset_has_corn);CHKERRQ(ierr);

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
    ierr = PetscMalloc2(j,&send_buffer,k,&recv_buffer);CHKERRQ(ierr);

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
        ierr = PetscMPIIntCast(2*graph->subset_ref_node[i]+1,&tag);CHKERRQ(ierr);
        for (k=0;k<graph->count[j];k++) {
          ierr = PetscMPIIntCast(graph->neighbours_set[j][k],&neigh);CHKERRQ(ierr);
          ierr = MPI_Isend(&send_buffer[start_of_send],size_of_send,MPIU_INT,neigh,tag,interface_comm,&send_requests[sum_requests]);CHKERRMPI(ierr);
          ierr = MPI_Irecv(&recv_buffer[start_of_recv],size_of_send,MPIU_INT,neigh,tag,interface_comm,&recv_requests[sum_requests]);CHKERRMPI(ierr);
          start_of_recv += size_of_send;
          sum_requests++;
        }
        start_of_send += size_of_send;
      }
    }
    ierr = MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);

    /* refine connected components */
    start_of_recv = 0;
    /* allocate some temporary space */
    if (mss) {
      ierr = PetscMalloc1(mss,&refine_buffer);CHKERRQ(ierr);
      ierr = PetscMalloc2(mss*(mns+1),&refine_buffer[0],mss,&private_labels);CHKERRQ(ierr);
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
        ierr = PetscArrayzero(private_labels,buffer_size);CHKERRQ(ierr);
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
        ierr = PetscArraycpy(graph->queue+cum_queue,graph->subset_idxs[i],graph->subset_size[i]);CHKERRQ(ierr);
        ncc++;
        cum_queue += graph->subset_size[i];
        graph->cptr[ncc] = cum_queue;
      }
    }
    graph->cptr[ncc] = cum_queue;
    graph->ncc       = ncc;
    if (mss) {
      ierr = PetscFree2(refine_buffer[0],private_labels);CHKERRQ(ierr);
      ierr = PetscFree(refine_buffer);CHKERRQ(ierr);
    }
    ierr = PetscFree(labels);CHKERRQ(ierr);
    ierr = MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE);CHKERRMPI(ierr);
    ierr = PetscFree2(send_requests,recv_requests);CHKERRQ(ierr);
    ierr = PetscFree2(send_buffer,recv_buffer);CHKERRQ(ierr);
    ierr = PetscFree(cum_recv_counts);CHKERRQ(ierr);
    ierr = PetscBTDestroy(&subset_cc_adapt);CHKERRQ(ierr);
  }
  ierr = PetscBTDestroy(&cornerp);CHKERRQ(ierr);

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
    ierr = MPIU_Allreduce(&twodim,&graph->twodim,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)graph->l2gmap));CHKERRMPI(ierr);
    graph->twodimset = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PCBDDCGraphComputeCC_Private(PCBDDCGraph graph,PetscInt pid,PetscInt* queue_tip,PetscInt n_prev,PetscInt* n_added)
{
  PetscInt       i,j,n;
  PetscInt       *xadj = graph->xadj,*adjncy = graph->adjncy;
  PetscBT        touched = graph->touched;
  PetscBool      havecsr = (PetscBool)(!!xadj);
  PetscBool      havesubs = (PetscBool)(!!graph->n_local_subs);
  PetscErrorCode ierr;

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
            ierr = PetscBTSet(touched,dof);CHKERRQ(ierr);
            queue_tip[n] = dof;
            n++;
          }
        }
      } else {
        for (j=xadj[start_dof];j<xadj[start_dof+1];j++) {
          PetscInt dof = adjncy[j];
          if (!PetscBTLookup(touched,dof) && graph->subset[dof] == pid) {
            ierr = PetscBTSet(touched,dof);CHKERRQ(ierr);
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
            ierr = PetscBTSet(touched,dof);CHKERRQ(ierr);
            queue_tip[n] = dof;
            n++;
          }
        }
      } else {
        for (j=xadj[start_dof];j<xadj[start_dof+1];j++) {
          PetscInt dof = adjncy[j];
          if (!PetscBTLookup(touched,dof) && graph->subset[dof] == pid && graph->local_subs[dof] == sid) {
            ierr = PetscBTSet(touched,dof);CHKERRQ(ierr);
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
        ierr = PetscBTSet(touched,dof);CHKERRQ(ierr);
        queue_tip[n] = dof;
        n++;
      }
    }
  } else {
    for (j=0;j<graph->subset_size[pid-1];j++) { /* pid \in [1,graph->n_subsets] */
      PetscInt dof = graph->subset_idxs[pid-1][j];
      if (!PetscBTLookup(touched,dof) && graph->subset[dof] == pid) {
        ierr = PetscBTSet(touched,dof);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!graph->setupcalled) SETERRQ(PetscObjectComm((PetscObject)graph->l2gmap),PETSC_ERR_ORDER,"PCBDDCGraphSetUp should be called first");
  /* quiet return if there isn't any local info */
  if (!graph->xadj && !graph->n_local_subs) {
    PetscFunctionReturn(0);
  }

  /* reset any previous search of connected components */
  ierr = PetscBTMemzero(graph->nvtxs,graph->touched);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)graph->l2gmap),&commsize);CHKERRMPI(ierr);
  if (commsize > graph->commsizelimit) {
    PetscInt i;
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK || !graph->count[i]) {
        ierr = PetscBTSet(graph->touched,i);CHKERRQ(ierr);
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
        ierr = PetscBTSet(graph->touched,graph->subset_idxs[n][first]);CHKERRQ(ierr);
        graph->queue[cum_queue] = graph->subset_idxs[n][first];
        graph->cptr[ncc] = cum_queue;
        prev = 1;
        cum_queue++;
        found++;
        ncc_pid++;
        ncc++;
      }
      ierr = PCBDDCGraphComputeCC_Private(graph,pid,graph->queue + cum_queue,prev,&added);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

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
  ierr = PetscObjectGetComm((PetscObject)(graph->l2gmap),&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&commsize);CHKERRMPI(ierr);

  /* custom_minimal_size */
  graph->custom_minimal_size = custom_minimal_size;
  /* get info l2gmap and allocate work vectors  */
  ierr = ISLocalToGlobalMappingGetInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
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

    ierr = ISCreateStride(PETSC_COMM_SELF,graph->nvtxs,0,1,&to);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApplyIS(graph->l2gmap,to,&from);CHKERRQ(ierr);
    /* get arrays of local and global indices */
    ierr = PetscMalloc1(graph->nvtxs,&local_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(to,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = PetscArraycpy(local_indices,is_indices,graph->nvtxs);CHKERRQ(ierr);
    ierr = ISRestoreIndices(to,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = PetscMalloc1(graph->nvtxs,&global_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(from,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = PetscArraycpy(global_indices,is_indices,graph->nvtxs);CHKERRQ(ierr);
    ierr = ISRestoreIndices(from,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    /* allocate space for mirrors */
    ierr = PetscMalloc2(graph->nvtxs,&graph->mirrors,graph->nvtxs,&graph->mirrors_set);CHKERRQ(ierr);
    ierr = PetscArrayzero(graph->mirrors,graph->nvtxs);CHKERRQ(ierr);
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
    ierr = PetscMalloc1(k,&graph->mirrors_set[0]);CHKERRQ(ierr);
    for (i=1;i<graph->nvtxs;i++)
      graph->mirrors_set[i]=graph->mirrors_set[i-1]+graph->mirrors[i-1];

    /* fill arrays */
    ierr = PetscArrayzero(graph->mirrors,graph->nvtxs);CHKERRQ(ierr);
    for (j=0;j<n_shared[0];j++) {
      i=shared[0][j];
      if (graph->count[i] > 1)
        graph->mirrors_set[i][graph->mirrors[i]++]=global_indices[i];
    }
    ierr = PetscSortIntWithArray(graph->nvtxs,global_indices,local_indices);CHKERRQ(ierr);
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->mirrors[i] > 0) {
        ierr = PetscFindInt(graph->mirrors_set[i][0],graph->nvtxs,global_indices,&k);CHKERRQ(ierr);
        j = global_indices[k];
        while (k > 0 && global_indices[k-1] == j) k--;
        for (j=0;j<graph->mirrors[i];j++) {
          graph->mirrors_set[i][j]=local_indices[k+j];
        }
        ierr = PetscSortInt(graph->mirrors[i],graph->mirrors_set[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(local_indices);CHKERRQ(ierr);
    ierr = PetscFree(global_indices);CHKERRQ(ierr);
    ierr = ISDestroy(&to);CHKERRQ(ierr);
    ierr = ISDestroy(&from);CHKERRQ(ierr);
  }
  ierr = PetscArrayzero(graph->count,graph->nvtxs);CHKERRQ(ierr);

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
    ierr = PetscMalloc1(k,&graph->neighbours_set[0]);CHKERRQ(ierr);
  }
  for (i=1;i<graph->nvtxs;i++) { /* dont count myself */
    graph->neighbours_set[i]=graph->neighbours_set[i-1]+graph->count[i-1];
  }
  /* Get information for sharing subdomains */
  ierr = PetscArrayzero(graph->count,graph->nvtxs);CHKERRQ(ierr);
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
    ierr = PetscSortRemoveDupsInt(&graph->count[i],graph->neighbours_set[i]);CHKERRQ(ierr);
  }
  /* free memory allocated by ISLocalToGlobalMappingGetInfo */
  ierr = ISLocalToGlobalMappingRestoreInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);

  /*
     Get info for dofs splitting
     User can specify just a subset; an additional field is considered as a complementary field
  */
  for (i=0,k=0;i<n_ISForDofs;i++) {
    PetscInt bs;

    ierr = ISGetBlockSize(ISForDofs[i],&bs);CHKERRQ(ierr);
    k   += bs;
  }
  for (i=0;i<graph->nvtxs;i++) graph->which_dof[i] = k; /* by default a dof belongs to the complement set */
  for (i=0,k=0;i<n_ISForDofs;i++) {
    PetscInt bs;

    ierr = ISGetLocalSize(ISForDofs[i],&is_size);CHKERRQ(ierr);
    ierr = ISGetBlockSize(ISForDofs[i],&bs);CHKERRQ(ierr);
    ierr = ISGetIndices(ISForDofs[i],(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (j=0;j<is_size/bs;j++) {
      PetscInt b;

      for (b=0;b<bs;b++) {
        PetscInt jj = bs*j + b;

        if (is_indices[jj] > -1 && is_indices[jj] < graph->nvtxs) { /* out of bounds indices (if any) are skipped */
          graph->which_dof[is_indices[jj]] = k+b;
        }
      }
    }
    ierr = ISRestoreIndices(ISForDofs[i],(const PetscInt**)&is_indices);CHKERRQ(ierr);
    k   += bs;
  }

  /* Take into account Neumann nodes */
  if (neumann_is) {
    ierr = ISGetLocalSize(neumann_is,&is_size);CHKERRQ(ierr);
    ierr = ISGetIndices(neumann_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<is_size;i++) {
      if (is_indices[i] > -1 && is_indices[i] < graph->nvtxs) { /* out of bounds indices (if any) are skipped */
        graph->special_dof[is_indices[i]] = PCBDDCGRAPH_NEUMANN_MARK;
      }
    }
    ierr = ISRestoreIndices(neumann_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  }
  /* Take into account Dirichlet nodes (they overwrite any neumann boundary mark previously set) */
  if (dirichlet_is) {
    ierr = ISGetLocalSize(dirichlet_is,&is_size);CHKERRQ(ierr);
    ierr = ISGetIndices(dirichlet_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<is_size;i++) {
      if (is_indices[i] > -1 && is_indices[i] < graph->nvtxs) { /* out of bounds indices (if any) are skipped */
        if (commsize > graph->commsizelimit) { /* dirichlet nodes treated as internal */
          ierr = PetscBTSet(graph->touched,is_indices[i]);CHKERRQ(ierr);
          graph->subset[is_indices[i]] = 0;
        }
        graph->special_dof[is_indices[i]] = PCBDDCGRAPH_DIRICHLET_MARK;
      }
    }
    ierr = ISRestoreIndices(dirichlet_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  }
  /* mark local periodic nodes (if any) and adapt CSR graph (if any) */
  if (graph->mirrors) {
    for (i=0;i<graph->nvtxs;i++)
      if (graph->mirrors[i])
        graph->special_dof[i] = PCBDDCGRAPH_LOCAL_PERIODIC_MARK;

    if (graph->xadj) {
      PetscInt *new_xadj,*new_adjncy;
      /* sort CSR graph */
      for (i=0;i<graph->nvtxs;i++)
        ierr = PetscSortInt(graph->xadj[i+1]-graph->xadj[i],&graph->adjncy[graph->xadj[i]]);CHKERRQ(ierr);

      /* adapt local CSR graph in case of local periodicity */
      k = 0;
      for (i=0;i<graph->nvtxs;i++)
        for (j=graph->xadj[i];j<graph->xadj[i+1];j++)
          k += graph->mirrors[graph->adjncy[j]];

      ierr = PetscMalloc1(graph->nvtxs+1,&new_xadj);CHKERRQ(ierr);
      ierr = PetscMalloc1(k+graph->xadj[graph->nvtxs],&new_adjncy);CHKERRQ(ierr);
      new_xadj[0] = 0;
      for (i=0;i<graph->nvtxs;i++) {
        k = graph->xadj[i+1]-graph->xadj[i];
        ierr = PetscArraycpy(&new_adjncy[new_xadj[i]],&graph->adjncy[graph->xadj[i]],k);CHKERRQ(ierr);
        new_xadj[i+1] = new_xadj[i]+k;
        for (j=graph->xadj[i];j<graph->xadj[i+1];j++) {
          k = graph->mirrors[graph->adjncy[j]];
          ierr = PetscArraycpy(&new_adjncy[new_xadj[i+1]],graph->mirrors_set[graph->adjncy[j]],k);CHKERRQ(ierr);
          new_xadj[i+1] += k;
        }
        k = new_xadj[i+1]-new_xadj[i];
        ierr = PetscSortRemoveDupsInt(&k,&new_adjncy[new_xadj[i]]);CHKERRQ(ierr);
        new_xadj[i+1] = new_xadj[i]+k;
      }
      /* set new CSR into graph */
      ierr = PetscFree(graph->xadj);CHKERRQ(ierr);
      ierr = PetscFree(graph->adjncy);CHKERRQ(ierr);
      graph->xadj = new_xadj;
      graph->adjncy = new_adjncy;
    }
  }

  /* mark special nodes (if any) -> each will become a single node equivalence class */
  if (custom_primal_vertices) {
    ierr = ISGetLocalSize(custom_primal_vertices,&is_size);CHKERRQ(ierr);
    ierr = ISGetIndices(custom_primal_vertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0,j=0;i<is_size;i++) {
      if (is_indices[i] > -1 && is_indices[i] < graph->nvtxs  && graph->special_dof[is_indices[i]] != PCBDDCGRAPH_DIRICHLET_MARK) { /* out of bounds indices (if any) are skipped */
        graph->special_dof[is_indices[i]] = PCBDDCGRAPH_SPECIAL_MARK-j;
        j++;
      }
    }
    ierr = ISRestoreIndices(custom_primal_vertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
  }

  /* mark interior nodes (if commsize > graph->commsizelimit) as touched and belonging to partition number 0 */
  if (commsize > graph->commsizelimit) {
    for (i=0;i<graph->nvtxs;i++) {
      if (!graph->count[i]) {
        ierr = PetscBTSet(graph->touched,i);CHKERRQ(ierr);
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
    ierr = PetscMalloc2(graph->nvtxs+1,&graph->cptr,graph->nvtxs,&graph->queue);CHKERRQ(ierr);
  } else {
    PetscInt nused = graph->nvtxs - nodes_touched;
    ierr = PetscMalloc2(nused+1,&graph->cptr,nused,&graph->queue);CHKERRQ(ierr);
  }

  while (nodes_touched<graph->nvtxs) {
    /*  find first untouched node in local ordering */
    while (PetscBTLookup(graph->touched,i)) i++;
    ierr = PetscBTSet(graph->touched,i);CHKERRQ(ierr);
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
          ierr = PetscBTSet(graph->touched,j);CHKERRQ(ierr);
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
  ierr = PetscMalloc1(graph->n_subsets,&graph->subset_ncc);CHKERRQ(ierr);
  for (i=0;i<graph->n_subsets;i++) {
    graph->subset_ncc[i] = 1;
  }
  /* final pointer */
  graph->cptr[graph->ncc] = total_counts;

  /* For consistency reasons (among neighbours), I need to sort (by global ordering) each connected component */
  /* Get a reference node (min index in global ordering) for each subset for tagging messages */
  ierr = PetscMalloc1(graph->ncc,&graph->subset_ref_node);CHKERRQ(ierr);
  ierr = PetscMalloc1(graph->cptr[graph->ncc],&queue_global);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(graph->l2gmap,graph->cptr[graph->ncc],graph->queue,queue_global);CHKERRQ(ierr);
  for (j=0;j<graph->ncc;j++) {
    ierr = PetscSortIntWithArray(graph->cptr[j+1]-graph->cptr[j],&queue_global[graph->cptr[j]],&graph->queue[graph->cptr[j]]);CHKERRQ(ierr);
    graph->subset_ref_node[j] = graph->queue[graph->cptr[j]];
  }
  ierr = PetscFree(queue_global);CHKERRQ(ierr);
  graph->queue_sorted = PETSC_TRUE;

  /* save information on subsets (needed when analyzing the connected components) */
  if (graph->ncc) {
    ierr = PetscMalloc2(graph->ncc,&graph->subset_size,graph->ncc,&graph->subset_idxs);CHKERRQ(ierr);
    ierr = PetscMalloc1(graph->cptr[graph->ncc],&graph->subset_idxs[0]);CHKERRQ(ierr);
    ierr = PetscArrayzero(graph->subset_idxs[0],graph->cptr[graph->ncc]);CHKERRQ(ierr);
    for (j=1;j<graph->ncc;j++) {
      graph->subset_size[j-1] = graph->cptr[j] - graph->cptr[j-1];
      graph->subset_idxs[j] = graph->subset_idxs[j-1] + graph->subset_size[j-1];
    }
    graph->subset_size[graph->ncc-1] = graph->cptr[graph->ncc] - graph->cptr[graph->ncc-1];
    ierr = PetscArraycpy(graph->subset_idxs[0],graph->queue,graph->cptr[graph->ncc]);CHKERRQ(ierr);
  }

  /* renumber reference nodes */
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)(graph->l2gmap)),graph->ncc,graph->subset_ref_node,PETSC_COPY_VALUES,&subset_n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(graph->l2gmap,subset_n,&subset);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);
  ierr = ISRenumber(subset,NULL,NULL,&subset_n);CHKERRQ(ierr);
  ierr = ISDestroy(&subset);CHKERRQ(ierr);
  ierr = ISGetLocalSize(subset_n,&k);CHKERRQ(ierr);
  if (k != graph->ncc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid size of new subset! %D != %D",k,graph->ncc);
  ierr = ISGetIndices(subset_n,&is_indices);CHKERRQ(ierr);
  ierr = PetscArraycpy(graph->subset_ref_node,is_indices,graph->ncc);CHKERRQ(ierr);
  ierr = ISRestoreIndices(subset_n,&is_indices);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);

  /* free workspace */
  graph->setupcalled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphResetCoords(PCBDDCGraph graph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!graph) PetscFunctionReturn(0);
  ierr = PetscFree(graph->coords);CHKERRQ(ierr);
  graph->cdim  = 0;
  graph->cnloc = 0;
  graph->cloc  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphResetCSR(PCBDDCGraph graph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!graph) PetscFunctionReturn(0);
  if (graph->freecsr) {
    ierr = PetscFree(graph->xadj);CHKERRQ(ierr);
    ierr = PetscFree(graph->adjncy);CHKERRQ(ierr);
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
  ierr = ISLocalToGlobalMappingDestroy(&graph->l2gmap);CHKERRQ(ierr);
  ierr = PetscFree(graph->subset_ncc);CHKERRQ(ierr);
  ierr = PetscFree(graph->subset_ref_node);CHKERRQ(ierr);
  if (graph->nvtxs) {
    ierr = PetscFree(graph->neighbours_set[0]);CHKERRQ(ierr);
  }
  ierr = PetscBTDestroy(&graph->touched);CHKERRQ(ierr);
  ierr = PetscFree5(graph->count,
                    graph->neighbours_set,
                    graph->subset,
                    graph->which_dof,
                    graph->special_dof);CHKERRQ(ierr);
  ierr = PetscFree2(graph->cptr,graph->queue);CHKERRQ(ierr);
  if (graph->mirrors) {
    ierr = PetscFree(graph->mirrors_set[0]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(graph->mirrors,graph->mirrors_set);CHKERRQ(ierr);
  if (graph->subset_idxs) {
    ierr = PetscFree(graph->subset_idxs[0]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(graph->subset_size,graph->subset_idxs);CHKERRQ(ierr);
  ierr = ISDestroy(&graph->dirdofs);CHKERRQ(ierr);
  ierr = ISDestroy(&graph->dirdofsB);CHKERRQ(ierr);
  if (graph->n_local_subs) {
    ierr = PetscFree(graph->local_subs);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(graph,1);
  PetscValidHeaderSpecific(l2gmap,IS_LTOGM_CLASSID,2);
  PetscValidLogicalCollectiveInt(l2gmap,N,3);
  PetscValidLogicalCollectiveInt(l2gmap,maxcount,4);
  /* raise an error if already allocated */
  if (graph->nvtxs_global) SETERRQ(PetscObjectComm((PetscObject)l2gmap),PETSC_ERR_PLIB,"BDDCGraph already initialized");
  /* set number of vertices */
  ierr = PetscObjectReference((PetscObject)l2gmap);CHKERRQ(ierr);
  graph->l2gmap = l2gmap;
  ierr = ISLocalToGlobalMappingGetSize(l2gmap,&n);CHKERRQ(ierr);
  graph->nvtxs = n;
  graph->nvtxs_global = N;
  /* allocate used space */
  ierr = PetscBTCreate(graph->nvtxs,&graph->touched);CHKERRQ(ierr);
  ierr = PetscMalloc5(graph->nvtxs,&graph->count,graph->nvtxs,&graph->neighbours_set,graph->nvtxs,&graph->subset,graph->nvtxs,&graph->which_dof,graph->nvtxs,&graph->special_dof);CHKERRQ(ierr);
  /* zeroes memory */
  ierr = PetscArrayzero(graph->count,graph->nvtxs);CHKERRQ(ierr);
  ierr = PetscArrayzero(graph->subset,graph->nvtxs);CHKERRQ(ierr);
  /* use -1 as a default value for which_dof array */
  for (n=0;n<graph->nvtxs;n++) graph->which_dof[n] = -1;
  ierr = PetscArrayzero(graph->special_dof,graph->nvtxs);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCBDDCGraphResetCSR(*graph);CHKERRQ(ierr);
  ierr = PCBDDCGraphResetCoords(*graph);CHKERRQ(ierr);
  ierr = PCBDDCGraphReset(*graph);CHKERRQ(ierr);
  ierr = PetscFree(*graph);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCGraphCreate(PCBDDCGraph *graph)
{
  PCBDDCGraph    new_graph;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&new_graph);CHKERRQ(ierr);
  new_graph->custom_minimal_size = 1;
  new_graph->commsizelimit = 1;
  *graph = new_graph;
  PetscFunctionReturn(0);
}
