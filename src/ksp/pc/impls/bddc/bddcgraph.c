#include <petsc/private/petscimpl.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <../src/ksp/pc/impls/bddc/bddcstructs.h>

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphGetDirichletDofsB"
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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphGetDirichletDofs"
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

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphASCIIView"
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
  if (verbosity_level > 1) {
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
      if (verbosity_level > 2) {
        if (graph->xadj && graph->adjncy) {
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
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  %d (neighs:",i);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (j=0;j<graph->count[node_num];j++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d",graph->neighbours_set[node_num][j]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"):");CHKERRQ(ierr);
    if (verbosity_level > 1 || graph->twodim) {
      printcc = PETSC_TRUE;
    } else if (graph->count[node_num] > 1 || (graph->count[node_num] == 1 && graph->special_dof[node_num] == PCBDDCGRAPH_NEUMANN_MARK)) {
      printcc = PETSC_TRUE;
    }
    if (printcc) {
      for (j=graph->cptr[i];j<graph->cptr[i+1];j++) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d (%d)",graph->queue[j],queue_in_global_numbering[j]);CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISetTab(viewer,tabs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscFree(queue_in_global_numbering);CHKERRQ(ierr);
  if (graph->custom_minimal_size > 1 && verbosity_level > 1) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Custom minimal size %d\n",graph->custom_minimal_size);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphGetCandidatesIS"
PetscErrorCode PCBDDCGraphGetCandidatesIS(PCBDDCGraph graph, PetscInt *n_faces, IS *FacesIS[], PetscInt *n_edges, IS *EdgesIS[], IS *VerticesIS)
{
  IS             *ISForFaces,*ISForEdges,ISForVertices;
  PetscInt       i,nfc,nec,nvc,*idx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* loop on ccs to evalute number of faces, edges and vertices */
  nfc = 0;
  nec = 0;
  nvc = 0;
  for (i=0;i<graph->ncc;i++) {
    PetscInt repdof = graph->queue[graph->cptr[i]];
    if (graph->cptr[i+1]-graph->cptr[i] > graph->custom_minimal_size) {
      if (graph->count[repdof] == 1 && graph->special_dof[repdof] != PCBDDCGRAPH_NEUMANN_MARK) {
        nfc++;
      } else { /* note that nec will be zero in 2d */
        nec++;
      }
    } else {
      nvc += graph->cptr[i+1]-graph->cptr[i];
    }
  }

  /* check if we are in 2D or 3D */
  if (graph->twodim) { /* we are in a 2D case -> edges are shared by 2 subregions and faces don't exist */
    nec = nfc;
    nfc = 0;
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
    PetscInt repdof = graph->queue[graph->cptr[i]];
    if (graph->cptr[i+1]-graph->cptr[i] > graph->custom_minimal_size) {
      if (graph->count[repdof] == 1 && graph->special_dof[repdof] != PCBDDCGRAPH_NEUMANN_MARK) {
        if (graph->twodim) {
          if (EdgesIS) {
            ierr = ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],&graph->queue[graph->cptr[i]],PETSC_COPY_VALUES,&ISForEdges[nec]);CHKERRQ(ierr);
          }
          nec++;
        } else {
          if (FacesIS) {
            ierr = ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],&graph->queue[graph->cptr[i]],PETSC_COPY_VALUES,&ISForFaces[nfc]);CHKERRQ(ierr);
          }
          nfc++;
        }
      } else {
        if (EdgesIS) {
          ierr = ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],&graph->queue[graph->cptr[i]],PETSC_COPY_VALUES,&ISForEdges[nec]);CHKERRQ(ierr);
        }
        nec++;
      }
    }
  }
  /* index set for vertices */
  if (VerticesIS) {
    nvc = 0;
    for (i=0;i<graph->ncc;i++) {
      if (graph->cptr[i+1]-graph->cptr[i] <= graph->custom_minimal_size) {
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
  /* get back info */
  if (n_faces) *n_faces = nfc;
  if (FacesIS) {
    *FacesIS = ISForFaces;
  }
  if (n_edges) *n_edges = nec;
  if (EdgesIS) {
    *EdgesIS = ISForEdges;
  }
  if (VerticesIS) {
    *VerticesIS = ISForVertices;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphComputeConnectedComponents"
PetscErrorCode PCBDDCGraphComputeConnectedComponents(PCBDDCGraph graph)
{
  PetscBool      adapt_interface_reduced;
  MPI_Comm       interface_comm;
  PetscMPIInt    size;
  PetscInt       i;
  PetscBool      twodim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* compute connected components locally */
  ierr = PetscObjectGetComm((PetscObject)(graph->l2gmap),&interface_comm);CHKERRQ(ierr);
  ierr = PCBDDCGraphComputeConnectedComponentsLocal(graph);CHKERRQ(ierr);
  /* check consistency of connected components among neighbouring subdomains -> it adapt them in case it is needed */
  ierr = MPI_Comm_size(interface_comm,&size);CHKERRQ(ierr);
  adapt_interface_reduced = PETSC_FALSE;
  if (size > 1) {
    PetscInt i;
    PetscBool adapt_interface = PETSC_FALSE;
    for (i=0;i<graph->n_subsets;i++) {
      /* We are not sure that on a given subset of the local interface,
         with two connected components, the latters be the same among sharing subdomains */
      if (graph->subset_ncc[i] > 1) {
        adapt_interface = PETSC_TRUE;
        break;
      }
    }
    ierr = MPIU_Allreduce(&adapt_interface,&adapt_interface_reduced,1,MPIU_BOOL,MPI_LOR,interface_comm);CHKERRQ(ierr);
  }

  if (graph->n_subsets && adapt_interface_reduced) {
    /* old csr stuff */
    PetscInt    *aux_new_xadj,*new_xadj,*new_adjncy;
    PetscInt    *old_xadj,*old_adjncy;
    PetscBT     subset_cc_adapt;
    /* MPI */
    MPI_Request *send_requests,*recv_requests;
    PetscInt    *send_buffer,*recv_buffer;
    PetscInt    sum_requests,start_of_recv,start_of_send;
    PetscInt    *cum_recv_counts;
    /* others */
    PetscInt    *labels;
    PetscInt    global_subset_counter;
    PetscInt    j,k,s;

    /* Retrict adjacency graph using information from previously computed connected components */
    ierr = PetscMalloc1(graph->nvtxs,&aux_new_xadj);CHKERRQ(ierr);
    for (i=0;i<graph->nvtxs;i++) {
      aux_new_xadj[i]=1;
    }
    for (i=0;i<graph->ncc;i++) {
      k = graph->cptr[i+1]-graph->cptr[i];
      for (j=0;j<k;j++) {
        aux_new_xadj[graph->queue[graph->cptr[i]+j]]=k;
      }
    }
    j = 0;
    for (i=0;i<graph->nvtxs;i++) {
      j += aux_new_xadj[i];
    }
    ierr = PetscMalloc1(graph->nvtxs+1,&new_xadj);CHKERRQ(ierr);
    ierr = PetscMalloc1(j,&new_adjncy);CHKERRQ(ierr);
    new_xadj[0]=0;
    for (i=0;i<graph->nvtxs;i++) {
      new_xadj[i+1]=new_xadj[i]+aux_new_xadj[i];
      if (aux_new_xadj[i]==1) {
        new_adjncy[new_xadj[i]]=i;
      }
    }
    ierr = PetscFree(aux_new_xadj);CHKERRQ(ierr);
    ierr = PetscMalloc1(graph->nvtxs,&labels);CHKERRQ(ierr);
    ierr = PetscMemzero(labels,graph->nvtxs*sizeof(*labels));CHKERRQ(ierr);
    for (i=0;i<graph->ncc;i++) {
      k = graph->cptr[i+1]-graph->cptr[i];
      for (j=0;j<k;j++) {
        ierr = PetscMemcpy(&new_adjncy[new_xadj[graph->queue[graph->cptr[i]+j]]],&graph->queue[graph->cptr[i]],k*sizeof(PetscInt));CHKERRQ(ierr);
        labels[graph->queue[graph->cptr[i]+j]] = i;
      }
    }
    /* set temporarly new CSR into graph */
    old_xadj = graph->xadj;
    old_adjncy = graph->adjncy;
    graph->xadj = new_xadj;
    graph->adjncy = new_adjncy;
    /* allocate some space */
    ierr = PetscMalloc1(graph->n_subsets+1,&cum_recv_counts);CHKERRQ(ierr);
    ierr = PetscMemzero(cum_recv_counts,(graph->n_subsets+1)*sizeof(*cum_recv_counts));CHKERRQ(ierr);
    /* first count how many neighbours per connected component I will receive from */
    cum_recv_counts[0]=0;
    for (i=0;i<graph->n_subsets;i++) {
      cum_recv_counts[i+1]=cum_recv_counts[i]+graph->count[graph->subsets[i][0]];
    }
    ierr = PetscMalloc1(cum_recv_counts[graph->n_subsets],&recv_buffer);CHKERRQ(ierr);
    ierr = PetscMalloc2(cum_recv_counts[graph->n_subsets],&send_requests,cum_recv_counts[graph->n_subsets],&recv_requests);CHKERRQ(ierr);
    for (i=0;i<cum_recv_counts[graph->n_subsets];i++) {
      send_requests[i]=MPI_REQUEST_NULL;
      recv_requests[i]=MPI_REQUEST_NULL;
    }
    /* exchange with my neighbours the number of my connected components on the subset of interface */
    sum_requests = 0;
    for (i=0;i<graph->n_subsets;i++) {
      PetscMPIInt neigh,tag;

      j = graph->subsets[i][0];
      ierr = PetscMPIIntCast(2*graph->subset_ref_node[i],&tag);CHKERRQ(ierr);
      for (k=0;k<graph->count[j];k++) {
        ierr = PetscMPIIntCast(graph->neighbours_set[j][k],&neigh);CHKERRQ(ierr);
        ierr = MPI_Isend(&graph->subset_ncc[i],1,MPIU_INT,neigh,tag,interface_comm,&send_requests[sum_requests]);CHKERRQ(ierr);
        ierr = MPI_Irecv(&recv_buffer[sum_requests],1,MPIU_INT,neigh,tag,interface_comm,&recv_requests[sum_requests]);CHKERRQ(ierr);
        sum_requests++;
      }
    }
    ierr = MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    /* determine the subsets I have to adapt */
    ierr = PetscBTCreate(graph->n_subsets,&subset_cc_adapt);CHKERRQ(ierr);
    ierr = PetscBTMemzero(graph->n_subsets,subset_cc_adapt);CHKERRQ(ierr);
    for (i=0;i<graph->n_subsets;i++) {
      for (j=cum_recv_counts[i];j<cum_recv_counts[i+1];j++){
        /* adapt if more than one cc is present */
         if (graph->subset_ncc[i] > 1 || recv_buffer[j] > 1) {
          ierr = PetscBTSet(subset_cc_adapt,i);CHKERRQ(ierr);
          break;
        }
      }
    }
    ierr = PetscFree(recv_buffer);CHKERRQ(ierr);
    /* determine send/recv buffers sizes */
    j = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (PetscBTLookup(subset_cc_adapt,i)) {
        j += graph->subsets_size[i];
      }
    }
    k = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (PetscBTLookup(subset_cc_adapt,i)) {
        k += (cum_recv_counts[i+1]-cum_recv_counts[i])*graph->subsets_size[i];
      }
    }
    ierr = PetscMalloc2(j,&send_buffer,k,&recv_buffer);CHKERRQ(ierr);

    /* fill send buffer */
    j = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (PetscBTLookup(subset_cc_adapt,i)) {
        for (k=0;k<graph->subsets_size[i];k++) {
          send_buffer[j++] = labels[graph->subsets[i][k]];
        }
      }
    }
    ierr = PetscFree(labels);CHKERRQ(ierr);

    /* now exchange the data */
    start_of_recv = 0;
    start_of_send = 0;
    sum_requests = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (PetscBTLookup(subset_cc_adapt,i)) {
        PetscMPIInt neigh,tag;
        PetscInt    size_of_send = graph->subsets_size[i];

        j = graph->subsets[i][0];
        ierr = PetscMPIIntCast(2*graph->subset_ref_node[i]+1,&tag);CHKERRQ(ierr);
        for (k=0;k<graph->count[j];k++) {
          ierr = PetscMPIIntCast(graph->neighbours_set[j][k],&neigh);CHKERRQ(ierr);
          ierr = MPI_Isend(&send_buffer[start_of_send],size_of_send,MPIU_INT,neigh,tag,interface_comm,&send_requests[sum_requests]);CHKERRQ(ierr);
          ierr = MPI_Irecv(&recv_buffer[start_of_recv],size_of_send,MPIU_INT,neigh,tag,interface_comm,&recv_requests[sum_requests]);CHKERRQ(ierr);
          start_of_recv += size_of_send;
          sum_requests++;
        }
        start_of_send += size_of_send;
      }
    }
    ierr = MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = PetscFree2(send_requests,recv_requests);CHKERRQ(ierr);

    start_of_recv = 0;
    global_subset_counter = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (PetscBTLookup(subset_cc_adapt,i)) {
        PetscInt **temp_buffer,temp_buffer_size,*private_label;

        /* allocate some temporary space */
        temp_buffer_size = graph->subsets_size[i];
        ierr = PetscMalloc1(temp_buffer_size,&temp_buffer);CHKERRQ(ierr);
        ierr = PetscMalloc1(temp_buffer_size*(cum_recv_counts[i+1]-cum_recv_counts[i]),&temp_buffer[0]);CHKERRQ(ierr);
        ierr = PetscMemzero(temp_buffer[0],temp_buffer_size*(cum_recv_counts[i+1]-cum_recv_counts[i])*sizeof(PetscInt));CHKERRQ(ierr);
        for (j=1;j<temp_buffer_size;j++) {
          temp_buffer[j] = temp_buffer[j-1]+cum_recv_counts[i+1]-cum_recv_counts[i];
        }
        /* analyze contributions from neighbouring subdomains for i-th subset
           temp buffer structure:
           supposing part of the interface has dimension 5
           e.g with global dofs 0,1,2,3,4, locally ordered on the current process as 0,4,3,1,2
           3 neighs procs having connected components:
             neigh 0: [0 1 4], [2 3], labels [4,7]  (2 connected components)
             neigh 1: [0 1], [2 3 4], labels [3 2]  (2 connected components)
             neigh 2: [0 4], [1], [2 3], labels [1 5 6] (3 connected components)
           tempbuffer (row-oriented) will be filled as:
             [ 4, 3, 1;
               4, 2, 1;
               7, 2, 6;
               4, 3, 5;
               7, 2, 6; ];
           This way we can simply find intersections of ccs among neighs.
           The graph->subset array will need to be modified. The output for the example is [0], [1], [2 3], [4];
                                                                                                                                   */
        for (j=0;j<cum_recv_counts[i+1]-cum_recv_counts[i];j++) {
          for (k=0;k<temp_buffer_size;k++) {
            temp_buffer[k][j] = recv_buffer[start_of_recv+k];
          }
          start_of_recv += temp_buffer_size;
        }
        ierr = PetscMalloc1(temp_buffer_size,&private_label);CHKERRQ(ierr);
        ierr = PetscMemzero(private_label,temp_buffer_size*sizeof(*private_label));CHKERRQ(ierr);
        for (j=0;j<temp_buffer_size;j++) {
          if (!private_label[j]) { /* found a new cc  */
            PetscBool same_set;

            global_subset_counter++;
            private_label[j] = global_subset_counter;
            for (k=j+1;k<temp_buffer_size;k++) { /* check for other nodes in new cc */
              same_set = PETSC_TRUE;
              for (s=0;s<cum_recv_counts[i+1]-cum_recv_counts[i];s++) {
                if (temp_buffer[j][s] != temp_buffer[k][s]) {
                  same_set = PETSC_FALSE;
                  break;
                }
              }
              if (same_set) {
                private_label[k] = global_subset_counter;
              }
            }
          }
        }
        /* insert private labels in graph structure */
        for (j=0;j<graph->subsets_size[i];j++) {
          graph->subset[graph->subsets[i][j]] = graph->n_subsets+private_label[j];
        }
        ierr = PetscFree(private_label);CHKERRQ(ierr);
        ierr = PetscFree(temp_buffer[0]);CHKERRQ(ierr);
        ierr = PetscFree(temp_buffer);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree2(send_buffer,recv_buffer);CHKERRQ(ierr);
    ierr = PetscFree(cum_recv_counts);CHKERRQ(ierr);
    ierr = PetscBTDestroy(&subset_cc_adapt);CHKERRQ(ierr);
    /* We are ready to find for connected components which are consistent among neighbouring subdomains */
    if (global_subset_counter) {
      ierr = PetscBTMemzero(graph->nvtxs,graph->touched);CHKERRQ(ierr);
      global_subset_counter = 0;
      for (i=0;i<graph->nvtxs;i++) {
        if (graph->subset[i] && !PetscBTLookup(graph->touched,i)) {
          global_subset_counter++;
          for (j=i+1;j<graph->nvtxs;j++) {
            if (!PetscBTLookup(graph->touched,j) && graph->subset[j]==graph->subset[i]) {
              graph->subset[j] = global_subset_counter;
              ierr = PetscBTSet(graph->touched,j);CHKERRQ(ierr);
            }
          }
          graph->subset[i] = global_subset_counter;
          ierr = PetscBTSet(graph->touched,i);CHKERRQ(ierr);
        }
      }
      /* refine connected components locally */
      ierr = PCBDDCGraphComputeConnectedComponentsLocal(graph);CHKERRQ(ierr);
    }
    /* restore original CSR graph of dofs */
    ierr = PetscFree(new_xadj);CHKERRQ(ierr);
    ierr = PetscFree(new_adjncy);CHKERRQ(ierr);
    graph->xadj = old_xadj;
    graph->adjncy = old_adjncy;
  }

  /* Determine if we are in 2D or 3D */
  twodim  = PETSC_TRUE;
  for (i=0;i<graph->ncc;i++) {
    PetscInt repdof = graph->queue[graph->cptr[i]];
    if (graph->cptr[i+1]-graph->cptr[i] > graph->custom_minimal_size) {
      if (graph->count[repdof] > 1 || graph->special_dof[repdof] == PCBDDCGRAPH_NEUMANN_MARK) {
        twodim = PETSC_FALSE;
        break;
      }
    }
  }
  ierr = MPIU_Allreduce(&twodim,&graph->twodim,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)graph->l2gmap));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* The following code has been adapted from function IsConnectedSubdomain contained
   in source file contig.c of METIS library (version 5.0.1)
   It finds connected components for each subset  */
#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphComputeConnectedComponentsLocal"
PetscErrorCode PCBDDCGraphComputeConnectedComponentsLocal(PCBDDCGraph graph)
{
  PetscInt       i,j,k,first,last,nleft,ncc,pid,cum_queue,n,ncc_pid;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* quiet return if no csr info is available */
  if (!graph->xadj || !graph->adjncy) {
    PetscFunctionReturn(0);
  }

  /* reset any previous search of connected components */
  ierr = PetscBTMemzero(graph->nvtxs,graph->touched);CHKERRQ(ierr);
  graph->n_subsets = 0;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)graph->l2gmap),&size);CHKERRQ(ierr);
  if (size == 1) {
    if (graph->nvtxs) {
      graph->n_subsets = 1;
      for (i=0;i<graph->nvtxs;i++) {
        graph->subset[i] = 1;
      }
    }
  } else {
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->special_dof[i] == PCBDDCGRAPH_DIRICHLET_MARK || !graph->count[i]) {
        ierr = PetscBTSet(graph->touched,i);CHKERRQ(ierr);
        graph->subset[i] = 0;
      }
      graph->n_subsets = PetscMax(graph->n_subsets,graph->subset[i]);
    }
  }
  ierr = PetscFree(graph->subset_ncc);CHKERRQ(ierr);
  ierr = PetscMalloc1(graph->n_subsets,&graph->subset_ncc);CHKERRQ(ierr);

  /* begin search for connected components */
  cum_queue = 0;
  ncc = 0;
  for (n=0;n<graph->n_subsets;n++) {
    pid = n+1;  /* partition labeled by 0 is discarded */
    nleft = 0;
    for (i=0;i<graph->nvtxs;i++) {
      if (graph->subset[i] == pid) {
        nleft++;
      }
    }
    for (i=0; i<graph->nvtxs; i++) {
      if (graph->subset[i] == pid) {
        break;
      }
    }
    ierr = PetscBTSet(graph->touched,i);CHKERRQ(ierr);
    graph->queue[cum_queue] = i;
    first = 0;
    last = 1;
    graph->cptr[ncc] = cum_queue;
    ncc_pid = 0;
    while (first != nleft) {
      if (first == last) {
        graph->cptr[++ncc] = first+cum_queue;
        ncc_pid++;
        for (i=0; i<graph->nvtxs; i++) { /* TODO-> use a while! */
          if (graph->subset[i] == pid && !PetscBTLookup(graph->touched,i)) {
            break;
          }
        }
        graph->queue[cum_queue+last] = i;
        last++;
        ierr = PetscBTSet(graph->touched,i);CHKERRQ(ierr);
      }
      i = graph->queue[cum_queue+first];
      first++;
      for (j=graph->xadj[i];j<graph->xadj[i+1];j++) {
        k = graph->adjncy[j];
        if (graph->subset[k] == pid && !PetscBTLookup(graph->touched,k)) {
          graph->queue[cum_queue+last] = k;
          last++;
          ierr = PetscBTSet(graph->touched,k);CHKERRQ(ierr);
        }
      }
    }
    graph->cptr[++ncc] = first+cum_queue;
    ncc_pid++;
    cum_queue = graph->cptr[ncc];
    graph->subset_ncc[n] = ncc_pid;
  }
  graph->ncc = ncc;
  graph->queue_sorted = PETSC_FALSE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphSetUp"
PetscErrorCode PCBDDCGraphSetUp(PCBDDCGraph graph, PetscInt custom_minimal_size, IS neumann_is, IS dirichlet_is, PetscInt n_ISForDofs, IS ISForDofs[], IS custom_primal_vertices)
{
  VecScatter     scatter_ctx;
  Vec            local_vec,local_vec2,global_vec;
  IS             to,from,subset,subset_n;
  MPI_Comm       comm;
  PetscScalar    *array,*array2;
  const PetscInt *is_indices;
  PetscInt       n_neigh,*neigh,*n_shared,**shared,*queue_global;
  PetscInt       i,j,k,s,total_counts,nodes_touched,is_size;
  PetscMPIInt    size;
  PetscBool      same_set,mirrors_found;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  graph->has_dirichlet = PETSC_FALSE;
  if (dirichlet_is) {
    PetscCheckSameComm(graph->l2gmap,1,dirichlet_is,4);
    graph->has_dirichlet = PETSC_TRUE;
  }
  ierr = PetscObjectGetComm((PetscObject)(graph->l2gmap),&comm);CHKERRQ(ierr);
  /* custom_minimal_size */
  graph->custom_minimal_size = PetscMax(graph->custom_minimal_size,custom_minimal_size);
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
  /* create some workspace objects */
  local_vec = NULL;
  local_vec2 = NULL;
  global_vec = NULL;
  to = NULL;
  from = NULL;
  scatter_ctx = NULL;
  if (n_ISForDofs || dirichlet_is || neumann_is || custom_primal_vertices) {
    ierr = VecCreate(PETSC_COMM_SELF,&local_vec);CHKERRQ(ierr);
    ierr = VecSetSizes(local_vec,PETSC_DECIDE,graph->nvtxs);CHKERRQ(ierr);
    ierr = VecSetType(local_vec,VECSTANDARD);CHKERRQ(ierr);
    ierr = VecDuplicate(local_vec,&local_vec2);CHKERRQ(ierr);
    ierr = VecCreate(comm,&global_vec);CHKERRQ(ierr);
    ierr = VecSetSizes(global_vec,PETSC_DECIDE,graph->nvtxs_global);CHKERRQ(ierr);
    ierr = VecSetType(global_vec,VECSTANDARD);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,graph->nvtxs,0,1,&to);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApplyIS(graph->l2gmap,to,&from);CHKERRQ(ierr);
    ierr = VecScatterCreate(global_vec,from,local_vec,to,&scatter_ctx);CHKERRQ(ierr);
  } else if (mirrors_found) {
    ierr = ISCreateStride(PETSC_COMM_SELF,graph->nvtxs,0,1,&to);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApplyIS(graph->l2gmap,to,&from);CHKERRQ(ierr);
  }
  /* compute local mirrors (if any) */
  if (mirrors_found) {
    PetscInt *local_indices,*global_indices;
    /* get arrays of local and global indices */
    ierr = PetscMalloc1(graph->nvtxs,&local_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(to,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = PetscMemcpy(local_indices,is_indices,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(to,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = PetscMalloc1(graph->nvtxs,&global_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(from,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = PetscMemcpy(global_indices,is_indices,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(from,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    /* allocate space for mirrors */
    ierr = PetscMalloc2(graph->nvtxs,&graph->mirrors,graph->nvtxs,&graph->mirrors_set);CHKERRQ(ierr);
    ierr = PetscMemzero(graph->mirrors,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
    graph->mirrors_set[0] = 0;

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
    ierr = PetscMemzero(graph->mirrors,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
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
        while ( k > 0 && global_indices[k-1] == j) k--;
        for (j=0;j<graph->mirrors[i];j++) {
          graph->mirrors_set[i][j]=local_indices[k+j];
        }
        ierr = PetscSortInt(graph->mirrors[i],graph->mirrors_set[i]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(local_indices);CHKERRQ(ierr);
    ierr = PetscFree(global_indices);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(graph->count,graph->nvtxs*sizeof(*graph->count));CHKERRQ(ierr);
  ierr = ISDestroy(&to);CHKERRQ(ierr);
  ierr = ISDestroy(&from);CHKERRQ(ierr);

  /* Count total number of neigh per node */
  k=0;
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
  ierr = PetscMemzero(graph->count,graph->nvtxs*sizeof(*graph->count));CHKERRQ(ierr);
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
  for (i=0;i<graph->nvtxs;i++) {
    graph->which_dof[i] = n_ISForDofs; /* by default a dof belongs to the complement set */
  }
  if (n_ISForDofs) {
    ierr = VecSet(local_vec,-1.0);CHKERRQ(ierr);
  }
  for (i=0;i<n_ISForDofs;i++) {
    ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
    ierr = ISGetLocalSize(ISForDofs[i],&is_size);CHKERRQ(ierr);
    ierr = ISGetIndices(ISForDofs[i],(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (j=0;j<is_size;j++) {
      if (is_indices[j] > -1 && is_indices[j] < graph->nvtxs) { /* out of bounds indices (if any) are skipped */
        graph->which_dof[is_indices[j]] = i;
        array[is_indices[j]] = 1.*i;
      }
    }
    ierr = ISRestoreIndices(ISForDofs[i],(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);
  }
  /* Check consistency among neighbours */
  if (n_ISForDofs) {
    ierr = VecScatterBegin(scatter_ctx,local_vec,global_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter_ctx,local_vec,global_vec,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter_ctx,global_vec,local_vec2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter_ctx,global_vec,local_vec2,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
    ierr = VecGetArray(local_vec2,&array2);CHKERRQ(ierr);
    for (i=0;i<graph->nvtxs;i++){
      PetscInt field1,field2;

      field1 = (PetscInt)PetscRealPart(array[i]);
      field2 = (PetscInt)PetscRealPart(array2[i]);
      if (field1 != field2) SETERRQ3(comm,PETSC_ERR_USER,"Local node %D have been assigned two different field ids %D and %D at the same time\n",i,field1,field2);
    }
    ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);
    ierr = VecRestoreArray(local_vec2,&array2);CHKERRQ(ierr);
  }
  if (neumann_is || dirichlet_is) {
    /* Take into account Neumann nodes */
    ierr = VecSet(local_vec,0.0);CHKERRQ(ierr);
    ierr = VecSet(local_vec2,0.0);CHKERRQ(ierr);
    if (neumann_is) {
      ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
      ierr = ISGetLocalSize(neumann_is,&is_size);CHKERRQ(ierr);
      ierr = ISGetIndices(neumann_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
      for (i=0;i<is_size;i++) {
        if (is_indices[i] > -1 && is_indices[i] < graph->nvtxs) { /* out of bounds indices (if any) are skipped */
          array[is_indices[i]] = 1.0;
        }
      }
      ierr = ISRestoreIndices(neumann_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
      ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);
    }
    /* Neumann nodes: impose consistency among neighbours */
    ierr = VecSet(global_vec,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter_ctx,local_vec,global_vec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter_ctx,local_vec,global_vec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter_ctx,global_vec,local_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter_ctx,global_vec,local_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
    for (i=0;i<graph->nvtxs;i++) {
      if (PetscRealPart(array[i]) > 0.1) {
        graph->special_dof[i] = PCBDDCGRAPH_NEUMANN_MARK;
      }
    }
    ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);
    /* Take into account Dirichlet nodes */
    ierr = VecSet(local_vec2,0.0);CHKERRQ(ierr);
    if (dirichlet_is) {
      ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
      ierr = VecGetArray(local_vec2,&array2);CHKERRQ(ierr);
      ierr = ISGetLocalSize(dirichlet_is,&is_size);CHKERRQ(ierr);
      ierr = ISGetIndices(dirichlet_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
      for (i=0;i<is_size;i++){
        if (is_indices[i] > -1 && is_indices[i] < graph->nvtxs) { /* out of bounds indices (if any) are skipped */
          k = is_indices[i];
          if (graph->count[k] && !PetscBTLookup(graph->touched,k)) {
            if (PetscRealPart(array[k]) > 0.1) SETERRQ1(comm,PETSC_ERR_USER,"BDDC cannot have boundary nodes which are marked Neumann and Dirichlet at the same time! Local node %D is wrong\n",k);
            array2[k] = 1.0;
          }
        }
      }
      ierr = ISRestoreIndices(dirichlet_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
      ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);
      ierr = VecRestoreArray(local_vec2,&array2);CHKERRQ(ierr);
    }
    /* Dirichlet nodes: impose consistency among neighbours */
    ierr = VecSet(global_vec,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter_ctx,local_vec2,global_vec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter_ctx,local_vec2,global_vec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter_ctx,global_vec,local_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter_ctx,global_vec,local_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
    for (i=0;i<graph->nvtxs;i++) {
      if (PetscRealPart(array[i]) > 0.1) {
        ierr = PetscBTSet(graph->touched,i);CHKERRQ(ierr);
        graph->subset[i] = 0; /* dirichlet nodes treated as internal -> is it ok? */
        graph->special_dof[i] = PCBDDCGRAPH_DIRICHLET_MARK;
      }
    }
    ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);
  }
  /* mark local periodic nodes (if any) and adapt CSR graph (if any) */
  if (graph->mirrors) {
    for (i=0;i<graph->nvtxs;i++)
      if (graph->mirrors[i])
        graph->special_dof[i] = PCBDDCGRAPH_LOCAL_PERIODIC_MARK;

    if (graph->xadj && graph->adjncy) {
      PetscInt *new_xadj,*new_adjncy;
      /* sort CSR graph */
      for (i=0;i<graph->nvtxs;i++)
        ierr = PetscSortInt(graph->xadj[i+1]-graph->xadj[i],&graph->adjncy[graph->xadj[i]]);CHKERRQ(ierr);

      /* adapt local CSR graph in case of local periodicity */
      k=0;
      for (i=0;i<graph->nvtxs;i++)
        for (j=graph->xadj[i];j<graph->xadj[i+1];j++)
          k += graph->mirrors[graph->adjncy[j]];

      ierr = PetscMalloc1(graph->nvtxs+1,&new_xadj);CHKERRQ(ierr);
      ierr = PetscMalloc1(k+graph->xadj[graph->nvtxs],&new_adjncy);CHKERRQ(ierr);
      new_xadj[0]=0;
      for (i=0;i<graph->nvtxs;i++) {
        k = graph->xadj[i+1]-graph->xadj[i];
        ierr = PetscMemcpy(&new_adjncy[new_xadj[i]],&graph->adjncy[graph->xadj[i]],k*sizeof(PetscInt));CHKERRQ(ierr);
        new_xadj[i+1]=new_xadj[i]+k;
        for (j=graph->xadj[i];j<graph->xadj[i+1];j++) {
          k = graph->mirrors[graph->adjncy[j]];
          ierr = PetscMemcpy(&new_adjncy[new_xadj[i+1]],graph->mirrors_set[graph->adjncy[j]],k*sizeof(PetscInt));CHKERRQ(ierr);
          new_xadj[i+1]+=k;
        }
        k = new_xadj[i+1]-new_xadj[i];
        ierr = PetscSortRemoveDupsInt(&k,&new_adjncy[new_xadj[i]]);CHKERRQ(ierr);
        new_xadj[i+1]=new_xadj[i]+k;
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
    ierr = VecSet(local_vec,0.0);CHKERRQ(ierr);
    ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
    ierr = ISGetLocalSize(custom_primal_vertices,&is_size);CHKERRQ(ierr);
    ierr = ISGetIndices(custom_primal_vertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<is_size;i++){
      if (is_indices[i] > -1 && is_indices[i] < graph->nvtxs) { /* out of bounds indices (if any) are skipped */
        array[is_indices[i]] = 1.0;
      }
    }
    ierr = ISRestoreIndices(custom_primal_vertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);
    /* special nodes: impose consistency among neighbours */
    ierr = VecSet(global_vec,0.0);CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter_ctx,local_vec,global_vec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter_ctx,local_vec,global_vec,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter_ctx,global_vec,local_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter_ctx,global_vec,local_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
    j = 0;
    for (i=0;i<graph->nvtxs;i++) {
      if (PetscRealPart(array[i]) > 0.1 && graph->special_dof[i] != PCBDDCGRAPH_DIRICHLET_MARK) {
        graph->special_dof[i] = PCBDDCGRAPH_SPECIAL_MARK-j;
        j++;
      }
    }
    ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);
  }

  /* mark interior nodes as touched and belonging to partition number 0 */
  for (i=0;i<graph->nvtxs;i++) {
    if (!graph->count[i]) {
      ierr = PetscBTSet(graph->touched,i);CHKERRQ(ierr);
      graph->subset[i] = 0;
    }
  }
  /* init graph structure and compute default subsets */
  nodes_touched=0;
  for (i=0;i<graph->nvtxs;i++) {
    if (PetscBTLookup(graph->touched,i)) {
      nodes_touched++;
    }
  }
  i = 0;
  graph->ncc = 0;
  total_counts = 0;

  /* allocated space for queues */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size == 1) {
    ierr = PetscMalloc2(graph->nvtxs+1,&graph->cptr,graph->nvtxs,&graph->queue);CHKERRQ(ierr);
  } else {
    PetscInt nused = graph->nvtxs - nodes_touched;
    ierr = PetscMalloc2(nused+1,&graph->cptr,nused,&graph->queue);CHKERRQ(ierr);
  }

  while (nodes_touched<graph->nvtxs) {
    /*  find first untouched node in local ordering */
    while (PetscBTLookup(graph->touched,i)) {
      i++;
    }
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
        same_set=PETSC_TRUE;
        for (k=0;k<graph->count[j];k++){
          if (graph->neighbours_set[i][k]!=graph->neighbours_set[j][k]) {
            same_set=PETSC_FALSE;
          }
        }
        /* I found a friend of mine */
        if (same_set) {
          graph->subset[j]=graph->ncc+1;
          ierr = PetscBTSet(graph->touched,j);CHKERRQ(ierr);
          nodes_touched++;
          graph->queue[total_counts] = j;
          total_counts++;
        }
      }
    }
    graph->ncc++;
  }
  /* set default number of subsets (at this point no info on csr graph has been taken into account, so n_subsets = ncc */
  graph->n_subsets = graph->ncc;
  ierr = PetscMalloc1(graph->n_subsets,&graph->subset_ncc);CHKERRQ(ierr);
  for (i=0;i<graph->n_subsets;i++) {
    graph->subset_ncc[i] = 1;
  }
  /* final pointer */
  graph->cptr[graph->ncc] = total_counts;

  /* save information on subsets (needed if we have to adapt the connected components later) */
  /* For consistency reasons (among neighbours), I need to sort (by global ordering) each subset */
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
  if (graph->ncc) {
    ierr = PetscMalloc2(graph->ncc,&graph->subsets_size,graph->ncc,&graph->subsets);CHKERRQ(ierr);
    ierr = PetscMalloc1(graph->cptr[graph->ncc],&graph->subsets[0]);CHKERRQ(ierr);
    ierr = PetscMemzero(graph->subsets[0],graph->cptr[graph->ncc]*sizeof(PetscInt));CHKERRQ(ierr);
    for (j=1;j<graph->ncc;j++) {
      graph->subsets_size[j-1] = graph->cptr[j] - graph->cptr[j-1];
      graph->subsets[j] = graph->subsets[j-1] + graph->subsets_size[j-1];
    }
    graph->subsets_size[graph->ncc-1] = graph->cptr[graph->ncc] - graph->cptr[graph->ncc-1];
    for (j=0;j<graph->ncc;j++) {
      ierr = PetscMemcpy(graph->subsets[j],&graph->queue[graph->cptr[j]],graph->subsets_size[j]*sizeof(PetscInt));CHKERRQ(ierr);
    }
  }
  /* renumber reference nodes */
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)(graph->l2gmap)),graph->ncc,graph->subset_ref_node,PETSC_COPY_VALUES,&subset_n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(graph->l2gmap,subset_n,&subset);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);
  ierr = PCBDDCSubsetNumbering(subset,NULL,NULL,&subset_n);CHKERRQ(ierr);
  ierr = ISDestroy(&subset);CHKERRQ(ierr);
  ierr = ISGetLocalSize(subset_n,&k);CHKERRQ(ierr);
  if (k != graph->ncc) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid size of new subset! %D != %D",k,graph->ncc);
  ierr = ISGetIndices(subset_n,&is_indices);CHKERRQ(ierr);
  ierr = PetscMemcpy(graph->subset_ref_node,is_indices,graph->ncc*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISRestoreIndices(subset_n,&is_indices);CHKERRQ(ierr);
  ierr = ISDestroy(&subset_n);CHKERRQ(ierr);

  /* free workspace */
  ierr = VecDestroy(&local_vec);CHKERRQ(ierr);
  ierr = VecDestroy(&local_vec2);CHKERRQ(ierr);
  ierr = VecDestroy(&global_vec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&scatter_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphResetCSR"
PetscErrorCode PCBDDCGraphResetCSR(PCBDDCGraph graph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(graph->xadj);CHKERRQ(ierr);
  ierr = PetscFree(graph->adjncy);CHKERRQ(ierr);
  graph->nvtxs_csr = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphReset"
PetscErrorCode PCBDDCGraphReset(PCBDDCGraph graph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
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
  if (graph->subsets) {
    ierr = PetscFree(graph->subsets[0]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(graph->subsets_size,graph->subsets);CHKERRQ(ierr);
  ierr = ISDestroy(&graph->dirdofs);CHKERRQ(ierr);
  ierr = ISDestroy(&graph->dirdofsB);CHKERRQ(ierr);
  graph->has_dirichlet = PETSC_FALSE;
  graph->nvtxs = 0;
  graph->nvtxs_global = 0;
  graph->n_subsets = 0;
  graph->custom_minimal_size = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphInit"
PetscErrorCode PCBDDCGraphInit(PCBDDCGraph graph, ISLocalToGlobalMapping l2gmap, PetscInt N)
{
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(graph,1);
  PetscValidHeaderSpecific(l2gmap,IS_LTOGM_CLASSID,2);
  PetscValidLogicalCollectiveInt(l2gmap,N,3);
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
  ierr = PetscMalloc5(graph->nvtxs,&graph->count,
                      graph->nvtxs,&graph->neighbours_set,
                      graph->nvtxs,&graph->subset,
                      graph->nvtxs,&graph->which_dof,
                      graph->nvtxs,&graph->special_dof);CHKERRQ(ierr);
  /* zeroes memory */
  ierr = PetscMemzero(graph->count,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(graph->subset,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  /* use -1 as a default value for which_dof array */
  for (n=0;n<graph->nvtxs;n++) graph->which_dof[n] = -1;
  ierr = PetscMemzero(graph->special_dof,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  /* zeroes first pointer to neighbour set */
  if (graph->nvtxs) {
    graph->neighbours_set[0] = 0;
  }
  /* zeroes workspace for values of ncc */
  graph->subset_ncc = 0;
  graph->subset_ref_node = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphDestroy"
PetscErrorCode PCBDDCGraphDestroy(PCBDDCGraph* graph)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCBDDCGraphReset(*graph);CHKERRQ(ierr);
  ierr = PetscFree(*graph);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphCreate"
PetscErrorCode PCBDDCGraphCreate(PCBDDCGraph *graph)
{
  PCBDDCGraph    new_graph;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&new_graph);CHKERRQ(ierr);
  new_graph->custom_minimal_size = 1;
  *graph = new_graph;
  PetscFunctionReturn(0);
}
