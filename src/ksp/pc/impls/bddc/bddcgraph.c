#include <petsc-private/petscimpl.h>
#include "bddcprivate.h"
#include "bddcstructs.h"

/* special marks: they cannot be enums, since special marks should in principle range from -4 to -max_int */
#define NEUMANN_MARK -1
#define DIRICHLET_MARK -2
#define LOCAL_PERIODIC_MARK -3
#define SPECIAL_MARK -4

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphASCIIView"
PetscErrorCode PCBDDCGraphASCIIView(PCBDDCGraph graph, PetscInt verbosity_level, PetscViewer viewer)
{
  PetscInt       i,j,tabs;
  PetscInt*      queue_in_global_numbering;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerASCIISynchronizedAllow(viewer,PETSC_TRUE);CHKERRQ(ierr);
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
        }
      }
      ierr = PetscViewerASCIISynchronizedPrintf(viewer,"   interface subset id: %d\n",graph->subset[i]);CHKERRQ(ierr);
      if (graph->subset[i] && graph->subset_ncc) {
        ierr = PetscViewerASCIISynchronizedPrintf(viewer,"   ncc for subset: %d\n",graph->subset_ncc[graph->subset[i]-1]);CHKERRQ(ierr);
      }
    }
  }
  ierr = PetscViewerASCIISynchronizedPrintf(viewer,"Total number of connected components %d\n",graph->ncc);CHKERRQ(ierr);
  ierr = PetscMalloc(graph->cptr[graph->ncc]*sizeof(*queue_in_global_numbering),&queue_in_global_numbering);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(graph->l2gmap,graph->cptr[graph->ncc],graph->queue,queue_in_global_numbering);CHKERRQ(ierr);
  for (i=0;i<graph->ncc;i++) {
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"  %d (neighs:",i);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    PetscInt node_num=graph->queue[graph->cptr[i]];
    for (j=0;j<graph->count[node_num];j++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d",graph->neighbours_set[node_num][j]);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIISynchronizedPrintf(viewer,"):");CHKERRQ(ierr);
    for (j=graph->cptr[i];j<graph->cptr[i+1];j++) {
      ierr = PetscViewerASCIISynchronizedPrintf(viewer," %d (%d)",graph->queue[j],queue_in_global_numbering[j]);CHKERRQ(ierr);
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
PetscErrorCode PCBDDCGraphGetCandidatesIS(PCBDDCGraph graph, PetscBool use_faces, PetscBool use_edges, PetscBool use_vertices, PetscInt *n_faces, IS *FacesIS[], PetscInt *n_edges, IS *EdgesIS[], IS *VerticesIS)
{
  IS             *ISForFaces,*ISForEdges,ISForVertices;
  PetscInt       i,j,nfc,nec,nvc,*idx;
  PetscBool      twodim_flag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* loop on ccs to evalute number of faces, edges and vertices */
  nfc = 0;
  nec = 0;
  nvc = 0;
  twodim_flag = PETSC_FALSE;
  for (i=0;i<graph->ncc;i++) {
    if (graph->cptr[i+1]-graph->cptr[i] > graph->custom_minimal_size) {
      if (graph->count[graph->queue[graph->cptr[i]]] == 1 && graph->special_dof[graph->queue[graph->cptr[i]]] != NEUMANN_MARK) {
        nfc++;
      } else { /* note that nec will be zero in 2d */
        nec++;
      }
    } else {
      nvc += graph->cptr[i+1]-graph->cptr[i];
    }
  }
  if (!nec) { /* we are in a 2d case -> no faces, only edges */
    nec = nfc;
    nfc = 0;
    twodim_flag = PETSC_TRUE;
  }
  /* allocate IS arrays for faces, edges. Vertices need a single index set. */
  ISForFaces = 0;
  ISForEdges = 0;
  ISForVertices = 0;
  if (use_faces && nfc) {
    ierr = PetscMalloc(nfc*sizeof(IS),&ISForFaces);CHKERRQ(ierr);
  }
  if (use_edges && nec) {
    ierr = PetscMalloc(nec*sizeof(IS),&ISForEdges);CHKERRQ(ierr);
  }
  if (use_vertices && nvc) {
    ierr = PetscMalloc(nvc*sizeof(PetscInt),&idx);CHKERRQ(ierr);
  }
  /* loop on ccs to compute index sets for faces and edges */
  nfc = 0;
  nec = 0;
  for (i=0;i<graph->ncc;i++) {
    if (graph->cptr[i+1]-graph->cptr[i] > graph->custom_minimal_size) {
      if (graph->count[graph->queue[graph->cptr[i]]] == 1 && graph->special_dof[graph->queue[graph->cptr[i]]] != NEUMANN_MARK) {
        if (twodim_flag) {
          if (use_edges) {
            ierr = ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],&graph->queue[graph->cptr[i]],PETSC_COPY_VALUES,&ISForEdges[nec]);CHKERRQ(ierr);
            nec++;
          }
        } else {
          if (use_faces) {
            ierr = ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],&graph->queue[graph->cptr[i]],PETSC_COPY_VALUES,&ISForFaces[nfc]);CHKERRQ(ierr);
            nfc++;
          }
        }
      } else {
        if (use_edges) {
          ierr = ISCreateGeneral(PETSC_COMM_SELF,graph->cptr[i+1]-graph->cptr[i],&graph->queue[graph->cptr[i]],PETSC_COPY_VALUES,&ISForEdges[nec]);CHKERRQ(ierr);
          nec++;
        }
      }
    }
  }
  /* index set for vertices */
  if (use_vertices && nvc) {
    nvc = 0;
    for (i=0;i<graph->ncc;i++) {
      if (graph->cptr[i+1]-graph->cptr[i] <= graph->custom_minimal_size) {
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
  *n_faces = nfc;
  *FacesIS = ISForFaces;
  *n_edges = nec;
  *EdgesIS = ISForEdges;
  *VerticesIS = ISForVertices;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphComputeConnectedComponents"
PetscErrorCode PCBDDCGraphComputeConnectedComponents(PCBDDCGraph graph)
{
  PetscInt    adapt_interface,adapt_interface_reduced;
  MPI_Comm    interface_comm;
  MPI_Request *send_requests;
  MPI_Request *recv_requests;
  PetscInt    *aux_new_xadj,*new_xadj,*new_adjncy,**temp_buffer;
  PetscInt    i,j,k,s,sum_requests,buffer_size,size_of_recv,temp_buffer_size;
  PetscMPIInt rank,neigh,tag,mpi_buffer_size;
  PetscInt    *cum_recv_counts,*subset_to_nodes_indices,*recv_buffer_subset,*nodes_to_temp_buffer_indices;
  PetscInt    *send_buffer,*recv_buffer,*queue_in_global_numbering,*sizes_of_sends,*add_to_subset;
  PetscInt    start_of_recv,start_of_send,size_of_send,global_subset_counter,ins_val;
  PetscBool   *subset_cc_adapt,same_set;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* compute connected components locally */
  ierr = PetscObjectGetComm((PetscObject)(graph->l2gmap),&interface_comm);CHKERRQ(ierr);
  ierr = PCBDDCGraphComputeConnectedComponentsLocal(graph);CHKERRQ(ierr);
  /* check consistency of connected components among neighbouring subdomains -> it adapt them in case it is needed */
  adapt_interface = 0;
  adapt_interface_reduced = 0;
  for (i=0;i<graph->n_subsets;i++) {
    /* We are not sure that on a given subset of the local interface,
       with two connected components, the latters be the same among sharing subdomains */
    if (graph->subset_ncc[i] > 1) {
      adapt_interface=1;
      break;
    }
  }
  ierr = MPI_Allreduce(&adapt_interface,&adapt_interface_reduced,1,MPIU_INT,MPI_LOR,interface_comm);CHKERRQ(ierr);

  if (graph->n_subsets && adapt_interface_reduced) {
    /* Retrict adjacency graph using information from previously computed connected components */
    ierr = PetscMalloc(graph->nvtxs*sizeof(PetscInt),&aux_new_xadj);CHKERRQ(ierr);
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
    ierr = PetscMalloc((graph->nvtxs+1)*sizeof(PetscInt),&new_xadj);CHKERRQ(ierr);
    ierr = PetscMalloc(j*sizeof(PetscInt),&new_adjncy);CHKERRQ(ierr);
    new_xadj[0]=0;
    for (i=0;i<graph->nvtxs;i++) {
      new_xadj[i+1]=new_xadj[i]+aux_new_xadj[i];
      if (aux_new_xadj[i]==1) {
        new_adjncy[new_xadj[i]]=i;
      }
    }
    ierr = PetscFree(aux_new_xadj);CHKERRQ(ierr);
    for (i=0;i<graph->ncc;i++) {
      k = graph->cptr[i+1]-graph->cptr[i];
      for (j=0;j<k;j++) {
        ierr = PetscMemcpy(&new_adjncy[new_xadj[graph->queue[graph->cptr[i]+j]]],&graph->queue[graph->cptr[i]],k*sizeof(PetscInt));CHKERRQ(ierr);
      }
    }
    /* set new CSR into graph */
    ierr = PetscFree(graph->xadj);CHKERRQ(ierr);
    ierr = PetscFree(graph->adjncy);CHKERRQ(ierr);
    graph->xadj = new_xadj;
    graph->adjncy = new_adjncy;
    /* allocate some space */
    ierr = MPI_Comm_rank(interface_comm,&rank);CHKERRQ(ierr);
    ierr = PetscMalloc((graph->n_subsets+1)*sizeof(*cum_recv_counts),&cum_recv_counts);CHKERRQ(ierr);
    ierr = PetscMemzero(cum_recv_counts,(graph->n_subsets+1)*sizeof(*cum_recv_counts));CHKERRQ(ierr);
    ierr = PetscMalloc(graph->n_subsets*sizeof(*subset_to_nodes_indices),&subset_to_nodes_indices);CHKERRQ(ierr);
    /* first count how many neighbours per connected component I will receive from */
    cum_recv_counts[0]=0;
    for (i=1;i<graph->n_subsets+1;i++) {
      j = 0;
      while (graph->subset[j] != i) {
        j++;
      }
      subset_to_nodes_indices[i-1]=j;
      /* We don't want sends/recvs_to/from_self -> here I don't count myself  */
      cum_recv_counts[i]=cum_recv_counts[i-1]+graph->count[j];
    }
    ierr = PetscMalloc(2*cum_recv_counts[graph->n_subsets]*sizeof(*recv_buffer_subset),&recv_buffer_subset);CHKERRQ(ierr);
    ierr = PetscMalloc(cum_recv_counts[graph->n_subsets]*sizeof(MPI_Request),&send_requests);CHKERRQ(ierr);
    ierr = PetscMalloc(cum_recv_counts[graph->n_subsets]*sizeof(MPI_Request),&recv_requests);CHKERRQ(ierr);
    for (i=0;i<cum_recv_counts[graph->n_subsets];i++) {
      send_requests[i]=MPI_REQUEST_NULL;
      recv_requests[i]=MPI_REQUEST_NULL;
    }
    /* exchange with my neighbours the number of my connected components on the shared interface */
    sum_requests = 0;
    for (i=0;i<graph->n_subsets;i++) {
      j = subset_to_nodes_indices[i];
      for (k=0;k<graph->count[j];k++) {
        ierr = PetscMPIIntCast(graph->neighbours_set[j][k],&neigh);CHKERRQ(ierr);
        ierr = PetscMPIIntCast((PetscInt)(rank+1)*graph->count[j],&tag);CHKERRQ(ierr);
        ierr = MPI_Isend(&graph->subset_ncc[i],1,MPIU_INT,neigh,tag,interface_comm,&send_requests[sum_requests]);CHKERRQ(ierr);
        ierr = PetscMPIIntCast((PetscInt)(neigh+1)*graph->count[j],&tag);CHKERRQ(ierr);
        ierr = MPI_Irecv(&recv_buffer_subset[sum_requests],1,MPIU_INT,neigh,tag,interface_comm,&recv_requests[sum_requests]);CHKERRQ(ierr);
        sum_requests++;
      }
    }
    ierr = MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    /* determine the connected component I need to adapt */
    ierr = PetscMalloc(graph->n_subsets*sizeof(*subset_cc_adapt),&subset_cc_adapt);CHKERRQ(ierr);
    ierr = PetscMemzero(subset_cc_adapt,graph->n_subsets*sizeof(*subset_cc_adapt));CHKERRQ(ierr);
    for (i=0;i<graph->n_subsets;i++) {
      for (j=cum_recv_counts[i];j<cum_recv_counts[i+1];j++){
        /* The first condition is natural (someone has a different number of ccs than me), the second one is just to be safe */
        if ( graph->subset_ncc[i] != recv_buffer_subset[j] || graph->subset_ncc[i] > 1 ) {
          subset_cc_adapt[i] = PETSC_TRUE;
          break;
        }
      }
    }
    buffer_size = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (subset_cc_adapt[i]) {
        for (j=i;j<graph->ncc;j++) {
          if (graph->subset[graph->queue[graph->cptr[j]]] == i+1) { /* WARNING -> subset values goes from 1 to graph->n_subsets included */
            buffer_size += 1 + graph->cptr[j+1]-graph->cptr[j];
          }
        }
      }
    }
    ierr = PetscMalloc(buffer_size*sizeof(*send_buffer),&send_buffer);CHKERRQ(ierr);
    /* now get from neighbours their ccs (in global numbering) and adapt them (in case it is needed) */
    ierr = PetscMalloc(graph->cptr[graph->ncc]*sizeof(*queue_in_global_numbering),&queue_in_global_numbering);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApply(graph->l2gmap,graph->cptr[graph->ncc],graph->queue,queue_in_global_numbering);CHKERRQ(ierr);
    /* determine how much data to send (size of each queue plus the global indices) and communicate it to neighbours */
    ierr = PetscMalloc(graph->n_subsets*sizeof(*sizes_of_sends),&sizes_of_sends);CHKERRQ(ierr);
    ierr = PetscMemzero(sizes_of_sends,graph->n_subsets*sizeof(*sizes_of_sends));CHKERRQ(ierr);
    sum_requests = 0;
    start_of_send = 0;
    start_of_recv = cum_recv_counts[graph->n_subsets];
    for (i=0;i<graph->n_subsets;i++) {
      if (subset_cc_adapt[i]) {
        size_of_send = 0;
        for (j=i;j<graph->ncc;j++) {
          if (graph->subset[graph->queue[graph->cptr[j]]] == i+1) { /* WARNING -> subset values goes from 1 to graph->n_subsets included */
            send_buffer[start_of_send+size_of_send]=graph->cptr[j+1]-graph->cptr[j];
            size_of_send += 1;
            ierr = PetscMemcpy(&send_buffer[start_of_send+size_of_send],
                               &queue_in_global_numbering[graph->cptr[j]],
                               (graph->cptr[j+1]-graph->cptr[j])*sizeof(*send_buffer));CHKERRQ(ierr);
            size_of_send = size_of_send+graph->cptr[j+1]-graph->cptr[j];
          }
        }
        j = subset_to_nodes_indices[i];
        sizes_of_sends[i] = size_of_send;
        for (k=0;k<graph->count[j];k++) {
          ierr = PetscMPIIntCast(graph->neighbours_set[j][k],&neigh);CHKERRQ(ierr);
          ierr = PetscMPIIntCast((PetscInt)(rank+1)*graph->count[j],&tag);CHKERRQ(ierr);
          ierr = MPI_Isend(&sizes_of_sends[i],1,MPIU_INT,neigh,tag,interface_comm,&send_requests[sum_requests]);CHKERRQ(ierr);
          ierr = PetscMPIIntCast((PetscInt)(neigh+1)*graph->count[j],&tag);CHKERRQ(ierr);
          ierr = MPI_Irecv(&recv_buffer_subset[sum_requests+start_of_recv],1,MPIU_INT,neigh,tag,interface_comm,&recv_requests[sum_requests]);CHKERRQ(ierr);
          sum_requests++;
        }
        start_of_send += size_of_send;
      }
    }
    ierr = MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    buffer_size = 0;
    for (k=0;k<sum_requests;k++) {
      buffer_size += recv_buffer_subset[start_of_recv+k];
    }
    ierr = PetscMalloc(buffer_size*sizeof(*recv_buffer),&recv_buffer);CHKERRQ(ierr);
    /* now exchange the data */
    start_of_recv = 0;
    start_of_send = 0;
    sum_requests = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (subset_cc_adapt[i]) {
        size_of_send = sizes_of_sends[i];
        j = subset_to_nodes_indices[i];
        for (k=0;k<graph->count[j];k++) {
          ierr = PetscMPIIntCast(graph->neighbours_set[j][k],&neigh);CHKERRQ(ierr);
          ierr = PetscMPIIntCast((PetscInt)(rank+1)*graph->count[j],&tag);CHKERRQ(ierr);
          ierr = PetscMPIIntCast(size_of_send,&mpi_buffer_size);CHKERRQ(ierr);
          ierr = MPI_Isend(&send_buffer[start_of_send],mpi_buffer_size,MPIU_INT,neigh,tag,interface_comm,&send_requests[sum_requests]);CHKERRQ(ierr);
          size_of_recv = recv_buffer_subset[cum_recv_counts[graph->n_subsets]+sum_requests];
          ierr = PetscMPIIntCast((PetscInt)(neigh+1)*graph->count[j],&tag);CHKERRQ(ierr);
          ierr = PetscMPIIntCast(size_of_recv,&mpi_buffer_size);CHKERRQ(ierr);
          ierr = MPI_Irecv(&recv_buffer[start_of_recv],mpi_buffer_size,MPIU_INT,neigh,tag,interface_comm,&recv_requests[sum_requests]);CHKERRQ(ierr);
          start_of_recv += size_of_recv;
          sum_requests++;
        }
        start_of_send += size_of_send;
      }
    }
    ierr = MPI_Waitall(sum_requests,recv_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    ierr = MPI_Waitall(sum_requests,send_requests,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
    for (j=0;j<buffer_size;) {
       ierr = ISGlobalToLocalMappingApply(graph->l2gmap,IS_GTOLM_MASK,recv_buffer[j],&recv_buffer[j+1],&recv_buffer[j],&recv_buffer[j+1]);CHKERRQ(ierr);
       /* we need to adapt the output of GlobalToLocal mapping if there are mirrored nodes */
       if (graph->mirrors) {
         PetscBool mirrored_found=PETSC_FALSE;
         for (k=0;k<recv_buffer[j];k++) {
           if (graph->mirrors[recv_buffer[j+k+1]]) {
             mirrored_found=PETSC_TRUE;
             recv_buffer[j+k+1]=graph->mirrors_set[recv_buffer[j+k+1]][0];
           }
         }
         if (mirrored_found) {
           ierr = PetscSortInt(recv_buffer[j],&recv_buffer[j+1]);CHKERRQ(ierr);
           k=0;
           while (k<recv_buffer[j]) {
             for (s=1;s<graph->mirrors[recv_buffer[j+1+k]];s++) {
               recv_buffer[j+1+k+s] = graph->mirrors_set[recv_buffer[j+1+k]][s];
             }
             k+=graph->mirrors[recv_buffer[j+1+k]]+s;
           }
         }
       }
       k = recv_buffer[j]+1;
       j += k;
    }
    sum_requests = cum_recv_counts[graph->n_subsets];
    start_of_recv = 0;
    ierr = PetscMalloc(graph->nvtxs*sizeof(*nodes_to_temp_buffer_indices),&nodes_to_temp_buffer_indices);CHKERRQ(ierr);
    global_subset_counter = 0;
    for (i=0;i<graph->n_subsets;i++) {
      if (subset_cc_adapt[i]) {
        temp_buffer_size = 0;
        /* find nodes on the shared interface we need to adapt */
        for (j=0;j<graph->nvtxs;j++) {
          if (graph->subset[j]==i+1) {
            nodes_to_temp_buffer_indices[j] = temp_buffer_size;
            temp_buffer_size++;
          } else {
            nodes_to_temp_buffer_indices[j] = -1;
          }
        }
        /* allocate some temporary space */
        ierr = PetscMalloc(temp_buffer_size*sizeof(PetscInt*),&temp_buffer);CHKERRQ(ierr);
        ierr = PetscMalloc(temp_buffer_size*(cum_recv_counts[i+1]-cum_recv_counts[i])*sizeof(PetscInt),&temp_buffer[0]);CHKERRQ(ierr);
        ierr = PetscMemzero(temp_buffer[0],temp_buffer_size*(cum_recv_counts[i+1]-cum_recv_counts[i])*sizeof(PetscInt));CHKERRQ(ierr);
        for (j=1;j<temp_buffer_size;j++) {
          temp_buffer[j] = temp_buffer[j-1]+cum_recv_counts[i+1]-cum_recv_counts[i];
        }
        /* analyze contributions from neighbouring subdomains for i-th conn comp
           temp buffer structure:
           supposing part of the interface has dimension 5 (for example with global dofs 0,1,2,3,4)
           3 neighs procs with structured connected components:
             neigh 0: [0 1 4], [2 3];  (2 connected components)
             neigh 1: [0 1], [2 3 4];  (2 connected components)
             neigh 2: [0 4], [1], [2 3]; (3 connected components)
           tempbuffer (row-oriented) will be filled as:
             [ 0, 0, 0;
               0, 0, 1;
               1, 1, 2;
               1, 1, 2;
               0, 1, 0; ];
           This way we can simply find intersections of ccs among neighs.
           For the example above, the graph->subset array will be modified to reproduce the following 4 connected components [0], [1], [2 3], [4];
                                                                                                                                   */
        for (j=0;j<cum_recv_counts[i+1]-cum_recv_counts[i];j++) {
          ins_val = 0;
          size_of_recv = recv_buffer_subset[sum_requests];  /* total size of recv from neighs */
          for (buffer_size=0;buffer_size<size_of_recv;) {  /* loop until all data from neighs has been taken into account */
            for (k=1;k<recv_buffer[buffer_size+start_of_recv]+1;k++) { /* filling properly temp_buffer using data from a single recv */
              temp_buffer[nodes_to_temp_buffer_indices[recv_buffer[start_of_recv+buffer_size+k]]][j] = ins_val;
            }
            buffer_size += k;
            ins_val++;
          }
          start_of_recv += size_of_recv;
          sum_requests++;
        }
        ierr = PetscMalloc(temp_buffer_size*sizeof(*add_to_subset),&add_to_subset);CHKERRQ(ierr);
        ierr = PetscMemzero(add_to_subset,temp_buffer_size*sizeof(*add_to_subset));CHKERRQ(ierr);
        for (j=0;j<temp_buffer_size;j++) {
          if (!add_to_subset[j]) { /* found a new cc  */
            global_subset_counter++;
            add_to_subset[j] = global_subset_counter;
            for (k=j+1;k<temp_buffer_size;k++) { /* check for other nodes in new cc */
              same_set = PETSC_TRUE;
              for (s=0;s<cum_recv_counts[i+1]-cum_recv_counts[i];s++) {
                if (temp_buffer[j][s]!=temp_buffer[k][s]) {
                  same_set = PETSC_FALSE;
                  break;
                }
              }
              if (same_set) {
                add_to_subset[k] = global_subset_counter;
              }
            }
          }
        }
        /* insert new data in subset array */
        temp_buffer_size = 0;
        for (j=0;j<graph->nvtxs;j++) {
          if (graph->subset[j]==i+1) {
            graph->subset[j] = graph->n_subsets+add_to_subset[temp_buffer_size];
            temp_buffer_size++;
          }
        }
        ierr = PetscFree(temp_buffer[0]);CHKERRQ(ierr);
        ierr = PetscFree(temp_buffer);CHKERRQ(ierr);
        ierr = PetscFree(add_to_subset);CHKERRQ(ierr);
      }
    }
    ierr = PetscFree(nodes_to_temp_buffer_indices);CHKERRQ(ierr);
    ierr = PetscFree(sizes_of_sends);CHKERRQ(ierr);
    ierr = PetscFree(send_requests);CHKERRQ(ierr);
    ierr = PetscFree(recv_requests);CHKERRQ(ierr);
    ierr = PetscFree(recv_buffer);CHKERRQ(ierr);
    ierr = PetscFree(recv_buffer_subset);CHKERRQ(ierr);
    ierr = PetscFree(send_buffer);CHKERRQ(ierr);
    ierr = PetscFree(cum_recv_counts);CHKERRQ(ierr);
    ierr = PetscFree(subset_to_nodes_indices);CHKERRQ(ierr);
    ierr = PetscFree(subset_cc_adapt);CHKERRQ(ierr);
    /* We are ready to find for connected components consistent among neighbouring subdomains */
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
    ierr = PetscFree(queue_in_global_numbering);CHKERRQ(ierr);
  }
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
  PetscInt       *queue_global;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* quiet return if no csr info is available */
  if (!graph->xadj || !graph->adjncy) {
    PetscFunctionReturn(0);
  }

  /* reset any previous search of connected components */
  ierr = PetscBTMemzero(graph->nvtxs,graph->touched);CHKERRQ(ierr);
  graph->n_subsets = 0;
  for (i=0;i<graph->nvtxs;i++) {
    if (graph->special_dof[i] == DIRICHLET_MARK || !graph->count[i]) {
      ierr = PetscBTSet(graph->touched,i);CHKERRQ(ierr);
      graph->subset[i] = 0;
    }
    graph->n_subsets = PetscMax(graph->n_subsets,graph->subset[i]);
  }
  ierr = PetscFree(graph->subset_ncc);CHKERRQ(ierr);
  ierr = PetscMalloc(graph->n_subsets*sizeof(*graph->subset_ncc),&graph->subset_ncc);CHKERRQ(ierr);
  ierr = PetscMemzero(graph->subset_ncc,graph->n_subsets*sizeof(*graph->subset_ncc));CHKERRQ(ierr);
  ierr = PetscMemzero(graph->cptr,(graph->nvtxs+1)*sizeof(*graph->cptr));CHKERRQ(ierr);
  ierr = PetscMemzero(graph->queue,graph->nvtxs*sizeof(*graph->queue));CHKERRQ(ierr);

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
  /* For consistency among neighbours, I need to sort (by global ordering) each connected component */
  ierr = PetscMalloc(graph->cptr[graph->ncc]*sizeof(*queue_global),&queue_global);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApply(graph->l2gmap,graph->cptr[graph->ncc],graph->queue,queue_global);CHKERRQ(ierr);
  for (i=0;i<graph->ncc;i++) {
    ierr = PetscSortIntWithArray(graph->cptr[i+1]-graph->cptr[i],&queue_global[graph->cptr[i]],&graph->queue[graph->cptr[i]]);CHKERRQ(ierr);
  }
  ierr = PetscFree(queue_global);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphSetUp"
PetscErrorCode PCBDDCGraphSetUp(PCBDDCGraph graph, PetscInt custom_minimal_size, IS neumann_is, IS dirichlet_is, PetscInt n_ISForDofs, IS ISForDofs[], IS custom_primal_vertices)
{
  VecScatter     scatter_ctx;
  Vec            local_vec,local_vec2,global_vec;
  IS             to,from;
  MPI_Comm       comm;
  PetscScalar    *array,*array2;
  const PetscInt *is_indices;
  PetscInt       n_neigh,*neigh,*n_shared,**shared;
  PetscInt       i,j,k,s,total_counts,nodes_touched,is_size;
  PetscErrorCode ierr;
  PetscBool      same_set,mirrors_found;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)(graph->l2gmap),&comm);CHKERRQ(ierr);
  /* custom_minimal_size */
  graph->custom_minimal_size = PetscMax(graph->custom_minimal_size,custom_minimal_size);
  /* get info l2gmap and allocate work vectors  */
  ierr = ISLocalToGlobalMappingGetInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingGetIndices(graph->l2gmap,&is_indices);CHKERRQ(ierr);
  j = 0;
  for (i=0;i<graph->nvtxs;i++) {
    j = PetscMax(j,is_indices[i]);
  }
  ierr = MPI_Allreduce(&j,&i,1,MPIU_INT,MPI_MAX,comm);CHKERRQ(ierr);
  i++;
  ierr = VecCreate(PETSC_COMM_SELF,&local_vec);CHKERRQ(ierr);
  ierr = VecSetSizes(local_vec,PETSC_DECIDE,graph->nvtxs);CHKERRQ(ierr);
  ierr = VecSetType(local_vec,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecDuplicate(local_vec,&local_vec2);CHKERRQ(ierr);
  ierr = VecCreate(comm,&global_vec);CHKERRQ(ierr);
  ierr = VecSetSizes(global_vec,PETSC_DECIDE,i);CHKERRQ(ierr);
  ierr = VecSetType(global_vec,VECSTANDARD);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,graph->nvtxs,0,1,&to);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(graph->l2gmap,to,&from);CHKERRQ(ierr);
  ierr = VecScatterCreate(global_vec,from,local_vec,to,&scatter_ctx);CHKERRQ(ierr);

  /* get local periodic nodes */
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
    PetscInt *local_indices,*global_indices;
    /* get arrays of local and global indices */
    ierr = PetscMalloc(graph->nvtxs*sizeof(PetscInt),&local_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(to,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = PetscMemcpy(local_indices,is_indices,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(to,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = PetscMalloc(graph->nvtxs*sizeof(PetscInt),&global_indices);CHKERRQ(ierr);
    ierr = ISGetIndices(from,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    ierr = PetscMemcpy(global_indices,is_indices,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(from,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    /* allocate space for mirrors */
    ierr = PetscMalloc2(graph->nvtxs,PetscInt,&graph->mirrors,
                        graph->nvtxs,PetscInt*,&graph->mirrors_set);CHKERRQ(ierr);
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
    ierr = PetscMalloc(k*sizeof(PetscInt*),&graph->mirrors_set[0]);CHKERRQ(ierr);
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
    ierr = PetscMalloc(k*sizeof(PetscInt),&graph->neighbours_set[0]);CHKERRQ(ierr);
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
  /* Get info for dofs splitting */
  for (i=0;i<n_ISForDofs;i++) {
    ierr = ISGetSize(ISForDofs[i],&is_size);CHKERRQ(ierr);
    ierr = ISGetIndices(ISForDofs[i],(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (j=0;j<is_size;j++) {
      graph->which_dof[is_indices[j]]=i;
    }
    ierr = ISRestoreIndices(ISForDofs[i],(const PetscInt**)&is_indices);CHKERRQ(ierr);
  }
  /* Take into account Neumann nodes */
  ierr = VecSet(local_vec,0.0);CHKERRQ(ierr);
  ierr = VecSet(local_vec2,0.0);CHKERRQ(ierr);
  if (neumann_is) {
    ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
    ierr = ISGetSize(neumann_is,&is_size);CHKERRQ(ierr);
    ierr = ISGetIndices(neumann_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<is_size;i++) {
      array[is_indices[i]] = 1.0;
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
    if (PetscRealPart(array[i]) > 0.0) {
      graph->special_dof[i] = NEUMANN_MARK;
    }
  }
  ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);
  /* Take into account Dirichlet nodes */
  ierr = VecSet(local_vec2,0.0);CHKERRQ(ierr);
  if (dirichlet_is) {
    ierr = VecGetArray(local_vec,&array);CHKERRQ(ierr);
    ierr = VecGetArray(local_vec2,&array2);CHKERRQ(ierr);
    ierr = ISGetSize(dirichlet_is,&is_size);CHKERRQ(ierr);
    ierr = ISGetIndices(dirichlet_is,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<is_size;i++){
      k = is_indices[i];
      if (graph->count[k] && !PetscBTLookup(graph->touched,k)) {
        if (PetscRealPart(array[k]) > 0.0) {
          SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"BDDC cannot have boundary nodes which are marked Neumann and Dirichlet at the same time! Local node %d is wrong!\n",k);
        }
        array2[k] = 1.0;
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
    if (PetscRealPart(array[i]) > 0.0) {
      ierr = PetscBTSet(graph->touched,i);CHKERRQ(ierr);
      graph->subset[i] = 0; /* dirichlet nodes treated as internal -> is it ok? */
      graph->special_dof[i] = DIRICHLET_MARK;
    }
  }
  ierr = VecRestoreArray(local_vec,&array);CHKERRQ(ierr);

  /* mark local periodic nodes (if any) and adapt CSR graph */
  if (graph->mirrors) {
    PetscInt *new_xadj,*new_adjncy;

    for (i=0;i<graph->nvtxs;i++)
      if (graph->mirrors[i])
        graph->special_dof[i] = LOCAL_PERIODIC_MARK;

    /* sort CSR graph */
    for (i=0;i<graph->nvtxs;i++)
      ierr = PetscSortInt(graph->xadj[i+1]-graph->xadj[i],&graph->adjncy[graph->xadj[i]]);CHKERRQ(ierr);

    /* adapt local CSR graph in case of local periodicity */
    k=0;
    for (i=0;i<graph->nvtxs;i++)
      for (j=graph->xadj[i];j<graph->xadj[i+1];j++)
        k += graph->mirrors[graph->adjncy[j]];

    ierr = PetscMalloc((graph->nvtxs+1)*sizeof(PetscInt),&new_xadj);CHKERRQ(ierr);
    ierr = PetscMalloc((k+graph->xadj[graph->nvtxs])*sizeof(PetscInt),&new_adjncy);CHKERRQ(ierr);
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

  /* mark special nodes -> each will become a single node equivalence class */
  if (custom_primal_vertices) {
    ierr = ISGetSize(custom_primal_vertices,&is_size);CHKERRQ(ierr);
    ierr = ISGetIndices(custom_primal_vertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
    for (i=0;i<is_size;i++) {
      graph->special_dof[is_indices[i]] = SPECIAL_MARK-i;
    }
    ierr = ISRestoreIndices(custom_primal_vertices,(const PetscInt**)&is_indices);CHKERRQ(ierr);
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
  ierr = PetscMalloc(graph->n_subsets*sizeof(*graph->subset_ncc),&graph->subset_ncc);CHKERRQ(ierr);
  for (i=0;i<graph->n_subsets;i++) {
    graph->subset_ncc[i] = 1;
  }
  /* final pointer */
  graph->cptr[graph->ncc] = total_counts;
  /* free memory allocated by ISLocalToGlobalMappingGetInfo */
  ierr = ISLocalToGlobalMappingRestoreInfo(graph->l2gmap,&n_neigh,&neigh,&n_shared,&shared);CHKERRQ(ierr);
  /* free objects */
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
  if (graph->nvtxs) {
    ierr = PetscFree(graph->neighbours_set[0]);CHKERRQ(ierr);
  }
  ierr = PetscBTDestroy(&graph->touched);CHKERRQ(ierr);
  ierr = PetscFree7(graph->count,
                    graph->neighbours_set,
                    graph->subset,
                    graph->which_dof,
                    graph->cptr,
                    graph->queue,
                    graph->special_dof);CHKERRQ(ierr);
  if (graph->mirrors) {
    ierr = PetscFree(graph->mirrors_set[0]);CHKERRQ(ierr);
  }
  ierr = PetscFree2(graph->mirrors,graph->mirrors_set);CHKERRQ(ierr);
  graph->nvtxs = 0;
  graph->n_subsets = 0;
  graph->custom_minimal_size = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCGraphInit"
PetscErrorCode PCBDDCGraphInit(PCBDDCGraph graph, ISLocalToGlobalMapping l2gmap)
{
  PetscInt       n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(graph,1);
  PetscValidHeaderSpecific(l2gmap,IS_LTOGM_CLASSID,2);
  /* raise an error if already allocated */
  if (graph->nvtxs) {
    SETERRQ(PetscObjectComm((PetscObject)l2gmap),PETSC_ERR_PLIB,"BDDCGraph already initialized");
  }
  /* set number of vertices */
  ierr = PetscObjectReference((PetscObject)l2gmap);CHKERRQ(ierr);
  graph->l2gmap = l2gmap;
  ierr = ISLocalToGlobalMappingGetSize(l2gmap,&n);CHKERRQ(ierr);
  graph->nvtxs = n;
  /* allocate used space */
  ierr = PetscBTCreate(graph->nvtxs,&graph->touched);CHKERRQ(ierr);
  ierr = PetscMalloc7(graph->nvtxs,PetscInt,&graph->count,
                      graph->nvtxs,PetscInt*,&graph->neighbours_set,
                      graph->nvtxs,PetscInt,&graph->subset,
                      graph->nvtxs,PetscInt,&graph->which_dof,
                      graph->nvtxs+1,PetscInt,&graph->cptr,
                      graph->nvtxs,PetscInt,&graph->queue,
                      graph->nvtxs,PetscInt,&graph->special_dof);CHKERRQ(ierr);
  /* zeroes memory */
  ierr = PetscMemzero(graph->count,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(graph->subset,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(graph->which_dof,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(graph->cptr,(graph->nvtxs+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(graph->queue,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMemzero(graph->special_dof,graph->nvtxs*sizeof(PetscInt));CHKERRQ(ierr);
  /* zeroes first pointer to neighbour set */
  if (graph->nvtxs) {
    graph->neighbours_set[0] = 0;
  }
  /* zeroes workspace for values of ncc */
  graph->subset_ncc = 0;
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
  ierr = PetscMalloc(sizeof(*new_graph),&new_graph);CHKERRQ(ierr);
  /* local to global mapping of dofs */
  new_graph->l2gmap = 0;
  /* vertex size */
  new_graph->nvtxs = 0;
  new_graph->n_subsets = 0;
  new_graph->custom_minimal_size = 1;
  /* zeroes ponters */
  new_graph->mirrors = 0;
  new_graph->mirrors_set = 0;
  new_graph->neighbours_set = 0;
  new_graph->subset = 0;
  new_graph->which_dof = 0;
  new_graph->special_dof = 0;
  new_graph->cptr = 0;
  new_graph->queue = 0;
  new_graph->count = 0;
  new_graph->subset_ncc = 0;
  new_graph->touched = 0;
  /* zeroes pointers to csr graph of local nodes connectivity (optional data) */
  new_graph->nvtxs_csr = 0;
  new_graph->xadj = 0;
  new_graph->adjncy = 0;
  *graph = new_graph;
  PetscFunctionReturn(0);
}
