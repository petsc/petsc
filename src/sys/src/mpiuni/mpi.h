/* $Id: mpi.h,v 1.26 1996/03/18 15:20:14 balay Exp bsmith $ */

/*
 * This is a special set of bindings for uni-processor use of MPI
 * Does not implement send to self.
 * Does not implement attributes correctly.
 */

#ifndef __MPI_BINDINGS
#define __MPI_BINDINGS

#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#if defined(PARCH_sun4) && defined(__cplusplus)
extern "C" {
extern void exit(const int);
}
#endif


extern double MPIUNI_DUMMY;
extern void   *MPIUNI_TMP;
extern double MPI_Wtime();

#define MPI_COMM_WORLD       1
#define MPI_COMM_SELF        2
#define MPI_COMM_NULL        0
#define MPI_SUCCESS          0
#define MPI_IDENT            0
#define MPI_UNEQUAL          3
#define MPI_ANY_SOURCE     (-2)
#define MPI_KEYVAL_INVALID   0
#define MPI_ERR_UNKNOWN     18
#define MPI_ERR_INTERN      21
#define MPI_ERR_OTHER        1
#define MPI_TAG_UB           0


/* External types */
typedef long   MPI_Comm;         
typedef void   *MPI_Request;
typedef void   *MPI_Group;
typedef struct {int MPI_TAG, MPI_SOURCE, MPI_ERROR;} MPI_Status;
typedef char*   MPI_Errhandler;

/* In order to handle datatypes, we make them into "sizeof(raw-type)";
    this allows us to do the PetscMemcpy's easily */
#define MPI_Datatype      int
#define MPI_FLOAT         sizeof(float)
#define MPI_DOUBLE        sizeof(double)
#define MPI_CHAR          sizeof(char)
#define MPI_INT           sizeof(int)
#define MPI_UNSIGNED_LONG sizeof(unsigned long)

/* Routines */
#define MPI_Send( buf, count, datatype, dest, tag, comm)  \
                     (MPIUNI_TMP = (void *) (buf), \
                     MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (dest), \
                     MPIUNI_TMP = (void *) (tag), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPI_SUCCESS)
#define MPI_Recv( buf, count, datatype, source, tag, comm, status) \
                     (MPIUNI_TMP = (void *) (buf), \
                     MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (source), \
                     MPIUNI_TMP = (void *) (tag), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPIUNI_TMP = (void *) (status), \
                     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Get_count(status,  datatype, count) \
                     (MPIUNI_TMP = (void *) (status), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (count), \
                     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Bsend( buf, count, datatype, dest, tag, comm)  \
                     (MPIUNI_TMP = (void *) (buf), \
                     MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (dest), \
                     MPIUNI_TMP = (void *) (tag), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPI_SUCCESS)
#define MPI_Ssend( buf, count,  datatype, dest, tag, comm) \
                     (MPIUNI_TMP = (void *) (buf), \
                     MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (dest), \
                     MPIUNI_TMP = (void *) (tag), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPI_SUCCESS)
#define MPI_Rsend( buf, count,  datatype, dest, tag, comm) \
                     (MPIUNI_TMP = (void *) (buf), \
                     MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (dest), \
                     MPIUNI_TMP = (void *) (tag), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPI_SUCCESS)
#define MPI_Buffer_attach( buffer, size) \
                     (MPIUNI_TMP = (void *) (buffer), \
                     MPIUNI_TMP = (void *) (size), \
                     MPI_SUCCESS)
#define MPI_Buffer_detach( buffer, size)\
                     (MPIUNI_TMP = (void *) (buffer), \
                     MPIUNI_TMP = (void *) (size), \
                     MPI_SUCCESS)
#define MPI_Ibsend( buf, count,  datatype, dest, tag, comm, request) \
                    ( MPIUNI_TMP = (void *) (buf), \
                     MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (dest), \
                     MPIUNI_TMP = (void *) (tag), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPIUNI_TMP = (void *) (request), \
                     MPI_SUCCESS)
#define MPI_Issend( buf, count,  datatype, dest, tag, comm, request) \
                     (MPIUNI_TMP = (void *) (buf), \
                     MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (dest), \
                     MPIUNI_TMP = (void *) (tag), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPIUNI_TMP = (void *) (request), \
                     MPI_SUCCESS)
#define MPI_Irsend( buf, count,  datatype, dest, tag, comm, request) \
                     (MPIUNI_TMP = (void *) (buf), \
                     MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (dest), \
                     MPIUNI_TMP = (void *) (tag), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPIUNI_TMP = (void *) (request), \
                     MPI_SUCCESS)
#define MPI_Irecv( buf, count,  datatype, source, tag, comm, request) \
                     (MPIUNI_TMP = (void *) (buf), \
                     MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (source), \
                     MPIUNI_TMP = (void *) (tag), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPIUNI_TMP = (void *) (request), \
                     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Isend( buf, count,  datatype, dest, tag, comm, request) \
                     (MPIUNI_TMP = (void *) (buf), \
                     MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (dest), \
                     MPIUNI_TMP = (void *) (tag), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPIUNI_TMP = (void *) (request), \
                     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Wait(request, status) \
                     (MPIUNI_TMP = (void *) (request), \
                     MPIUNI_TMP = (void *) (status), \
                   MPI_SUCCESS)
#define MPI_Test(request, flag, status) \
                     (MPIUNI_TMP = (void *) (request), \
                     MPIUNI_TMP = (void *) (status), \
                     *(flag) = 0 \
                     MPI_SUCCESS)
#define MPI_Request_free(request) \
                     (MPIUNI_TMP = (void *) (request), \
                       MPI_SUCCESS)
#define MPI_Waitany(a, b, c, d) \
                     (MPIUNI_TMP = (void *) (a), \
                     MPIUNI_TMP = (void *) (b), \
                     MPIUNI_TMP = (void *) (c), \
                     MPIUNI_TMP = (void *) (d), \
                    MPI_SUCCESS)
#define MPI_Testany(a, b, c, d, e) \
                    (MPIUNI_TMP = (void *) (a), \
                     MPIUNI_TMP = (void *) (b), \
                     MPIUNI_TMP = (void *) (c), \
                     MPIUNI_TMP = (void *) (d), \
                     MPIUNI_TMP = (void *) (e), \
                    MPI_SUCCESS)
#define MPI_Waitall(count, array_of_requests, array_of_statuses) \
                     (MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (array_of_requests), \
                     MPIUNI_TMP = (void *) (array_of_statuses), \
                     MPI_SUCCESS)
#define MPI_Testall(count, array_of_requests, flag, array_of_statuses) \
                     (MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (array_of_requests), \
                     MPIUNI_TMP = (void *) (flag), \
                     MPIUNI_TMP = (void *) (array_of_statuses), \
                     MPI_SUCCESS)
#define MPI_Waitsome(incount, array_of_requests, outcount, \
		 array_of_indices, array_of_statuses) \
                     (MPIUNI_TMP = (void *) (incount), \
                     MPIUNI_TMP = (void *) (array_of_requests), \
                     MPIUNI_TMP = (void *) (outcount), \
                     MPIUNI_TMP = (void *) (array_of_indices), \
                     MPIUNI_TMP = (void *) (array_of_statuses), \
                     MPI_SUCCESS)
#define MPI_Testsome(incount, array_of_requests, outcount, \
		 array_of_indices, array_of_statuses) MPI_SUCCESS
#define MPI_Iprobe(source, tag, comm, flag, status)  \
                     *(flag)=0,  MPI_SUCCESS)
#define MPI_Probe(source, tag, comm, status) MPI_SUCCESS
#define MPI_Cancel(request) MPI_SUCCESS
#define MPI_Test_cancelled(status, flag)  \
                     (*(flag)=0, MPI_SUCCESS)
#define MPI_Send_init( buf, count,  datatype, dest, tag, comm, request) \
    MPI_SUCCESS
#define MPI_Bsend_init( buf, count,  datatype, dest, tag, comm, request) \
    MPI_SUCCESS
#define MPI_Ssend_init( buf, count,  datatype, dest, tag, comm, request) \
    MPI_SUCCESS
#define MPI_Rsend_init( buf, count,  datatype, dest, tag, comm, request) \
    MPI_SUCCESS
#define MPI_Recv_init( buf, count,  datatype, source, tag, comm, request) \
    MPI_SUCCESS
#define MPI_Start(request) MPI_SUCCESS
#define MPI_Startall(count, array_of_requests) MPI_SUCCESS
/* Need to determine sizeof "sendtype" */
#define MPI_Sendrecv(sendbuf, sendcount,  sendtype, \
		 dest, sendtag, recvbuf, recvcount, \
		  recvtype, source, recvtag, \
		 comm, status) \
            PetscMemcpy( recvbuf, sendbuf, (sendcount) * (sendtype) )
#define MPI_Sendrecv_replace( buf, count,  datatype, dest, sendtag, \
              source, recvtag, comm, status) MPI_SUCCESS
#define MPI_Type_contiguous(count,  oldtype, newtype) MPI_SUCCESS
#define MPI_Type_vector(count, blocklength, stride, oldtype,  newtype) \
    MPI_SUCCESS
#define MPI_Type_hvector(count, blocklength, stride, oldtype,  newtype) \
    MPI_SUCCESS
#define MPI_Type_indexed(count, array_of_blocklengths, \
		     array_of_displacements,  oldtype, \
		      newtype) MPI_SUCCESS
#define MPI_Type_hindexed(count, array_of_blocklengths, \
		      array_of_displacements,  oldtype, \
		       newtype) MPI_SUCCESS
#define MPI_Type_struct(count, array_of_blocklengths, \
		    array_of_displacements, \
		     array_of_types,  newtype) MPI_SUCCESS
#define MPI_Address( location, address) \
    (*(address) = (long)(char *)(location),MPI_SUCCESS)
#define MPI_Type_extent( datatype, extent) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Type_size( datatype, size) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Type_lb( datatype, displacement) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Type_ub( datatype, displacement) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Type_commit( datatype) MPI_SUCCESS
#define MPI_Type_free( datatype) MPI_SUCCESS
#define MPI_Get_elements(status,  datatype, count) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Pack( inbuf, incount,  datatype, outbuf, \
	     outsize, position,  comm) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Unpack( inbuf, insize, position, outbuf, \
	       outcount,  datatype, comm) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Pack_size(incount,  datatype, comm, size) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Barrier(comm ) \
                     (MPIUNI_TMP = (void *) (comm), \
                     MPI_SUCCESS)
#define MPI_Bcast( buffer, count, datatype, root, comm ) \
                     (MPIUNI_TMP = (void *) (buffer), \
                     MPIUNI_TMP = (void *) (count), \
                     MPIUNI_TMP = (void *) (datatype), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPI_SUCCESS)
#define MPI_Gather( sendbuf, sendcount,  sendtype, \
                   recvbuf, recvcount,  recvtype, \
                   root, comm) \
                   MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Gatherv( sendbuf, sendcount,  sendtype, \
		 recvbuf, recvcounts, displs, \
		 recvtype, root, comm) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Scatter( sendbuf, sendcount,  sendtype, \
		 recvbuf, recvcount,  recvtype, \
		root, comm) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Scatterv( sendbuf, sendcounts, displs, \
		  sendtype,  recvbuf, recvcount, \
		  recvtype, root, comm) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Allgather( sendbuf, sendcount,  sendtype, \
		   recvbuf, recvcount,  recvtype, comm) \
                     (MPIUNI_TMP = (void *) (recvcount), \
                     MPIUNI_TMP = (void *) (recvtype), \
                     MPIUNI_TMP = (void *) (comm), \
                     PetscMemcpy(recvbuf,sendbuf,(sendcount)* (sendtype)), \
                     MPI_SUCCESS)
#define MPI_Allgatherv( sendbuf, sendcount,  sendtype, \
		    recvbuf, recvcounts, displs, recvtype, comm) \
                     (MPIUNI_TMP = (void *) (sendbuf), \
                     MPIUNI_TMP = (void *) (sendcount), \
                     MPIUNI_TMP = (void *) (sendtype), \
                     MPIUNI_TMP = (void *) (recvbuf), \
                     MPIUNI_TMP = (void *) (recvcounts), \
                     MPIUNI_TMP = (void *) (displs), \
                     MPIUNI_TMP = (void *) (recvtype), \
                     MPIUNI_TMP = (void *) (comm), \
                     MPI_Abort(MPI_COMM_WORLD,0))
#define MPI_Alltoall( sendbuf, sendcount,  sendtype, \
		  recvbuf, recvcount,  recvtype, \
		 comm) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Alltoallv( sendbuf, sendcounts, sdispls, \
		   sendtype,  recvbuf, recvcounts, \
		  rdispls,  recvtype, comm) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Reduce( sendbuf,  recvbuf, count, \
	        datatype, op, root, comm) \
		PetscMemcpy(recvbuf,sendbuf,(count)*( datatype))
#define MPI_Op_create(function, commute, op) MPI_SUCCESS
#define MPI_Op_free( op) MPI_SUCCESS
#define MPI_Allreduce( sendbuf,  recvbuf, count, datatype, op, comm) \
          (PetscMemcpy( recvbuf, sendbuf, (count)*(datatype)), MPI_SUCCESS)
#define MPI_Reduce_scatter( sendbuf,  recvbuf, recvcounts, \
		        datatype, op, comm) \
                        MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Scan( sendbuf,  recvbuf, count,  datatype, \
	     op, comm ) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Group_size(group, size) (*(size)=1,MPI_SUCCESS)
#define MPI_Group_rank(group, rank) (*(rank)=0,MPI_SUCCESS)
#define MPI_Group_translate_ranks (group1, n, ranks1, \
			       group2, ranks2) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Group_compare(group1, group2, result) \
                (*(result)=1,MPI_SUCCESS)
#define MPI_Comm_group(comm, group) MPI_SUCCESS
#define MPI_Group_union(group1, group2, newgroup) MPI_SUCCESS
#define MPI_Group_intersection(group1, group2, newgroup) MPI_SUCCESS
#define MPI_Group_difference(group1, group2, newgroup) MPI_SUCCESS
#define MPI_Group_incl(group, n, ranks, newgroup) MPI_SUCCESS
#define MPI_Group_excl(group, n, ranks, newgroup) MPI_SUCCESS
#define MPI_Group_range_incl(group, n, ranges,newgroup) MPI_SUCCESS
#define MPI_Group_range_excl(group, n, ranges, newgroup) MPI_SUCCESS
#define MPI_Group_free(group) MPI_SUCCESS
#define MPI_Comm_size(comm, size) \
                      (MPIUNI_TMP = (void *) (comm), \
                       *(size)=1, \
                       MPI_SUCCESS)
#define MPI_Comm_rank(comm, rank) \
                      (MPIUNI_TMP = (void *) (comm), \
                      *(rank)=0, \
                      MPI_SUCCESS)
#define MPI_Comm_compare(comm1, comm2, result) \
                      (MPIUNI_TMP = (void *) (comm1), \
                      MPIUNI_TMP = (void *) (comm2), \
                      *(result)=MPI_IDENT, \
                      MPI_SUCCESS )
#define MPI_Comm_create(comm, group, newcomm) MPI_SUCCESS
#define MPI_Comm_split(comm, color, key, newcomm) MPI_SUCCESS
#define MPI_Comm_free(comm) \
                      (MPIUNI_TMP = (void *) (comm), \
                       MPI_SUCCESS)
#define MPI_Comm_dup(comm, newcomm) *(newcomm) = comm, MPI_SUCCESS
#define MPI_Comm_test_inter(comm, flag) (*(flag)=1,MPI_SUCCESS)
#define MPI_Comm_remote_size(comm, size) (*(size)=1,MPI_SUCCESS)
#define MPI_Comm_remote_group(comm, group) MPI_SUCCESS
#define MPI_Intercomm_create(local_comm, local_leader, peer_comm, \
                    remote_leader, tag, newintercomm) MPI_SUCCESS
#define MPI_Intercomm_merge(intercomm, high, newintracomm) MPI_SUCCESS
#define MPI_Keyval_create(copy_fn, delete_fn, keyval, extra_state) \
                      (MPIUNI_TMP = (void *) (copy_fn), \
                      MPIUNI_TMP = (void *) (delete_fn), \
                      MPIUNI_TMP = (void *) (keyval), \
                      MPIUNI_TMP = (void *) (extra_state), \
                      MPI_SUCCESS)
#define MPI_Keyval_free(keyval) MPI_SUCCESS
#define MPI_Attr_put(comm, keyval, attribute_val) \
                      (MPIUNI_TMP = (void *) (comm), \
                      MPIUNI_TMP = (void *) (keyval), \
                      MPIUNI_TMP = (void *) (attribute_val), \
                      MPI_SUCCESS)
#define MPI_Attr_get(comm, keyval, attribute_val, flag)  \
                      (MPIUNI_TMP = (void *) (comm), \
                      MPIUNI_TMP = (void *) (keyval), \
                      (*(void**)attribute_val)= (void *)&MPIUNI_DUMMY, \
                      *(flag) = 1, \
                      MPI_SUCCESS) 
#define MPI_Attr_delete(comm, keyval) MPI_SUCCESS
#define MPI_Topo_test(comm, status) MPI_SUCCESS
#define MPI_Cart_create(comm_old, ndims, dims, periods,\
		    reorder, comm_cart) MPI_SUCCESS
#define MPI_Dims_create(nnodes, ndims, dims) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_create(comm, a, b, c, d, e) MPI_SUCCESS
#define MPI_Graphdims_Get(comm, nnodes, nedges) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_get(comm, a, b, c, d) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cartdim_get(comm, ndims) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_get(comm, maxdims, dims, periods, coords) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_rank(comm, coords, rank) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_coords(comm, rank, maxdims, coords) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_neighbors_count(comm, rank, nneighbors) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_neighbors(comm, rank, maxneighbors,neighbors) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_shift(comm, direction, disp, rank_source, rank_dest) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_sub(comm, remain_dims, newcomm) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Cart_map(comm, ndims, dims, periods, newrank) \
    MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Graph_map(comm, a, b, c, d) MPI_Abort(MPI_COMM_WORLD,0)
#define MPI_Get_processor_name(name, result_len) \
      (PetscStrcpy(name,"localhost"), *(result_len) =10)
#define MPI_Errhandler_create(function, errhandler) \
                     (MPIUNI_TMP = (void *) (errhandler), \
                     MPI_SUCCESS)
#define MPI_Errhandler_set(comm, errhandler) \
                     (MPIUNI_TMP = (void *) (comm), \
                     MPIUNI_TMP = (void *) (errhandler), \
                     MPI_SUCCESS)
#define MPI_Errhandler_get(comm, errhandler) MPI_SUCCESS
#define MPI_Errhandler_free(errhandler) MPI_SUCCESS
#define MPI_Error_string(errorcode, string, result_len) MPI_SUCCESS
#define MPI_Error_class(errorcode, errorclass) MPI_SUCCESS
#define MPI_Wtick() 1.0
#define MPI_Init(argc, argv) MPI_SUCCESS
#define MPI_Finalize() MPI_SUCCESS
#define MPI_Initialized(flag) (*(flag)=1,MPI_SUCCESS)
#define MPI_Pcontrol(level) MPI_SUCCESS

#define MPI_Abort(comm, errorcode) \
                        (MPIUNI_TMP = (void *) (comm),\
     PetscError(__LINE__,__DIR__,__FILE__,errorcode,"[0] Aborting program!"), \
               exit(errorcode))
#define MPI_NULL_COPY_FN   0
#define MPI_NULL_DELETE_FN 0

#endif

