#ifndef __DATA_EXCHANGER_H__
#define __DATA_EXCHANGER_H__

#include <petscvec.h>
#include <petscmat.h>

typedef enum { DEOBJECT_INITIALIZED=0, DEOBJECT_FINALIZED, DEOBJECT_STATE_UNKNOWN } DEObjectState;

typedef struct _p_DataEx* DataEx;
struct  _p_DataEx {
	PetscInt       instance;
	MPI_Comm       comm;
	PetscMPIInt    rank;
	PetscMPIInt    n_neighbour_procs;
	PetscMPIInt    *neighbour_procs; /* [n_neighbour_procs] */
	PetscInt       *messages_to_be_sent; /* [n_neighbour_procs] */
	PetscInt       *message_offsets; /* [n_neighbour_procs] */
	PetscInt       *messages_to_be_recvieved; /* [n_neighbour_procs] */
	size_t         unit_message_size;
	void           *send_message;
	PetscInt       send_message_length;
	void           *recv_message;
	PetscInt       recv_message_length;
	PetscMPIInt    *send_tags, *recv_tags;
	PetscInt       total_pack_cnt;
	PetscInt       *pack_cnt; /* [n_neighbour_procs] */
	DEObjectState  topology_status;
	DEObjectState  message_lengths_status;
	DEObjectState  packer_status;
	DEObjectState  communication_status;
	MPI_Status     *_stats;
	MPI_Request    *_requests;
};


/* OBJECT_STATUS */
/* #define OBJECT_INITIALIZED    0 */
/* #define OBJECT_FINALIZED      1 */
/* #define OBJECT_STATE_UNKNOWN  2 */

extern const char *status_names[];

PetscErrorCode DataExCreate(MPI_Comm,const PetscInt, DataEx *);
PetscErrorCode DataExView(DataEx);
PetscErrorCode DataExDestroy(DataEx);
PetscErrorCode DataExTopologyInitialize(DataEx);
PetscErrorCode DataExTopologyAddNeighbour(DataEx,const PetscMPIInt);
PetscErrorCode DataExTopologyFinalize(DataEx);
PetscErrorCode DataExInitializeSendCount(DataEx);
PetscErrorCode DataExAddToSendCount(DataEx,const PetscMPIInt,const PetscInt);
PetscErrorCode DataExFinalizeSendCount(DataEx);
PetscErrorCode DataExPackInitialize(DataEx,size_t);
PetscErrorCode DataExPackData(DataEx,PetscMPIInt,PetscInt,void*);
PetscErrorCode DataExPackFinalize(DataEx);
PetscErrorCode DataExBegin(DataEx);
PetscErrorCode DataExEnd(DataEx);
PetscErrorCode DataExGetSendData(DataEx,PetscInt*,void**);
PetscErrorCode DataExGetRecvData(DataEx,PetscInt*,void**);
PetscErrorCode DataExTopologyGetNeighbours(DataEx,PetscMPIInt*,PetscMPIInt *[]);


#endif

