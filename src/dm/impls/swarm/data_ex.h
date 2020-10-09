#if !defined(__DMSWARM_DATA_EXCHANGER_H__)
#define __DMSWARM_DATA_EXCHANGER_H__

#include <petscvec.h>
#include <petscmat.h>

typedef enum { DEOBJECT_INITIALIZED=0, DEOBJECT_FINALIZED, DEOBJECT_STATE_UNKNOWN } DMSwarmDEObjectState;

typedef struct _p_DMSwarmDataEx* DMSwarmDataEx;
struct  _p_DMSwarmDataEx {
        PetscInt              instance;
        MPI_Comm              comm;
        PetscMPIInt           rank;
        PetscMPIInt           n_neighbour_procs;
        PetscMPIInt           *neighbour_procs;          /* [n_neighbour_procs] */
        PetscInt              *messages_to_be_sent;      /* [n_neighbour_procs] */
        PetscInt              *message_offsets;          /* [n_neighbour_procs] */
        PetscInt              *messages_to_be_recvieved; /* [n_neighbour_procs] */
        size_t                unit_message_size;
        void                  *send_message;
        PetscInt              send_message_length;
        void                  *recv_message;
        PetscInt              recv_message_length;
        PetscMPIInt           *send_tags, *recv_tags;
        PetscInt              total_pack_cnt;
        PetscInt              *pack_cnt;                 /* [n_neighbour_procs] */
        DMSwarmDEObjectState  topology_status;
        DMSwarmDEObjectState  message_lengths_status;
        DMSwarmDEObjectState  packer_status;
        DMSwarmDEObjectState  communication_status;
        MPI_Status            *_stats;
        MPI_Request           *_requests;
};

/* OBJECT_STATUS */
/* #define OBJECT_INITIALIZED    0 */
/* #define OBJECT_FINALIZED      1 */
/* #define OBJECT_STATE_UNKNOWN  2 */

extern const char *status_names[];

PETSC_INTERN PetscErrorCode DMSwarmDataExCreate(MPI_Comm,const PetscInt, DMSwarmDataEx *);
PETSC_INTERN PetscErrorCode DMSwarmDataExView(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExDestroy(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyInitialize(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyAddNeighbour(DMSwarmDataEx,const PetscMPIInt);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyFinalize(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExInitializeSendCount(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExAddToSendCount(DMSwarmDataEx,const PetscMPIInt,const PetscInt);
PETSC_INTERN PetscErrorCode DMSwarmDataExFinalizeSendCount(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExPackInitialize(DMSwarmDataEx,size_t);
PETSC_INTERN PetscErrorCode DMSwarmDataExPackData(DMSwarmDataEx,PetscMPIInt,PetscInt,void*);
PETSC_INTERN PetscErrorCode DMSwarmDataExPackFinalize(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExBegin(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExEnd(DMSwarmDataEx);
PETSC_INTERN PetscErrorCode DMSwarmDataExGetSendData(DMSwarmDataEx,PetscInt*,void**);
PETSC_INTERN PetscErrorCode DMSwarmDataExGetRecvData(DMSwarmDataEx,PetscInt*,void**);
PETSC_INTERN PetscErrorCode DMSwarmDataExTopologyGetNeighbours(DMSwarmDataEx,PetscMPIInt*,PetscMPIInt *[]);

#endif
