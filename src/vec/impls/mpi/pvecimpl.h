/* 
 */

#ifndef __PVECIMPL
#define __PVECIMPL

#include "ptscimpl.h"
#include "vecimpl.h"
#include "dvecimpl.h"

/* The first two elements of this structure should remain the same */
typedef struct {
    int         n;           /* Length of LOCAL vector */
    Scalar      *array;
    int         N;           /* length of total vector */
    MPI_Comm    comm;        /* Vector is distributed on these processors */
    int         numtids,mytid,*ownership;
    InsertMode  insertmode;
    struct      {int nmax, n, *idx; Scalar *array;} stash;
    MPI_Request *send_waits,*recv_waits;
    int         nsends,nrecvs;
    Scalar      *svalues,*rvalues;
    int         rmax,pad2;
} Vec_MPI;

#endif



