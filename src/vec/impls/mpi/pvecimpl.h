/* $Id: pvecimpl.h,v 1.7 1995/11/01 19:08:38 bsmith Exp bsmith $ */
/* 
 */

#ifndef __PVECIMPL
#define __PVECIMPL

#include "vecimpl.h"
#include "dvecimpl.h"

/* The first two elements of this structure should remain the same */
typedef struct {
    VECHEADER
    int         N;           /* length of total vector */
    int         size,rank,*ownership;
    InsertMode  insertmode;
    struct      {int nmax, n, *idx; Scalar *array;} stash;
    MPI_Request *send_waits,*recv_waits;
    int         nsends,nrecvs;
    Scalar      *svalues,*rvalues;
    int         rmax;
} Vec_MPI;

extern int VecNorm_Seq(Vec, NormType, double *work );

#endif



