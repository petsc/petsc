/* $Id: pvecimpl.h,v 1.5 1995/08/07 21:57:45 bsmith Exp curfman $ */
/* 
 */

#ifndef __PVECIMPL
#define __PVECIMPL

#include "vecimpl.h"
#include "dvecimpl.h"

/* The first two elements of this structure should remain the same */
typedef struct {
    int         n;           /* Length of LOCAL vector */
    Scalar      *array;
    int         N;           /* length of total vector */
    int         size,rank,*ownership;
    InsertMode  insertmode;
    struct      {int nmax, n, *idx; Scalar *array;} stash;
    MPI_Request *send_waits,*recv_waits;
    int         nsends,nrecvs;
    Scalar      *svalues,*rvalues;
    int         rmax;
} Vec_MPI;

#endif



