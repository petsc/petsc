/* $Id: dvec2.c,v 1.11 1995/06/07 17:27:28 bsmith Exp $ */
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
    int         numtids,mytid,*ownership;
    InsertMode  insertmode;
    struct      {int nmax, n, *idx; Scalar *array;} stash;
    MPI_Request *send_waits,*recv_waits;
    int         nsends,nrecvs;
    Scalar      *svalues,*rvalues;
    int         rmax;
} Vec_MPI;

#endif



