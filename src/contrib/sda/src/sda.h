
/* $Id: sda.h,v 1.8 2000/05/05 22:19:45 balay Exp bsmith $ */
/*
    Defines the interface object for the simplified distributed array
    */

#ifndef __SDA_H
#define __SDA_H

#include "petscda.h"

typedef struct _SDA* SDA;

EXTERN int SDACreate3d(MPI_Comm,DAPeriodicType,DAStencilType,
                int,int,int,int,int,int,int,int,int *,int *,int *,SDA *);
EXTERN int SDACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,
                int,int,int,int,int,int,int *,int *,SDA *);
EXTERN int SDACreate1d(MPI_Comm,DAPeriodicType,int,int,int,int*,SDA *);
EXTERN int SDADestroy(SDA);
EXTERN int SDALocalToLocalBegin(SDA,Scalar*,InsertMode,Scalar*);
EXTERN int SDALocalToLocalEnd(SDA,Scalar*,InsertMode,Scalar*);

EXTERN int SDAGetCorners(SDA,int*,int*,int*,int*,int*,int*);
EXTERN int SDAGetGhostCorners(SDA,int*,int*,int*,int*,int*,int*);

#endif
