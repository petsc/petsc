
/* $Id: sda.h,v 1.7 2000/01/11 21:03:35 bsmith Exp balay $ */
/*
    Defines the interface object for the simplified distributed array
    */

#ifndef __SDA_H
#define __SDA_H

#include "petscda.h"

typedef struct _SDA* SDA;

extern int SDACreate3d(MPI_Comm,DAPeriodicType,DAStencilType,
                int,int,int,int,int,int,int,int,int *,int *,int *,SDA *);
extern int SDACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,
                int,int,int,int,int,int,int *,int *,SDA *);
extern int SDACreate1d(MPI_Comm,DAPeriodicType,int,int,int,int*,SDA *);
extern int SDADestroy(SDA);
extern int SDALocalToLocalBegin(SDA,Scalar*,InsertMode,Scalar*);
extern int SDALocalToLocalEnd(SDA,Scalar*,InsertMode,Scalar*);

extern int SDAGetCorners(SDA,int*,int*,int*,int*,int*,int*);
extern int SDAGetGhostCorners(SDA,int*,int*,int*,int*,int*,int*);

#endif
