
/* $Id: sda.h,v 1.1 1996/02/04 22:27:58 bsmith Exp bsmith $ */
/*
    Defines the interface object for the simplified distributed array
    */

#ifndef __SDA_H
#define __SDA_H

#include "da.h"

typedef struct _SDA* SDA;

extern int SDACreate3d(MPI_Comm,DAPeriodicType,DAStencilType,
                int, int, int,int,int, int,int, int, SDA *);
extern int SDACreate2d(MPI_Comm,DAPeriodicType,DAStencilType,
                int, int, int,int, int, int, SDA *);
extern int SDACreate1d(MPI_Comm,DAPeriodicType,int,int,int,SDA *);
extern int SDADestroy(SDA);
extern int SDALocalToLocalBegin(SDA,Scalar*,InsertMode,Scalar*);
extern int SDALocalToLocalEnd(SDA,Scalar*,InsertMode,Scalar*);

#endif
