
/* $Id: sda.h,v 1.9 2000/05/10 16:43:48 bsmith Exp bsmith $ */
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
EXTERN int SDALocalToLocalBegin(SDA,PetscScalar*,InsertMode,PetscScalar*);
EXTERN int SDALocalToLocalEnd(SDA,PetscScalar*,InsertMode,PetscScalar*);

EXTERN int SDAGetCorners(SDA,int*,int*,int*,int*,int*,int*);
EXTERN int SDAGetGhostCorners(SDA,int*,int*,int*,int*,int*,int*);

#endif
