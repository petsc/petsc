/* $Id: phead.h,v 1.28 1996/02/09 01:56:42 bsmith Exp bsmith $ */

/*
    Defines the basic format of all data types. 
*/

#if !defined(_PHEAD_H)
#define _PHEAD_H
#include "petsc.h"  

extern int  PetscCommDup_Private(MPI_Comm,MPI_Comm*,int*);
extern int  PetscCommFree_Private(MPI_Comm*);

extern int PetscRegisterCookie(int *);

/*
     All Major PETSc Data structures have a common core; this 
   is defined below by PETSCHEADER. 

     PetscHeaderCreate should be used whenever you create a PETSc structure.

     PetscCheckSameType checks if your PETSc structures are of same type.
*/

#define PETSCHEADER                        \
  double      flops,time,mem;              \
  int         cookie;                      \
  int         type;                        \
  int         id;                          \
  int         refct;                       \
  int         tag;                         \
  int         (*destroy)(PetscObject);     \
  int         (*view)(PetscObject,Viewer); \
  MPI_Comm    comm;                        \
  PetscObject parent;                      \
  char*       name;                        \
  char        *prefix;                     \
  void *      child;                       \
  int         (*childcopy)(void *,void**);     
  /*  ... */                               

#define  PETSCFREEDHEADER -1

#define PetscHeaderCreate(h,tp,cook,t,com)                         \
      {h = (struct tp *) PetscNew(struct tp);                      \
       CHKPTRQ((h));                                               \
       PetscMemzero(h,sizeof(struct tp));                          \
       (h)->cookie = cook;                                         \
       (h)->type   = t;                                            \
       (h)->prefix = 0;                                            \
       PetscCommDup_Private(com,&(h)->comm,&(h)->tag);}
#define PetscHeaderDestroy(h)                                      \
       {PetscCommFree_Private(&(h)->comm);                         \
        (h)->cookie = PETSCFREEDHEADER;                            \
        if ((h)->prefix) PetscFree((h)->prefix);                   \
        PetscFree(h);          }

/* 
  PetscLow and PetscHigh are a poor person's way of checking if 
  an address if out of range. They are set in src/sys/src/tr.c
*/
extern void *PetscLow,*PetscHigh;

#if defined(PETSC_BOPT_g) && !defined(PETSC_INSIGHT)
#define PetscValidHeaderSpecific(h,ck)                             \
  {if (!h) {SETERRQ(PETSC_ERR_OBJ,"Null Object");}                 \
  if (PetscLow > (void *) h || PetscHigh < (void *)h){             \
    SETERRQ(PETSC_ERR_OBJ,"Invalid Pointer to Object");            \
  }                                                                \
  if (((PetscObject)(h))->cookie != ck) {                          \
    if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {          \
      SETERRQ(PETSC_ERR_OBJ,"Object already free");                \
    }                                                              \
    else {                                                         \
      SETERRQ(PETSC_ERR_OBJ,"Invalid or Wrong Object");            \
    }                                                              \
  }}
#define PetscValidHeader(h)                                        \
  {if (!h) {SETERRQ(1,"Null Object");}                             \
  else if (PetscLow > (void *) h || PetscHigh < (void *)h){        \
    SETERRQ(PETSC_ERR_OBJ,"Invalid Pointer to Object");            \
  }                                                                \
  else if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {       \
      SETERRQ(PETSC_ERR_OBJ,"Object already free");                \
  }                                                                \
  else if (((PetscObject)(h))->cookie < PETSC_COOKIE ||            \
      ((PetscObject)(h))->cookie > LARGEST_PETSC_COOKIE) {         \
      SETERRQ(PETSC_ERR_OBJ,"Invalid or Wrong Object");            \
  }}
#else
#define PetscValidHeaderSpecific(h,ck)                             \
  {if (!h) {SETERRQ(PETSC_ERR_OBJ,"Null Object");}                 \
  if (((PetscObject)(h))->cookie != ck) {                          \
    if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {          \
      SETERRQ(PETSC_ERR_OBJ,"Object already free");                \
    }                                                              \
    else {                                                         \
      SETERRQ(PETSC_ERR_OBJ,"Invalid or Wrong Object");            \
    }                                                              \
  }}
#define PetscValidHeader(h)                                        \
  {if (!h) {SETERRQ(PETSC_ERR_OBJ,"Null Object");}                 \
  else if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {       \
      SETERRQ(PETSC_ERR_OBJ,"Object already free");                \
  }                                                                \
  else if (((PetscObject)(h))->cookie < PETSC_COOKIE ||            \
      ((PetscObject)(h))->cookie > LARGEST_PETSC_COOKIE) {         \
      SETERRQ(PETSC_ERR_OBJ,"Invalid or Wrong Object");            \
  }}
#endif

#define PetscCheckSameType(a,b) \
  if ((a)->type != (b)->type) SETERRQ(3,"Objects not of same type");

/*
      All PETSc objects begin with the fields defined in PETSCHEADER,
   the PetscObject is a way of examining these fields regardless of 
   the specific object. In C++ this could be a base abstract class
   from which all objects are derived.
*/
struct _PetscObject {
  PETSCHEADER
};

extern int PetscObjectSetPrefix(PetscObject,char*);
extern int PetscObjectAppendPrefix(PetscObject,char*);
extern int PetscObjectGetPrefix(PetscObject,char**);

#endif
