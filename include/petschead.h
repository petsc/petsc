/* $Id: petschead.h,v 1.38 1996/11/06 04:57:53 bsmith Exp bsmith $ */

/*
    Defines the basic header of all PETSc objects.
*/

#if !defined(_PHEAD_H)
#define _PHEAD_H
#include "petsc.h"  

extern int PetscCommDup_Private(MPI_Comm,MPI_Comm*,int*);
extern int PetscCommFree_Private(MPI_Comm*);

extern int PetscRegisterCookie(int *);

/*
   All Major PETSc Data structures have a common core; this 
   is defined below by PETSCHEADER. 

   PetscHeaderCreate() should be used whenever you create a PETSc structure.
*/

#define PETSCHEADER                         \
  double      flops,time,mem;               \
  int         cookie;                       \
  int         type;                         \
  int         id;                           \
  int         refct;                        \
  int         tag;                          \
  int         (*destroy)(PetscObject);      \
  int         (*view)(PetscObject,Viewer);  \
  MPI_Comm    comm;                         \
  PetscObject parent;                       \
  char*       name;                         \
  char        *prefix;                      \
  void*       child;                        \
  int         (*childcopy)(void *,void**);  \
  int         (*childdestroy)(void *);     
  /*  ... */                               

#define  PETSCFREEDHEADER -1

#define PetscHeaderCreate(h,tp,cook,t,com)                         \
      {h = (struct tp *) PetscNew(struct tp);CHKPTRQ((h));         \
       PetscMemzero(h,sizeof(struct tp));                          \
       (h)->cookie = cook;                                         \
       (h)->type   = t;                                            \
       (h)->prefix = 0;                                            \
       PetscCommDup_Private(com,&(h)->comm,&(h)->tag);}
#define PetscHeaderDestroy(h)                                      \
       {PetscCommFree_Private(&(h)->comm);                         \
        (h)->cookie = PETSCFREEDHEADER;                            \
        if ((h)->prefix) PetscFree((h)->prefix);                   \
        if ((h)->child) (*(h)->childdestroy)((h)->child);          \
        PetscFree(h);          }

/* 
  PetscLow and PetscHigh are a way of checking if an address is 
  out of range. They are set in src/sys/src/tr.c
*/
extern void *PetscLow,*PetscHigh;

#if defined(PETSC_BOPT_g) && !defined(PETSC_INSIGHT)
#define PetscValidHeaderSpecific(h,ck)                              \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null Object");}          \
  if ((unsigned long)h & (unsigned long)3) {                        \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object");     \
  }                                                                 \
  if (PetscLow > (void *) h || PetscHigh < (void *)h){              \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object:out of range");\
  }                                                                 \
  if (((PetscObject)(h))->cookie != ck) {                           \
    if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {           \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Object already free");         \
    }                                                               \
    else {                                                          \
      SETERRQ(PETSC_ERR_ARG_WRONG,"Wrong Object");                  \
    }                                                               \
  }}
#define PetscValidHeader(h)                                         \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null Object");}          \
  if ((unsigned long)h & (unsigned long)3) {                        \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object");     \
  }                                                                 \
  else if (PetscLow > (void *) h || PetscHigh < (void *)h){         \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object:out of range");\
  }                                                                 \
  else if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {        \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Object already free");         \
  }                                                                 \
  else if (((PetscObject)(h))->cookie < PETSC_COOKIE ||             \
      ((PetscObject)(h))->cookie > LARGEST_PETSC_COOKIE) {          \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Object");              \
  }}
#else
#define PetscValidHeaderSpecific(h,ck)                              \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null Object");}          \
  if ((unsigned long)h & (unsigned long)3) {                        \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object");     \
  }                                                                 \
  if (((PetscObject)(h))->cookie != ck) {                           \
    if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {           \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Object already free");         \
    }                                                               \
    else {                                                          \
      SETERRQ(PETSC_ERR_ARG_WRONG,"Wrong Object");                  \
    }                                                               \
  }} 
#define PetscValidHeader(h)                                         \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_CORRUPT,"Null Object");}          \
  if ((unsigned long)h & (unsigned long)3) {                        \
    SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Pointer to Object");     \
  }                                                                 \
  else if (((PetscObject)(h))->cookie == PETSCFREEDHEADER) {        \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Object already free");         \
  }                                                                 \
  else if (((PetscObject)(h))->cookie < PETSC_COOKIE ||             \
      ((PetscObject)(h))->cookie > LARGEST_PETSC_COOKIE) {          \
      SETERRQ(PETSC_ERR_ARG_CORRUPT,"Invalid Object");              \
  }}
#endif

#define PetscValidIntPointer(h)                                     \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}          \
  if ((unsigned long)h & (unsigned long)3){                         \
    SETERRQ(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to Int");         \
  }}
#define PetscValidPointer(h)                                        \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}          \
  if ((unsigned long)h & (unsigned long)3){                         \
    SETERRQ(PETSC_ERR_ARG_BADPTR,"Invalid Pointer");                \
  }}

/*
   Some machines do not double align doubles 
*/
#if defined(PARCH_freebsd) || defined(PARCH_rs6000) || defined(PARCH_linux)
#define PetscValidScalarPointer(h)                                  \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}          \
  if ((unsigned long)h & (unsigned long)3) {                        \
    SETERRQ(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to Scalar");      \
  }}
#else
#define PetscValidScalarPointer(h)                                  \
  {if (!h) {SETERRQ(PETSC_ERR_ARG_BADPTR,"Null Pointer");}          \
  if ((unsigned long)h & (unsigned long)7) {                        \
    SETERRQ(PETSC_ERR_ARG_BADPTR,"Invalid Pointer to Scalar");      \
  }}
#endif

/*
    For example, in the dot product between two vectors,
  both vectors must be either Seq or MPI, not one of each 
*/
#define PetscCheckSameType(a,b) \
  if ((a)->type != (b)->type) SETERRQ(PETSC_ERR_ARG_NOTSAMETYPE,"Objects not of same type");

/*
   All PETSc objects begin with the fields defined in PETSCHEADER.
   The PetscObject is a way of examining these fields regardless of 
   the specific object. In C++ this could be a base abstract class
   from which all objects are derived.
*/
struct _PetscObject {
  PETSCHEADER
};

extern int PetscObjectSetOptionsPrefix(PetscObject,char*);
extern int PetscObjectAppendOptionsPrefix(PetscObject,char*);
extern int PetscObjectGetOptionsPrefix(PetscObject,char**);

#endif


