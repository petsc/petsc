/* $Id: phead.h,v 1.18 1995/09/30 19:31:53 bsmith Exp bsmith $ */

/*
    Defines the basic format of all data types. 
*/

#if !defined(_PHEAD_H)
#define _PHEAD_H
#include "petsc.h"  


/*
     All Major PETSc Data structures have a common core; this 
   is defined below by PETSCHEADER. 

     PETSCHEADERCREATE should be used whenever you create a PETSc structure.

     CHKSAME checks if your PETSc structures are of same type.
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
  /*  ... */                               \

#define  PETSCFREEDHEADER -1

#define PETSCHEADERCREATE(h,tp,cook,t,com)                         \
      {h = (struct tp *) PETSCNEW(struct tp);                      \
       CHKPTRQ((h));                                               \
       PetscMemzero(h,sizeof(struct tp));                          \
       (h)->cookie = cook;                                         \
       (h)->type = t;                                              \
       MPIU_Comm_dup(com,&(h)->comm,&(h)->tag);}
#define PETSCHEADERDESTROY(h)                                      \
       {MPIU_Comm_free(&(h)->comm);                                \
        (h)->cookie = PETSCFREEDHEADER;                            \
        PETSCFREE(h);          }

extern void *PetscLow,*PetscHigh;

#if defined(PETSC_BOPT_g) && !defined(PETSC_INSIGHT)
#define PETSCVALIDHEADERSPECIFIC(h,ck)                             \
  {if (!h) {SETERRQ(PETSC_ERR_OBJ,"Null Object");}                 \
  if (PetscLow > (void *) h || PetscHigh < (void *)h){             \
    SETERRQ(PETSC_ERR_OBJ,"Invalid Pointer to Object");            \
  }                                                                \
  if ((h)->cookie != ck) {                                         \
    if ((h)->cookie == PETSCFREEDHEADER) {                         \
      SETERRQ(PETSC_ERR_OBJ,"Object already free");                \
    }                                                              \
    else {                                                         \
      SETERRQ(PETSC_ERR_OBJ,"Invalid or Wrong Object");            \
    }                                                              \
  }}
#define PETSCVALIDHEADER(h)                                        \
  {if (!h) {SETERRQ(1,"Null Object");}                             \
  else if (PetscLow > (void *) h || PetscHigh < (void *)h){        \
    SETERRQ(PETSC_ERR_OBJ,"Invalid Pointer to Object");            \
  }                                                                \
  else if ((h)->cookie == PETSCFREEDHEADER) {                      \
      SETERRQ(PETSC_ERR_OBJ,"Object already free");                \
  }                                                                \
  else if ((h)->cookie < PETSC_COOKIE ||                           \
      (h)->cookie > PETSC_COOKIE+20) {                             \
      SETERRQ(PETSC_ERR_OBJ,"Invalid or Wrong Object");            \
  }}
#else
#define PETSCVALIDHEADERSPECIFIC(h,ck)                             \
  {if (!h) {SETERRQ(PETSC_ERR_OBJ,"Null Object");}                 \
  if ((h)->cookie != ck) {                                         \
    if ((h)->cookie == PETSCFREEDHEADER) {                         \
      SETERRQ(PETSC_ERR_OBJ,"Object already free");                \
    }                                                              \
    else {                                                         \
      SETERRQ(PETSC_ERR_OBJ,"Invalid or Wrong Object");            \
    }                                                              \
  }}
#define PETSCVALIDHEADER(h)                                        \
  {if (!h) {SETERRQ(PETSC_ERR_OBJ,"Null Object");}                 \
  else if ((h)->cookie == PETSCFREEDHEADER) {                      \
      SETERRQ(PETSC_ERR_OBJ,"Object already free");                \
  }                                                                \
  else if ((h)->cookie < PETSC_COOKIE ||                           \
      (h)->cookie > PETSC_COOKIE+20) {                             \
      SETERRQ(PETSC_ERR_OBJ,"Invalid or Wrong Object");            \
  }}
#endif

#define CHKSAME(a,b) \
  if ((a)->type != (b)->type) SETERRQ(3,"Objects not of same type");

struct _PetscObject {
  PETSCHEADER
};


#endif
