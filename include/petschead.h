/* $Id: snes.h,v 1.17 1995/06/02 21:05:19 bsmith Exp $ */

/*
    Defines the basic format of all data types. 
*/

#if !defined(_PETSCIMPL)
#define _PETSCIMPL
#include "petsc.h"  
#include "plog.h"
#include <stdio.h>


/*
     All Major PETSc Data structures have a common core; this 
   is defined below by PETSCHEADER. 

     PETSCHEADERCREATE should be used whenever you create a PETSc structure.

     CHKSAME checks if you PETSc structures are of same type.

*/

#define PETSCHEADER                        \
  double      flops,time;                  \
  int         cookie;                      \
  int         type;                        \
  int         id;                          \
  int         refct;                       \
  int         (*destroy)(PetscObject);     \
  int         (*view)(PetscObject,Viewer); \
  MPI_Comm    comm;                        \
  PetscObject parent;                      \
  char*       name;                        \
  /*  ... */                               \

#define  FREEDHEADER -1

#define PETSCHEADERCREATE(h,tp,cook,t,com)                         \
      {h = (struct tp *) NEW(struct tp);                           \
       CHKPTR((h));                                                \
       MEMSET(h,0,sizeof(struct tp));                              \
       (h)->cookie = cook;                                         \
       (h)->type = t;                                              \
       MPI_Comm_dup(com,&(h)->comm);}
#define PETSCHEADERDESTROY(h)                                      \
       {MPI_Comm_free(&(h)->comm);                                 \
        (h)->cookie = FREEDHEADER;                                 \
        FREE(h);          }

extern void *PetscLow,*PetscHigh;

#if defined(PETSC_MALLOC) && !defined(PETSC_INSIGHT)
#define VALIDHEADER(h,ck)                             \
  {if (!h) {SETERR(1,"Null Object");}                 \
  if (PetscLow > (void *) h || PetscHigh < (void *)h){\
    SETERR(3,"Invalid Pointer to Object");            \
  }                                                   \
  if ((h)->cookie != ck) {                            \
    if ((h)->cookie == FREEDHEADER) {                 \
      SETERR(1,"Object already free");                \
    }                                                 \
    else {                                            \
      SETERR(2,"Invalid or Wrong Object");            \
    }                                                 \
  }}
#else
#define VALIDHEADER(h,ck)                             \
  {if (!h) {SETERR(1,"Null Object");}                 \
  if ((h)->cookie != ck) {                            \
    if ((h)->cookie == FREEDHEADER) {                 \
      SETERR(1,"Object already free");                \
    }                                                 \
    else {                                            \
      SETERR(2,"Invalid or Wrong Object");            \
    }                                                 \
  }}
#endif

#define CHKSAME(a,b) \
  if ((a)->type != (b)->type) SETERR(3,"Objects not of same type");

#define CHKTYPE(a,b) (((a)->type & (b)) ? 1 : 0)

struct _PetscObject {
  PETSCHEADER
};


#endif
