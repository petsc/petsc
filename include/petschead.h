/* $Id: ptscimpl.h,v 1.10 1995/06/07 16:33:54 bsmith Exp bsmith $ */

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
      {h = (struct tp *) PETSCNEW(struct tp);                           \
       CHKPTRQ((h));                                                \
       PETSCMEMSET(h,0,sizeof(struct tp));                              \
       (h)->cookie = cook;                                         \
       (h)->type = t;                                              \
       MPI_Comm_dup(com,&(h)->comm);}
#define PETSCHEADERDESTROY(h)                                      \
       {MPI_Comm_free(&(h)->comm);                                 \
        (h)->cookie = FREEDHEADER;                                 \
        PETSCFREE(h);          }

extern void *PetscLow,*PetscHigh;

#if defined(PETSC_MALLOC) && !defined(PETSC_INSIGHT)
#define VALIDHEADER(h,ck)                             \
  {if (!h) {SETERRQ(1,"Null Object");}                 \
  if (PetscLow > (void *) h || PetscHigh < (void *)h){\
    SETERRQ(3,"Invalid Pointer to Object");            \
  }                                                   \
  if ((h)->cookie != ck) {                            \
    if ((h)->cookie == FREEDHEADER) {                 \
      SETERRQ(1,"Object already free");                \
    }                                                 \
    else {                                            \
      SETERRQ(2,"Invalid or Wrong Object");            \
    }                                                 \
  }}
#else
#define VALIDHEADER(h,ck)                             \
  {if (!h) {SETERRQ(1,"Null Object");}                 \
  if ((h)->cookie != ck) {                            \
    if ((h)->cookie == FREEDHEADER) {                 \
      SETERRQ(1,"Object already free");                \
    }                                                 \
    else {                                            \
      SETERRQ(2,"Invalid or Wrong Object");            \
    }                                                 \
  }}
#endif

#define CHKSAME(a,b) \
  if ((a)->type != (b)->type) SETERRQ(3,"Objects not of same type");

#define CHKTYPE(a,b) (((a)->type & (b)) ? 1 : 0)

struct _PetscObject {
  PETSCHEADER
};


#endif
