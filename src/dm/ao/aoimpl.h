/* $Id: aoimpl.h,v 1.7 1997/10/10 04:07:40 bsmith Exp bsmith $ */
/* 
   This private file should not be included in users' code.
*/

#ifndef __AOIMPL 
#define __AOIMPL
#include "ao.h"

/*
    Defines the abstract AO operations
*/
struct _AOOps {
  int  (*petsctoapplication)(AO,int,int*),  /* map a set of integers to application order */
       (*applicationtopetsc)(AO,int,int*);   
};

struct _p_AO {
  PETSCHEADER                            /* general PETSc header */
  struct _AOOps ops;                     /* AO operations */
  void          *data;                   /* implementation-specific data */
  int           N, n;                    /* global, local vector size */
};

/*
    Defines the abstract AOData operations
*/
struct _AODataOps {
  int (*segmentadd)(AOData,char *,char *,int,int,int*,void*,PetscDataType);
  int (*segmentget)(AOData,char *,char*,int,int*,void**);
  int (*segmentrestore)(AOData,char *,char *,int,int*,void**);
  int (*segmentgetlocal)(AOData,char *,char*,int,int*,void**);
  int (*segmentrestorelocal)(AOData,char *,char *,int,int*,void**);
  int (*segmentgetreduced)(AOData,char *,char*,int,int*,IS *);
  int (*keyremap)(AOData,char *,AO);
  int (*keygetadjacency)(AOData,char *,Mat*);
};

/*
      A AODate object consists of 

           - key1 
	       * name      = name of first key
               * N         = number of local keys 
               * nsegments = number of segments in first key  
               * ltog      = local to global mapping 
               - segment1 
                  * name      = name of first segment in first key
                  * bs        = blocksize of first segment in first key
                  * datatype  = datatype of first segment in first key

               - segment2

                  ....

            - key2

                ....
*/       
typedef struct {
  void              *data;                   /* implementation-specific data */
  char              *name;
  int               bs;                      /* block size of basic chunk */
  PetscDataType     datatype;                /* type of data item, int, double etc */
} AODataSegment;

typedef struct {
  void                   *data;                   /* implementation-specific data */
  char                   *name;
  int                    N;                       /* number of keys */
  int                    nsegments;               /* number of segments in key */
  int                    nsegments_max;           /* number of segments allocated for */
  AODataSegment          *segments;
  ISLocalToGlobalMapping ltog;

  /* should the following be so public? */

  int                    nlocal;                  /* number of keys owned locally */
  int                    *rowners;                /* ownership range of each processor */
  int                    rstart,rend;             /* first and 1 + last owned locally */
} AODataKey;

struct _p_AOData {
  PETSCHEADER                                /* general PETSc header */
  struct _AODataOps ops;                     /* AOData operations */
  int               nkeys_max;               /* number of keys allocated for */
  int               nkeys;                   /* current number of keys */
  AODataKey         *keys;
  void              *data;
  int               datacomplete;            /* indicates all AOData object is fully set */
};

extern int AODataKeyFind_Private(AOData, char *, int *,int *);
extern int AODataSegmentFind_Private(AOData,char *, char *, int *,int *,int *);


#endif
