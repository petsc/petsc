/* $Id: pcimpl.h,v 1.28 2000/08/01 20:02:56 bsmith Exp bsmith $ */

#ifndef _PCIMPL
#define _PCIMPL

#include "petscksp.h"
#include "petscpc.h"

typedef struct _PCOps *PCOps;
struct _PCOps {
  int          (*setup)(PC);
  int          (*apply)(PC,Vec,Vec);
  int          (*applyrichardson)(PC,Vec,Vec,Vec,int);
  int          (*applyBA)(PC,int,Vec,Vec,Vec);
  int          (*applytranspose)(PC,Vec,Vec);
  int          (*applyBAtranspose)(PC,int,Vec,Vec,Vec);
  int          (*setfromoptions)(PC);
  int          (*presolve)(PC,KSP,Vec,Vec);
  int          (*postsolve)(PC,KSP,Vec,Vec);  
  int          (*getfactoredmatrix)(PC,Mat*);
  int          (*applysymmetricleft)(PC,Vec,Vec);
  int          (*applysymmetricright)(PC,Vec,Vec);
  int          (*setuponblocks)(PC);
  int          (*destroy)(PC);
  int          (*view)(PC,Viewer);
};

/*
   Preconditioner context
*/
struct _p_PC {
  PETSCHEADER(struct _PCOps)
  int           setupcalled;
  MatStructure  flag;
  Mat           mat,pmat;
  Vec           vec;
  MatNullSpace  nullsp;
  int           (*modifysubmatrices)(PC,int,IS*,IS*,Mat*,void*); /* user provided routine */
  void          *modifysubmatricesP; /* context for user routine */
  void          *data;
};



#endif
