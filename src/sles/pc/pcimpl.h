/* $Id: pcimpl.h,v 1.25 1999/08/23 19:14:07 curfman Exp bsmith $ */

#ifndef _PCIMPL
#define _PCIMPL

#include "ksp.h"
#include "pc.h"

typedef struct _PCOps *PCOps;
struct _PCOps {
  int          (*setup)(PC);
  int          (*apply)(PC,Vec,Vec);
  int          (*applyrichardson)(PC,Vec,Vec,Vec,int);
  int          (*applyBA)(PC,int,Vec,Vec,Vec);
  int          (*applytranspose)(PC,Vec,Vec);
  int          (*applyBAtranspose)(PC,int,Vec,Vec,Vec);
  int          (*setfromoptions)(PC);
  int          (*printhelp)(PC,char*);
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
  int          setupcalled;
  MatStructure flag;
  Mat          mat,pmat;
  Vec          vec;
  PCNullSpace  nullsp;
  int          (*modifysubmatrices)(PC,int,IS*,IS*,Mat*,void*); /* user provided routine */
  void         *modifysubmatricesP; /* context for user routine */
  void         *data;
};

/*
   Null space context for preconditioner
*/
struct _p_PCNullSpace {
  PETSCHEADER(int)
  int         has_cnst;
  int         n;
  Vec*        vecs;
};


#endif
