/* $Id: pcimpl.h,v 1.24 1999/04/21 18:17:08 bsmith Exp curfman $ */

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
  int          (*applytrans)(PC,Vec,Vec);
  int          (*applyBAtrans)(PC,int,Vec,Vec,Vec);
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
