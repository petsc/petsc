/* $Id: pcimpl.h,v 1.17 1997/05/23 15:46:15 balay Exp balay $ */

#ifndef _PCIMPL
#define _PCIMPL

#include "vec.h"
#include "mat.h"
#include "ksp.h"
#include "pc.h"

/*
   Preconditioner context
*/
struct _p_PC {
  PETSCHEADER
  int          setupcalled;
  MatStructure flag;
  int          (*apply)(PC,Vec,Vec),(*setup)(PC),(*applyrich)(PC,Vec,Vec,Vec,int),
               (*applyBA)(PC,int,Vec,Vec,Vec),(*setfrom)(PC),(*printhelp)(PC,char*),
               (*applytrans)(PC,Vec,Vec),(*applyBAtrans)(PC,int,Vec,Vec,Vec);
  int          (*presolve)(PC,KSP),(*postsolve)(PC,KSP);
  Mat          mat,pmat;
  Vec          vec;
  void         *data;
  int          (*getfactoredmatrix)(PC,Mat*);
  PCNullSpace  nullsp;
  int          (*applysymmetricleft)(PC,Vec,Vec),(*applysymmetricright)(PC,Vec,Vec);
  int          (*setuponblocks)(PC);
  int          (*modifysubmatrices)(PC,int,IS*,IS*,Mat*,void*);
  void         *modifysubmatricesP;
};

struct _p_PCNullSpace {
  PETSCHEADER
  int         has_cnst;
  int         n;
  Vec*        vecs;
};


#endif
