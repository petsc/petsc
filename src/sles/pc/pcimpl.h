/* $Id: pcimpl.h,v 1.19 1998/01/12 15:54:59 bsmith Exp bsmith $ */

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
  PETSCHEADER(int dummy)
  int          setupcalled;
  MatStructure flag;
  int          (*apply)(PC,Vec,Vec),(*setup)(PC),(*applyrich)(PC,Vec,Vec,Vec,int),
               (*applyBA)(PC,int,Vec,Vec,Vec),(*setfromoptions)(PC),(*printhelp)(PC,char*),
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
  PETSCHEADER(int dummy)
  int         has_cnst;
  int         n;
  Vec*        vecs;
};


#endif
