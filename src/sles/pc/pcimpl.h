/* $Id: pcimpl.h,v 1.11 1996/01/12 03:52:27 bsmith Exp bsmith $ */

#ifndef _PCIMPL
#define _PCIMPL

#include "vec.h"
#include "mat.h"
#include "ksp.h"
#include "pc.h"

/*
   Preconditioner context
*/
struct _PC {
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
  int          (*applysymmleft)(PC,Vec,Vec),(*applysymmright)(PC,Vec,Vec);
};

struct _PCNullSpace {
  PETSCHEADER
  int         has_cnst;
  int         n;
  Vec*        vecs;
};

#endif
