/* $Id: pcimpl.h,v 1.13 1996/03/04 05:15:17 bsmith Exp bsmith $ */

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
  int          (*applysymmetricleft)(PC,Vec,Vec),(*applysymmetricright)(PC,Vec,Vec);
  int          (*setuponblocks)(PC);
};

struct _PCNullSpace {
  PETSCHEADER
  int         has_cnst;
  int         n;
  Vec*        vecs;
};

#define USE_PRECONDITIONER_MATRIX 0
#define USE_TRUE_MATRIX           1

#endif
