
#ifndef _PCIMPL
#define _PCIMPL

#include "ptscimpl.h"
#include "vec.h"
#include "mat.h"
#include "pc.h"

#define PC_COOKIE         0x505050

/*
   Preconditioner context
*/
struct _PC {
  PETSCHEADER
  int  setupcalled;
  int  (*apply)(PC,Vec,Vec),(*setup)(PC),(*applyrich)(PC,Vec,Vec,Vec,int),
       (*setfrom)(PC),(*printhelp)(PC);
  Mat  mat;
  Vec  vec;
  void *data;
  char *namemethod;
};

#endif
