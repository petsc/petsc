/* $Id: drawimpl.h,v 1.4 1995/07/07 17:16:34 bsmith Exp bsmith $ */

#if !defined(_DRAWIMPL_H)
#define _DRAWIMPL_H

#include "draw.h"

struct _DrawOps {
  int (*doublebuff)(DrawCtx);
  int (*flush)(DrawCtx);
  int (*drawline)(DrawCtx,double,double,double,double,int);
  int (*drawlinewidth)(DrawCtx,double);
  int (*drawpoint)(DrawCtx,double,double,int);
  int (*drawpointsize)(DrawCtx,double);
  int (*drawtext)(DrawCtx,double,double,int,char*);
  int (*drawtextvert)(DrawCtx,double,double,int,char*);
  int (*drawtextsize)(DrawCtx,double,double);
  int (*drawtextgetsize)(DrawCtx,double*,double*);
  int (*viewport)(DrawCtx,double,double,double,double);
  int (*clear)(DrawCtx);
  int (*sflush)(DrawCtx);
  int (*rectangle)(DrawCtx,double,double,double,double,int,int,int,int);
  int (*triangle)(DrawCtx,double,double,double,double,double,double,int,int,int);
};

struct _DrawCtx {
  PETSCHEADER
  struct _DrawOps *ops;
  int             pause;       /* sleep time after a sync flush */
  double          port_xl,port_yl,port_xr,port_yr;
  double          coor_xl,coor_yl,coor_xr,coor_yr;
  void            *data;
};

#endif
