
#if !defined(_DRAWIMPL_H)
#define _DRAWIMPL_H

#include "ptscimpl.h"
#include "draw.h"

#define DRAW_COOKIE 0x101010

 
struct _DrawOps2d {
  int (*drawline)(DrawCtx,double,double,double,double,int,int);
  int (*drawtext)(DrawCtx,double,double,double,double,int,int);
};
struct _DrawOps3d {
  int (*drawline)(DrawCtx,double,double,double,double,int,int);
};

struct _DrawCtx {
  PETSCHEADER
  struct _DrawOps2d *ops2d;
  struct _DrawOps3d *ops3d;
  double            port_xl,port_yl,port_xr,port_yr;
  double            coor_xl,coor_yl,coor_xr,coor_yr;
  void              *data;
};

#endif
