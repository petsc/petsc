#if !defined(__WIN32DRAW_H)
#define __WIN32DRAW_H

#include <stdio.h>
#include <windows.h>
#include <windowsx.h>
#include <math.h>
#include <time.h>
#include <conio.h>
#include <stdlib.h>
#include "petscdraw.h"

/* Nodes that record mouse actions when needed */
typedef struct _p_MouseNode *MouseNode;
struct _p_MouseNode{
  PetscDrawButton Button;
  POINT      user;
  POINT      phys;
  MouseNode  mnext;
  int        Length;
};

/* nodes that contain handle to all user created windows */
typedef struct _p_WindowNode *WindowNode;
struct _p_WindowNode {
  HWND       hWnd;
  WindowNode wnext,wprev;
  HANDLE     event;
  MouseNode  MouseListHead;
  MouseNode  MouseListTail;
  BOOL       IsGetMouseOn;
  PetscTruth DoubleBuffered;
  HDC        Buffer,DoubleBuffer;
  HBITMAP    BufferBit,DoubleBufferBit;
  HGDIOBJ    store,dbstore;
  int        bitwidth,bitheight;
};

/* Nodes that hold all information about a windows device context */
typedef struct  {
  HDC        hdc;
  HWND       hWnd;
  int        linewidth;
  int        pointdiameter;
  COLORREF   currentcolor;
  int        stringheight;
  int        stringwidth;
  int        pause;
  PetscTruth haveresized;
  HANDLE     hReadyEvent;
  int        x,y,w,h;  /* Size and location of window */
  WindowNode node;/* so we can grab windownode info if needed */
  DWORD      popup,caption,overlapped;
 
} PetscDraw_Win32;



#endif
