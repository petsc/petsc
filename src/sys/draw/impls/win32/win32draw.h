#if !defined(__WIN32DRAW_H)
#define __WIN32DRAW_H

#include <stdio.h>
#if defined(PETSC_HAVE_WINDOWS_H)
#include <windows.h>
#endif
#if defined(PETSC_HAVE_WINDOWSX_H)
#include <windowsx.h>
#endif
#include <math.h>
#if defined(PETSC_HAVE_TIME_H)
#include <time.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include <petscdraw.h>

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
  PetscBool  DoubleBuffered;
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
  PetscBool  haveresized;
  HANDLE     hReadyEvent;
  int        x,y,w,h;  /* Size and location of window */
  WindowNode node;/* so we can grab windownode info if needed */
  DWORD      popup,caption,overlapped;
 
} PetscDraw_Win32;



#endif
