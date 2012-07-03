
#include <petscsys.h>
#include <../src/sys/draw/drawimpl.h>
#include <../src/sys/draw/impls/win32/win32draw.h>

#define IDC_FOUR       109
#define IDI_FOUR       107
#define IDM_EXIT       105
#define IDR_POPUP      103
#define MAX_LOADSTRING 100

#if !defined(SelectPen)
#define SelectPen(hdc, hpen)      ((HPEN)SelectObject((hdc), (HGDIOBJ)(HPEN)(hpen)))
#endif
#if !defined(SelectFont)
#define SelectFont(hdc,hfont)    ((HFONT)SelectObject((hdc), (HGDIOBJ)(HFONT)(hfont)))
#endif
#if !defined(SelectBrush)
#define SelectBrush(hdc,hbrush) ((HBRUSH)SelectObject((hdc), (HGDIOBJ)(HBRUSH)(hbrush)))
#endif
#if !defined(GetStockBrush)
#define GetStockBrush(i)      ((HBRUSH)GetStockObject(i))
#endif

#define XTRANS(draw,win,x) \
   (int)(((win)->w)*((draw)->port_xl + (((x - (draw)->coor_xl)*\
                                   ((draw)->port_xr - (draw)->port_xl))/\
                                   ((draw)->coor_xr - (draw)->coor_xl))))
#define YTRANS(draw,win,y) \
   (int)(((win)->h)*(1.0-(draw)->port_yl - (((y - (draw)->coor_yl)*\
                                   ((draw)->port_yr - (draw)->port_yl))/\
                                   ((draw)->coor_yr - (draw)->coor_yl))))

HINSTANCE     hInst;
HANDLE        g_hWindowListMutex = NULL;
WindowNode    WindowListHead     = NULL;

/* Hard coded color hue until hue.c works with this */
unsigned char RedMap[]   = {255,0,255,0,0,0,255,127,34,255,238,165,255,255,190,255,255,238,0,255,105,154,135,0,0,244,152,176,220,216,50,255};
unsigned char GreenMap[] = {255,0,0,255,255,0,0,255,139,165,130,42,182,127,190,255,215,162,197,246,105,205,206,100,0,164,245,224,17,191,205,240};
unsigned char BlueMap[]  = {255,0,0,0,255,255,225,212,34,0,238,42,193,80,190,0,0,173,205,143,105,50,235,0,128,96,255,230,120,216,50,245};

/* Foward declarations of functions included in this code module: */
LRESULT  CALLBACK PetscWndProc(HWND, UINT, WPARAM, LPARAM);
static PetscErrorCode TranslateColor_Win32(PetscDraw,int);
static PetscErrorCode AverageColorRectangle_Win32(PetscDraw,int,int,int,int);
static PetscErrorCode AverageColorTriangle_Win32(PetscDraw,int,int,int);
static PetscErrorCode deletemouselist_Win32(WindowNode);
static void OnPaint_Win32(HWND);
static void OnDestroy_Win32(HWND);
static PetscErrorCode MouseRecord_Win32(HWND,PetscDrawButton);
static PetscErrorCode PetscDrawGetPopup_Win32(PetscDraw,PetscDraw *);

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetDoubleBuffer_Win32" 
static PetscErrorCode PetscDrawSetDoubleBuffer_Win32(PetscDraw draw)
{
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  HDC             hdc      = GetDC(windraw->hWnd);
  
  PetscFunctionBegin;
  windraw->node->DoubleBuffer = CreateCompatibleDC(hdc);
  windraw->node->DoubleBufferBit = CreateCompatibleBitmap(hdc,windraw->w,windraw->h);
  windraw->node->dbstore = SelectObject(windraw->node->DoubleBuffer,windraw->node->DoubleBufferBit);
  /* Fill background of second buffer */
  ExtFloodFill(windraw->node->DoubleBuffer,0,0,COLOR_WINDOW,FLOODFILLBORDER);
  /* Copy current buffer into seconf buffer and set window data as double buffered */
  BitBlt(windraw->node->DoubleBuffer,
         0,0,
         windraw->w,windraw->h,
         windraw->node->Buffer,
         0,0,
         SRCCOPY);

  windraw->node->DoubleBuffered = PETSC_TRUE;
  ReleaseDC(windraw->hWnd,hdc);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawFlush_Win32" 
static PetscErrorCode PetscDrawFlush_Win32(PetscDraw draw)
{
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  HDC             hdc = GetDC(windraw->hWnd);
  
  PetscFunctionBegin;
  /* flush double buffer into primary buffer */
  BitBlt(windraw->node->Buffer,
         0,0,
         windraw->w,windraw->h,
         windraw->node->DoubleBuffer,
         0,0,
         SRCCOPY);
  /* flush double buffer into window */
  BitBlt(hdc,
         0,0,
         windraw->w,windraw->h,
         windraw->node->DoubleBuffer,
         0,0,
         SRCCOPY);
  ReleaseDC(windraw->hWnd,hdc);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "deletemouselist_Win32" 
static PetscErrorCode deletemouselist_Win32(WindowNode deletelist)
{ 
  /* Called upon window close. Frees memory of linked list of stored mouse commands */
  MouseNode node;
  
  while(deletelist->MouseListHead != NULL) {       
    node = deletelist->MouseListHead;
    if(deletelist->MouseListHead->mnext != NULL) {
      deletelist->MouseListHead = deletelist->MouseListHead->mnext;
    }
    PetscFree(node);
  }
  deletelist->MouseListHead = deletelist->MouseListTail = NULL;
  if (deletelist->wprev != NULL) {
    deletelist->wprev->wnext = deletelist->wnext;
  }
  if (deletelist->wnext != NULL) {
    deletelist->wnext->wprev = deletelist->wprev;
  }
  PetscFree(deletelist);
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawGetMouseButton_Win32" 
static PetscErrorCode PetscDrawGetMouseButton_Win32(PetscDraw draw, PetscDrawButton *button,PetscReal *x_user,PetscReal *y_user,PetscReal *x_phys,PetscReal *y_phys)
{
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  WindowNode      current;
  MouseNode       node=0;
  
  PetscFunctionBegin;
  /* Make sure no other code is using the linked list at this moment */
  WaitForSingleObject(g_hWindowListMutex, INFINITE);
  /* Look for the node that matches the window you are using */
  current = WindowListHead;
  while (current != NULL) {
    if(current->hWnd == windraw->hWnd) {       
      current->IsGetMouseOn = TRUE;
      break;
    } else {
      current = current->wnext;
    }
  }
  /* If no actions have occured, wait for one */
  node = current->MouseListHead;
  if (!node) {
    ReleaseMutex(g_hWindowListMutex);
    WaitForSingleObject(current->event, INFINITE);
    WaitForSingleObject(g_hWindowListMutex, INFINITE);
  }
  /* once we have the information, assign the pointers to it */
  *button = current->MouseListHead->Button;
  *x_user = current->MouseListHead->user.x;
  *y_user = current->MouseListHead->user.y;
  /* optional arguments */
  if (x_phys) *x_phys = current->MouseListHead->phys.x;
  if (y_phys) *y_phys = current->MouseListHead->phys.y;
  /* remove set of information from sub linked-list, delete the node */
  current->MouseListHead = current->MouseListHead->mnext;
  if (!current->MouseListHead) {
    ResetEvent(current->event);
    current->MouseListTail = NULL;
  }
  if (node) PetscFree(node);

  /* Release mutex so that  other code can use
     the linked list now that we are done with it */
  ReleaseMutex(g_hWindowListMutex);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawPause_Win32" 
static PetscErrorCode PetscDrawPause_Win32(PetscDraw draw)
{
  PetscFunctionBegin;
  PetscSleep(draw->pause);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "TranslateColor_Win32" 
static PetscErrorCode TranslateColor_Win32(PetscDraw draw,int color)
{
  /* Maps single color value into the RGB colors in our tables */
  PetscDraw_Win32 *windraw   = (PetscDraw_Win32*)draw->data;
  windraw->currentcolor = RGB(RedMap[color],GreenMap[color],BlueMap[color]);
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "AverageColorRectangle_Win32"
static PetscErrorCode AverageColorRectangle_Win32(PetscDraw draw,int c1,int c2, int c3, int c4)
{
  /* Averages colors given at points of rectangle and sets color from color table
    will be changed once the color gradient problem is worked out */
  PetscDraw_Win32 *windraw   = (PetscDraw_Win32*)draw->data;
  windraw->currentcolor = RGB(((RedMap[c1]+RedMap[c2]+RedMap[c3]+RedMap[c4])/4),
                              ((GreenMap[c1]+GreenMap[c2]+GreenMap[c3]+GreenMap[c4])/4),
                              ((BlueMap[c1]+BlueMap[c2]+BlueMap[c3]+BlueMap[c4])/4));
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "AverageColorTriangle_Win32"
static PetscErrorCode AverageColorTriangle_Win32(PetscDraw draw,int c1,int c2,int c3)
{
  /* Averages colors given at points of rectangle and sets color from color table
    will be changed once the color gradient problem is worked out */
  PetscDraw_Win32 *windraw   = (PetscDraw_Win32*)draw->data;
  windraw->currentcolor = RGB((RedMap[c1]+RedMap[c2]+RedMap[c3])/3,
                              (GreenMap[c1]+GreenMap[c2]+GreenMap[c3])/3,
                              (BlueMap[c1]+BlueMap[c2]+BlueMap[c3])/3); 
  return 0;
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawRectangle_Win32"
static PetscErrorCode PetscDrawRectangle_Win32(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int c1,int c2,int c3,int c4)
{
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  HBRUSH          hbrush;
  RECT            rect;
  int             x1,yone,x2,y2;
  HDC             hdc;
  
  PetscFunctionBegin;
  x1 = XTRANS(draw,windraw,xl);
  x2 = XTRANS(draw,windraw,xr);
  yone = YTRANS(draw,windraw,yl);
  y2 = YTRANS(draw,windraw,yr);
  SetRect(&rect,x1,y2,x2,yone);        
  if (c1==c2 && c2==c3 && c3==c4) {         
    TranslateColor_Win32(draw,c1);
  } else {                                   
    AverageColorRectangle_Win32(draw,c1,c2,c3,c4);
  }
  hbrush = CreateSolidBrush(windraw->currentcolor);
  
  if(windraw->node->DoubleBuffered) {
    hdc = windraw->node->DoubleBuffer;
  } else {
    hdc = windraw->node->Buffer;
  }
  FillRect(hdc,&rect,hbrush);
  /* Forces a WM_PAINT message and erases background */
  InvalidateRect(windraw->hWnd,NULL,TRUE);
  UpdateWindow(windraw->hWnd);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLine_Win32"
static PetscErrorCode PetscDrawLine_Win32(PetscDraw draw,PetscReal xl,PetscReal yl,PetscReal xr,PetscReal yr,int color)
{
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  HPEN            hpen;
  int             x1,yone,x2,y2;
  HDC             hdc;
  
  PetscFunctionBegin;
  TranslateColor_Win32(draw,color);
  x1   = XTRANS(draw,windraw,xl);x2  = XTRANS(draw,windraw,xr); 
  yone   = YTRANS(draw,windraw,yl);y2  = YTRANS(draw,windraw,yr); 
  hpen = CreatePen (PS_SOLID, windraw->linewidth, windraw->currentcolor);
  if(windraw->node->DoubleBuffered) {
    hdc = windraw->node->DoubleBuffer;
  } else {
    hdc = windraw->node->Buffer;
  }
  SelectPen(hdc,hpen);
  MoveToEx(hdc,x1,yone,NULL);
  LineTo(hdc,x2,y2);
  /* Forces a WM_PAINT message and erases background */
  InvalidateRect(windraw->hWnd,NULL,TRUE);
  UpdateWindow(windraw->hWnd);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLineSetWidth_Win32"
static PetscErrorCode PetscDrawLineSetWidth_Win32(PetscDraw draw,PetscReal width)
{
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  int             averagesize,finalwidth;
  RECT            rect;
  
  PetscFunctionBegin;
  GetClientRect(windraw->hWnd,&rect);
  averagesize = ((rect.right - rect.left)+(rect.bottom - rect.top))/2;
  finalwidth  = (int)floor(averagesize*width);
  if (finalwidth < 1) {
    finalwidth = 1; /* minimum size PetscDrawLine can except */
  }
  windraw->linewidth = finalwidth;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawLineGetWidth_Win32"
static PetscErrorCode PetscDrawLineGetWidth_Win32(PetscDraw draw,PetscReal *width)
{
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  
  PetscFunctionBegin;
  *width = (PetscReal)windraw->linewidth;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawPoint_Win32"
static PetscErrorCode PetscDrawPoint_Win32(PetscDraw draw,PetscReal x,PetscReal y,int color)
{       
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  HBRUSH          hbrush;
  HRGN            hrgn;
  int             radius;
  int             x1,yone;
  HDC             hdc;
  
  PetscFunctionBegin;
  TranslateColor_Win32(draw,color);
  x1     = XTRANS(draw,windraw,x);   
  yone     = YTRANS(draw,windraw,y);
  hbrush = CreateSolidBrush(windraw->currentcolor);
  if(windraw->node->DoubleBuffered) {
    hdc = windraw->node->DoubleBuffer;
  } else {
    hdc = windraw->node->Buffer;
  }
  /* desired size is one logical pixel so just turn it on */
  if (windraw->pointdiameter == 1) {
    SetPixelV(hdc,x1,yone,windraw->currentcolor);
  } else {
    /* draw point around position determined */
    radius = windraw->pointdiameter/2; /* integer division */
    hrgn   = CreateEllipticRgn(x1-radius,yone-radius,x1+radius,yone+radius);
    FillRgn(hdc,hrgn,hbrush);
  }
  /* Forces a WM_PAINT and erases background */
  InvalidateRect(windraw->hWnd,NULL,TRUE);
  UpdateWindow(windraw->hWnd);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawPointSetSize_Win32"
static PetscErrorCode PetscDrawPointSetSize_Win32(PetscDraw draw,PetscReal width)
{
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  int             averagesize,diameter;
  RECT            rect;
  
  PetscFunctionBegin;
  GetClientRect(windraw->hWnd,&rect);
  averagesize = ((rect.right - rect.left)+(rect.bottom - rect.top))/2;
  diameter    = (int)floor(averagesize*width);
  if (diameter < 1) diameter = 1;
  windraw->pointdiameter     = diameter;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawString_Win32"
static PetscErrorCode PetscDrawString_Win32(PetscDraw draw,PetscReal x,PetscReal y,int color,const char *text)
{       
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  RECT            r;
  HFONT           hfont;                                                                    
  LOGFONT         logfont; 
  int             x1,yone;
  HDC             hdc;
  
  PetscFunctionBegin;
  x1              = XTRANS(draw,windraw,x);
  yone              = YTRANS(draw,windraw,y);
  r.bottom        = yone;
  r.left          = x1;
  r.right         = x1 + 1; 
  r.top           = yone + 1;
  logfont.lfHeight         = windraw->stringheight;
  logfont.lfWidth          = windraw->stringwidth;
  logfont.lfEscapement     = 0;
  logfont.lfOrientation    = 0;
  logfont.lfCharSet        = 0;
  logfont.lfClipPrecision  = 0;
  logfont.lfItalic         = 0;
  logfont.lfOutPrecision   = 0;
  logfont.lfPitchAndFamily = DEFAULT_PITCH;
  logfont.lfQuality        = DEFAULT_QUALITY;
  logfont.lfStrikeOut      = 0;
  logfont.lfUnderline      = 0;
  logfont.lfWeight         = FW_NORMAL;
  hfont = CreateFontIndirect(&logfont); 
  TranslateColor_Win32(draw,color);
  if(windraw->node->DoubleBuffered) {
    hdc = windraw->node->DoubleBuffer;
  } else {
    hdc = windraw->node->Buffer;
  }
  SelectFont(hdc,hfont);
  SetTextColor(hdc,windraw->currentcolor);
  DrawText(hdc,text,lstrlen(text),&r,DT_NOCLIP);
  DeleteObject(hfont);
  /* Forces a WM_PAINT message and erases background */
  InvalidateRect(windraw->hWnd,NULL,TRUE);
  UpdateWindow(windraw->hWnd);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringVertical_Win32"
static PetscErrorCode PetscDrawStringVertical_Win32(PetscDraw draw,PetscReal x,PetscReal y,int color,const char *text)
{       
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  RECT            r;
  HFONT           hfont;                                                                                    
  LOGFONT         logfont;
  int             x1,yone;
  HDC             hdc;
  
  PetscFunctionBegin;
  x1           = XTRANS(draw,windraw,x);
  yone           = YTRANS(draw,windraw,y);
  r.left       = x1;
  r.bottom     = yone + 30;
  r.right      = x1 + 1;
  r.top        = yone - 30;
  logfont.lfEscapement     = 2700; /* Causes verticle text drawing */
  logfont.lfHeight         = windraw->stringheight;
  logfont.lfWidth          = windraw->stringwidth;
  logfont.lfOrientation    = 0;
  logfont.lfCharSet        = DEFAULT_CHARSET;
  logfont.lfClipPrecision  = 0;
  logfont.lfItalic         = 0;
  logfont.lfOutPrecision   = 0;
  logfont.lfPitchAndFamily = DEFAULT_PITCH;
  logfont.lfQuality        = DEFAULT_QUALITY;
  logfont.lfStrikeOut      = 0;
  logfont.lfUnderline      = 0;
  logfont.lfWeight         = FW_NORMAL;
  hfont = CreateFontIndirect(&logfont);
  TranslateColor_Win32(draw,color);
  if(windraw->node->DoubleBuffered) {
    hdc = windraw->node->DoubleBuffer;
  } else {
    hdc = windraw->node->Buffer;
  }
  SelectFont(hdc,hfont);
  SetTextColor(hdc,windraw->currentcolor);
  DrawText(hdc,text,lstrlen(text),&r,DT_NOCLIP | DT_SINGLELINE );
  DeleteObject(hfont);
  /* Forces a WM_PAINT message and erases background */
  InvalidateRect(windraw->hWnd,NULL,TRUE);
  UpdateWindow(windraw->hWnd);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringSetSize_Win32"
static PetscErrorCode PetscDrawStringSetSize_Win32(PetscDraw draw,PetscReal width,PetscReal height)
{       
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  int             w,h;
  
  PetscFunctionBegin;
  w = (int)((windraw->w)*width *(draw->port_xr - draw->port_xl)/(draw->coor_xr - draw->coor_xl));
  h = (int)((windraw->h)*height*(draw->port_yr - draw->port_yl)/(draw->coor_yr - draw->coor_yl));
  if (h < 1) h = 1;
  if (w < 1) w = 1;
  windraw->stringheight = h;
  windraw->stringwidth  = w;
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawStringGetSize_Win32"
static PetscErrorCode PetscDrawStringGetSize_Win32(PetscDraw draw,PetscReal *width,PetscReal *height)
{       
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  double scaleX = (draw->coor_xr - draw->coor_xl)/(draw->w)*(draw->port_xr - draw->port_xl);
  double scaleY = (draw->coor_yr - draw->coor_yl)/(draw->h)*(draw->port_yr - draw->port_yl);
  
  PetscFunctionBegin;
  *height = (double)windraw->stringheight*scaleY;
  *width  = (double)windraw->stringwidth*scaleX;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawResizeWindow_Win32"
static PetscErrorCode PetscDrawResizeWindow_Win32(PetscDraw draw,int w,int h)
{       
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  RECT            r;
  
  PetscFunctionBegin;
  GetWindowRect(windraw->hWnd,&r);
  MoveWindow(windraw->hWnd,r.left,r.top,(int)w,(int)h,TRUE);
  /* set all variable dealing with window dimensions */
  windraw->node->bitheight = windraw->h = draw->h = h;
  windraw->node->bitwidth  = windraw->w = draw->w = w;
  /* set up graphic buffers with the new size of window */
  SetBitmapDimensionEx(windraw->node->BufferBit,w,h,NULL);
  if(windraw->node->DoubleBuffered) {
    SetBitmapDimensionEx(windraw->node->DoubleBufferBit,w,h,NULL);
  }
  windraw->haveresized = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawCheckResizeWindow_Win32"
static PetscErrorCode PetscDrawCheckResizedWindow_Win32(PetscDraw draw)
{       
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  
  PetscFunctionBegin;
  if (windraw->haveresized == 1) {
    PetscFunctionReturn(1);
  } else {
    PetscFunctionReturn(0);
  }
  
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSetTitle_Win32"
static PetscErrorCode PetscDrawSetTitle_Win32(PetscDraw draw, const char title[])
{
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  
  PetscFunctionBegin;
  SetWindowText(windraw->hWnd,title);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawClear_Win32"
static PetscErrorCode PetscDrawClear_Win32(PetscDraw draw)
{       
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  
  PetscFunctionBegin;
  /* clear primary buffer */
  ExtFloodFill(windraw->node->Buffer,0,0,COLOR_WINDOW,FLOODFILLBORDER);
  /* if exists clear secondary buffer */
  if(windraw->node->DoubleBuffered) {
    ExtFloodFill(windraw->node->DoubleBuffer,0,0,COLOR_WINDOW,FLOODFILLBORDER);
  }
  /* force WM_PAINT message so cleared buffer will show */
  InvalidateRect(windraw->hWnd,NULL,TRUE);
  UpdateWindow(windraw->hWnd);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawTriangle_Win32"
static PetscErrorCode PetscDrawTriangle_Win32(PetscDraw draw,PetscReal x1,PetscReal yone,PetscReal x2,PetscReal y2,
			      PetscReal x3,PetscReal y3,int c1,int c2,int c3)
{       
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  HBRUSH          hbrush;
  HPEN            hpen;
  int             p1x,p1y,p2x,p2y,p3x,p3y;
  HDC             bit;
  
  PetscFunctionBegin;
  AverageColorTriangle_Win32(draw,c1,c2,c3); 
  hbrush = CreateSolidBrush(windraw->currentcolor);
  hpen   = CreatePen(PS_SOLID,0,windraw->currentcolor);
  p1x = XTRANS(draw,windraw,x1);
  p2x = XTRANS(draw,windraw,x2);
  p3x = XTRANS(draw,windraw,x3);
  p1y = YTRANS(draw,windraw,yone);
  p2y = YTRANS(draw,windraw,y2);
  p3y = YTRANS(draw,windraw,y3);
  
  if(windraw->node->DoubleBuffered) {
    bit = windraw->node->DoubleBuffer;
  } else {
    bit = windraw->node->Buffer;
  }
  BeginPath(bit);
  MoveToEx(bit,p1x,p1y,NULL);
  LineTo(bit,p2x,p2y);
  LineTo(bit,p3x,p3y);
  LineTo(bit,p1x,p1y);
  EndPath(bit);
  SelectPen(bit,hpen);
  SelectBrush(bit,hbrush);
  StrokeAndFillPath(bit);
  /* Forces a WM_PAINT message and erases background */
  InvalidateRect(windraw->hWnd,NULL,TRUE);
  UpdateWindow(windraw->hWnd);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PopMessageLoopThread_Win32"
void PopMessageLoopThread_Win32(PetscDraw popdraw)
{
  PetscDraw_Win32 *pop = (PetscDraw_Win32*)popdraw->data;
  MSG             msg;
  HWND            hWnd = NULL;
  char            PopClassName [MAX_LOADSTRING + 1]; 
  RECT            r;
  int             width,height;
  WNDCLASSEX      myclass;
  LPVOID          lpMsgBuf;
  
  PetscFunctionBegin;
  /* initialize window class parameters */
  myclass.cbSize        = sizeof(WNDCLASSEX);
  myclass.style         = CS_OWNDC;
  myclass.lpfnWndProc   = (WNDPROC)PetscWndProc;
  myclass.cbClsExtra    = 0;
  myclass.cbWndExtra    = 0;
  myclass.hInstance     = NULL;
  myclass.hIcon         = NULL;
  myclass.hCursor       = LoadCursor(NULL, IDC_ARROW);
  myclass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
  myclass.lpszMenuName  = NULL;
  myclass.lpszClassName = PopClassName;
  myclass.hIconSm       = NULL;
  
  RegisterClassEx(&myclass);
  
  SetRect(&r,0,0,450,450);
  
  width    = (r.right - r.left) / 3;
  height   = (r.bottom - r.top) / 3;
  
  hWnd = CreateWindowEx(0,
                        PopClassName,
                        NULL, 
                        WS_POPUPWINDOW | WS_CAPTION,
                        0,0, 
                        width,height,
                        NULL,
                        NULL,
                        hInst,
                        NULL);
  pop->x = 0;
  pop->y = 0;
  pop->w = width;
  pop->h = height;
  
  if(!hWnd) {
    lpMsgBuf = (LPVOID)"Window Not Succesfully Created";
    MessageBox( NULL, (LPCTSTR)lpMsgBuf, "Error", MB_OK | MB_ICONINFORMATION );
    LocalFree( lpMsgBuf );
    exit(0);
  }
  pop->hWnd = hWnd;
  /* display and update new popup window */
  ShowWindow(pop->hWnd, SW_SHOWNORMAL);
  UpdateWindow(pop->hWnd);
  SetEvent(pop->hReadyEvent);
  
  while (GetMessage(&msg, pop->hWnd, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
  PetscFunctionReturnVoid();
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawDestroy_Win32"
static PetscErrorCode PetscDrawDestroy_Win32(PetscDraw draw)
{
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  
  PetscFunctionBegin;
  SendMessage(windraw->hWnd,WM_DESTROY,0,0);
  PetscFree(windraw);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSynchronizedFlush_Win32"
static PetscErrorCode PetscDrawSynchronizedFlush_Win32(PetscDraw draw)
{
  /* Multi Processor is not implemeted yet */
  PetscFunctionBegin;
  PetscDrawFlush_Win32(draw);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawSynchronizedClear_Win32"
static PetscErrorCode PetscDrawSynchronizedClear_Win32(PetscDraw draw)
{
  /* Multi Processor is not implemeted yet */
  PetscFunctionBegin;
  PetscDrawClear_Win32(draw);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MessageLoopThread_Win32"
void MessageLoopThread_Win32(PetscDraw draw)
{
  PetscDraw_Win32 *windraw = (PetscDraw_Win32*)draw->data;
  MSG             msg;
  HWND            hWnd = NULL;
  char            classname[MAX_LOADSTRING + 1];
  WNDCLASSEX      wclass;
  LPVOID          lpMsgBuf;
  
  PetscFunctionBegin;
  /* initialize window class parameters */
  wclass.cbSize         = sizeof(WNDCLASSEX);
  wclass.style          = CS_SAVEBITS | CS_HREDRAW | CS_VREDRAW;
  wclass.lpfnWndProc    = (WNDPROC)PetscWndProc;
  wclass.cbClsExtra     = 0;
  wclass.cbWndExtra     = 0;
  wclass.hInstance      = NULL;
  wclass.hIcon          = LoadIcon(NULL,IDI_APPLICATION);
  wclass.hCursor        = LoadCursor(NULL,IDC_ARROW);
  wclass.hbrBackground  = GetStockBrush(WHITE_BRUSH);
  wclass.lpszMenuName   = NULL;
  wclass.lpszClassName  = classname;
  wclass.hIconSm        = NULL;
  
  RegisterClassEx(&wclass);
  
  
  hWnd = CreateWindowEx(0,
                        classname,
                        NULL,
                        WS_OVERLAPPEDWINDOW,
                        draw->x,
                        draw->y, 
                        draw->w,
                        draw->h, 
                        NULL,
                        NULL,
                        hInst,
                        NULL);
  
  if (!hWnd) {
    lpMsgBuf = (LPVOID)"Window Not Succesfully Created";
    MessageBox( NULL, (LPCTSTR)lpMsgBuf, "Error", MB_OK | MB_ICONINFORMATION );
    LocalFree( lpMsgBuf );
    exit(0);
  }
  windraw->hWnd = hWnd;
  /* display and update new window */
  ShowWindow(hWnd,SW_SHOWNORMAL);
  UpdateWindow(hWnd);
  SetEvent(windraw->hReadyEvent);
  
  while (GetMessage(&msg,hWnd, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
  PetscFunctionReturnVoid();
}


static struct _PetscDrawOps DvOps = { PetscDrawSetDoubleBuffer_Win32,
                                 PetscDrawFlush_Win32,
                                 PetscDrawLine_Win32,
                                 PetscDrawLineSetWidth_Win32,
                                 PetscDrawLineGetWidth_Win32,
                                 PetscDrawPoint_Win32,
                                 PetscDrawPointSetSize_Win32,
                                 PetscDrawString_Win32,
                                 PetscDrawStringVertical_Win32,
                                 PetscDrawStringSetSize_Win32,
                                 PetscDrawStringGetSize_Win32,
                                 0,
                                 PetscDrawClear_Win32,
                                 PetscDrawSynchronizedFlush_Win32,
                                 PetscDrawRectangle_Win32,
                                 PetscDrawTriangle_Win32,
                                 0,
                                 PetscDrawGetMouseButton_Win32,
                                 PetscDrawPause_Win32,
                                 PetscDrawSynchronizedClear_Win32,
                                 0,
                                 0,
                                 PetscDrawGetPopup_Win32,
                                 PetscDrawSetTitle_Win32,
                                 PetscDrawCheckResizedWindow_Win32,
                                 PetscDrawResizeWindow_Win32,
                                 PetscDrawDestroy_Win32,
                                 0,
                                 0,
                                 0,
                                 0};

#undef __FUNCT__
#define __FUNCT__ "PetscDrawGetPopup_Win32"
static PetscErrorCode PetscDrawGetPopup_Win32(PetscDraw draw,PetscDraw *popdraw)
{
  PetscDraw_Win32 *pop;
  HANDLE          hThread = NULL;
  WindowNode      newnode;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = PetscNew(PetscDraw_Win32,&pop);CHKERRQ(ierr);
  (*popdraw)->data = pop;
  
  /* the following is temporary fix for initializing a global datastructure */
  if(!g_hWindowListMutex) {
    g_hWindowListMutex = CreateMutex(NULL,FALSE,NULL);
  }
  ierr = PetscMemcpy((*popdraw)->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  
  pop->hReadyEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
  CreateThread(NULL, 0,(LPTHREAD_START_ROUTINE)PopMessageLoopThread_Win32,*popdraw,0,(unsigned long*)hThread);
  CloseHandle(hThread);
  WaitForSingleObject(pop->hReadyEvent, INFINITE);
  CloseHandle(pop->hReadyEvent);
  WaitForSingleObject(g_hWindowListMutex, INFINITE);
  
  draw->popup             = (*popdraw);
  ierr                    = PetscNew(struct _p_WindowNode,&newnode);CHKERRQ(ierr);
  newnode->MouseListHead  = NULL;
  newnode->MouseListTail  = NULL;
  newnode->wnext          = WindowListHead;
  newnode->wprev          = NULL;
  newnode->hWnd           = pop->hWnd;
  if(WindowListHead != NULL) {
    WindowListHead->wprev = newnode;
  }
  WindowListHead          = newnode;
  pop->hdc                = GetDC(pop->hWnd);
  
  pop->stringheight   = 10; 
  pop->stringwidth    = 6;
  pop->linewidth      = 1;   /* default pixel sizes of graphics until user changes them */
  pop->pointdiameter  = 1;
  pop->node           = newnode;
  
  newnode->bitwidth  = pop->w;
  newnode->bitheight = pop->h;
  
  /* Create and initialize primary graphics buffer */
  newnode->Buffer = CreateCompatibleDC(pop->hdc);
  newnode->BufferBit = CreateCompatibleBitmap(pop->hdc,pop->w,pop->h);
  newnode->store = SelectObject(newnode->Buffer,newnode->BufferBit);
  ExtFloodFill(newnode->Buffer,0,0,COLOR_WINDOW,FLOODFILLBORDER);
  
  
  newnode->event          = CreateEvent(NULL, TRUE, FALSE, NULL);
  newnode->DoubleBuffered = PETSC_FALSE;
  
  ReleaseDC(pop->hWnd,pop->hdc);
  ReleaseMutex(g_hWindowListMutex);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawCreate_Win32" 
PetscErrorCode  PetscDrawCreate_Win32(PetscDraw draw)
{       
  PetscDraw_Win32 *windraw;
  HANDLE          hThread = NULL;
  PetscErrorCode ierr;
  WindowNode      newnode;
  
  PetscFunctionBegin;
  ierr        = PetscNew(PetscDraw_Win32,&windraw);CHKERRQ(ierr);
  draw->data  = windraw;
  
  /* the following is temporary fix for initializing a global datastructure */
  if(!g_hWindowListMutex) {
    g_hWindowListMutex = CreateMutex(NULL,FALSE,NULL);
  }
  ierr = PetscMemcpy(draw->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  
  windraw->hReadyEvent = CreateEvent(NULL,TRUE,FALSE,NULL);
  /* makes call to MessageLoopThread to creat window and attach a thread */
  CreateThread(NULL,0,(LPTHREAD_START_ROUTINE)MessageLoopThread_Win32,draw,0,(unsigned long*)hThread);
  CloseHandle(hThread);
  WaitForSingleObject(windraw->hReadyEvent,INFINITE);
  CloseHandle(windraw->hReadyEvent);
  WaitForSingleObject(g_hWindowListMutex,INFINITE);
  
  ierr                    = PetscNew(struct _p_WindowNode,&newnode);CHKERRQ(ierr);
  newnode->MouseListHead  = NULL;
  newnode->MouseListTail  = NULL;
  newnode->wnext          = WindowListHead;
  newnode->wprev          = NULL;
  newnode->hWnd           = windraw->hWnd;
  if(WindowListHead != NULL) {
    WindowListHead->wprev = newnode;
  }
  WindowListHead          = newnode;
  windraw->hdc            = GetDC(windraw->hWnd);
  
  windraw->stringheight   = 10; 
  windraw->stringwidth    = 6;
  windraw->linewidth      = 1;   /* default pixel sizes of graphics until user changes them */
  windraw->pointdiameter  = 1;
  windraw->node           = newnode;
  
  windraw->x = draw->x;
  windraw->y = draw->y;
  windraw->w = newnode->bitwidth    = draw->w;
  windraw->h = newnode->bitheight   = draw->h;  
  
  /* Create and initialize primary graphics buffer */
  newnode->Buffer = CreateCompatibleDC(windraw->hdc);
  newnode->BufferBit = CreateCompatibleBitmap(windraw->hdc,windraw->w,windraw->h);
  newnode->store = SelectObject(newnode->Buffer,newnode->BufferBit);
  ExtFloodFill(newnode->Buffer,0,0,COLOR_WINDOW,FLOODFILLBORDER);
  
  newnode->event          = CreateEvent(NULL,TRUE,FALSE,NULL);
  newnode->DoubleBuffered = PETSC_FALSE;
  
  ReleaseDC(windraw->hWnd,windraw->hdc);
  ReleaseMutex(g_hWindowListMutex);
  PetscFunctionReturn(0);
}
EXTERN_C_END


/* FUNCTION: PetscWndProc(HWND, unsigned, WORD, LONG)
   PURPOSE:  Processes messages for the main window.
   WM_COMMAND  - process the application menu
   WM_PAINT    - Paint the main window
   WM_DESTROY  - post a quit message and return */

#undef __FUNCT__
#define __FUNCT__ "PetscWndProc"
LRESULT  CALLBACK PetscWndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
  int         wmId, wmEvent;
  
  PetscFunctionBegin;
  switch (message) {
    HANDLE_MSG(hWnd,WM_PAINT,OnPaint_Win32);
    HANDLE_MSG(hWnd,WM_DESTROY,OnDestroy_Win32);
  case WM_COMMAND:
    wmId    = LOWORD(wParam); 
    wmEvent = HIWORD(wParam); 
    /* Parse the menu selections:*/
    switch (wmId) {
    case IDM_EXIT:
      DestroyWindow(hWnd);
      break;
    default:
      return DefWindowProc(hWnd, message, wParam, lParam);
    }
    break;
  case WM_LBUTTONUP:
    MouseRecord_Win32(hWnd,PETSC_BUTTON_LEFT);
    break;
  case WM_RBUTTONUP:
    MouseRecord_Win32(hWnd,PETSC_BUTTON_RIGHT);
    break;
  case WM_MBUTTONUP:
    MouseRecord_Win32(hWnd,PETSC_BUTTON_CENTER);
    break; 
  default:
    PetscFunctionReturn(DefWindowProc(hWnd, message, wParam, lParam));
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OnPaint_Win32"
static void OnPaint_Win32(HWND hWnd) 
{
  PAINTSTRUCT ps;
  HDC         hdc;
  WindowNode  current = NULL;

  PetscFunctionBegin;
  InvalidateRect(hWnd,NULL,TRUE); 
  WaitForSingleObject(g_hWindowListMutex, INFINITE);
  current = WindowListHead;
  hdc     = BeginPaint(hWnd, &ps);
  
  while(current != NULL) {
    if (current->hWnd == hWnd) { 
      /* flushes primary buffer to window */
      BitBlt(hdc,
             0,0,
             GetDeviceCaps(hdc,HORZRES),
             GetDeviceCaps(hdc,VERTRES),
             current->Buffer,
             0,0,
             SRCCOPY);
      
      /* StretchBlt(hdc,
        0,0,
        w,h,
        current->Buffer,
        0,0,
        current->bitwidth,
        current->bitheight,
        SRCCOPY); */
      break;
    }
    current = current->wnext;
  }
  EndPaint(hWnd, &ps);
  ReleaseMutex(g_hWindowListMutex);
  PetscFunctionReturnVoid();
}

#undef __FUNCT__
#define __FUNCT__ "MouseRecord_Win32"
static PetscErrorCode MouseRecord_Win32(HWND hWnd,PetscDrawButton button) 
{
  /* Called by all three mouse button actions
    Records needed mouse data in windows data structure */
  WindowNode current = NULL;
  MouseNode  newnode;
  POINT      mousepos;
  PetscErrorCode ierr; 
  
  PetscFunctionBegin;
  WaitForSingleObject(g_hWindowListMutex, INFINITE);
  current = WindowListHead;
  if(current->IsGetMouseOn == TRUE) {
    
    SetEvent(current->event);
    while (current != NULL) {   
      if(current->hWnd == hWnd) {       
        
        ierr            = PetscNew(struct _p_MouseNode,&newnode);CHKERRQ(ierr);
        newnode->Button = button;
        GetCursorPos(&mousepos);
        newnode->user.x = mousepos.x;
        newnode->user.y = mousepos.y;
        ScreenToClient(hWnd,&mousepos);
        newnode->phys.x = mousepos.x;
        newnode->phys.y = mousepos.y;
        if (!current->MouseListTail) {
          current->MouseListHead = newnode;
          current->MouseListTail = newnode;
        } else {
          current->MouseListTail->mnext = newnode;
          current->MouseListTail = newnode;
        }
        newnode->mnext = NULL;
        
        break;
      } 
      current = current->wnext;
    }
  }
  ReleaseMutex(g_hWindowListMutex);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OnDestroy_Win32"
static void OnDestroy_Win32(HWND hWnd) 
{
  /* searches linked list of window data and frees corresponding memory */
  WindowNode current;
  
  PetscFunctionBegin;
  WaitForSingleObject(g_hWindowListMutex, INFINITE);
  current = WindowListHead;
  
  SetEvent(current->event);
  while (current != NULL) { 
    if(current->hWnd == hWnd) {
      if(current->wprev != NULL) {
        current->wprev->wnext = current->wnext;
      } else {
        WindowListHead = current->wnext;
      }
      if(current->MouseListHead) {
        deletemouselist_Win32(current);
      } else {
        PetscFree(current);
      }
      break;
    }
    current = current->wnext;
  }
  ReleaseMutex(g_hWindowListMutex);
  PostQuitMessage(0);
  PetscFunctionReturnVoid();
}
