/* $Id: ximpl.h,v 1.14 2000/05/05 22:13:35 balay Exp $ */

#include "src/sys/src/draw/drawimpl.h"
#include "src/sys/src/draw/impls/win32/win32draw.h"

#define IDC_FOUR       109
#define IDI_FOUR       107
#define IDM_ABOUT      104
#define IDM_EXIT       105
#define IDR_POPUP      103
#define MAX_LOADSTRING 100  

#define SelectPen(hdc, hpen) ((HPEN)SelectObject((hdc), (HGDIOBJ)(HPEN)(hpen)))
#define SelectFont(hdc,hfont) ((HFONT)SelectObject((hdc), (HGDIOBJ)(HFONT)(hfont)))
#define GetStockBrush(i)        ((HBRUSH)GetStockObject(i))


HINSTANCE     hInst;
HANDLE        g_hWindowListMutex = NULL;
WindowNode    WindowListHead = NULL;
PetscTruth    quitflag = PETSC_TRUE;
static double Gamma = 2.0;
const int     mapsize = 255;
unsigned char RedMap[]  = {255,0,255,0,0,0,255,127,34,255,238,165,255,255,190,255,255,238,0,255,105,154,135,0,0,244,152,176,220,216,50,255};
unsigned char GreenMap[] = {255,0,0,255,255,0,0,255,139,165,130,42,182,127,190,255,215,162,197,246,105,205,206,100,0,164,245,224,17,191,205,240};
unsigned char BlueMap[] = {255,0,0,0,255,255,225,212,34,0,238,42,193,80,190,0,0,173,205,143,105,50,235,0,128,96,255,230,120,216,50,245};

/* Foward declarations of functions included in this code module: */
LRESULT CALLBACK  WndProc(HWND, UINT, WPARAM, LPARAM);
extern int DrawRectangle_Win32(Draw,double,double,double,double,int,int,int,int);
extern int DrawLine_Win32(Draw,double,double,double,double,int);
extern int DrawLineSetWidth_Win32(Draw ,double);
extern int DrawLineGetWidth_Win32(Draw,double *);
extern int DrawPoint_Win32(Draw,double,double,int);
extern int DrawPointSetSize_Win32(Draw,double);
extern int TranslateColor_Win32(Draw,int);
extern int AverageColorRectangle_Win32(Draw,int,int,int,int);
extern int AverageColorTriangle_Win32(Draw,int,int,int);
extern int DrawString_Win32(Draw,double,double,int,char *);
extern int DrawStringVertical_Win32(Draw,double,double,int,char *);
extern int DrawStringSetSize_Win32(Draw,double,double);
extern int DrawStringGetSize_Win32(Draw,double *,double *);
extern int DrawSetCoordinates_Win32(Draw,double,double,double,double);
extern int DrawGetCoordinates_Win32(Draw,double *, double *, double *, double *);
extern int DrawResizeWindow_Win32(Draw,int,int);
extern int DrawCheckResizedWindow_Win32(Draw);
extern int DrawGetTitle_Win32(Draw, char **);
extern int DrawSetTitle_Win32(Draw, char *);
extern int DrawGetPopup_Win32(Draw,Draw *);
extern int DrawClear_Win32(Draw);
extern int DrawSetPause_Win32(Draw,int);
extern int DrawCreate_Win32(Draw);
extern int DrawDestroy_Win32(Draw);
extern int DrawPause_Win32(Draw);
extern int DrawGetMouseButton_Win32(Draw, DrawButton *,double *,double *,double *,double *);
extern int DrawGetPause_Win32(Draw,int *);
extern int DrawTriangle_Win32(Draw,double,double,double,double,double,double,int,int,int);
extern int DrawScalePopup_Win32(Draw,double,double);
extern void MessageLoopThread(Draw_Win32 *);
extern int deletemouselist_Win32(WindowNode);

/*
FUNCTION: deletemouselist_Win32
        
          input:*deletelist  //pointer to a window node 
                        
          description: At the destruction(closing) of a window this function is 
                       called to release the memory for the linked list of recorded
                       mouse actions
*/
int deletemouselist_Win32(WindowNode deletelist)
{
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

/*
FUNCTION: DrawGetMouseButton_Win32
        
          input:*x_user,*y_user  //pointer to screen coordinates of received mouse action
                        *x_phys,*y_phys  //pointer to window coordinates of received mouse action
                        *button          //pointer to what mouse button was pressed in action
                        *ctx                     //pointer to structure for window that is being used

          description: Looks at global linked list to retreive information on requested mouse 
                                   action. If none has happened yet, wait for a mouse action to occur.
*/
int DrawGetMouseButton_Win32(Draw ctx, DrawButton *button,double *x_user,double *y_user, \
                             double *x_phys,double *y_phys)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  WindowNode current;
  MouseNode  node;

  /*Make sure no other code is using the linked list at this moment*/
  WaitForSingleObject(g_hWindowListMutex, INFINITE);
  /*Look for the node that matches the window you are using*/
  current = WindowListHead;
  while (current != NULL) {
    if(current->hWnd == draw->hWnd) {       
      current->IsGetMouseOn = TRUE;
      break;
    } else {
      current = current->wnext;
    }
  }
  /*If no actions have occured, wait for one*/
  node = current->MouseListHead;
  if (node == NULL) {
    ReleaseMutex(g_hWindowListMutex);
    WaitForSingleObject(current->event, INFINITE);
    WaitForSingleObject(g_hWindowListMutex, INFINITE);
  }
  /*once we have the information, assign the pointers to it */
  *button = current->MouseListHead->Button;
  *x_user = current->MouseListHead->user.x;
  *y_user = current->MouseListHead->user.y;
  *x_phys = current->MouseListHead->phys.x;
  *y_phys = current->MouseListHead->phys.y;
  /*remove set of information from sub linked-list, delete the node*/
  current->MouseListHead = current->MouseListHead->mnext;
  if (current->MouseListHead == NULL) {
    ResetEvent(current->event);
    current->MouseListTail = NULL;
  }
  PetscFree(node);

  /*Release mutex so that  other code can use
    the linked list now that we are done with it*/
  ReleaseMutex(g_hWindowListMutex);
  return 0;
}

/*
FUNCTION: DrawSetPause_Win32
        
          input:*ctx            //pointer to a window  
                        pausetime       //integer value in seconds
          description: Access the window strucure and stores the desired
                                        pausetime. This value is what will be used for the
                                        length of the pause when DrawPause_Win32 is called.
*/

int DrawSetPause_Win32(Draw ctx,int pausetime)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  draw->pause = pausetime;
  return 0;
}

/*
FUNCTION: DrawGetPause_Win32
        
          input:*ctx            //pointer to a window  
                        &pausetime      //referenced integer variable
          description:  Access the window strucure and retrieves the current set
                                        pausetime for that window, returns it by reference through 
                                        pausetime
*/
int DrawGetPause_Win32(Draw ctx, int *pausetime)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  *pausetime = draw->pause;
  return 0;
}
/*
FUNCTION: DrawPause_Win32
        
          input:*ctx            //pointer to a window  
                        
          description: Access the window strucure and determined the 
                                        value that was set by the user for that window
                                        and waits for that amount of seconds before
                                        continuing.
*/
int DrawPause_Win32(Draw ctx)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  clock_t goal;
  goal = ((clock_t)(draw->pause * CLOCKS_PER_SEC)) + clock();   
  while( goal > clock() )  ;
  return 0;
}

/*
FUNCTION: TranslateColor_Win32
        
          input:*ctx            //pointer to a window  
                        color   //integer value from 1 to 32
          description: Since the existing code for PETSc handels color
                                        from a hue table that takes values 1 to 32. Since here 
                                        that was not available we are currently taking the integer
                                        value, comparing to a Red,Green and Blue array we have
                                        previously put in the RGB values of that corresponding
                                        hue number. It retreives these values from the array and
                                        calls the RGB internal translator, retreving a single number
                                        and stores this number in the currentcolor field of that
                                        window
*/
int TranslateColor_Win32(Draw ctx,int color)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  draw->currentcolor = RGB(RedMap[color],GreenMap[color],BlueMap[color]);
  return 0;
}

/*
FUNCTION: AverageColorRectangle_Win32
        
          input:*ctx            //pointer to a window  
                        c1,c2,c3,c4     //integer value from 1 to 32 corresponding
                                                to the color value at each point of a rectangle
          description: A simple function removing this code from the DrawRectangle_Win32
                                        function. Averages and translates the inputed color points for a 
                                        rectangle. Will be changed because the desired effect is a gradient
                                        thoughout the rectangle based on the color value at each point
          Overload : Does the same for the three points of a triangle
*/
int AverageColorRectangle_Win32(Draw ctx,int c1,int c2, int c3, int c4)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;

  draw->currentcolor = RGB(((RedMap[c1]+RedMap[c2]+RedMap[c3]+RedMap[c4])/4),
                           ((GreenMap[c1]+GreenMap[c2]+GreenMap[c3]+GreenMap[c4])/4),
                           ((BlueMap[c1]+BlueMap[c2]+BlueMap[c3]+BlueMap[c4])/4));
  return 0;
}

int AverageColorTriangle_Win32(Draw ctx,int c1,int c2,int c3)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  draw->currentcolor = RGB((RedMap[c1]+RedMap[c2]+RedMap[c3])/3,
                           (GreenMap[c1]+GreenMap[c2]+GreenMap[c3])/3,
                           (BlueMap[c1]+BlueMap[c2]+BlueMap[c3])/3); 
  return 0;
}
/*
FUNCTION: DrawRectangle_Win32

        inputs: xl,xr,yl,yr             //coordinates for the lower-left and upper-right hand corners
                        c1,c2,c3,c4             //color assigned to each point
                         *ctx                                   //pointer to window
        description: Retrieves the given windows device context, 
                                draws a rectangle from the upper-left corner
                                to the lower-right corner in that context, 
                                then calls previous functions to set color 
                                and paints in the region with that color.Since in the
                                original PETSc code you supply DrawRectangle with the 
                                lower-left corner and upper-right hand corner we kept
                                the syntax of the function the same, however since windows
                                takes there dimensions differently we flip the values through
                                assignment to gain correct perspective
*/
int DrawRectangle_Win32(Draw ctx,double xl,double yl,double xr,double yr,int c1,int c2,int c3,int c4)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  
  HBRUSH hbrush;
  
  RECT rect;  /* retrieve tag for a rect, then make it */
  /* assigning values this way flips to correct prespective */
  SetRect(&rect,(int)xl,(int)yr,(int)xr,(int)yl); 
  
  draw->hdc = GetDC(draw->hWnd);          
  if (c1==c2 && c2==c3 && c3==c4) {         /* if all colors equal, use that color */
    TranslateColor_Win32(ctx,c1);
  } else {                                   /* else get average of the colors */
    AverageColorRectangle_Win32(ctx,c1,c2,c3,c4);
  }
  hbrush = CreateSolidBrush(draw->currentcolor);
  FillRect(draw->hdc,&rect,hbrush);
  ReleaseDC(draw->hWnd,draw->hdc);
  return 0;
}

/*
FUNCTION: DrawLine_Win32

        inputs: xl,xr,yl,yr     //coordinates for the left and right hand points of the line
                        color           //color assigned to line
                        *ctx            //pointer to window
        description: Retrieves the given windows device context, 
                                draws a rectangle from the upper-left corner
                                to the lower-right corner in that context, 
                                then calls previous functions to set color 
                                and paints in the region with that color
*/
int DrawLine_Win32(Draw ctx,double xl,double yl,double xr,double yr,int color)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  HPEN hpen;
  TranslateColor_Win32(ctx,color);
  hpen = CreatePen (PS_SOLID, draw->linewidth, draw->currentcolor);
  draw->hdc = GetDC(draw->hWnd);
  SelectPen(draw->hdc,hpen);
  MoveToEx(draw->hdc,(int)xl,(int)yl,NULL);
  LineTo(draw->hdc,(int)xr,(int)yr);
  ReleaseDC(draw->hWnd, draw->hdc);
  return 0;
}

/*
FUNCTION: DrawLineSetWidth_Win32

        inputs: width   //double value corresponding to the desired 
                                        percentage of viewport the width of the line should
                                        take up
                        *ctx    //pointer to window
        description: Retrieves the given windows current size, both width and 
                                 height, in pixels then calculates the average, finds the 
                                 desired percent of those pixels and stores that number of
                                 pixels in linewidth variable of the window struct 
*/
int DrawLineSetWidth_Win32(Draw ctx,double width)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  int averagesize,finalwidth;
  
  RECT rect;
  GetClientRect(draw->hWnd,&rect);
  /*this takes the horizontal and vertical size of the 
    viewport and averages them, then takes our percent*/
  averagesize = ((rect.right - rect.left)+(rect.bottom - rect.top))/2;
  finalwidth = (int)floor(averagesize*width);
  if (finalwidth < 1) {
    finalwidth = 1; /* this is the minimum size that DrawLine can except */
  }
  draw->linewidth = finalwidth;
  return 0;
}

/*
FUNCTION: DrawLineGetWidth_Win32

        inputs: &width  //referenced double variable through which the currently
                                          stored linewidth value will be passed
                        *ctx    //pointer to window
        description: Retrieves the given windows current linewidth setting
                                 and passes to the referenced variable width
*/
int DrawLineGetWidth_Win32(Draw ctx,PetscReal *width)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;

  *width = (PetscReal)draw->linewidth;
  return 0;
}
/*
FUNCTION: DrawPoint_Win32

        inputs: x,y       // position for center of point
                        color // number from 1 to 32 representing color to be created
                        *ctx    //pointer to window
        description: Retrieves the given windows current pointdiameter setting. If 1 
                                it colors in one pixel, otherwise it draws a circle the desired 
                                amount of pixels in diameter around the point and colors it in
*/
int DrawPoint_Win32(Draw ctx,double x,double y,int color)
{       
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  HBRUSH hbrush;
  HRGN hrgn;
  int radius;

  radius = (int)floor(draw->pointdiameter/2);
  TranslateColor_Win32(ctx,color);

  hbrush = CreateSolidBrush(draw->currentcolor);
        
  draw->hdc = GetDC(draw->hWnd);
  /*If requested percentage of the viewport is less than or
    one pixel, then make it one pixel and set it*/
  if (draw->pointdiameter == 1) {
    SetPixelV(draw->hdc,(int)x,(int)y,draw->currentcolor);  
  } else {
    /*else make a circle around the point requested that is the correct
      percentage of the viewport*/
    hrgn = CreateEllipticRgn((int)x-radius,(int)y-radius,(int)x+radius,(int)y+radius);
    FillRgn(draw->hdc,hrgn,hbrush);
  }
  ReleaseDC(draw->hWnd,draw->hdc);
  return 0;
}

/*
FUNCTION: DrawPointSetSize_Win32

        inputs: width   //double value corresponding to the desired 
                                        percentage of viewport the diameter of the point should
                                        take up
                        *ctx    //pointer to window
        description: Retrieves the given windows current size, calculates the given
                                 percentage of this and stores it in the windows pointdiameter setting
*/
int DrawPointSetSize_Win32(Draw ctx,double width)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  int averagesize,diameter;
  RECT rect;
  GetClientRect(draw->hWnd,&rect);
  /*This takes the horizontal and vertical sizes of the viewport 
    and averages them,then takes the correct percentage of them*/
  averagesize = ((rect.right - rect.left)+(rect.bottom - rect.top))/2;
  
  diameter = (int)floor(averagesize*width);
  if (diameter < 1) diameter = 1;
  draw->pointdiameter = diameter;
  return 0;
}
/*
FUNCTION: DrawString_Win32

        inputs: color   //1 to 32 value for the color
                        x,y             //coordinated that upper-left hand corner of 
                                          string box will start
                        *text   //text that will be drawn on screen
                        *ctx    //pointer to window
        description: Creates a minimized rectangle to draw text within
                                 then creates a font using mostly default paramaters
                                 except for height and width and then writes it in the color
                                 asked for
*/
int DrawString_Win32(Draw ctx,double x,double y,int color,char *text)
{       
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  RECT r;
  HFONT hfont;                                                                    
  LOGFONT logfont;                                                                        


  r.bottom        = (int)x;
  r.left          = (int)y;
  r.right         = (int)x + 1; 
  r.top           = (int)y + 1;/*Because of the DT_NOCLIP parameter
                                 in the DrawText call below, all we 
                                 need is a non-zero rect*/
  draw->hdc = GetDC(draw->hWnd);
        
  logfont.lfHeight         = draw->stringheight;/*sets the height according to % of viewport*/
  logfont.lfWidth          = draw->stringwidth;/*set the width the same way*/
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
  
  hfont = CreateFontIndirect(&logfont); /* creates font from our parameters */
  SelectFont(draw->hdc,hfont);
        
  TranslateColor_Win32(ctx,color);
  SetTextColor(draw->hdc,draw->currentcolor);
  DrawText(draw->hdc,text,lstrlen(text),&r,DT_NOCLIP);
  DeleteObject(hfont);
  ReleaseDC(draw->hWnd,draw->hdc);
  return 0;
}
/*
FUNCTION: DrawStringVertical_Win32

        inputs: color   //1 to 32 value for the color
                        x,y             //coordinated that upper-left hand corner of 
                                          string box will start
                        *text   //text that will be drawn on screen
                        *ctx    //pointer to window
        description: The same as DrawString_Win32 except for the text is drawn vertical
*/
int DrawStringVertical_Win32(Draw ctx,double x,double y,int color,char *text)
{       
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  RECT r;
  HFONT hfont;                                                                                    
  LOGFONT logfont;                                                                                

  r.bottom = (int)x;
  r.left   = (int)y + 30;
  r.right  = (int)x + 1; /*Because of NO_CLIP all we need is a non-zero rect*/
  r.top    = (int)y - 30;

  draw->hdc = GetDC(draw->hWnd);
        
  logfont.lfEscapement     = 2700; /*Causes verticle text drawing*/
  logfont.lfHeight         = draw->stringheight;/*sets the height according to % of viewport*/
  logfont.lfWidth          = draw->stringwidth;/*set the width the same way*/ 
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

  hfont = CreateFontIndirect(&logfont);//creates font according to our parameters
  SelectFont(draw->hdc,hfont);

  TranslateColor_Win32(ctx,color);
  SetTextColor(draw->hdc,draw->currentcolor);
  DrawText(draw->hdc,text,lstrlen(text),&r,DT_NOCLIP | DT_SINGLELINE );

  DeleteObject(hfont);
  ReleaseDC(draw->hWnd,draw->hdc);
  return 0;
}

/*
FUNCTION: DrawStringSetSize_Win32

        inputs: width   //percentage of windows width the text font will be set to
                        height  //percentage of windows height the text font will be set to 
                        *ctx    //pointer to window
        description: Retreives the windows size and calculates the percentage of the
                                 windows dimensions by pixels, then stores them in the windows struct
*/
int DrawStringSetSize_Win32(Draw ctx,double width,double height)
{       
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  int sizeheight, sizewidth;
  /*Here we get the dimensions of the current user window
    so that we can apply our given percentage to the number
    of pixels for the value of stringheight and stringwidth*/
  RECT r;
  GetWindowRect(draw->hWnd,&r);
  sizeheight = (int)floor(height*(r.bottom - r.top));
  sizewidth  = (int)floor(width*(r.right - r.left));
  if (sizeheight < 1) sizeheight = 1;
  if (sizewidth < 1)  sizewidth = 1;
  draw->stringheight = sizeheight;
  draw->stringwidth  = sizewidth;
  return 0;
}

/*
FUNCTION: DrawStringGetSize_Win32

        inputs: *width  //reference variable current width setting will be passed to
                        *height //reference variable current height setting will be passed to
                        *ctx    //pointer to window
        description: Retreives the windows string height and width setting and passes 
        them back through the given reference variables
*/
int DrawStringGetSize_Win32(Draw ctx,double *width,double *height)
{       
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  *height = (double)draw->stringheight;
  *width  = (double)draw->stringwidth;
  return 0;
}
/*
FUNCTION: DrawSetCoordinates_Win32

        inputs: xl,yl,xr,yr     //upper-left and lower-right point that window will be
                                                  placed at
                        *ctx    //pointer to window
        description: Moves windows start and end points to desired locations
*/
int DrawSetCoordinates_Win32(Draw ctx,double xl,double yl,double xr,double yr)
{       
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  MoveWindow(draw->hWnd,(int)xl,(int)yr,(int)(xr-xl),(int)(yl-yr),TRUE);
  return 0;
}

/*
FUNCTION: DrawGetCoordinates_Win32

        inputs: *xl,*yl,*xr,*yr //reference variables that are passed the windows position
                        *ctx                    //pointer to window
        description: Retreives current window position and passes back through reference 
                                 variables
*/
int DrawGetCoordinates_Win32(Draw ctx,double *xl, double *yl, double *xr, double *yr)
{       
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  RECT r;
  GetWindowRect(draw->hWnd,&r);
  *xl = (double)r.left;
  *yl = (double)r.bottom;
  *xr = (double)r.right;
  *yr = (double)r.top;
  return 0;
}
/*
FUNCTION: DrawResizeWindow_Win32

        inputs: w,h             // width and height in pixels that window will be resized to
                        *ctx    //pointer to window
        description: Resizes window from original point with given height and width. 
                                 Since windows are drawn from upper-left to lower-right corners,
                                to change the size we will grab the curent value of the upper left 
                                corner and redraw the window with the desired height and width from 
                                that point.It also turns the haveresized flag for that window to true
*/
int DrawResizeWindow_Win32(Draw ctx,int w,int h)
{       
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  RECT r;
  GetWindowRect(draw->hWnd,&r);
  MoveWindow(draw->hWnd,r.left,r.top,(int)w,(int)h,TRUE);
  draw->haveresized = 1;//throw the flag that we have resized this window
  return 0;
}
/*
FUNCTION: DrawCheckResizeWindow_Win32

        inputs:         *ctx    //pointer to window
        description: Returns a 1 if this windows has a recorded resizing and 0 if not.
*/
int DrawCheckResizedWindow_Win32(Draw ctx)
{       
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  if (draw->haveresized == 1) return 1;
  else return 0;
}
/*
FUNCTION: DrawGetTitle_Win32

        inputs:         *ctx    //pointer to window
                                **title //pointer for title chars
        description: returns a pointer to the pointer for the windows title
*/
int DrawGetTitle_Win32(Draw ctx, char **title)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  GetWindowText(draw->hWnd,*title, 30);/*title shouldn't be more than 30 characters???*/
  return 0;
}
/*
FUNCTION: DrawSetTitle_Win32

        inputs:         *ctx    //pointer to window
                                *title //pointer for chars that will be set as the new window title
        description: Sets the current windows title to chars it was passed
*/
int DrawSetTitle_Win32(Draw ctx, char *title)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  SetWindowText(draw->hWnd,title);
  return 0;
}

/*
FUNCTION: DrawClear_Win32

        inputs:         *ctx    //pointer to window
        description: Redraws window blank
*/
int DrawClear_Win32(Draw ctx)
{       
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  RECT r;
  GetClientRect(draw->hWnd,&r);
  RedrawWindow(draw->hWnd,&r,NULL,RDW_INVALIDATE | RDW_ERASE);
  UpdateWindow(draw->hWnd);
  return 0;
}
/*
FUNCTION: DrawTriangle_Win32

 inputs: x1,x2,x3,y1,y2,y3      //coordinates for the three points
                 c1,c2,c3                       //color assigned to each point
                 *ctx                           //pointer to stucture for window being used

 description: Uses points given to create a triangle out of lines,
                          averages assigned colors and fills in are with a solid
                          brush of that color.
*/
int DrawTriangle_Win32(Draw ctx,double x1,double y1,double x2,double y2,double x3,double y3,
                       int c1,int c2,int c3)
{       
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  HBRUSH hbrush;
  HPEN hpen;
  AverageColorTriangle_Win32(ctx,c1,c2,c3); 
        
  hbrush = CreateSolidBrush(draw->currentcolor);
  hpen = CreatePen(PS_SOLID,0,draw->currentcolor);
  /*Retrieve the correct windows device context 
    and draw lines connecting the three points*/
  draw->hdc = GetDC(draw->hWnd);
  BeginPath(draw->hdc);
  MoveToEx(draw->hdc,(int)x1,(int)y1,NULL);
  LineTo(draw->hdc,(int)x2,(int)y2);
  LineTo(draw->hdc,(int)x3,(int)y3);
  LineTo(draw->hdc, (int)x1,(int)y1);
  EndPath(draw->hdc);
  /*Fill the Triangle with averaged color 
    and release handle to device context*/
  SelectPen(draw->hdc,hpen);
  SelectBrush(draw->hdc,hbrush);
  StrokeAndFillPath(draw->hdc);
  ReleaseDC(draw->hWnd,draw->hdc);
  return 0;
}

void PopMessageLoopThread_Win32(Draw popdraw)
{
  Draw_Win32 *pop = (Draw_Win32*)popdraw->data;
  MSG msg;

  TCHAR PopClassName [MAX_LOADSTRING + 1]; 
  RECT r;
  int width,height;
  POINT origin;
  WNDCLASSEX myclass;
  myclass.cbSize        = sizeof(WNDCLASSEX);
  myclass.style         = CS_OWNDC;
  myclass.lpfnWndProc   = (WNDPROC)WndProc;
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
  origin.x = 0;
  origin.y = 0;
        
  pop->hWnd = CreateWindow(PopClassName,NULL, WS_POPUPWINDOW | WS_CAPTION,
                           0, 0, width,height, NULL, NULL,NULL, NULL);
                                        
  ShowWindow(pop->hWnd, SW_SHOWNOACTIVATE);
  UpdateWindow(pop->hWnd);
  SetEvent(pop->hReadyEvent);
        
  while (GetMessage(&msg, pop->hWnd, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
}
/*
FUNCTION: DrawGetPopup_Win32

        inputs:         *ctx     //pointer to window that pop-up window will belong to
                                *popdraw //pointer to the created popup window
        description: Creates all the same structures and definition for a regular window.
                                Creates the pop-up window within its own new thread. How it is the ctx
                                window has no control or ownership of the pop-up window.Mutex is used
                                to ensure that nothing can access the structs for the window before
                                it is done setting them up
*/
int DrawGetPopup_Win32(Draw ctx,Draw *popdraw)
{
  Draw_Win32 *pop;
  HANDLE hThread;
  WindowNode newnode;

  pop = (Draw_Win32 *) PetscMalloc(sizeof (Draw_Win32));
  (*popdraw)->data = pop;

  pop->hReadyEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
        
  hThread = CreateThread(NULL, 0,(LPTHREAD_START_ROUTINE)PopMessageLoopThread_Win32,*popdraw,0, NULL);
  CloseHandle(hThread);
  WaitForSingleObject(pop->hReadyEvent, INFINITE);
  CloseHandle(pop->hReadyEvent);

  WaitForSingleObject(g_hWindowListMutex, INFINITE);
  newnode = (WindowNode)PetscMalloc(sizeof(struct _p_WindowNode));CHKPTRQ(newnode);

/* WindowNode newnode = new _p_WindowNode; */
  newnode->MouseListHead = NULL;
  newnode->MouseListTail = NULL;
  newnode->wnext = WindowListHead;
  newnode->wprev = NULL;
  newnode->hWnd = pop->hWnd;
  if(WindowListHead != NULL) {
    WindowListHead->wprev = newnode;
  }
  newnode->event = CreateEvent(NULL, TRUE, FALSE, NULL);
  WindowListHead = newnode;
  ReleaseMutex(g_hWindowListMutex);
  return 0;
}



void MessageLoopThread_Win32(Draw ctx)
{
  Draw_Win32 *draw    = (Draw_Win32*)ctx->data;
  TCHAR classname[MAX_LOADSTRING];
  MSG msg;
        
  WNDCLASSEX wcex;
  wcex.cbSize         = sizeof(WNDCLASSEX); 
  wcex.style          = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
  wcex.lpfnWndProc    = (WNDPROC)WndProc;
  wcex.cbClsExtra     = 0;
  wcex.cbWndExtra     = 0;
  wcex.hInstance      = NULL;
  wcex.hIcon          = LoadIcon(NULL, (LPCTSTR)IDI_FOUR);
  wcex.hCursor        = LoadCursor(NULL, IDC_ARROW);
  wcex.hbrBackground  = (HBRUSH)(COLOR_WINDOW+1);
  wcex.lpszMenuName   = (LPCSTR)IDC_FOUR;
  wcex.lpszClassName  = classname;
  wcex.hIconSm        = NULL;

  RegisterClassEx(&wcex);
        
  draw->hWnd = CreateWindow(classname,NULL, WS_OVERLAPPEDWINDOW,CW_USEDEFAULT, 0, 
                            CW_USEDEFAULT, 0, NULL, NULL,NULL, NULL);

  ShowWindow(draw->hWnd, SW_SHOW);
  UpdateWindow(draw->hWnd);
  SetEvent(draw->hReadyEvent);
        
  while (GetMessage(&msg, draw->hWnd, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
}
static struct _DrawOps DvOps = { 0,
                                 0,DrawLine_Win32,
                                 DrawLineSetWidth_Win32,
                                 0,
                                 DrawPoint_Win32,
                                 0,
                                 DrawString_Win32,
                                 DrawStringVertical_Win32,
                                 DrawStringSetSize_Win32,
                                 DrawStringGetSize_Win32,
                                 0,
                                 DrawClear_Win32,
                                 0,
                                 DrawRectangle_Win32,
                                 DrawTriangle_Win32,
                                 DrawGetMouseButton_Win32,
                                 DrawPause_Win32,
                                 0,
				 0,
                                 0,
                                 DrawGetPopup_Win32,
                                 DrawSetTitle_Win32,
                                 DrawCheckResizedWindow_Win32,
                                 DrawResizeWindow_Win32,
                                 DrawDestroy_Win32,
                                 0,
                                 0,
                                 0 };


/*
FUNCTION: DrawCreate_Win32

        inputs:         *ctx     //pointer to window that will be created
        description: Creates a window in a new thread
*/
int DrawCreate_Win32(Draw ctx)
{       
  Draw_Win32  *draw;
  HANDLE      hThread;
  int         ierr;
  WindowNode newnode;
  draw      = (Draw_Win32*)PetscMalloc(sizeof(Draw_Win32));CHKPTRQ(draw);
  ctx->data = draw;

  /* the following is temporary fix for initializing a global datastructure */
  if (!g_hWindowListMutex) {
  g_hWindowListMutex = CreateMutex(NULL, FALSE, NULL);
  }

  ierr = PetscMemcpy(ctx->ops,&DvOps,sizeof(DvOps));CHKERRQ(ierr);
  
  draw->hReadyEvent = CreateEvent(NULL, TRUE, FALSE, NULL);
        
  hThread = CreateThread(NULL, 0,(LPTHREAD_START_ROUTINE)MessageLoopThread_Win32, ctx,0, NULL);
  CloseHandle(hThread);
  WaitForSingleObject(draw->hReadyEvent, INFINITE);
  CloseHandle(draw->hReadyEvent);

  WaitForSingleObject(g_hWindowListMutex, INFINITE);

  newnode = (WindowNode)PetscMalloc(sizeof(struct _p_WindowNode));CHKPTRQ(newnode);
  newnode->MouseListHead = NULL;
  newnode->MouseListTail = NULL;
  newnode->wnext = WindowListHead;
  newnode->wprev = NULL;
  newnode->hWnd = draw->hWnd;
  if(WindowListHead != NULL) {
    WindowListHead->wprev = newnode;
  }
  newnode->event = CreateEvent(NULL, TRUE, FALSE, NULL);
  WindowListHead = newnode;

  ReleaseMutex(g_hWindowListMutex);
  return 0;
}

/*
FUNCTION: DrawDestroy_Win32

        inputs:         *ctx     //pointer to window that will be destoyed
        description: Releases all memory and threads to that window and removes it
*/
int DrawDestroy_Win32(Draw ctx)
{
  Draw_Win32 *draw = (Draw_Win32*)ctx->data;
  SendMessage(draw->hWnd,WM_DESTROY,0,0);
  PetscFree(draw);
  return 0;
}


//  FUNCTION: WndProc(HWND, unsigned, WORD, LONG)
//
//  PURPOSE:  Processes messages for the main window.
//
//  WM_COMMAND  - process the application menu
//  WM_PAINT    - Paint the main window
//  WM_DESTROY  - post a quit message and return
LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam)
{
  int wmId, wmEvent;
  PAINTSTRUCT ps;
  HDC hdc;
  POINT mousepos;
  WindowNode current = NULL; 
  MouseNode newnode;

  switch (message) {
  case WM_COMMAND:
    wmId    = LOWORD(wParam); 
    wmEvent = HIWORD(wParam); 
    // Parse the menu selections:
    switch (wmId) {
    case IDM_EXIT:
      DestroyWindow(hWnd);         
      break;
    default:
      return DefWindowProc(hWnd, message, wParam, lParam);
    }
    break;
    /*What is happening here is that once a window has been opened
      every mouse click action is stored in a global linked list. Then when 
      GetMouseButton is called the first occurence is used then deleted. If 
      there has not been an action performed than it waits.
      We have set up the mouse actions this way so that mouse clicks are
      not lost in the message loops or execution time.Also this way we are 
      able to throw the IsGetMouseOn flag so that actions that come before
      you want to record them are not recorded or used at a later time*/
  case WM_LBUTTONUP:
    WaitForSingleObject(g_hWindowListMutex, INFINITE);
    current = WindowListHead;
    SetEvent(current->event);
    while (current != NULL) {   
      if(current->hWnd == hWnd) {       
        if(current->IsGetMouseOn == TRUE) {
          newnode = (MouseNode)PetscMalloc(sizeof(struct _p_MouseNode));CHKPTRQ(newnode);

          newnode->Button = BUTTON_LEFT;
          GetCursorPos(&mousepos);
          newnode->user.x = mousepos.x;
          newnode->user.y = mousepos.y;
          ScreenToClient(hWnd,&mousepos);
          newnode->phys.x = mousepos.x;
          newnode->phys.y = mousepos.y;

          if (current->MouseListTail == NULL) {
            current->MouseListHead = newnode;
            current->MouseListTail = newnode;
          } else {
            current->MouseListTail->mnext = newnode;
            current->MouseListTail = newnode;
          }
          newnode->mnext = NULL;
          ReleaseMutex(g_hWindowListMutex);
        }
        break;
      } else {
        current = current->wnext;
      }
    }
    break;
  case WM_MBUTTONUP:
    WaitForSingleObject(g_hWindowListMutex, INFINITE);
    current = WindowListHead;
    SetEvent(current->event);
    while (current != NULL) {   
      if(current->hWnd == hWnd) {       
        if(current->IsGetMouseOn == TRUE) {
          newnode = (MouseNode)PetscMalloc(sizeof(struct _p_MouseNode));CHKPTRQ(newnode);

          newnode->Button = BUTTON_CENTER;
          GetCursorPos(&mousepos);
          newnode->user.x = mousepos.x;
          newnode->user.y = mousepos.y;
          ScreenToClient(hWnd,&mousepos);
          newnode->phys.x = mousepos.x;
          newnode->phys.y = mousepos.y;

          if (current->MouseListTail == NULL) {
            current->MouseListHead = newnode;
            current->MouseListTail = newnode;
          } else {
            current->MouseListTail->mnext = newnode;
            current->MouseListTail = newnode;
          }
          newnode->mnext = NULL;
          ReleaseMutex(g_hWindowListMutex);
        }
        break;
      }
      else {
        current = current->wnext;
      }
    }
    break;
  case WM_RBUTTONUP:
    WaitForSingleObject(g_hWindowListMutex, INFINITE);
    current = WindowListHead;
    SetEvent(current->event);
    while (current != NULL) {   
      if(current->hWnd == hWnd) {       
        if(current->IsGetMouseOn == TRUE) {
          newnode = (MouseNode)PetscMalloc(sizeof(struct _p_MouseNode));CHKPTRQ(newnode);

          newnode->Button = BUTTON_RIGHT;
          GetCursorPos(&mousepos);
          newnode->user.x = mousepos.x;
          newnode->user.y = mousepos.y;
          ScreenToClient(hWnd,&mousepos);
          newnode->phys.x = mousepos.x;
          newnode->phys.y = mousepos.y;

          if (current->MouseListTail == NULL) {
            current->MouseListHead = newnode;
            current->MouseListTail = newnode;
          } else {
            current->MouseListTail->mnext = newnode;
            current->MouseListTail = newnode;
          }
          newnode->mnext = NULL;
          ReleaseMutex(g_hWindowListMutex);
        }
        break;
      }
      else {
        current = current->wnext;
      }
    }
    break;
  case WM_PAINT:
    hdc = BeginPaint(hWnd, &ps);
    EndPaint(hWnd, &ps);
    break;
    /*Since each window now has several linked lists associated with it
      we destroy these when a window is closed to free the memory. Both the 
      list for the window and the mouse action list are destroyed*/
  case WM_DESTROY:
    WaitForSingleObject(g_hWindowListMutex, INFINITE);
    current = WindowListHead;
    while (current != NULL) {   
      if((current->hWnd != NULL) && (current->hWnd == hWnd)) {       
        break;
      } else {
        current = current->wnext;
      }
    }
    if(current != NULL) {
      deletemouselist_Win32(current);
    } else {
      PetscFree(current);
    }
    ReleaseMutex(g_hWindowListMutex);
    quitflag = PETSC_FALSE;
    PostQuitMessage(0);
    break;

  default:
    return DefWindowProc(hWnd, message, wParam, lParam);
  }
  return 0;
}
