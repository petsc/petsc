/*
       Provides the calling sequences for all the basic PetscDraw routines.
*/
#include <petsc/private/drawimpl.h>  /*I "petscdraw.h" I*/

/*@C
   PetscDrawString - PetscDraws text onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of lower left corner of text
.  cl - the color of the text
-  text - the text to draw

   Level: beginner

.seealso: PetscDrawStringVertical(), PetscDrawStringCentered(), PetscDrawStringBoxed(), PetscDrawStringSetSize(),
          PetscDrawStringGetSize(), PetscDrawLine(), PetscDrawRectangle(), PetscDrawTriangle(), PetscDrawEllipse(),
          PetscDrawMarker(), PetscDrawPoint()

@*/
PetscErrorCode  PetscDrawString(PetscDraw draw,PetscReal xl,PetscReal yl,int cl,const char text[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidCharPointer(text,5);
  PetscAssertFalse(!draw->ops->string,PETSC_COMM_SELF,PETSC_ERR_SUP,"This draw type %s does not support drawing strings",((PetscObject)draw)->type_name);
  ierr = (*draw->ops->string)(draw,xl,yl,cl,text);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawStringVertical - PetscDraws text onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl,yl - the coordinates of upper left corner of text
.  cl - the color of the text
-  text - the text to draw

   Level: beginner

.seealso: PetscDrawString(), PetscDrawStringCentered(), PetscDrawStringBoxed(), PetscDrawStringSetSize(),
          PetscDrawStringGetSize()

@*/
PetscErrorCode  PetscDrawStringVertical(PetscDraw draw,PetscReal xl,PetscReal yl,int cl,const char text[])
{
  int            i;
  char           chr[2] = {0, 0};
  PetscReal      tw,th;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidCharPointer(text,5);

  if (draw->ops->stringvertical) {
    ierr = (*draw->ops->stringvertical)(draw,xl,yl,cl,text);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = PetscDrawStringGetSize(draw,&tw,&th);CHKERRQ(ierr);
  for (i = 0; (chr[0] = text[i]); i++) {
    ierr = PetscDrawString(draw,xl,yl-th*(i+1),cl,chr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawStringCentered - PetscDraws text onto a drawable centered at a point

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xc - the coordinates of right-left center of text
.  yl - the coordinates of lower edge of text
.  cl - the color of the text
-  text - the text to draw

   Level: beginner

.seealso: PetscDrawStringVertical(), PetscDrawString(), PetscDrawStringBoxed(), PetscDrawStringSetSize(),
          PetscDrawStringGetSize()

@*/
PetscErrorCode  PetscDrawStringCentered(PetscDraw draw,PetscReal xc,PetscReal yl,int cl,const char text[])
{
  PetscErrorCode ierr;
  size_t         len;
  PetscReal      tw,th;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidCharPointer(text,5);

  ierr = PetscDrawStringGetSize(draw,&tw,&th);CHKERRQ(ierr);
  ierr = PetscStrlen(text,&len);CHKERRQ(ierr);
  xc   = xc - len*tw/2;
  ierr = PetscDrawString(draw,xc,yl,cl,text);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   PetscDrawStringBoxed - Draws a string with a box around it

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  sxl - the coordinates of center of the box
.  syl - the coordinates of top line of box
.  sc - the color of the text
.  bc - the color of the bounding box
-  text - the text to draw

   Output Parameter:
.   w,h - width and height of resulting box (optional)

   Level: beginner

.seealso: PetscDrawStringVertical(), PetscDrawString(), PetscDrawStringCentered(), PetscDrawStringSetSize(),
          PetscDrawStringGetSize()

@*/
PetscErrorCode  PetscDrawStringBoxed(PetscDraw draw,PetscReal sxl,PetscReal syl,int sc,int bc,const char text[],PetscReal *w,PetscReal *h)
{
  PetscErrorCode ierr;
  PetscReal      top,left,right,bottom,tw,th;
  size_t         len,mlen = 0;
  char           **array;
  int            cnt,i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscValidCharPointer(text,6);

  if (draw->ops->boxedstring) {
    ierr = (*draw->ops->boxedstring)(draw,sxl,syl,sc,bc,text,w,h);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscStrToArray(text,'\n',&cnt,&array);CHKERRQ(ierr);
  for (i=0; i<cnt; i++) {
    ierr = PetscStrlen(array[i],&len);CHKERRQ(ierr);
    mlen = PetscMax(mlen,len);
  }

  ierr = PetscDrawStringGetSize(draw,&tw,&th);CHKERRQ(ierr);

  top    = syl;
  left   = sxl - .5*(mlen + 2)*tw;
  right  = sxl + .5*(mlen + 2)*tw;
  bottom = syl - (1.0 + cnt)*th;
  if (w) *w = right - left;
  if (h) *h = top - bottom;

  /* compute new bounding box */
  draw->boundbox_xl = PetscMin(draw->boundbox_xl,left);
  draw->boundbox_xr = PetscMax(draw->boundbox_xr,right);
  draw->boundbox_yl = PetscMin(draw->boundbox_yl,bottom);
  draw->boundbox_yr = PetscMax(draw->boundbox_yr,top);

  /* top, left, bottom, right lines */
  ierr = PetscDrawLine(draw,left,top,right,top,bc);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,left,bottom,left,top,bc);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,right,bottom,right,top,bc);CHKERRQ(ierr);
  ierr = PetscDrawLine(draw,left,bottom,right,bottom,bc);CHKERRQ(ierr);

  for  (i=0; i<cnt; i++) {
    ierr = PetscDrawString(draw,left + tw,top - (1.5 + i)*th,sc,array[i]);CHKERRQ(ierr);
  }
  ierr = PetscStrToArrayDestroy(cnt,array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PetscDrawStringSetSize - Sets the size for character text.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  width - the width in user coordinates
-  height - the character height in user coordinates

   Level: advanced

   Note:
   Only a limited range of sizes are available.

.seealso: PetscDrawStringVertical(), PetscDrawString(), PetscDrawStringCentered(), PetscDrawStringBoxed(),
          PetscDrawStringGetSize()

@*/
PetscErrorCode  PetscDrawStringSetSize(PetscDraw draw,PetscReal width,PetscReal height)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  if (draw->ops->stringsetsize) {
    ierr = (*draw->ops->stringsetsize)(draw,width,height);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   PetscDrawStringGetSize - Gets the size for character text.  The width is
   relative to the user coordinates of the window.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  width - the width in user coordinates
-  height - the character height

   Level: advanced

.seealso: PetscDrawStringVertical(), PetscDrawString(), PetscDrawStringCentered(), PetscDrawStringBoxed(),
          PetscDrawStringSetSize()

@*/
PetscErrorCode  PetscDrawStringGetSize(PetscDraw draw,PetscReal *width,PetscReal *height)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw,PETSC_DRAW_CLASSID,1);
  PetscAssertFalse(!draw->ops->stringgetsize,PETSC_COMM_SELF,PETSC_ERR_SUP,"This draw type %s does not support getting string size",((PetscObject)draw)->type_name);
  ierr = (*draw->ops->stringgetsize)(draw,width,height);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

