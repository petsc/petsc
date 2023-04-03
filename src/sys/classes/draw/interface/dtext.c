
#include <petsc/private/drawimpl.h> /*I "petscdraw.h" I*/

/*@C
   PetscDrawString - draws text onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl - coordinate of lower left corner of text
.  yl - coordinate of lower left corner of text
.  cl - the color of the text
-  text - the text to draw

   Level: beginner

.seealso: `PetscDraw`, `PetscDrawStringVertical()`, `PetscDrawStringCentered()`, `PetscDrawStringBoxed()`, `PetscDrawStringSetSize()`,
          `PetscDrawStringGetSize()`, `PetscDrawLine()`, `PetscDrawRectangle()`, `PetscDrawTriangle()`, `PetscDrawEllipse()`,
          `PetscDrawMarker()`, `PetscDrawPoint()`
@*/
PetscErrorCode PetscDrawString(PetscDraw draw, PetscReal xl, PetscReal yl, int cl, const char text[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscValidCharPointer(text, 5);
  PetscUseTypeMethod(draw, string, xl, yl, cl, text);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscDrawStringVertical - draws text onto a drawable.

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xl - coordinate of upper left corner of text
.  yl - coordinate of upper left corner of text
.  cl - the color of the text
-  text - the text to draw

   Level: beginner

.seealso: `PetscDraw`, `PetscDrawString()`, `PetscDrawStringCentered()`, `PetscDrawStringBoxed()`, `PetscDrawStringSetSize()`,
          `PetscDrawStringGetSize()`
@*/
PetscErrorCode PetscDrawStringVertical(PetscDraw draw, PetscReal xl, PetscReal yl, int cl, const char text[])
{
  int       i;
  char      chr[2] = {0, 0};
  PetscReal tw, th;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscValidCharPointer(text, 5);

  if (draw->ops->stringvertical) PetscUseTypeMethod(draw, stringvertical, xl, yl, cl, text);
  else {
    PetscCall(PetscDrawStringGetSize(draw, &tw, &th));
    for (i = 0; (chr[0] = text[i]); i++) PetscCall(PetscDrawString(draw, xl, yl - th * (i + 1), cl, chr));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PetscDrawStringCentered - draws text onto a drawable centered at a point

   Not Collective

   Input Parameters:
+  draw - the drawing context
.  xc - the coordinates of right-left center of text
.  yl - the coordinates of lower edge of text
.  cl - the color of the text
-  text - the text to draw

   Level: beginner

.seealso: `PetscDraw`, `PetscDrawStringVertical()`, `PetscDrawString()`, `PetscDrawStringBoxed()`, `PetscDrawStringSetSize()`,
          `PetscDrawStringGetSize()`
@*/
PetscErrorCode PetscDrawStringCentered(PetscDraw draw, PetscReal xc, PetscReal yl, int cl, const char text[])
{
  size_t    len;
  PetscReal tw, th;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscValidCharPointer(text, 5);

  PetscCall(PetscDrawStringGetSize(draw, &tw, &th));
  PetscCall(PetscStrlen(text, &len));
  xc = xc - len * tw / 2;
  PetscCall(PetscDrawString(draw, xc, yl, cl, text));
  PetscFunctionReturn(PETSC_SUCCESS);
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

.seealso: `PetscDraw`, `PetscDrawStringVertical()`, `PetscDrawString()`, `PetscDrawStringCentered()`, `PetscDrawStringSetSize()`,
          `PetscDrawStringGetSize()`
@*/
PetscErrorCode PetscDrawStringBoxed(PetscDraw draw, PetscReal sxl, PetscReal syl, int sc, int bc, const char text[], PetscReal *w, PetscReal *h)
{
  PetscReal top, left, right, bottom, tw, th;
  size_t    len, mlen = 0;
  char    **array;
  int       cnt, i;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscValidCharPointer(text, 6);

  if (draw->ops->boxedstring) {
    PetscUseTypeMethod(draw, boxedstring, sxl, syl, sc, bc, text, w, h);
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscStrToArray(text, '\n', &cnt, &array));
  for (i = 0; i < cnt; i++) {
    PetscCall(PetscStrlen(array[i], &len));
    mlen = PetscMax(mlen, len);
  }

  PetscCall(PetscDrawStringGetSize(draw, &tw, &th));

  top    = syl;
  left   = sxl - .5 * (mlen + 2) * tw;
  right  = sxl + .5 * (mlen + 2) * tw;
  bottom = syl - (1.0 + cnt) * th;
  if (w) *w = right - left;
  if (h) *h = top - bottom;

  /* compute new bounding box */
  draw->boundbox_xl = PetscMin(draw->boundbox_xl, left);
  draw->boundbox_xr = PetscMax(draw->boundbox_xr, right);
  draw->boundbox_yl = PetscMin(draw->boundbox_yl, bottom);
  draw->boundbox_yr = PetscMax(draw->boundbox_yr, top);

  /* top, left, bottom, right lines */
  PetscCall(PetscDrawLine(draw, left, top, right, top, bc));
  PetscCall(PetscDrawLine(draw, left, bottom, left, top, bc));
  PetscCall(PetscDrawLine(draw, right, bottom, right, top, bc));
  PetscCall(PetscDrawLine(draw, left, bottom, right, bottom, bc));

  for (i = 0; i < cnt; i++) PetscCall(PetscDrawString(draw, left + tw, top - (1.5 + i) * th, sc, array[i]));
  PetscCall(PetscStrToArrayDestroy(cnt, array));
  PetscFunctionReturn(PETSC_SUCCESS);
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

.seealso: `PetscDraw`, `PetscDrawStringVertical()`, `PetscDrawString()`, `PetscDrawStringCentered()`, `PetscDrawStringBoxed()`,
          `PetscDrawStringGetSize()`
@*/
PetscErrorCode PetscDrawStringSetSize(PetscDraw draw, PetscReal width, PetscReal height)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscTryTypeMethod(draw, stringsetsize, width, height);
  PetscFunctionReturn(PETSC_SUCCESS);
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

.seealso: `PetscDraw`, `PetscDrawStringVertical()`, `PetscDrawString()`, `PetscDrawStringCentered()`, `PetscDrawStringBoxed()`,
          `PetscDrawStringSetSize()`
@*/
PetscErrorCode PetscDrawStringGetSize(PetscDraw draw, PetscReal *width, PetscReal *height)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(draw, PETSC_DRAW_CLASSID, 1);
  PetscUseTypeMethod(draw, stringgetsize, width, height);
  PetscFunctionReturn(PETSC_SUCCESS);
}
