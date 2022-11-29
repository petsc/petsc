
/*
    Code for drawing color interpolated triangles using X-windows.
*/
#include <../src/sys/classes/draw/impls/x/ximpl.h>

PETSC_INTERN PetscErrorCode PetscDrawInterpolatedTriangle_X(PetscDraw_X *, int, int, int, int, int, int, int, int, int);

#define SHIFT_VAL 6

PetscErrorCode PetscDrawInterpolatedTriangle_X(PetscDraw_X *win, int x1, int y_1, int t1, int x2, int y2, int t2, int x3, int y3, int t3)
{
  PetscReal rfrac, lfrac;
  PetscReal R_y2_y_1, R_y3_y_1, R_y3_y2;
  int       lc, rc = 0, lx, rx = 0, xx, y, c;
  int       rc_lc, rx_lx, t2_t1, x2_x1, t3_t1, x3_x1, t3_t2, x3_x2;

  PetscFunctionBegin;
  /*
        Is triangle even visible in window?
  */
  if (x1 < 0 && x2 < 0 && x3 < 0) PetscFunctionReturn(0);
  if (y_1 < 0 && y2 < 0 && y3 < 0) PetscFunctionReturn(0);
  if (x1 > win->w && x2 > win->w && x3 > win->w) PetscFunctionReturn(0);
  if (y_1 > win->h && y2 > win->h && y3 > win->h) PetscFunctionReturn(0);

  t1 = t1 << SHIFT_VAL;
  t2 = t2 << SHIFT_VAL;
  t3 = t3 << SHIFT_VAL;

  /* Sort the vertices */
#define SWAP(a, b) \
  { \
    int _a; \
    _a = a; \
    a  = b; \
    b  = _a; \
  }
  if (y_1 > y2) {
    SWAP(y_1, y2);
    SWAP(t1, t2);
    SWAP(x1, x2);
  }
  if (y_1 > y3) {
    SWAP(y_1, y3);
    SWAP(t1, t3);
    SWAP(x1, x3);
  }
  if (y2 > y3) {
    SWAP(y2, y3);
    SWAP(t2, t3);
    SWAP(x2, x3);
  }
  /* This code is decidely non-optimal; it is intended to be a start at
   an implementation */

  if (y2 != y_1) R_y2_y_1 = 1.0 / ((double)(y2 - y_1));
  else R_y2_y_1 = 0.0;
  if (y3 != y_1) R_y3_y_1 = 1.0 / ((double)(y3 - y_1));
  else R_y3_y_1 = 0.0;
  t2_t1 = t2 - t1;
  x2_x1 = x2 - x1;
  t3_t1 = t3 - t1;
  x3_x1 = x3 - x1;
  for (y = y_1; y <= y2; y++) {
    /* PetscDraw a line with the correct color from t1-t2 to t1-t3 */
    /* Left color is (y-y_1)/(y2-y_1) * (t2-t1) + t1 */
    lfrac = ((double)(y - y_1)) * R_y2_y_1;
    lc    = (int)(lfrac * (t2_t1) + t1);
    lx    = (int)(lfrac * (x2_x1) + x1);
    /* Right color is (y-y_1)/(y3-y_1) * (t3-t1) + t1 */
    rfrac = ((double)(y - y_1)) * R_y3_y_1;
    rc    = (int)(rfrac * (t3_t1) + t1);
    rx    = (int)(rfrac * (x3_x1) + x1);
    /* PetscDraw the line */
    rc_lc = rc - lc;
    rx_lx = rx - lx;
    if (rx > lx) {
      for (xx = lx; xx <= rx; xx++) {
        c = (((xx - lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        PetscDrawXiSetColor(win, c);
        XDrawPoint(win->disp, PetscDrawXiDrawable(win), win->gc.set, xx, y);
      }
    } else if (rx < lx) {
      for (xx = lx; xx >= rx; xx--) {
        c = (((xx - lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        PetscDrawXiSetColor(win, c);
        XDrawPoint(win->disp, PetscDrawXiDrawable(win), win->gc.set, xx, y);
      }
    } else {
      c = lc >> SHIFT_VAL;
      PetscDrawXiSetColor(win, c);
      XDrawPoint(win->disp, PetscDrawXiDrawable(win), win->gc.set, lx, y);
    }
  }

  /* For simplicity,"move" t1 to the intersection of t1-t3 with the line y=y2.
     We take advantage of the previous iteration. */
  if (y2 >= y3) PetscFunctionReturn(0);
  if (y_1 < y2) {
    t1  = rc;
    y_1 = y2;
    x1  = rx;

    t3_t1 = t3 - t1;
    x3_x1 = x3 - x1;
  }
  t3_t2 = t3 - t2;
  x3_x2 = x3 - x2;
  if (y3 != y2) R_y3_y2 = 1.0 / ((double)(y3 - y2));
  else R_y3_y2 = 0.0;
  if (y3 != y_1) R_y3_y_1 = 1.0 / ((double)(y3 - y_1));
  else R_y3_y_1 = 0.0;

  for (y = y2; y <= y3; y++) {
    /* PetscDraw a line with the correct color from t2-t3 to t1-t3 */
    /* Left color is (y-y_1)/(y2-y_1) * (t2-t1) + t1 */
    lfrac = ((double)(y - y2)) * R_y3_y2;
    lc    = (int)(lfrac * (t3_t2) + t2);
    lx    = (int)(lfrac * (x3_x2) + x2);
    /* Right color is (y-y_1)/(y3-y_1) * (t3-t1) + t1 */
    rfrac = ((double)(y - y_1)) * R_y3_y_1;
    rc    = (int)(rfrac * (t3_t1) + t1);
    rx    = (int)(rfrac * (x3_x1) + x1);
    /* PetscDraw the line */
    rc_lc = rc - lc;
    rx_lx = rx - lx;
    if (rx > lx) {
      for (xx = lx; xx <= rx; xx++) {
        c = (((xx - lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        PetscDrawXiSetColor(win, c);
        XDrawPoint(win->disp, PetscDrawXiDrawable(win), win->gc.set, xx, y);
      }
    } else if (rx < lx) {
      for (xx = lx; xx >= rx; xx--) {
        c = (((xx - lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
        PetscDrawXiSetColor(win, c);
        XDrawPoint(win->disp, PetscDrawXiDrawable(win), win->gc.set, xx, y);
      }
    } else {
      c = lc >> SHIFT_VAL;
      PetscDrawXiSetColor(win, c);
      XDrawPoint(win->disp, PetscDrawXiDrawable(win), win->gc.set, lx, y);
    }
  }
  PetscFunctionReturn(0);
}
