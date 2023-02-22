
/*
   This file contains simple code to manage access to fonts, insuring that
   library routines access/load fonts only once
 */

#include <../src/sys/classes/draw/impls/x/ximpl.h>

static PetscErrorCode PetscDrawXiInitFonts(PetscDraw_X *);
static PetscErrorCode PetscDrawXiLoadFont(PetscDraw_X *, PetscDrawXiFont *);
static PetscErrorCode PetscDrawXiMatchFontSize(PetscDrawXiFont *, int, int);

/*
    PetscDrawXiFontFixed - Return a pointer to the selected font.

    Warning: Loads a new font for each window. This should be
   ok because there will never be many windows and the graphics
   are not intended to be high performance.
*/
PetscErrorCode PetscDrawXiFontFixed(PetscDraw_X *XBWin, int w, int h, PetscDrawXiFont **outfont)
{
  static PetscDrawXiFont *curfont = NULL, *font;

  PetscFunctionBegin;
  if (!curfont) PetscCall(PetscDrawXiInitFonts(XBWin));
  PetscCall(PetscNew(&font));
  PetscCall(PetscDrawXiMatchFontSize(font, w, h));
  PetscCall(PetscDrawXiLoadFont(XBWin, font));

  curfont  = font;
  *outfont = curfont;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* this is set by XListFonts at startup */
#define NFONTS 20
static struct {
  int w, h, descent;
} nfonts[NFONTS];
static int act_nfonts = 0;

/*
  These routines determine the font to be used based on the requested size,
  and load it if necessary
*/

static PetscErrorCode PetscDrawXiLoadFont(PetscDraw_X *XBWin, PetscDrawXiFont *font)
{
  char         font_name[100];
  XFontStruct *FontInfo;
  XGCValues    values;

  PetscFunctionBegin;
  PetscCall(PetscSNPrintf(font_name, PETSC_STATIC_ARRAY_LENGTH(font_name), "%dx%d", font->font_w, font->font_h));
  font->fnt = XLoadFont(XBWin->disp, font_name);

  /* The font->descent may not have been set correctly; get it now that
      the font has been loaded */
  FontInfo           = XQueryFont(XBWin->disp, font->fnt);
  font->font_descent = FontInfo->descent;
  font->font_w       = FontInfo->max_bounds.rbearing - FontInfo->min_bounds.lbearing;
  font->font_h       = FontInfo->max_bounds.ascent + FontInfo->max_bounds.descent;

  XFreeFontInfo(NULL, FontInfo, 1);

  /* Set the current font in the CG */
  values.font = font->fnt;
  XChangeGC(XBWin->disp, XBWin->gc.set, GCFont, &values);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Code to find fonts and their characteristics */
static PetscErrorCode PetscDrawXiInitFonts(PetscDraw_X *XBWin)
{
  char       **names;
  int          cnt, i, j;
  XFontStruct *info;

  PetscFunctionBegin;
  /* This just gets the most basic fixed-width fonts */
  names = XListFontsWithInfo(XBWin->disp, "?x??", NFONTS, &cnt, &info);
  j     = 0;
  for (i = 0; i < cnt; i++) {
    names[i][1]       = '\0';
    nfonts[j].w       = info[i].max_bounds.width;
    nfonts[j].h       = info[i].ascent + info[i].descent;
    nfonts[j].descent = info[i].descent;
    if (nfonts[j].w <= 0 || nfonts[j].h <= 0) continue;
    j++;
    if (j >= NFONTS) break;
  }
  act_nfonts = j;
  if (cnt > 0) XFreeFontInfo(names, info, cnt);

  /* If the above fails,try this: */
  if (!act_nfonts) {
    /* This just gets the most basic fixed-width fonts */
    names = XListFontsWithInfo(XBWin->disp, "?x", NFONTS, &cnt, &info);
    j     = 0;
    for (i = 0; i < cnt; i++) {
      size_t len;

      PetscCall(PetscStrlen(names[i], &len));
      if (len != 2) continue;
      names[i][1] = '\0';
      nfonts[j].w = info[i].max_bounds.width;
      /* nfonts[j].w         = info[i].max_bounds.lbearing + info[i].max_bounds.rbearing; */
      nfonts[j].h       = info[i].ascent + info[i].descent;
      nfonts[j].descent = info[i].descent;
      if (nfonts[j].w <= 0 || nfonts[j].h <= 0) continue;
      j++;
      if (j >= NFONTS) break;
    }
    act_nfonts = j;
    XFreeFontInfo(names, info, cnt);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscDrawXiMatchFontSize(PetscDrawXiFont *font, int w, int h)
{
  int i, max, imax, tmp;

  PetscFunctionBegin;
  for (i = 0; i < act_nfonts; i++) {
    if (nfonts[i].w == w && nfonts[i].h == h) {
      font->font_w       = w;
      font->font_h       = h;
      font->font_descent = nfonts[i].descent;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }

  /* determine closest fit,per max. norm */
  imax = 0;
  max  = PetscMax(PetscAbsInt(nfonts[0].w - w), PetscAbsInt(nfonts[0].h - h));
  for (i = 1; i < act_nfonts; i++) {
    tmp = PetscMax(PetscAbsInt(nfonts[i].w - w), PetscAbsInt(nfonts[i].h - h));
    if (tmp < max) {
      max  = tmp;
      imax = i;
    }
  }

  /* should use font with closest match */
  font->font_w       = nfonts[imax].w;
  font->font_h       = nfonts[imax].h;
  font->font_descent = nfonts[imax].descent;
  PetscFunctionReturn(PETSC_SUCCESS);
}
