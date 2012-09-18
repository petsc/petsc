
/*
   This file contains simple code to manage access to fonts, insuring that
   library routines access/load fonts only once
 */

#include <../src/sys/draw/impls/x/ximpl.h>

PetscErrorCode PetscDrawXiInitFonts(PetscDraw_X *);
PetscErrorCode PetscDrawXiMatchFontSize(PetscDrawXiFont*,int,int);
PetscErrorCode PetscDrawXiLoadFont(PetscDraw_X*,PetscDrawXiFont*);
/*
    PetscDrawXiFontFixed - Return a pointer to the selected font.

    Warning: Loads a new font for each window. This should be
   ok because there will never be many windows and the graphics
   are not intended to be high performance.
*/
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiFontFixed"
PetscErrorCode PetscDrawXiFontFixed(PetscDraw_X *XBWin,int w,int h,PetscDrawXiFont **outfont)
{
  static PetscDrawXiFont *curfont = 0,*font;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if (!curfont) { ierr = PetscDrawXiInitFonts(XBWin);CHKERRQ(ierr);}
  ierr = PetscNew(PetscDrawXiFont,&font);CHKERRQ(ierr);
  ierr = PetscDrawXiMatchFontSize(font,w,h);CHKERRQ(ierr);
  ierr = PetscDrawXiLoadFont(XBWin,font);CHKERRQ(ierr);
  curfont = font;
  *outfont = curfont;
  PetscFunctionReturn(0);
}

/* this is set by XListFonts at startup */
#define NFONTS 20
static struct {
    int w,h,descent;
} nfonts[NFONTS];
static int act_nfonts = 0;

/*
  These routines determine the font to be used based on the requested size,
  and load it if necessary
*/

#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiLoadFont"
PetscErrorCode PetscDrawXiLoadFont(PetscDraw_X *XBWin,PetscDrawXiFont *font)
{
  char        font_name[100];
  XFontStruct *FontInfo;
  XGCValues   values ;

  PetscFunctionBegin;
  (void) sprintf(font_name,"%dx%d",font->font_w,font->font_h);
  font->fnt  = XLoadFont(XBWin->disp,font_name);

  /* The font->descent may not have been set correctly; get it now that
      the font has been loaded */
  FontInfo   = XQueryFont(XBWin->disp,font->fnt);
  font->font_descent   = FontInfo->descent;
  font->font_w         = FontInfo->max_bounds.rbearing - FontInfo->min_bounds.lbearing;
  font->font_h         = FontInfo->max_bounds.ascent + FontInfo->max_bounds.descent;

  XFreeFontInfo(0,FontInfo,1);

  /* Set the current font in the CG */
  values.font = font->fnt ;
  XChangeGC(XBWin->disp,XBWin->gc.set,GCFont,&values);
  PetscFunctionReturn(0);
}

/* Code to find fonts and their characteristics */
#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiInitFonts"
PetscErrorCode PetscDrawXiInitFonts(PetscDraw_X *XBWin)
{
  char         **names;
  int          cnt,i,j;
  XFontStruct  *info;

  PetscFunctionBegin;
  /* This just gets the most basic fixed-width fonts */
  names   = XListFontsWithInfo(XBWin->disp,"?x??",NFONTS,&cnt,&info);
  j       = 0;
  for (i=0; i<cnt; i++) {
    names[i][1]         = '\0';
    nfonts[j].w         = info[i].max_bounds.width ;
    nfonts[j].h         = info[i].ascent + info[i].descent;
    nfonts[j].descent   = info[i].descent;
    if (nfonts[j].w <= 0 || nfonts[j].h <= 0) continue;
    j++;
    if (j >= NFONTS) break;
  }
  act_nfonts    = j;
  if (cnt > 0)  {
    XFreeFontInfo(names,info,cnt);
  }
  /* If the above fails,try this: */
  if (!act_nfonts) {
    /* This just gets the most basic fixed-width fonts */
    names   = XListFontsWithInfo(XBWin->disp,"?x",NFONTS,&cnt,&info);
    j       = 0;
    for (i=0; i<cnt; i++) {
        PetscErrorCode ierr;
        size_t len;

        ierr = PetscStrlen(names[i],&len);CHKERRQ(ierr);
        if (len != 2) continue;
        names[i][1]         = '\0';
	nfonts[j].w         = info[i].max_bounds.width ;
        /* nfonts[j].w         = info[i].max_bounds.lbearing + info[i].max_bounds.rbearing; */
        nfonts[j].h         = info[i].ascent + info[i].descent;
        nfonts[j].descent   = info[i].descent;
        if (nfonts[j].w <= 0 || nfonts[j].h <= 0) continue;
        j++;
	if (j >= NFONTS) break;
    }
    act_nfonts    = j;
    XFreeFontInfo(names,info,cnt);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscDrawXiMatchFontSize"
PetscErrorCode PetscDrawXiMatchFontSize(PetscDrawXiFont *font,int w,int h)
{
  int i,max,imax,tmp;

  PetscFunctionBegin;
  for (i=0; i<act_nfonts; i++) {
    if (nfonts[i].w == w && nfonts[i].h == h) {
      font->font_w        = w;
      font->font_h        = h;
      font->font_descent  = nfonts[i].descent;
      PetscFunctionReturn(0);
    }
  }

  /* determine closest fit,per max. norm */
  imax = 0;
  max  = PetscMax(PetscAbsInt(nfonts[0].w - w),PetscAbsInt(nfonts[0].h - h));
  for (i=1; i<act_nfonts; i++) {
    tmp = PetscMax(PetscAbsInt(nfonts[i].w - w),PetscAbsInt(nfonts[i].h - h));
    if (tmp < max) {max = tmp; imax = i;}
  }

  /* should use font with closest match */
  font->font_w        = nfonts[imax].w;
  font->font_h        = nfonts[imax].h;
  font->font_descent  = nfonts[imax].descent;
  PetscFunctionReturn(0);
}
