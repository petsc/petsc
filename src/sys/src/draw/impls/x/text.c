

/*
   This file contains simple code to manage access to fonts, insuring that
   library routines access/load fonts only once
 */

#include "ximpl.h"

extern XiFont *XiFontCreate();
static XiFont *curfont = 0,*font;
static int    fw = 0, fh = 0;

/*@
    XiFontFixed - Return a pointer to the selected font; loading it if 
                  necessary 

    Input Parameters:
.   XBWin - window
.   w,h   - requested width and height of a character
@*/
XiFont *XiFontFixed( XiWindow *XBWin,int w, int h )
{
  if (!curfont) { XBInitFonts( XBWin );}
  if (w != fw || h != fh) {
    if (!font)	font = XiFontCreate();
    XiMatchFontSize( font, w, h );
    fw = w;
    fh = h;
    /* if (curfont) ? unload current font ? */
    XiLoadFont( XBWin, font );
  }
  curfont = font;
  return curfont;
}

/*
   This file contains routines to manage fonts.  There are two separate
   approaches here:

   A set of default, fixed-width fonts is provided.

   The ability to load an arbitrary, variable-width font (not yet available)

   The approach is:

   XBInitFonts( XBWin ) - acquires info on (some of) the available fixed
                          width fonts.
   XBMatchFontSize( font, w, h ) - find a font that matches the width and
                                   height
   XBLoadFont( XBWin, font ) - loads a particular font
   XBDrawText( XBWin, font, x, y, chrs ) - write text
 */

/* this is set by XListFonts at startup */
#define NFONTS 20
static struct {
    int w, h, descent;
    } nfonts[NFONTS];
static int act_nfonts = 0;



/*
 * These routines determine the font to be used based on the requested size,
 * and load it if necessary
 */

int XiLoadFont( XiWindow *XBWin, XiFont *font )
{
  char        font_name[100];
  XFontStruct *FontInfo;
  XGCValues   values ;

  (void) sprintf(font_name, "%dx%d", font->font_w, font->font_h );
  /* printf( "font num is %d in loadfont\n", fnum ); */
  /* printf( "font names is %s\n", font_name ); */
  font->fnt  = XLoadFont( XBWin->disp, font_name );

  /* The font->descent may not have been set correctly; get it now that
      the font has been loaded */
  FontInfo   = XQueryFont( XBWin->disp, font->fnt );
  font->font_descent   = FontInfo->descent;

  /* For variable width, also get the max width and height */

  /* Storage leak; should probably just free FontInfo? */
  /* XFreeFontInfo( FontInfo ); */

  /* Set the current font in the CG */
  values.font = font->fnt ;
  XChangeGC( XBWin->disp, XBWin->gc.set, GCFont, &values ) ; 
  return 0;
}


/* Code to find fonts and their characteristics */
int XBInitFonts( XiWindow *XBWin )
{
  char         **names;
  int          cnt, i, j;
  XFontStruct  *info;

  /* This just gets the most basic fixed-width fonts */
  names   = XListFontsWithInfo( XBWin->disp, "?x??", NFONTS, &cnt, &info );
  j       = 0;
  for (i=0; i < cnt; i++) {
    /*printf( "in XBInitFonts - found font %s\n", names[i] ); */
    names[i][1]         = '\0';
    nfonts[j].w         = info[i].max_bounds.width ;
    nfonts[j].h         = info[i].ascent + info[i].descent;
    nfonts[j].descent   = info[i].descent;
    /* printf( "w = %d, h = %d\n", nfonts[j].w, nfonts[j].h ); */
    if (nfonts[j].w <= 0 || nfonts[j].h <= 0) continue;
    j++;
    if (j >= NFONTS) break;
  }
  act_nfonts    = j;
  if (cnt > 0)  {
    XFreeFontInfo( names, info, cnt );
#ifdef FOO
    /* Most recent documentation says that FreeFontNames is a SUBSET of
       FreeFontInfo */
    /* This causes IRIX and rs6000 to barf */
    XFreeFontNames( names );
#endif
  }
  /* If the above fails, try this: */
  if (act_nfonts == 0) {
    XFontStruct *info;
    /* This just gets the most basic fixed-width fonts */
    names   = XListFontsWithInfo( XBWin->disp, "?x", NFONTS, &cnt, &info );
    j       = 0;
    for (i=0; i < cnt; i++) {
        if (strlen(names[i]) != 2) continue;
        /* printf( "found font %s\n", names[i] ); */
        names[i][1]         = '\0';
	nfonts[j].w         = info[i].max_bounds.width ;
        /* nfonts[j].w         = info[i].max_bounds.lbearing +
                                    info[i].max_bounds.rbearing; */
        nfonts[j].h         = info[i].ascent + info[i].descent;
        nfonts[j].descent   = info[i].descent;
        /* printf( "w = %d, h = %d\n", nfonts[j].w, nfonts[j].h ); */
        if (nfonts[j].w <= 0 || nfonts[j].h <= 0) continue;
        j++;
	if (j >= NFONTS) break;
    }
    act_nfonts    = j;
    XFreeFontInfo( names, info, cnt );
#ifdef FOO
    XFreeFontNames( names );
#endif
  }
  return 0;
}

int XiMatchFontSize( XiFont *font, int w, int h )
{
  int i;

  for (i=0; i<act_nfonts; i++) {
    if (nfonts[i].w == w && nfonts[i].h == h) {
        font->font_w        = w;
        font->font_h        = h;
        font->font_descent  = nfonts[i].descent;
        return 0;
    }
  }

  fprintf( stderr, "Warning could not match to font of %d fonts\n", act_nfonts );
  fprintf( stderr, "Wanted Size %d %d Using %d %d \n", w, h, nfonts[0].w,
                                                           nfonts[0].h);
  /* should use font with closest match */
  font->font_w        = nfonts[0].w;
  font->font_h        = nfonts[0].h;
  font->font_descent  = nfonts[0].descent;
  return 0;
}

/*@
   XiFontWidth - Returns the width of the selected font
@*/
int XiFontWidth(XiFont *font )
{
  return font->font_w;
}

/*@
   XiFontHeight - Returns the height of the selected font
@*/
int XiFontHeight( XiFont *font )
{
  return font->font_h;
}

/*@
   XiFontDescent - Returns the descent of the selected font
@*/
int XiFontDescent( XiFont *font )
{
  return font->font_descent;
}

/*@
    XiFontCreate - Creates a new font structure
@*/
XiFont *XiFontCreate()
{
  XiFont *font;
  font = NEW(XiFont);  CHKPTR(font);
  return font;
}
