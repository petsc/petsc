/*$Id: xinit.c,v 1.69 2000/09/28 21:08:36 bsmith Exp $*/

#if defined(foo)

#include "../src/sys/draw/impls/x/ximpl.h"
/*
    Gets the image from a window and stores it as a .gif file
*/
static char grabmapR[256], grabmapG[256], grabmapB[256];  /* colormap */
static int  gWIDE,gHIGH;

int PetscDrawSave_X(PetscDraw draw,PetscViewer viewer)
{
  PetscDraw_X            *drawx = (PetscDraw_X*)draw;
  XImage            *image;
  XWindowAttributes xwa;
  XColor            *colors;
  int               ncolors, i, ix, iy;
  Window            win;

  ierr = XGetWindowAttributes(disp,win, &xwa);CHKERRQ(ierr);
  XTranslateCoordinates(theDisp, rootW, clickWin, x, y, &ix, &iy, &win);

  xerrcode = 0;
  image = XGetImage(theDisp, clickWin, ix, iy, (u_int) w, (u_int) h, 
		    AllPlanes, ZPixmap);
  i = convertImage(image, colors, ncolors, &xwa);

  /* DO *NOT* use xvDestroyImage(), as the 'data' field was alloc'd by X, not
     necessarily through 'malloc() / free()' */
  XDestroyImage(image);   

#define lowbit(x) ((x) & (~(x) + 1))
static int getxcolors(win_info, colors)
     XWindowAttributes *win_info;
     XColor **colors;
{
  int i, ncolors;
  Colormap cmap;

  *colors = (XColor *) NULL;

  if (win_info->visual->class == TrueColor) {
    if (DEBUG) fprintf(stderr,"TrueColor visual:  no colormap needed\n");
    return 0;
  }

  else if (!win_info->colormap) {
    if (DEBUG) fprintf(stderr,"no colormap associated with window\n");
    return 0;
  }

  ncolors = win_info->visual->map_entries;
  if (DEBUG) fprintf(stderr,"%d entries in colormap\n", ncolors);

  if (!(*colors = (XColor *) malloc (sizeof(XColor) * ncolors)))
    FatalError("malloc failed in getxcolors()");


  if (win_info->visual->class == DirectColor) {
    Pixel red, green, blue, red1, green1, blue1;

    if (DEBUG) fprintf(stderr,"DirectColor visual\n");

    red = green = blue = 0;
    red1   = lowbit(win_info->visual->red_mask);
    green1 = lowbit(win_info->visual->green_mask);
    blue1  = lowbit(win_info->visual->blue_mask);
    for (i=0; i<ncolors; i++) {
      (*colors)[i].pixel = red|green|blue;
      (*colors)[i].pad = 0;
      red += red1;
      if (red > win_info->visual->red_mask)     red = 0;
      green += green1;
      if (green > win_info->visual->green_mask) green = 0;
      blue += blue1;
      if (blue > win_info->visual->blue_mask)   blue = 0;
    }
  }
  else {
    for (i=0; i<ncolors; i++) {
      (*colors)[i].pixel = i;
      (*colors)[i].pad = 0;
    }
  }

  XQueryColors(theDisp, win_info->colormap, *colors, ncolors);

  return(ncolors);

union swapun {
  CARD32 l;
  CARD16 s;
  CARD8  b[sizeof(CARD32)];
};


/**************************************/
static int convertImage(image, colors, ncolors, xwap)
     XImage *image;
     XColor *colors;
     int     ncolors;
     XWindowAttributes *xwap;
{
  /* attempts to conver the image from whatever weird-ass format it might
     be in into something E-Z to deal with (either an 8-bit colormapped
     image, or a 24-bit image).  Returns '1' on success. */

  /* this code owes a lot to 'xwdtopnm.c', part of the pbmplus package,
     written by Jef Poskanzer */

  int             i, j;
  CARD8          *bptr, tmpbyte;
  CARD16         *sptr, sval;
  CARD32         *lptr, lval;
  CARD8          *pptr, *lineptr;
  int            bits_used, bits_per_item, bit_shift, bit_order;
  int            bits_per_pixel, byte_order;
  CARD32         pixvalue, pixmask, rmask, gmask, bmask;
  int            rshift, gshift, bshift, r8shift, g8shift, b8shift;
  CARD32         rval, gval, bval;
  union swapun   sw;
  int            isLsbMachine, flipBytes;
  Visual         *visual;
  char            errstr[256];
  static char    *foo[] = { "\nThat Sucks!" };


  /* quiet compiler warnings */
  sval = 0;
  lval = 0;
  bit_shift = 0;
  pixvalue  = 0;
  rmask  = gmask  = bmask = 0;
  rshift = gshift = bshift = 0;


  /* determine byte order of the machine we're running on */
  sw.l = 1;
  isLsbMachine = (sw.b[0]) ? 1 : 0;

  if (xwap && xwap->visual) visual = xwap->visual;
                       else visual = theVisual;

  if (DEBUG) {
    fprintf(stderr,"convertImage:\n");
    fprintf(stderr,"  %dx%d (offset %d), %s\n",
	    image->width, image->height, image->xoffset, 
	    (image->format == XYBitmap || image->format == XYPixmap) 
	    ? "XYPixmap" : "ZPixmap");

    fprintf(stderr,"byte_order = %s, bitmap_bit_order = %s, unit=%d, pad=%d\n",
	    (image->byte_order == LSBFirst) ? "LSBFirst" : "MSBFirst",
	    (image->bitmap_bit_order == LSBFirst) ? "LSBFirst" : "MSBFirst",
	    image->bitmap_unit, image->bitmap_pad);

    fprintf(stderr,"depth = %d, bperline = %d, bits_per_pixel = %d\n",
	    image->depth, image->bytes_per_line, image->bits_per_pixel);

    fprintf(stderr,"masks:  red %lx  green %lx  blue %lx\n",
	    image->red_mask, image->green_mask, image->blue_mask);

    if (isLsbMachine) fprintf(stderr,"This looks like an lsbfirst machine\n");
                 else fprintf(stderr,"This looks like an msbfirst machine\n");
  }


  if (image->bitmap_unit != 8 && image->bitmap_unit != 16 &&
      image->bitmap_unit != 32) {
    sprintf(errstr, "%s\nReturned image bitmap_unit (%d) non-standard.",
	    "Can't deal with this display.", image->bitmap_unit);
    ErrPopUp(errstr, "\nThat Sucks!");
    return 0;
  }

  if (!ncolors && visual->class != TrueColor) {
    sprintf(errstr, "%s\nOnly TrueColor displays can have no colormap.",
	    "Can't deal with this display.");
    ErrPopUp(errstr, "\nThat Sucks!");
    return 0;
  }


  /* build the 'global' grabPic stuff */
  gWIDE = image->width;  gHIGH = image->height;

  if (visual->class == TrueColor || visual->class == DirectColor ||
      ncolors > 256) {
    grabPic = (byte *) malloc((size_t) gWIDE * gHIGH * 3);
    gbits = 24;
  }
  else {
    grabPic = (byte *) malloc((size_t) gWIDE * gHIGH);
    gbits = 8;

    /* load up the colormap */
    for (i=0; i<ncolors; i++) {
      grabmapR[i] = colors[i].red   >> 8;
      grabmapG[i] = colors[i].green >> 8;
      grabmapB[i] = colors[i].blue  >> 8;
    }
  }
  
  if (!grabPic) FatalError("unable to malloc grabPic in convertImage()");
  pptr = grabPic;


  if (visual->class == TrueColor || visual->class == DirectColor) {
    unsigned int tmp;

    /* compute various shifty constants we'll need */
    rmask = image->red_mask;
    gmask = image->green_mask;
    bmask = image->blue_mask;

    rshift = lowbitnum((unsigned long) rmask);
    gshift = lowbitnum((unsigned long) gmask);
    bshift = lowbitnum((unsigned long) bmask);

    r8shift = 0;  tmp = rmask >> rshift;
    while (tmp >= 256) { tmp >>= 1;  r8shift -= 1; }
    while (tmp < 128)  { tmp <<= 1;  r8shift += 1; }

    g8shift = 0;  tmp = gmask >> gshift;
    while (tmp >= 256) { tmp >>= 1;  g8shift -= 1; }
    while (tmp < 128)  { tmp <<= 1;  g8shift += 1; }

    b8shift = 0;  tmp = bmask >> bshift;
    while (tmp >= 256) { tmp >>= 1;  b8shift -= 1; }
    while (tmp < 128)  { tmp <<= 1;  b8shift += 1; }

    if (DEBUG)
      fprintf(stderr,"True/DirectColor: shifts=%d,%d,%d  8shifts=%d,%d,%d\n",
	      rshift, gshift, bshift, r8shift, g8shift, b8shift);
  }


  bits_per_item = image->bitmap_unit;
  bits_used = bits_per_item;
  bits_per_pixel = image->bits_per_pixel;

  if (bits_per_pixel == 32) pixmask = 0xffffffff;
  else pixmask = (((CARD32) 1) << bits_per_pixel) - 1;

  bit_order = image->bitmap_bit_order;
  byte_order = image->byte_order;

  /* if we're on an lsbfirst machine, or the image came from an lsbfirst
     machine, we should flip the bytes around.  NOTE:  if we're on an
     lsbfirst machine *and* the image came from an lsbfirst machine, 
     *don't* flip bytes, as it should work out */

  /* pity we don't have a logical exclusive-or */
  flipBytes = ( isLsbMachine && byte_order != LSBFirst) ||
              (!isLsbMachine && byte_order == LSBFirst);

  for (i=0; i<image->height; i++) {
    lineptr = (byte *) image->data + (i * image->bytes_per_line);
    bptr = ((CARD8  *) lineptr) - 1;
    sptr = ((CARD16 *) lineptr) - 1;
    lptr = ((CARD32 *) lineptr) - 1;
    bits_used = bits_per_item;

    for (j=0; j<image->width; j++) {
      
      /* get the next pixel value from the image data */

      if (bits_used == bits_per_item) {  /* time to move on to next b/s/l */
	switch (bits_per_item) {
	case 8:  bptr++;  break;
	case 16: sptr++;  sval = *sptr;
	         if (flipBytes) {   /* swap CARD16 */
		   sw.s = sval;
		   tmpbyte = sw.b[0];
		   sw.b[0] = sw.b[1];
		   sw.b[1] = tmpbyte;
		   sval = sw.s;
		 }
	         break;
	case 32: lptr++;  lval = *lptr;
	         if (flipBytes) {   /* swap CARD32 */
		   sw.l = lval;
		   tmpbyte = sw.b[0];
		   sw.b[0] = sw.b[3];
		   sw.b[3] = tmpbyte;
		   tmpbyte = sw.b[1];
		   sw.b[1] = sw.b[2];
		   sw.b[2] = tmpbyte;
		   lval = sw.l;
		 }
	         break;
	}
		   
	bits_used = 0;
	if (bit_order == MSBFirst) bit_shift = bits_per_item - bits_per_pixel;
	                      else bit_shift = 0;
      }

      switch (bits_per_item) {
      case 8:  pixvalue = (*bptr >> bit_shift) & pixmask;  break;
      case 16: pixvalue = ( sval >> bit_shift) & pixmask;  break;
      case 32: pixvalue = ( lval >> bit_shift) & pixmask;  break;
      }

      if (bit_order == MSBFirst) bit_shift -= bits_per_pixel;
                            else bit_shift += bits_per_pixel;
      bits_used += bits_per_pixel;

      
      /* okay, we've got the next pixel value in 'pixvalue' */
      
      if (visual->class == TrueColor || visual->class == DirectColor) {
	/* in either case, we have to take the pixvalue and 
	   break it out into individual r,g,b components */
	rval = (pixvalue & rmask) >> rshift;
	gval = (pixvalue & gmask) >> gshift;
	bval = (pixvalue & bmask) >> bshift;

	if (visual->class == DirectColor) {
	  /* use rval, gval, bval as indicies into colors array */

	  *pptr++ = (rval < ncolors) ? (colors[rval].red   >> 8) : 0;
	  *pptr++ = (gval < ncolors) ? (colors[gval].green >> 8) : 0;
	  *pptr++ = (bval < ncolors) ? (colors[bval].blue  >> 8) : 0;
	}

	else {   /* TrueColor */
	  /* have to map rval,gval,bval into 0-255 range */
	  *pptr++ = (r8shift >= 0) ? (rval << r8shift) : (rval >> (-r8shift));
	  *pptr++ = (g8shift >= 0) ? (gval << g8shift) : (gval >> (-g8shift));
	  *pptr++ = (b8shift >= 0) ? (bval << b8shift) : (bval >> (-b8shift));
	}
      }

      else { /* all others: StaticGray, StaticColor, GrayScale, PseudoColor */
	/* use pixel value as an index into colors array */

	if (pixvalue >= ncolors) {
	  FatalError("convertImage(): pixvalue >= ncolors");
	}

	if (gbits == 24) {   /* too many colors for 8-bit colormap */
	  *pptr++ = (colors[pixvalue].red)   >> 8;
	  *pptr++ = (colors[pixvalue].green) >> 8;
	  *pptr++ = (colors[pixvalue].blue)  >> 8;
	}
	else *pptr++ = pixvalue & 0xff;

      }
    }
  }

  return 1;
}



/**************************************/
static int lowbitnum(ul)
     unsigned long ul;
{
  /* returns position of lowest set bit in 'ul' as an integer (0-31),
   or -1 if none */

  int i;
  for (i=0; ((ul&1) == 0) && i<32;  i++, ul>>=1);
  if (i==32) i = -1;
  return i;
}

#endif
