/*
 * xvgrab.c - image grabbing (from the display) functions for XV
 *
 *  Author:    John Bradley, University of Pennsylvania
 *
 *  Contains:
 *     int Grab()             - handles the GRAB command
 *     int LoadGrab();        - 'loads' the pic from the last succesful Grab
 *            
 */

#include "copyright.h"

#define NEEDSTIME
#include "xv.h"

static byte *grabPic = (byte *) NULL;
static int  gbits;                              /* either '8' or '24' */
static byte grabmapR[256], grabmapG[256], grabmapB[256];  /* colormap */
static int  gWIDE,gHIGH;
static int  grabInProgress=0;
static int  hidewins = 0;
static GC   rootGC;

static void   flashrect       PARM((int, int, int, int, int));
static void   startflash      PARM((void));
static void   endflash        PARM((void));
static int    grabImage       PARM((Window, int, int, int, int));
static void   ungrabX         PARM((void));
static int    convertImage    PARM((XImage *, XColor *, int, 
				    XWindowAttributes *));

static int    lowbitnum       PARM((unsigned long));
static int    getxcolors      PARM((XWindowAttributes *, XColor **));
static Window xvClientWindow  PARM((Display *, Window));




/***********************************/
int Grab()
{
  /* does UI part of Grab command.  returns '1' if a new image was grabbed,
     0 if cancelled */

  int          i, x, y, x1, y1, x2, y2, ix, iy, iw, ih, rv;
  int          rx, ry, pretendGotB1, autograb;
  int          oldaclose;
  Window       rW, cW, clickWin, tmpwin;
  unsigned int mask;
  XColor       fc, bc;

  pretendGotB1 = 0;

  if (grabInProgress) return 0;      /* avoid recursive grabs during delay */

  /* do the dialog box thing */
  i = GrabPopUp(&hidewins, &grabDelay);
  if (i==2) return 0;    /* cancelled */
  autograb = (i==1);

  if (hidewins) {                   /* unmap all XV windows */
    autoclose += 2;                 /* force it on once */
    if (mainW && dispMode==RMB_WINDOW) XUnmapWindow(theDisp, mainW);
    else if (ctrlW) CtrlBox(0);
  }


  XSync(theDisp, False);

  if (grabDelay>0) {    /* instead of sleep(), handle events while waiting */
    time_t startT, t;
    int  done;

    grabInProgress = 1; /* guard against recursive grabs during delay */
    time(&startT);
    while (1) {
      time(&t);  
      if (t >= startT + grabDelay) break;
      if (XPending(theDisp)>0) {
	XEvent evt;
	XNextEvent(theDisp, &evt);
	i = HandleEvent(&evt, &done);
	if (done) {                    /* only 'new image' cmd accepted=quit */
	  if (i==QUIT) Quit(0);
	  else XBell(theDisp, 0);
	}
      }
      Timer(100);
    }
    grabInProgress = 0;
  }

  
  rootGC   = DefaultGC(theDisp, theScreen);
  
  if (grabPic) {  /* throw away previous 'grabbed' pic, if there is one */
    free(grabPic);  grabPic = (byte *) NULL;
  }


  fc.flags = bc.flags = DoRed | DoGreen | DoBlue;
  fc.red = fc.green = fc.blue = 0xffff;  
  bc.red = bc.green = bc.blue = 0x0000;
  XRecolorCursor(theDisp, tcross, &fc, &bc);


  XBell(theDisp, 0);		/* beep once at start of grab */

  if (!autograb) XGrabButton(theDisp, (u_int) AnyButton, 0, rootW, False, 0, 
			     GrabModeAsync, GrabModeSync, None, tcross);
  
  if (autograb) {
    XGrabServer(theDisp);	 /* until we've done the grabImage */
    if (!XQueryPointer(theDisp,rootW,&rW,&cW,&rx,&ry,&x1,&y1,&mask)) {
      fprintf(stderr, "xv: XQueryPointer failed!\n");
      XUngrabServer(theDisp);
      rv = 0;
      goto exit;
    }
    else { pretendGotB1 = 1;  mask = Button1Mask; }
  }

  else {   /* !autograb */
    /* wait for a button press */
    while (1) {
      XEvent evt;  int done;

      /* recolor cursor to indicate that grabbing is active? */

      if (XQueryPointer(theDisp,rootW,&rW,&cW,&rx,&ry,&x1,&y1,&mask)) {
	if (mask & (Button1Mask | Button2Mask | Button3Mask)) break;
      }

      /* continue to handle events while waiting... */
      XNextEvent(theDisp, &evt);
      i = HandleEvent(&evt, &done);
      if (done) {                    /* only 'new image' cmd accepted=quit */
	if (i==QUIT) { 
	  XUngrabButton(theDisp, (u_int) AnyButton, 0, rootW);
	  Quit(0);
	}
	else XBell(theDisp, 0);
      }

    }
  }
  
  
  /***
   ***  got button click (or pretending we did, if autograb)
   ***/
  

  if (mask & Button3Mask || rW!=rootW) {        /* Button3: CANCEL GRAB */
    while (1) {      /* wait for button to be released */
      if (XQueryPointer(theDisp,rootW,&rW,&cW,&rx,&ry,&x1,&y1,&mask)) {
	if (!(mask & Button3Mask)) break;
      }
    }
    
    XUngrabButton(theDisp, (u_int) AnyButton, 0, rootW);
    XBell(theDisp, 0);
    XBell(theDisp, 0);
    rv = 0;
    goto exit;
  }



  if (mask & Button1Mask) {  /* Button1:  GRAB WINDOW (& FRAME, maybe)     */
    while (!pretendGotB1) {  /* wait for button to be released, if clicked */
      int rx,ry,x1,y1;  Window rW, cW;
      if (XQueryPointer(theDisp,rootW,&rW,&cW,&rx,&ry,&x1,&y1,&mask)) {
	if (!(mask & Button1Mask)) break;
      }
    }
    
    if (!cW || cW == rootW) clickWin = rootW;
    else {
      int xr, yr;    Window chwin;
      XTranslateCoordinates(theDisp, rW, cW, rx, ry, &xr, &yr, &chwin);
      if (chwin != None) {
	XWindowAttributes clickxwa, parentxwa;

	clickWin = xvClientWindow(theDisp, chwin);

	/* decide if we want to just grab clickWin, or cW.
	   basically, if they're different in any important way 
	   (depth, visual, colormap), grab 'clickWin' only, 
	   as it's the important part */

	if (!clickWin || 
	    (XGetWindowAttributes(theDisp, clickWin, &clickxwa)  &&
	     XGetWindowAttributes(theDisp, cW,       &parentxwa) &&
	     clickxwa.visual->class == parentxwa.visual->class   &&
	     clickxwa.colormap      == parentxwa.colormap        &&
	     clickxwa.depth         == parentxwa.depth)
	    )
	  clickWin = cW;   	  /* close enough! */
      }
      else clickWin = cW;
      
      if (DEBUG) 
	fprintf(stderr, "rW = %x, cW = %x, chwin = %x, clickWin = %x\n",
		(u_int) rW, (u_int) cW, (u_int) chwin, (u_int) clickWin);
    }
    
    
    if (clickWin == rootW) {   /* grab entire screen */
      if (DEBUG) fprintf(stderr,"Grab: clicked on root window.\n");
      ix = iy = 0;  iw = dispWIDE;  ih = dispHIGH;
    }
    else {
      int x,y;  Window win;   unsigned int rw,rh,rb,rd;
      
      if (XGetGeometry(theDisp,clickWin,&rW, &x, &y, &rw, &rh, &rb, &rd)) {
	iw = (int) rw;  ih = (int) rh;
	
	XTranslateCoordinates(theDisp, clickWin, rootW, 0, 0, &ix,&iy, &win);
	
	if (DEBUG) fprintf(stderr,"clickWin=0x%x: %d,%d %dx%d depth=%ud\n", 
			   (u_int) clickWin, ix, iy, iw, ih, rd);    
      }
      else {
	ix = iy = 0;  iw = dispWIDE;  ih = dispHIGH;  clickWin = rootW;
	if (DEBUG) fprintf(stderr,"XGetGeometry failed? (using root win)\n");
      }
    }
    
    
    /* range checking:  keep rectangle fully on-screen */
    if (ix<0) { iw += ix;  ix = 0; }
    if (iy<0) { ih += iy;  iy = 0; }
    if (ix+iw>dispWIDE) iw = dispWIDE-ix;
    if (iy+ih>dispHIGH) ih = dispHIGH-iy;
    
    
    if (DEBUG) fprintf(stderr,"using %d,%d (%dx%d)\n", ix, iy, iw, ih);
    
    /* flash the rectangle a bit... */
    startflash();
    for (i=0; i<5; i++) {
      flashrect(ix, iy, iw, ih, 1);
      XFlush(theDisp);  Timer(100);
      flashrect(ix, iy, iw, ih, 0);
      XFlush(theDisp);  Timer(100);
    }
    endflash();
  }


  else {  /* Button2:  TRACK A RECTANGLE */
    int    origrx, origry;
    Window origcW;

    clickWin = rootW;  origcW = cW;
    origrx = ix = x2 = rx;
    origry = iy = y2 = ry;
    iw = ih = 0;
    
    XGrabServer(theDisp);
    startflash();

    /* Wait for button release while tracking rectangle on screen */
    while (1) {
      if (XQueryPointer(theDisp,rootW,&rW,&cW,&rx,&ry,&x,&y,&mask)) {
	if (!(mask & Button2Mask)) break;
      }

      flashrect(ix, iy, iw, ih, 0);                /* turn off rect */

      if (rx!=x2 || ry!=y2) {                      /* resize rectangle */
	ix = (x1<rx) ? x1 : rx;
	iy = (y1<ry) ? y1 : ry;
	iw = abs(rx - x1);  ih = abs(ry - y1);
	x2 = rx;  y2 = ry;
      }
      
      if (iw>1 && ih>1) flashrect(ix,iy,iw,ih,1);  /* turn on rect */
    }

    flashrect(ix, iy, iw, ih, 0);                  /* turn off rect */
    endflash();
    
    XUngrabServer(theDisp);
    
    
    if (origcW == cW) {  /* maybe it's entirely in one window??? */
      if (cW) {    /* will be 0 if clicked in rootW */
	Window stwin, enwin, stwin1, enwin1;
	if (DEBUG) fprintf(stderr,"origcW=%x cW=%x   ", 
			   (u_int) origcW, (u_int) cW);
	XTranslateCoordinates(theDisp,rootW,cW, origrx,origry,&x,&y,&stwin);
	XTranslateCoordinates(theDisp,rootW,cW, rx,    ry,    &x,&y,&enwin);
	
	if (DEBUG) fprintf(stderr,"stwin=%x enwin=%x   ", 
			   (u_int) stwin, (u_int) enwin);
	if (stwin == enwin && stwin != None) {
	  stwin1 = xvClientWindow(theDisp, stwin);
	  enwin1 = xvClientWindow(theDisp, enwin);
	  if (DEBUG) fprintf(stderr,"stwin1=%x enwin1=%x   ", 
			     (u_int) stwin1, (u_int) enwin1);
	  
	  if (stwin1 == enwin1 && stwin1) clickWin = stwin1;
	  else clickWin = stwin;
	}
	if (DEBUG) fprintf(stderr,"\n");
      }
      else clickWin = rootW;
    }
  }
  

  /***
   ***  now that clickWin,ix,iy,iw,ih are known, try to grab the bits...
   ***/


  WaitCursor();

  if (!autograb) XGrabServer(theDisp);	 /* until we've done the grabImage */
  rv = grabImage(clickWin,ix,iy,iw,ih);  /* ungrabs the server & button */

  SetCursors(-1);


 exit:

  if (hidewins) {                   /* remap XV windows */
    autoclose += 2;                 /* force it on once */
    if (mainW && dispMode == RMB_WINDOW) {
      int state;

      XMapRaised(theDisp, mainW);
      XSync(theDisp, False);        /* get that damned window on screen */

      if (DEBUG) fprintf(stderr,"==remapped mainW.  waiting for Config.\n");

      /* sit here until we see a MapNotify on mainW followed by a 
	 ConfigureNotify on mainW */

      state = 0;
      while (1) {
	XEvent event;
	XNextEvent(theDisp, &event);
	HandleEvent(&event, &i);

	if (state==0 && event.type == MapNotify &&
	    event.xmap.window == mainW) state = 1;

	if (state==1 && event.type == ConfigureNotify && 
	    event.xconfigure.window == mainW) break;
      }

      if (DEBUG) fprintf(stderr,"==after remapping mainW, GOT Config.\n");
    }
      
    else if (ctrlW) CtrlBox(1);
  }

  return rv;
}


/***********************************/
static void flashrect(x,y,w,h,show)
     int x,y,w,h,show;
{
  static int isvis  = 0;
  static int maskno = 0;

  XSetPlaneMask(theDisp, rootGC, xorMasks[maskno]);

  if (!show) {     /* turn off rectangle */
    if (isvis) 
      XDrawRectangle(theDisp, rootW, rootGC, x, y, (u_int) w-1, (u_int) h-1);

    isvis = 0;
  }
  else {           /* show rectangle */
    if (!isvis && w>1 && h>1) {
      maskno = (maskno + 1) & 7;
      XSetPlaneMask(theDisp, rootGC, xorMasks[maskno]);
      XDrawRectangle(theDisp, rootW, rootGC, x, y, (u_int) w-1, (u_int) h-1);
      isvis = 1;
    }
  }
}


/***********************************/
static void startflash()
{  
  /* set up for drawing a flashing rectangle */
  XSetFunction(theDisp, rootGC, GXinvert);
  XSetSubwindowMode(theDisp, rootGC, IncludeInferiors);
}

/***********************************/
static void endflash()
{  
  XSetFunction(theDisp, rootGC, GXcopy);
  XSetSubwindowMode(theDisp, rootGC, ClipByChildren);
  XSetPlaneMask(theDisp, rootGC, AllPlanes);
  XSync(theDisp, False);
}


/***********************************/
static int grabImage(clickWin, x, y, w, h)
     Window clickWin;
     int    x, y, w, h;
{
  /* attempts to grab the specified rectangle of the root window
     returns '1' on success.  clickWin is used to figure out the depth
     and colormap to use */

  XImage *image;
  XWindowAttributes xwa;
  XColor *colors;
  int ncolors, i, ix, iy;
  char str[256];
  Window win;


  /* range checking */
  if (x<0) { w += x;  x = 0; }
  if (y<0) { h += y;  y = 0; }
  if (x+w>dispWIDE) w = dispWIDE-x;
  if (y+h>dispHIGH) h = dispHIGH-y;

  if (w==0 || h==0) {  /* selected nothing */
    ungrabX();
    return 0;
  }

  if (!XGetWindowAttributes(theDisp, clickWin, &xwa)) {
    sprintf(str,"Unable to get window attributes for clicked-on window\n");
    ungrabX();
    ErrPopUp(str, "\nThat Sucks!");
    return 0;
  }


  XTranslateCoordinates(theDisp, rootW, clickWin, x, y, &ix, &iy, &win);

  xerrcode = 0;
  image = XGetImage(theDisp, clickWin, ix, iy, (u_int) w, (u_int) h, 
		    AllPlanes, ZPixmap);
  if (xerrcode || !image || !image->data) {
    sprintf(str, "Unable to get image (%d,%d %dx%d) from display", ix,iy,w,h);
    ungrabX();
    ErrPopUp(str, "\nThat Sucks!");
    return 0;
  }

  ncolors = getxcolors(&xwa, &colors);

  ungrabX();

  if (ncolors && DEBUG) {
    fprintf(stderr, "Colormap:\n");
    for (i=0; i<ncolors; i++)
      fprintf(stderr,"%02x%02x%02x  ",colors[i].red>>8, colors[i].green>>8,
	      colors[i].blue>>8);
    fprintf(stderr,"\n");
  }


  XBell(theDisp, 0);    /* beep twice at end of grab */
  XBell(theDisp, 0);

  i = convertImage(image, colors, ncolors, &xwa);

  /* DO *NOT* use xvDestroyImage(), as the 'data' field was alloc'd by X, not
     necessarily through 'malloc() / free()' */
  XDestroyImage(image);   
  
  if (colors) free((char *) colors);

  return i;
}


static void ungrabX()
{
  XUngrabServer(theDisp);
  XUngrabButton(theDisp, (u_int) AnyButton, 0, rootW);
}





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



/**************************************/
/* following code snarfed from 'xwd.c' */
/**************************************/

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
}
    




/***********************************/
int LoadGrab(pinfo)
     PICINFO *pinfo;
{
  /* loads up (into XV structures) last image successfully grabbed.
     returns '0' on failure, '1' on success */

  int   i;

  if (!grabPic) return 0;   /* no image to use */

  if (gbits == 24) pinfo->type = PIC24;
  else {
    pinfo->type = PIC8;

    for (i=0; i<256; i++) {
      pinfo->r[i] = grabmapR[i];
      pinfo->g[i] = grabmapG[i];
      pinfo->b[i] = grabmapB[i];
    }
  }

  pinfo->pic     = grabPic;
  pinfo->normw   = pinfo->w   = gWIDE;
  pinfo->normh   = pinfo->h   = gHIGH;
  pinfo->frmType = -1;
  pinfo->colType = -1;

  sprintf(pinfo->fullInfo,"<%s internal>", 
	  (pinfo->type == PIC8) ? "8-bit" : "24-bit");
  
  sprintf(pinfo->shrtInfo,"%dx%d image.",gWIDE, gHIGH);
  
  pinfo->comment = (char *) NULL;

  grabPic = (byte *) NULL;

  return 1;
}





#include <X11/Xlib.h>
#include <X11/Xatom.h>

static Window TryChildren PARM((Display *, Window, Atom));

/* Find a window with WM_STATE, else return '0' */

static Window xvClientWindow (dpy, win)
    Display *dpy;
    Window win;
{
    Atom WM_STATE;
    Atom type = None;
    int format;
    unsigned long nitems, after;
    unsigned char *data;
    Window inf;

    WM_STATE = XInternAtom(dpy, "WM_STATE", True);
    if (!WM_STATE) return win;

    XGetWindowProperty(dpy, win, WM_STATE, 0L, 0L, False, AnyPropertyType,
		       &type, &format, &nitems, &after, &data);
    if (type) return win;

    inf = TryChildren(dpy, win, WM_STATE);

    return inf;
}

static Window TryChildren (dpy, win, WM_STATE)
    Display *dpy;
    Window win;
    Atom WM_STATE;
{
    Window root, parent;
    Window *children;
    unsigned int nchildren;
    unsigned int i;
    Atom type = None;
    int format;
    unsigned long nitems, after;
    unsigned char *data;
    Window inf = 0;

    if (!XQueryTree(dpy, win, &root, &parent, &children, &nchildren))
	return 0;

    for (i = 0; !inf && (i < nchildren); i++) {
	XGetWindowProperty(dpy, children[i], WM_STATE, 0L, 0L, False,
			   AnyPropertyType, &type, &format, &nitems,
			   &after, &data);
	if (type)
	  inf = children[i];
    }

    for (i = 0; !inf && (i < nchildren); i++)
      inf = TryChildren(dpy, children[i], WM_STATE);

    if (children) XFree((char *)children);
    return inf;
}
