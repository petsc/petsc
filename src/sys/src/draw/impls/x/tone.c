#ifndef lint
static char vcid[] = "$Id: tone.c,v 1.1 1994/03/18 00:22:34 gropp Exp $";
#endif
/*
    This is a standalong routine for displaying a matrix of values using
    a color plot under suntools.  The approach is:

    convert the matrix (of doubles) to a color map, using 1 byte colors.
    (Actually, an intensity map, with the colors linear maps of the
    intensity).  This map is then displayed using a ramp of colors.
    For simplicity, the map is organized as suntools desires it, and
    so that it can be displayed with a single call, rather than
    zillions of put_pixel calls.

    The matrix is organized as:

      (1,1)                                       (nx,1)
        x       x       x               x       x   x

        x       x       x               x       x   x

        x       x       x               x       x   x
        x       x       x               x       x   x

        x       x       x               x       x   x
      (1,ny)                                      (nx,ny)

    (notice the uneven spacing).
    In each "cell" of the matrix, bilinear interpolation is used to
    assign the color values.

 */

#include <stdio.h>
#include "tools.h"
#include <math.h>

#define SHIFT_VAL 6
/*
    XBSetcell - set an individual cell with a bi-linear interpolant
                    of the provided corner values
    input parameters:
        nx      - cell is part of ny by nx matrix
        mx,my   - cell is mx x my in size
        ul, ur, - upper left, right values
        ll, lr  - lower left, right values
        ncolor  - valid values are coff + [0..ncolor-1]
	coff    - index of MIN colormap entry
    output parameters
        cell    - cell to set
    Algorithm:
        This code uses a simplified Bressenham algorithm to do the intensity
        lines.  There are three such lines: one down each side, and one
        across (using the values from down the side).  Only the one across
        is done with Bresenham.
        To keep more accuracy, each corner value is multiplied by 2^SHIFT_VAL,
        and the values in the cell are shifted back by that amount.

    Finally, for efficiency reasons, the cases dy >,<,== 0 are handled
    separately.  Clipping the color is also checked for.  For additional
    speed, we could handle |dy| <= mx-1; in that case the while(err>=mx)
    turns into an if(err>=mx).  This should be a common case, since the
    change in color from one pixel to an adjacent one will usually be
    no greater than 1.  Other speed ups could work with words rather than
    bytes.

    Just to make things more exciting, very non-uniform data can have
    mx and/or my == 1.  In this case, the cell is degenerate, and
    we have to fake out some of the tests.
 */
void XBSetcell( cell, nx, mx, my, ul, ur, ll, lr, ncolor, coff )
unsigned char   *cell;
int             nx, my;
int             ul, ur, ll, lr;
int             mx;
int             ncolor;
{
int     i;
register unsigned char *c;
register unsigned char cc;
int     l, r;           /* left and right values */
int     diffleft, diffright, dl, dr;
register int    y, err;    /* err is (mx-1)*(y_actual - y) */
unsigned char   ncm1 = ncolor - 1 + coff;
/* This must always be at least 1, or the error delta will be zero */
register int    mxm1 = (mx > 1) ? mx - 1 : 1;
register int    tval;
int     off;
int     myby2, mxby2, sgnl, sgnr;
register int     dy, j;

off = 1 << (SHIFT_VAL - 1);
/*
   A substantial amount of time is spent doing the multiplications
   and divisions in the setup of this code; we move as much as possible
   out of the loop.
 */
diffleft = ll - ul;
diffright= lr - ur;
dl       = 0;
dr       = 0;
myby2    = (my-1)/2;
mxby2    = (mx-1)/2;
sgnl     = (diffleft >= 0) ? myby2 : -myby2;
sgnr     = (diffright >= 0) ? myby2 : -myby2;
for (i=0; i<my; i++) {
    c   = cell;
    cell += nx;
    /*
       If my is even, we want these to be symmetric: reversing
       (ul,ll) or (ur,lr) should exactly reverse the order of
       the generated values.  The approach here adds or subtracts
       1/2 before truncating; the sign is picked to make the
       operation symmetric.
     */
    if (my == 1) {
        l       = ul;
        r       = ur;
        }
    else {
        l       = (dl + sgnl) / (my - 1) + ul;
        r       = (dr + sgnr) / (my - 1) + ur;
        }
    dl      += diffleft;
    dr      += diffright;
    dy      = r - l;
    y       = l;
    err     = (dy >=0) ? mxby2 : -mxby2 ;
    j       = mx;
    y       = y + off;
    if (dy == 0) {
        cc      = y >> SHIFT_VAL;
        if (cc > ncm1) cc = ncm1;
        while (j--) {
            *c++ = cc;
            }
        }
    else if (dy > 0) {
        tval    = mx;
        if ( ((r + off) >> SHIFT_VAL) > ncm1) {
            while (j--) {
                cc      = y >> SHIFT_VAL;
                if (cc > ncm1) cc = ncm1;
                *c++    = cc;
                err     += dy;
                while (err >= tval) {
                    err -= mxm1;
                    y++;
                    }
                }
            }
        else {
            while (j--) {
                *c++    = y >> SHIFT_VAL;
                err     += dy;
                while (err >= tval) {
                    err -= mxm1;
                    y++;
                    }
                }
            }
        }
    else {      /* dy < 0 */
        tval    = -mx;
        if ( ((l + off) >> SHIFT_VAL) > ncm1) {
            while (j--) {
                cc      = y >> SHIFT_VAL;
                if (cc > ncm1) cc = ncm1;
                *c++    = cc;
                err     += dy;
                while (err <= tval) {
                    err += mxm1;
                    y--;
                    }
                }
            }
        else {
            while (j--) {
                *c++ = y >> SHIFT_VAL;
                err     += dy;
                while (err <= tval) {
                    err += mxm1;
                    y--;
                    }
                }
            }
        }
    }
}

/*
    XBGetimap - given a matrix (double) value, return the color
                    index to use.  This is done by finding the index in
                    an array, imap, that the value falls between
    input parameters:
        imap    - map of values
        ncolor  - valid indices are 0:ncolor-1
        v       - value to find
    Note: imap values are monotone increasing
    Further, we need to give some (SHIFT_VAL) additional bits to
    allow the bilinear interpolation to smooth the data.  The
    additional bits are computed by doing linear interpolation on imap.

    A special case is an imap that is constant;  In this case, the result
    is just 0;
 */
unsigned int XBGetimap( imap, ncolor, v )
double  *imap, v;
int     ncolor;
{
register int idx;
unsigned int vv, vpart;
double dx;
for (idx=0; idx<ncolor; idx++)
    if (v <= imap[idx]) {
        if (idx == 0) {
            dx  = (imap[1] - imap[0]);
            /* try to smooth this out, using the first gap as the 0th gap */
	    if (dx == 0.0) vpart = 0;
	    else 
		vpart   = (unsigned int) ( (1 << SHIFT_VAL) *
					  (v - (imap[0]-dx)) / dx );
            if (vpart >= (1 << SHIFT_VAL)) vpart = (1 << SHIFT_VAL) - 1;
            }
        else {
	    dx      = imap[idx] - imap[idx-1];
	    if (dx == 0.0) vpart = 0;
	    else 
		vpart   = (unsigned int) ( (1 << SHIFT_VAL) *
					   (v - imap[idx-1]) / dx );
            }
        vv   = (idx << SHIFT_VAL) + vpart;
        return vv;
        }
return ((unsigned int)(ncolor - 1)) << SHIFT_VAL;
}

/*
    XBGetloc - convert x locations into indices in the color map
    input parameters:
        nx      - number of columns in map
        mx      - number of columns in matrix
        x       - x values of columns in matrix
    output value
        matrix of size mx that give the cell indices
    Assumes that the entries are monotonically increasing.
 */
int *XBGetloc( nx, mx, x )
int     nx, mx;
double  *x;
{
register int i;
int     *p;
double  dx;

p   = (int *) MALLOC( (unsigned)( mx * sizeof(int) ) );   CHKPTRN(p)
if (!p) return p;

dx  = (nx - 1) / ( x[mx-1] - x[0] );
if (dx <= 0) {
    fprintf( stderr, "matrix coordinates not strictly increasing\n" );
    fprintf( stderr, "from %le to %le in %d\n", x[0], x[mx-1], nx );
    }
for (i=0; i<mx; i++)
    p[i]    = (x[i] - x[0]) * dx;
p[mx-1] = nx - 1;
return p;
}

/*
    XBSetmap - set a color map from a matrix
 */
void XBSetmap( map, nx, ny, matrix, mx, my, x, y, ncolor, imap, coff )
unsigned char *map;
int         nx, ny, mx, my, ncolor, coff;
double      *matrix, *x, *y;
double      *imap;
{
int             i, j;
double          *p;
unsigned char   *mp;
int             ul, ur, ll, lr, nnx, nny;
int             *ix, *iy;
int             cshift = (coff << SHIFT_VAL);
/* determine the sizes of the cells based on x, y */
ix  = XBGetloc( nx, mx, x );
iy  = XBGetloc( ny, my, y );
if (!ix || !iy) {
    if (!ix) SETERRC(1,"x indices invalid");
    if (!iy) SETERRC(1,"y indices invalid");
    return;
    }
/* printf( ">>>>>Starting setmap\n" ); */
for (j=0; j<my-1; j++) {
    p   = matrix + j * mx;
    if (iy[j] >= ny || iy[j] < 0) {
	fprintf( stderr,
		"in setmap, offsets to map (iy[%d]=%d) exceed (ny=%d)\n",
		j, iy[j], ny );
	continue;
	}
    ur  = XBGetimap( imap, ncolor, *p ) + cshift;
    lr  = XBGetimap( imap, ncolor, *(p + mx) ) + cshift;
    for (i=0; i<mx-1; i++) {
        mp  = map + iy[j] * nx + ix[i];
        if (ix[i] >= nx || ix[i] < 0) {
            fprintf( stderr,
                     "in setmap, offsets to map (ix[%d]=%d) exceed (nx=%d)\n",
                        i, ix[i], nx );
	    continue;
            }
        nnx = ix[i+1] - ix[i] + 1;
        nny = iy[j+1] - iy[j] + 1;
        ul  = ur;
        ur  = XBGetimap( imap, ncolor, *(p + 1) ) + cshift;
        ll  = lr;
        lr  = XBGetimap( imap, ncolor, *(p + mx + 1) ) + cshift;
/*
        printf( "setting cell size (%d,%d) at (%d,%d)\n",
                nnx, nny, ix[i], iy[j] );
        printf( "corner values (%d,%d) x (%d,%d)\n", *p, *(p+1), *(p+mx),
                *(p+mx+1) );
        printf( "corner colors (%d,%d) x (%d,%d)\n", ul, ur, ll, lr );
 */
        XBSetcell( mp, nx, nnx, nny, ul, ur, ll, lr, ncolor, coff );
/*
        printf( "after setcell\n" );
        printmap( map, nx, ny );
 */
        p++;
        }
    }

FREE( ix );
FREE( iy );
}

/*
  This is like XBSetmap, except:
      Data is organized bottom to top, rather than top to bottom
      There are nnx values on each row, of which only nx should be used.
  This routine permits data to be displayed without copying it into
  a separate buffer when the data is in the "natural coordinate system".
 */
void XBSetmap2( map, nx, ny, matrix, mmx, mx, my, x, y, ncolor, imap )
unsigned char *map;
int         nx, ny, mx, my, mmx, ncolor;
double      *matrix, *x, *y;
double      *imap;
{
int             i, j;
double          *p;
unsigned char   *mp;
int             ul, ur, ll, lr, nnx, nny;
int             *ix, *iy;

/* determine the sizes of the cells based on x, y */
ix  = XBGetloc( nx, mx, x );
iy  = XBGetloc( ny, my, y );
if (!ix || !iy) {
    if (!ix) SETERRC(1,"x indices invalid");
    if (!iy) SETERRC(1,"y indices invalid");
    return;
    }
/* printf( "###Starting setmap###\n" ); */
for (j=0; j<my-1; j++) {
    p   = matrix + (my-1-j) * mmx;
    ur  = XBGetimap( imap, ncolor, *p );
    lr  = XBGetimap( imap, ncolor, *(p - mmx) );
    for (i=0; i<mx-1; i++) {
        mp  = map + (iy[my-1] - iy[my-1-j]) * nx + ix[i];
        if (ix[i] >= nx || iy[my-1-j] >= ny ||
            ix[i] < 0   || iy[my-1-j] < 0) {
            fprintf( stderr,
                     "in setmap, offsets to map (%d,%d) exceed (%d,%d)\n",
                        ix[i], iy[my-1-j], nx, ny );
            }
        nnx = ix[i+1] - ix[i] + 1;
        nny = - iy[my-1-(j+1)] + iy[my-1-j] + 1;
        ul  = ur;
        ur  = XBGetimap( imap, ncolor, *(p + 1) );
        ll  = lr;
        lr  = XBGetimap( imap, ncolor, *(p - mmx + 1) );
/*
        printf( "setting cell size (%d,%d) at (%d,%d)\n",
                nnx, nny, ix[i], iy[j] );
        printf( "corner values (%f,%f) x (%f,%f)\n", *p, *(p+1), *(p-mmx),
                *(p-mmx+1) );
        printf( "corner colors (%d,%d) x (%d,%d)\n", ul, ur, ll, lr );
 */
        XBSetcell( mp, nx, nnx, nny, ul, ur, ll, lr, ncolor, 0 );
/*
        printf( "after setcell\n" );
        printmap( map, nx, ny );
 */
        p++;
        }
    }

FREE( ix );
FREE( iy );
}

/*
    XBSetTriangle - set an individual cell with a linear interpolant
                    of the provided corner values
    input parameters:
        map     - bitmap to draw in
	x1,y1   - location of 1st corner in map
	t1      - value of 1st corner
	x2,y2   - location of 2nd corner in map
	t2      - value of 2nd corner
	x3,y3   - location of 3rd corner in map
	t3      - value of 3rd corner
        ncolor  - valid values are coff + [0..ncolor-1]
	coff    - index of MIN colormap entry
    output parameters
        map    - cell to set
    Algorithm:
        This code uses a simplified Bressenham algorithm to do the intensity
        lines.  The vertices are reorder so that they look like this:
$                              t1
$
$                        t2
$                                       t3
$       Lines are then drawn for t1-t2 to t1-t3 until either t2 is reached;
        then lines for t2-t3 to t1-t3 are drawn.
 */
void XBSetTriangle( map, nx, x1, y1, t1, x2, y2, t2, x3, y3, t3, ncolor, coff )
unsigned char   *map;
int             nx, x1, y1, t1, x2, y2, t2, x3, y3, t3;
int             coff, ncolor;
{
double rfrac, lfrac;
int    lc, rc, lx, rx, xx, y, off;
unsigned char *lmap;

int    rc_lc, rx_lx, t2_t1, x2_x1, t3_t1, x3_x1, t3_t2, x3_x2;
double R_y2_y1, R_y3_y1, R_y3_y2;

off = 1 << (SHIFT_VAL - 1);

/* Sort the vertices */
#define SWAP(a,b) {int _a; _a=a; a=b; b=_a;}
if (y1 > y2) {
    SWAP(y1,y2);SWAP(t1,t2); SWAP(x1,x2);
    }
if (y1 > y3) {
    SWAP(y1,y3);SWAP(t1,t3); SWAP(x1,x3);
    }
if (y2 > y3) {
    SWAP(y2,y3);SWAP(t2,t3); SWAP(x2,x3);
    }
/* This code is decidely non-optimal; it is intended to be a start at
   an implementation */

if (y2 != y1) R_y2_y1 = 1.0/((double)(y2-y1)); else R_y2_y1 = 0.0; 
if (y3 != y1) R_y3_y1 = 1.0/((double)(y3-y1)); else R_y3_y1 = 0.0;
t2_t1   = t2 - t1;
x2_x1   = x2 - x1;
t3_t1   = t3 - t1;
x3_x1   = x3 - x1;
for (y=y1; y<=y2; y++) {
    /* Draw a line with the correct color from t1-t2 to t1-t3 */
    /* Left color is (y-y1)/(y2-y1) * (t2-t1) + t1 */
/*     lfrac = ((double)(y-y1)) / ((double)(y2-y1)); */
    lfrac = ((double)(y-y1)) * R_y2_y1; 
    lc    = lfrac * (t2_t1) + t1;
    lx    = lfrac * (x2_x1) + x1;
    /* Right color is (y-y1)/(y3-y1) * (t3-t1) + t1 */
/*    rfrac = ((double)(y - y1)) / ((double)(y3 - y1));*/
    rfrac = ((double)(y - y1)) * R_y3_y1; 
    rc    = rfrac * (t3_t1) + t1;
    rx    = rfrac * (x3_x1) + x1;
    /* Draw the line */
    lmap = map + y * nx;
    rc_lc = rc - lc; 
    rx_lx = rx - lx;
    if (rx > lx) {
	for (xx=lx; xx<=rx; xx++) {
	    lmap[xx] = 
		(((xx-lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
	    }
	}
    else if (rx < lx) {
	for (xx=lx; xx>=rx; xx--) {
	    lmap[xx] = 
		(((xx-lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
	    }
	}
    else
	lmap[lx] = lc >> SHIFT_VAL;
    }
/* For simplicity, "move" t1 to the intersection of t1-t3 with the line y=y2.
   We take advantage of the previous iteration. */
if (y2 >= y3) return;
if (y1 < y2) {
    t1 = rc;
    y1 = y2;
    x1 = rx;

    t3_t1   = t3 - t1;
    x3_x1   = x3 - x1;    
}
t3_t2 = t3 - t2;
x3_x2 = x3 - x2;
if (y3 != y2) R_y3_y2 = 1.0/((double)(y3-y2)); else R_y3_y2 = 0.0;
if (y3 != y1) R_y3_y1 = 1.0/((double)(y3-y1)); else R_y3_y1 = 0.0;
for (y=y2; y<=y3; y++) {
    /* Draw a line with the correct color from t2-t3 to t1-t3 */
    /* Left color is (y-y1)/(y2-y1) * (t2-t1) + t1 */
  /*  lfrac = ((double)(y-y2))/ ((double)(y3-y2));*/
    lfrac = ((double)(y-y2)) * R_y3_y2; 
    lc    = lfrac * (t3_t2) + t2;
    lx    = lfrac * (x3_x2) + x2;
    /* Right color is (y-y1)/(y3-y1) * (t3-t1) + t1 */
   /* rfrac = ((double)(y - y1)) / ((double)(y3 - y1)); */
     rfrac = ((double)(y - y1)) * R_y3_y1 ; 
    rc    = rfrac * (t3_t1) + t1;
    rx    = rfrac * (x3_x1) + x1;
    /* Draw the line */
    lmap = map + y * nx;
    rc_lc = rc - lc; 
    rx_lx = rx - lx;
    if (rx > lx) {
	for (xx=lx; xx<=rx; xx++) {
	    lmap[xx] = 
		(((xx-lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
	    }
	}
    else if (rx < lx) {
	for (xx=lx; xx>=rx; xx--) {
	    lmap[xx] = 
		(((xx-lx) * (rc_lc)) / (rx_lx) + lc) >> SHIFT_VAL;
	    }
	}
    else
	lmap[lx] = lc >> SHIFT_VAL;
    }
}

/*
    XBGetlocUnsorted - convert x locations into indices in the color map
    input parameters:
        nx      - number of columns in map
        mx      - number of columns in matrix
        x       - x values of columns in matrix
    output value
        matrix of size mx that give the cell indices
    Does NOT assume that the entries are monotonically increasing.
 */
int *XBGetlocUnsorted( nx, mx, x )
int     nx, mx;
double  *x;
{
register int i;
register double maxx, minx;
int     *p;
double  dx;

p   = (int *) MALLOC( (unsigned)( mx * sizeof(int) ) );   CHKPTRN(p)
if (!p) return p;

/* Find the limits */
maxx = minx = x[0];
for (i=1; i<mx; i++) {
    if (x[i] > maxx) maxx = x[i];
    else if (x[i] < minx) minx = x[i];
    }
dx  = (nx - 1) / ( maxx - minx );
if (dx <= 0) {
    fprintf( stderr, "All coordinates the same\n" );
    }
/* irint and rint are not available on many systems */
#if !defined(sun4)
for (i=0; i<mx; i++){
    p[i]    = (x[i] - minx) * dx;
#else
for (i=0; i<mx; i++){
    p[i]    = rint((x[i] - minx) * dx);
#endif
}
return p;
}
