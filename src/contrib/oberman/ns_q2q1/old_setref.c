#include "appctx.h"

/* The following functions set the reference element, and the local element for the quadrature.  Set reference element is called only once, at initialization, while set reference element must be called over each element.  */
int SetBiQuadReferenceElement(AppCtx* appctx){

  /* COULD HAVE just defined the 8 basis functions as functions
(and the two sets of partials), and set the 9
gauss points, then had a loop which evaluated them successively, 
assigning these values to reference element.
Then I would have had 8*3 functions to define, and 9 evaluation points. 
probably would have been a better idea. */


  /* 
The basis (eight elements):
0    .5     *  x    *   (1-x)   *   .5    *  y   * (1-y);
1   1+x  *  1    *   (1-x)   *   .5    *  y   * (1-y);
2   1+x  *  x    *     .5      *   .5    *  y   * (1-y);
3   1+x  *  x    *     .5      *(1+y)*  1   * (1-y);
4   1+x  *  x    *     .5      *(1+y)*  y   *   .5;
5   1+x  *  1    *    (1-x)  *(1+y)*  y   *   .5;
6     .5    *  x    *    (1-x)  *(1+y)*  y   *   .5;
7     .5    *  x    *    (1-x)  *(1+y)*  1   * (1-y);
The quadrature points (nine quadrature points):
  t = sqrt(.5);
0  (-t,  -t), 
1  ( 0, -t),
2  ( t,  -t),
3  ( t,  0),
4  ( t,  t),
5  ( 0, t),
6  (-t, t),
7  (-t, 0),
8  ( 0, 0).
*/

  AppElement *phi = &appctx->element;
  double  t;
  t = sqrt(.5);

  /* The reference element needs to have Val[i][j] = ith basis at jth quadrature point. */
phi->Val[0][0] =    .5     *  -t    *   (1--t)   *   .5    *  -t   * (1--t);
phi->Val[0][1] =     .5     *  0    *   (1-0)   *   .5    *  -t   * (1--t);
phi->Val[0][2] =      .5 *  t    *   (1-t)   *   .5    *  -t   * (1--t);
phi->Val[0][3] =  .5     *  t    *   (1-t)   *   .5    *  0   * (1-0);
phi->Val[0][4] =  .5     *  t    *   (1-t)   *   .5    *  t   * (1-t);
phi->Val[0][5] =  .5     *  0    *   (1-0)   *   .5    *  t   * (1-t);
phi->Val[0][6] =  .5     *  -t    *   (1--t)   *   .5    *  t   * (1-t);
phi->Val[0][7] =  .5     *  -t    *   (1--t)   *   .5    *  0   * (1-0);
phi->Val[0][8] =  .5     *  0   *   (1-0)   *   .5    *  0   * (1-0);

phi->Val[1][0] =  1-t  *  1    *   (1--t)   *   .5    *  -t   * (1--t);
phi->Val[1][1] =  1+0  *  1    *   (1-0)   *   .5    *  -t   * (1--t);
phi->Val[1][2] =  1+t  *  1    *   (1-t)   *   .5    *  -t   * (1--t);
phi->Val[1][3] =  1+t  *  1    *   (1-t)   *   .5    *  0   * (1-0);
phi->Val[1][4] =  1+t  *  1    *   (1-t)   *   .5    *  t   * (1-t);
phi->Val[1][5] =  1+0  *  1    *   (1-0)   *   .5    *  t   * (1-t);
phi->Val[1][6] =  1+-t  *  1    *   (1--t)   *   .5    *  t   * (1-t);
phi->Val[1][7] =  1+-t  *  1    *   (1--t)   *   .5    *  0   * (1-0);
phi->Val[1][8] =  1+0  *  1    *   (1-0)   *   .5    *  0   * (1-0);

phi->Val[2][0] = 1+-t *  -t  *     .5      *   .5    *  -t   * (1--t);
phi->Val[2][1] = 1+0  *  0   *     .5      *   .5    *  -t   * (1--t);
phi->Val[2][2] = 1+t  *  t   *     .5      *   .5    *  -t   * (1--t);
phi->Val[2][3] = 1+t  *  t   *     .5      *   .5    *  0   * (1-0);
phi->Val[2][4] = 1+t  *  t   *     .5      *   .5    *  t   * (1-t);
phi->Val[2][5] = 1+0  *  0   *     .5      *   .5    *  t   * (1-t);
phi->Val[2][6] = 1+-t *  -t  *     .5      *   .5    *  t   * (1-t);
phi->Val[2][7] = 1+-t *  -t  *     .5      *   .5    *  0   * (1-0);
phi->Val[2][8] = 1+0  *  0   *     .5      *   .5    *  0   * (1-0);

phi->Val[3][0] = 1+-t  *  -t    *     .5      *(1+-t)*  1   * (1--t);
phi->Val[3][1] = 1+0  *  0   *     .5      *(1+-t)*  1   * (1--t);
phi->Val[3][2] = 1+t  *  t    *     .5      *(1+-t)*  1   * (1--t);
phi->Val[3][3] = 1+t  *  t    *     .5      *(1+0)*  1   * (1-0);
phi->Val[3][4] = 1+t  *  t    *     .5      *(1+t)*  1   * (1-t);
phi->Val[3][5] = 1+0  *  0    *     .5      *(1+t)*  1   * (1-t);
phi->Val[3][6] = 1+-t  *  -t    *     .5      *(1+t)*  1   * (1-t);
phi->Val[3][7] = 1+-t  *  -t   *     .5      *(1+0)*  1   * (1-0);
phi->Val[3][8] = 1+0  *  0   *     .5      *(1+0)*  1   * (1-0);

phi->Val[4][0] = 1+-t  *  -t    *     .5      *(1+-t)*  -t   *   .5;
phi->Val[4][1] = 1+0  *  0    *     .5      *(1+-t)*  -t   *   .5;
phi->Val[4][2] = 1+t  *  t    *     .5      *(1+-t)*  -t   *   .5;
phi->Val[4][3] = 1+t  *  t    *     .5      *(1+0)*  0   *   .5;
phi->Val[4][4] = 1+t  *  t    *     .5      *(1+t)*  t   *   .5;
phi->Val[4][5] = 1+0  *  0    *     .5      *(1+t)*  t   *   .5;
phi->Val[4][6] = 1+-t  *  -t    *     .5      *(1+t)*  t   *   .5;
phi->Val[4][7] = 1+-t  *  -t   *     .5      *(1+0)*  0   *   .5;
phi->Val[4][8] = 1+0  *  0    *     .5      *(1+0)*  0   *   .5;

phi->Val[5][0] = 1+-t  *  1    *    (1--t)  * (1+-t) *  -t   *   .5;
phi->Val[5][1] = 1+0  *  1    *    (1-0)  * (1+-t) *  -t   *   .5;
phi->Val[5][2] = 1+t  *  1    *    (1-t)  * (1+-t) *  -t   *   .5;
phi->Val[5][3] = 1+t  *  1    *    (1-t)  * (1+0) *  0   *   .5;
phi->Val[5][4] = 1+t  *  1    *    (1-t)  * (1+t) *  t   *   .5;
phi->Val[5][5] = 1+0  *  1    *    (1-0)  * (1+t) *  t   *   .5;
phi->Val[5][6] = 1+-t  *  1    *    (1--t)  * (1+t) *  t   *   .5;
phi->Val[5][7] = 1+-t  *  1    *    (1--t)  * (1+0) *  0   *   .5;
phi->Val[5][8] = 1+0  *  1    *    (1-0)  * (1+0) *  0   *   .5;

phi->Val[6][0] = .5    *  -t    *    (1--t)  *(1+-t)*  -t   *   .5;
phi->Val[6][1] = .5    *  0    *    (1-0)  *(1+-t)*  -t   *   .5;
phi->Val[6][2] = .5    *  t    *    (1-t)  *(1+-t)*  -t   *   .5;
phi->Val[6][3] = .5    *  t    *    (1-t)  *(1+0)*  0   *   .5;
phi->Val[6][4] = .5    *  t    *    (1-t)  *(1+t)*  t   *   .5;
phi->Val[6][5] = .5    *  0    *    (1-0)  *(1+t)*  t   *   .5;
phi->Val[6][6] = .5    *  -t    *    (1--t)  *(1+t)*  t   *   .5;
phi->Val[6][7] = .5    *  -t   *    (1--t)  *(1+0)*  0   *   .5;
phi->Val[6][8] = .5    *  0    *    (1-0)  *(1+0)*  0   *   .5;

phi->Val[7][0] = .5    *  -t    *    (1--t)  *(1+-t)*  1   * (1--t);
phi->Val[7][0] = .5    *  0    *    (1-0)  *(1+-t)*  1   * (1--t);
phi->Val[7][0] = .5    *  t    *    (1-t)  *(1+-t)*  1   * (1--t);
phi->Val[7][0] = .5    *  t    *    (1-t)  *(1+0)*  1   * (1-0);
phi->Val[7][0] = .5    *  t    *    (1-t)  *(1+t)*  1   * (1-t);
phi->Val[7][0] = .5    *  0    *    (1-0)  *(1+t)*  1   * (1-t);
phi->Val[7][0] = .5    *  -t    *    (1--t)  *(1+t)*  1   * (1-t);
phi->Val[7][0] = .5    *  -t   *    (1--t)  *(1+0)*  1   * (1-0);
phi->Val[7][0] = .5    *  0    *    (1-0)  *(1+0)*  1   * (1-0);

/* Next have phi->Dx[i][j] = the x derivative of the ith element at jth quad point */
/*
The basis (eight elements):
0    .5     *  x    *   (1-x)   *   .5    *  y   * (1-y);
1   1+x  *  1    *   (1-x)   *   .5    *  y   * (1-y);
2   1+x  *  x    *     .5      *   .5    *  y   * (1-y);
3   1+x  *  x    *     .5      *(1+y)*  1   * (1-y);
4   1+x  *  x    *     .5      *(1+y)*  y   *   .5;
5   1+x  *  1    *    (1-x)  *(1+y)*  y   *   .5;
6     .5    *  x    *    (1-x)  *(1+y)*  y   *   .5;
7     .5    *  x    *    (1-x)  *(1+y)*  1   * (1-y);

The partials wrt x:

0    .5 * (1 - 2*x)     *   .5    *  y   * (1-y);
1     1  *     - 2*x    *   .5    *  y   * (1-y);
2    .5  * (1 + 2*x)    *   .5    *  y   * (1-y);
3    .5  * (1 + 2*x)    *(1+y)*  1   * (1-y);
4    .5  * (1 + 2*x)    *(1+y)*  y   *   .5;
5              - 2*x    *(1+y)*  y   *   .5;
6     .5    * (1 - 2*x) *(1+y)*  y   *   .5;
7     .5    * (1 - 2*x) *(1+y)*  1   * (1-y);

*/

phi->Dx[0][0] =    .5     *   (1 - 2*-t)  *   .5    *  -t   * (1--t);
phi->Dx[0][1] =     .5     *   (1 - 2*0)   *   .5    *  -t   * (1--t);
phi->Dx[0][2] =      .5 *    (1 - 2*t)  *   .5    *  -t   * (1--t);
phi->Dx[0][3] =  .5     *  (1 - 2*t)   *   .5    *  0   * (1-0);
phi->Dx[0][4] =  .5     *   (1 - 2*t)   *   .5    *  t   * (1-t);
phi->Dx[0][5] =  .5     *    (1 - 2*0)   *   .5    *  t   * (1-t);
phi->Dx[0][6] =  .5     *    (1 - 2*-t)   *   .5    *  t   * (1-t);
phi->Dx[0][7] =  .5     *    (1 - 2*-t)   *   .5    *  0   * (1-0);
phi->Dx[0][8] =  .5     *    (1 - 2*0)   *   .5    *  0   * (1-0);

phi->Dx[1][0] =  - 2*-t    *   .5    *  -t   * (1--t);
phi->Dx[1][1] =   - 2*0    *   .5    *  -t   * (1--t);
phi->Dx[1][2] =    - 2*t    *   .5    *  -t   * (1--t);
phi->Dx[1][3] =   - 2*t  *   .5    *  0   * (1-0);
phi->Dx[1][4] =    - 2*t   *   .5    *  t   * (1-t);
phi->Dx[1][5] =    - 2*0   *   .5    *  t   * (1-t);
phi->Dx[1][6] =    - 2*-t   *   .5    *  t   * (1-t);
phi->Dx[1][7] =    - 2*-t   *   .5    *  0   * (1-0);
phi->Dx[1][8] =     - 2*0   *   .5    *  0   * (1-0);

phi->Dx[2][0] =  (1 + 2*-t) *     .5      *   .5    *  -t   * (1--t);
phi->Dx[2][1] =   (1 + 2*0) *     .5      *   .5    *  -t   * (1--t);
phi->Dx[2][2] =  (1 + 2*t) *     .5      *   .5    *  -t   * (1--t);
phi->Dx[2][3] =   (1 + 2*t) *     .5      *   .5    *  0   * (1-0);
phi->Dx[2][4] =   (1 + 2*t) *     .5      *   .5    *  t   * (1-t);
phi->Dx[2][5] =   (1 + 2*0) *     .5      *   .5    *  t   * (1-t);
phi->Dx[2][6] =   (1 + 2*-t) *     .5      *   .5    *  t   * (1-t);
phi->Dx[2][7] =  (1 + 2*-t) *     .5      *   .5    *  0   * (1-0);
phi->Dx[2][8] =   (1 + 2*0)  *     .5      *   .5    *  0   * (1-0);

phi->Dx[3][0] =  (1 + 2*-t) *     .5      *(1+-t)*  1   * (1--t);
phi->Dx[3][1] =   (1 + 2*0) *     .5      *(1+-t)*  1   * (1--t);
phi->Dx[3][2] =   (1 + 2*t) *     .5      *(1+-t)*  1   * (1--t);
phi->Dx[3][3] =  (1 + 2*t)  *     .5      *(1+0)*  1   * (1-0);
phi->Dx[3][4] =  (1 + 2*t)   *     .5      *(1+t)*  1   * (1-t);
phi->Dx[3][5] =  (1 + 2*0)  *     .5      *(1+t)*  1   * (1-t);
phi->Dx[3][6] =   (1 + 2*-t)  *     .5      *(1+t)*  1   * (1-t);
phi->Dx[3][7] =   (1 + 2*-t)  *     .5      *(1+0)*  1   * (1-0);
phi->Dx[3][8] =   (1 + 2*0) *     .5      *(1+0)*  1   * (1-0);

phi->Dx[4][0] =   (1 + 2*-t) *     .5      *(1+-t)*  -t   *   .5;
phi->Dx[4][1] =    (1 + 2*0) *     .5      *(1+-t)*  -t   *   .5;
phi->Dx[4][2] =    (1 + 2*t)  *     .5      *(1+-t)*  -t   *   .5;
phi->Dx[4][3] =    (1 + 2*t)   *     .5      *(1+0)*  0   *   .5;
phi->Dx[4][4] =   (1 + 2*t)   *     .5      *(1+t)*  t   *   .5;
phi->Dx[4][5] =    (1 + 2*0) *     .5      *(1+t)*  t   *   .5;
phi->Dx[4][6] =    (1 + 2*-t)  *     .5      *(1+t)*  t   *   .5;
phi->Dx[4][7] =   (1 + 2*-t)  *     .5      *(1+0)*  0   *   .5;
phi->Dx[4][8] =    (1 + 2*0)  *     .5      *(1+0)*  0   *   .5;

phi->Dx[5][0] = - 2*-t   * (1+-t) *  -t   *   .5;
phi->Dx[5][1] = - 2*0    * (1+-t) *  -t   *   .5;
phi->Dx[5][2] = - 2*t  * (1+-t) *  -t   *   .5;
phi->Dx[5][3] = - 2*t  * (1+0) *  0   *   .5;
phi->Dx[5][4] = - 2*t  * (1+t) *  t   *   .5;
phi->Dx[5][5] = - 2*0  * (1+t) *  t   *   .5;
phi->Dx[5][6] =  - 2*-t  * (1+t) *  t   *   .5;
phi->Dx[5][7] =  - 2*-t * (1+0) *  0   *   .5;
phi->Dx[5][8] =  - 2*0 * (1+0) *  0   *   .5;

phi->Dx[6][0] =   .5    * (1 - 2*-t) *(1+-t)*  -t   *   .5;
phi->Dx[6][1] =   .5    * (1 - 2*0) *(1+-t)*  -t   *   .5;
phi->Dx[6][2] =   .5    * (1 - 2*t) *(1+-t)*  -t   *   .5;
phi->Dx[6][3] =   .5    * (1 - 2*t)  *(1+0)*  0   *   .5;
phi->Dx[6][4] =   .5    * (1 - 2*t)  *(1+t)*  t   *   .5;
phi->Dx[6][5] =   .5    * (1 - 2*0)  *(1+t)*  t   *   .5;
phi->Dx[6][6] =   .5    * (1 - 2*-t) *(1+t)*  t   *   .5;
phi->Dx[6][7] =   .5    * (1 - 2*-t)  *(1+0)*  0   *   .5;
phi->Dx[6][8] =   .5    * (1 - 2*0) *(1+0)*  0   *   .5;

phi->Dx[7][0] =  .5    * (1 - 2*-t)  *(1+-t)*  1   * (1--t);
phi->Dx[7][0] =  .5    * (1 - 2*0) *(1+-t)*  1   * (1--t);
phi->Dx[7][0] =  .5    * (1 - 2*t)  *(1+-t)*  1   * (1--t);
phi->Dx[7][0] =  .5    * (1 - 2*t) *(1+0)*  1   * (1-0);
phi->Dx[7][0] =  .5    * (1 - 2*t)   *(1+t)*  1   * (1-t);
phi->Dx[7][0] =   .5    * (1 - 2*0)  *(1+t)*  1   * (1-t);
phi->Dx[7][0] =   .5    * (1 - 2*-t)  *(1+t)*  1   * (1-t);
phi->Dx[7][0] =   .5    * (1 - 2*-t)  *(1+0)*  1   * (1-0);
phi->Dx[7][0] =   .5    * (1 - 2*0) *(1+0)*  1   * (1-0);

/* Next have phi->Dy[i][j] = the y derivative of the ith element at jth quad point */
/*
The basis (eight elements):
0    .5     *  x    *   (1-x)   *   .5    *  y   * (1-y);
1   1+x  *  1    *   (1-x)   *   .5    *  y   * (1-y);
2   1+x  *  x    *     .5      *   .5    *  y   * (1-y);
3   1+x  *  x    *     .5      *(1+y)*  1   * (1-y);
4   1+x  *  x    *     .5      *(1+y)*  y   *   .5;
5   1+x  *  1    *    (1-x)  *(1+y)*  y   *   .5;
6     .5    *  x    *    (1-x)  *(1+y)*  y   *   .5;
7     .5    *  x    *    (1-x)  *(1+y)*  1   * (1-y);

The partials wrt y:
0    .5     *  x    *   (1-x)   *   .5    * ( 1 - 2*y );
1   1+x  *  1    *   (1-x)   *   .5    * ( 1 - 2*y );
2   1+x  *  x    *     .5      *   .5    * ( 1 - 2*y );
3   1+x  *  x    *     .5      * -2*y;
4   1+x  *  x    *     .5      *  .5 * ( 1 + 2*y );
5   1+x  *  1    *    (1-x)  * .5 * ( 1 + 2*y );
6     .5    *  x    *    (1-x)  * .5 * ( 1 + 2*y );
7     .5    *  x    *    (1-x)  * -2*y;

*/

phi->Dy[0][0] =    .5     *  -t    *   (1--t)   * .5    * ( 1 - 2*-t );
phi->Dy[0][1] =     .5     *  0    *   (1-0)   * .5    * ( 1 - 2*-t );
phi->Dy[0][2] =      .5 *  t    *   (1-t)   *   .5    * ( 1 - 2*-t );
phi->Dy[0][3] =  .5     *  t    *   (1-t)   *   .5    * ( 1 - 2*0 );
phi->Dy[0][4] =  .5     *  t    *   (1-t)   *    .5    * ( 1 - 2*t );
phi->Dy[0][5] =  .5     *  0    *   (1-0)   *   .5    * ( 1 - 2*t );
phi->Dy[0][6] =  .5     *  -t    *   (1--t)   *   .5    * ( 1 - 2*t );
phi->Dy[0][7] =  .5     *  -t    *   (1--t)   *   .5    * ( 1 - 2*0 );
phi->Dy[0][8] =  .5     *  0   *   (1-0)   *    .5    * ( 1 - 2*0 );

phi->Dy[1][0] =  1-t  *  1    *   (1--t)   *   .5    * ( 1 - 2*-t );
phi->Dy[1][1] =  1+0  *  1    *   (1-0)   *  .5    * ( 1 - 2*-t );
phi->Dy[1][2] =  1+t  *  1    *   (1-t)   *   .5    * ( 1 - 2*-t );
phi->Dy[1][3] =  1+t  *  1    *   (1-t)   *   .5    * ( 1 - 2*0 );
phi->Dy[1][4] =  1+t  *  1    *   (1-t)   *  .5    * ( 1 - 2*t );
phi->Dy[1][5] =  1+0  *  1    *   (1-0)   *    .5    * ( 1 - 2*t );
phi->Dy[1][6] =  1+-t  *  1    *   (1--t)   *    .5    * ( 1 - 2*t );
phi->Dy[1][7] =  1+-t  *  1    *   (1--t)   *   .5    * ( 1 - 2*0 );
phi->Dy[1][8] =  1+0  *  1    *   (1-0)   *    .5    * ( 1 - 2*0 );

phi->Dy[2][0] = 1+-t *  -t  *     .5      *   .5    * ( 1 - 2*-t );
phi->Dy[2][1] = 1+0  *  0   *     .5      *  .5    * ( 1 - 2*-t );
phi->Dy[2][2] = 1+t  *  t   *     .5      *    .5    * ( 1 - 2*-t );
phi->Dy[2][3] = 1+t  *  t   *     .5      *  .5    * ( 1 - 2*0 );
phi->Dy[2][4] = 1+t  *  t   *     .5      *    .5    * ( 1 - 2*t );
phi->Dy[2][5] = 1+0  *  0   *     .5      *   .5    * ( 1 - 2*t );
phi->Dy[2][6] = 1+-t *  -t  *     .5      *    .5    * ( 1 - 2*t );
phi->Dy[2][7] = 1+-t *  -t  *     .5      *    .5    * ( 1 - 2*0 );
phi->Dy[2][8] = 1+0  *  0   *     .5      *  .5    * ( 1 - 2*0 );

phi->Dy[3][0] = 1+-t  *  -t    *     .5     * -2*-t;
phi->Dy[3][1] = 1+0  *  0   *     .5       * -2*-t;
phi->Dy[3][2] = 1+t  *  t    *     .5      * -2*-t;
phi->Dy[3][3] = 1+t  *  t    *     .5        * -2*0;
phi->Dy[3][4] = 1+t  *  t    *     .5       * -2*t;
phi->Dy[3][5] = 1+0  *  0    *     .5       * -2*t;
phi->Dy[3][6] = 1+-t  *  -t    *     .5     * -2*t;
phi->Dy[3][7] = 1+-t  *  -t   *     .5       * -2*0;
phi->Dy[3][8] = 1+0  *  0   *     .5       * -2*0;

phi->Dy[4][0] = 1+-t  *  -t    *     .5      *.5 * ( 1 + 2*-t );
phi->Dy[4][1] = 1+0  *  0    *     .5      *.5 * ( 1 + 2*-t );
phi->Dy[4][2] = 1+t  *  t    *     .5      *.5 * ( 1 + 2*-t );
phi->Dy[4][3] = 1+t  *  t    *     .5      *.5 * ( 1 + 2*0 );
phi->Dy[4][4] = 1+t  *  t    *     .5      *.5 * ( 1 + 2*t );
phi->Dy[4][5] = 1+0  *  0    *     .5      *.5 * ( 1 + 2*t );
phi->Dy[4][6] = 1+-t  *  -t    *     .5      *.5 * ( 1 + 2*t );
phi->Dy[4][7] = 1+-t  *  -t   *     .5      *.5 * ( 1 + 2*0 );
phi->Dy[4][8] = 1+0  *  0    *     .5      *.5 * ( 1 + 2*0 );

phi->Dy[5][0] = 1+-t  *  1    *    (1--t)  *.5 * ( 1 + 2*-t );
phi->Dy[5][1] = 1+0  *  1    *    (1-0)  *.5 * ( 1 + 2*-t );
phi->Dy[5][2] = 1+t  *  1    *    (1-t)  * .5 * ( 1 + 2*-t );
phi->Dy[5][3] = 1+t  *  1    *    (1-t)  * .5 * ( 1 + 2*0 );
phi->Dy[5][4] = 1+t  *  1    *    (1-t)  * .5 * ( 1 + 2*t );
phi->Dy[5][5] = 1+0  *  1    *    (1-0)  *.5 * ( 1 + 2*t );
phi->Dy[5][6] = 1+-t  *  1    *    (1--t)  * .5 * ( 1 + 2*t );
phi->Dy[5][7] = 1+-t  *  1    *    (1--t)  * .5 * ( 1 + 2*0 );
phi->Dy[5][8] = 1+0  *  1    *    (1-0)  * .5 * ( 1 + 2*0 );

phi->Dy[6][0] = .5    *  -t    *    (1--t)  *.5 * ( 1 + 2*-t );
phi->Dy[6][1] = .5    *  0    *    (1-0)  *.5 * ( 1 + 2*-t );
phi->Dy[6][2] = .5    *  t    *    (1-t)  *.5 * ( 1 + 2*-t );
phi->Dy[6][3] = .5    *  t    *    (1-t)  *.5 * ( 1 + 2*0 );
phi->Dy[6][4] = .5    *  t    *    (1-t)  *.5 * ( 1 + 2*t );
phi->Dy[6][5] = .5    *  0    *    (1-0)  *.5 * ( 1 + 2*t );
phi->Dy[6][6] = .5    *  -t    *    (1--t)  *.5 * ( 1 + 2*t );
phi->Dy[6][7] = .5    *  -t   *    (1--t)  *.5 * ( 1 + 2*0 );
phi->Dy[6][8] = .5    *  0    *    (1-0)  *.5 * ( 1 + 2*0 );

phi->Dy[7][0] = .5    *  -t    *    (1--t)  * -2*-t;
phi->Dy[7][0] = .5    *  0    *    (1-0)  * -2*-t;
phi->Dy[7][0] = .5    *  t    *    (1-t)  * -2*-t;
phi->Dy[7][0] = .5    *  t    *    (1-t)  * -2*0;
phi->Dy[7][0] = .5    *  t    *    (1-t)  * -2*t;
phi->Dy[7][0] = .5    *  0    *    (1-0)  * -2*t;
phi->Dy[7][0] = .5    *  -t    *    (1--t)  * -2*t;
phi->Dy[7][0] = .5    *  -t   *    (1--t)  * -2*0;
phi->Dy[7][0] = .5    *  0    *    (1-0)  * -2*0;

PetscFunctionReturn(0);
}

/* The following functions set the reference element, and the local element for the quadrature.  Set reference element is called only once, at initialization, while set reference element must be called over each element.  */
int SetBiLinReferenceElement(AppCtx* appctx){

  AppElement *phi = &appctx->element;
  double psi, psi_m, psi_p, psi_pp, psi_mp, psi_pm, psi_mm;

phi->dorhs = 0;

  psi = sqrt(3.0)/3.0;
  psi_p = 0.25*(1.0 + psi);   psi_m = 0.25*(1.0 - psi);
  psi_pp = 0.25*(1.0 + psi)*(1.0 + psi);  psi_pm = 0.25*(1.0 + psi)*(1.0 - psi); 
  psi_mp = 0.25*(1.0 - psi)*(1.0 + psi);  psi_mm = 0.25*(1.0 - psi)*(1.0 - psi);

phi->Values[0][0] = psi_pp; phi->Values[0][1] = psi_pm;phi->Values[0][2] = psi_mm;
phi->Values[0][3] = psi_mp;phi->Values[1][0] = psi_mp; phi->Values[1][1] = psi_pp;
phi->Values[1][2] = psi_pm;phi->Values[1][3] = psi_mm;phi->Values[2][0] = psi_mm; 
phi->Values[2][1] = psi_pm;phi->Values[2][2] = psi_pp;phi->Values[2][3] = psi_mp;
phi->Values[3][0] = psi_pm; phi->Values[3][1] = psi_mm;phi->Values[3][2] = psi_mp;
phi->Values[3][3] = psi_pp;

phi->DxValues[0][0] = -psi_p; phi->DxValues[0][1] = -psi_p;phi->DxValues[0][2] = -psi_m;
phi->DxValues[0][3] = -psi_m;phi->DxValues[1][0] = psi_p; phi->DxValues[1][1] = psi_p;
phi->DxValues[1][2] = psi_m;phi->DxValues[1][3] = psi_m;phi->DxValues[2][0] = psi_m; 
phi->DxValues[2][1] = psi_m;phi->DxValues[2][2] = psi_p;phi->DxValues[2][3] = psi_p;
phi->DxValues[3][0] = -psi_m; phi->DxValues[3][1] = -psi_m;phi->DxValues[3][2] = -psi_p;
phi->DxValues[3][3] = -psi_p;

phi->DyValues[0][0] = -psi_p; phi->DyValues[0][1] = -psi_m;phi->DyValues[0][2] = -psi_m;
phi->DyValues[0][3] = -psi_p;phi->DyValues[1][0] = -psi_m; phi->DyValues[1][1] = -psi_p;
phi->DyValues[1][2] = -psi_p;phi->DyValues[1][3] = -psi_m;phi->DyValues[2][0] = psi_m; 
phi->DyValues[2][1] = psi_p;phi->DyValues[2][2] = psi_p;phi->DyValues[2][3] = psi_m;
phi->DyValues[3][0] = psi_p; phi->DyValues[3][1] = psi_m;phi->DyValues[3][2] = psi_m;
phi->DyValues[3][3] = psi_p;

PetscFunctionReturn(0);
}


int SetLocalBiQuadElement(AppElement *phi, double *coords)
{
  /* the coords array consists of pairs (x[0],y[0],...,x[7],y[7]) representing the images of the
support points for the 8 basis functions */ 

  int i,j;
  double Dh[9][2][2], Dhinv[9][2][2];

  /* will set these to phi */
  double dx[8][9];  
  double dy[8][9];
  double detDh[9];
  double x[9], y[9];


 /* The function h takes the reference element to the local element.
                  h(x,y) = sum(i) of alpha_i*phi_i(x,y),
   where alpha_i is the image of the support point of the ith basis fn */

  /*Values */
  for(i=0;i<9;i++){ /* loop over the gauss points */
    x[i] = 0; y[i] = 0; 
    for(j=0;j<8,j++){/*loop over the basis functions, and support points */
      x[i] += coords[2*j]*phi->Val[j][i];
      y[i] += coords[2*j]*phi->Val[j][i];
    }
  }

  /* Jacobian */
  for(i=0;i<9;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(j=0; j<8; j++ ){/* loop over functions */
      Dh[i][0][0] += coords[2*j]*phi->Dx[j][i];
      Dh[i][0][1] += coords[2*j]*phi->Dy[j][i];
      Dh[i][1][0] += coords[2*j+1]*phi->Dx[j][i];
      Dh[i][1][1] += coords[2*j+1]*phi->Dy[j][i];    
    }
  }

  /* Determinant of the Jacobian */
  for( i=0; i<9; i++){   /* loop over Gauss points */
    detDh[i] = Dh[i][0][0]*Dh[i][1][1] - Dh[i][0][1]*Dh[i][1][0];
  }

  /* Inverse of the Jacobian */
    for( i=0; i<9; i++){   /* loop over Gauss points */
      Dhinv[i][0][0] = Dh[i][1][1]/detDh[i];
      Dhinv[i][0][1] = -Dh[i][0][1]/detDh[i];
      Dhinv[i][1][0] = -Dh[i][1][0]/detDh[i];
      Dhinv[i][1][1] = Dh[i][0][0]/detDh[i];
    }
    

    /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, (chain rule)
       so Dphi~ = Dphi*(Dh)inv    (multiply by (Dh)inv   */       
    /* partial of phi at h(gauss pt) times Dhinv */
    /* loop over gauss, the basis fns, then d/dx or d/dy */
    for( i=0;i<9;i++ ){  /* loop over Gauss points */
      for( j=0;j<8;j++ ){ /* loop over basis functions */
	dx[j][i] = phi->Dx[j][i]*Dhinv[i][0][0] + phi->Dy[j][i]*Dhinv[i][1][0];
	dy[j][i] = phi->Dx[j][i]*Dhinv[i][0][1] + phi->Dy[j][i]*Dhinv[i][1][1];
      }
    }

 /* set these to phi */
 phi->dx = dx;
 phi->dy = dy;
 phi->detDh  = detDh;
 phi->x = x; phi->y = y;

 PetscFunctionReturn(0);
}

int SetLocalBiLinElement(AppElement *phi, double *coords)
{
  int i,j,k,ii ;

  double Dh[4][2][2], Dhinv[4][2][2]; 
  
  /* will set these to phi */
  double bdx[4][4];  
  double bdy[4][4];
  double bdetDh[4];
  double bx[4], by[4];

  /* the image of the reference element is given by sum (coord i)*phi_i */

    for(j=0;j<4;j++){ /* loop over Gauss points */
      bx[j] = 0; by[j] = 0;
      for( k=0;k<4;k++ ){/* loop over functions */
	bx[j] += coords[2*k]*phi->Values[k][j];
	by[j] += coords[2*k+1]*phi->Values[k][j];
      }
    }

  /* Jacobian */
  for(i=0;i<4;i++){ /* loop over Gauss points */
    Dh[i][0][0] = 0; Dh[i][0][1] = 0; Dh[i][1][0] = 0; Dh[i][1][1] = 0;
    for(k=0; k<4; k++ ){/* loop over functions */
      Dh[i][0][0] += coords[2*k]*phi->DxValues[k][i];
      Dh[i][0][1] += coords[2*k]*phi->DyValues[k][i];
      Dh[i][1][0] += coords[2*k+1]*phi->DxValues[k][i];
      Dh[i][1][1] += coords[2*k+1]*phi->DyValues[k][i];    
    }
  }

  /* Determinant of the Jacobian */
  for( j=0; j<4; j++){   /* loop over Gauss points */
    bdetDh[j] = Dh[j][0][0]*Dh[j][1][1] - Dh[j][0][1]*Dh[j][1][0];
  }
  /* Inverse of the Jacobian */
    for( j=0; j<4; j++){   /* loop over Gauss points */
      Dhinv[j][0][0] = Dh[j][1][1]/bdetDh[j];
      Dhinv[j][0][1] = -Dh[j][0][1]/bdetDh[j];
      Dhinv[j][1][0] = -Dh[j][1][0]/bdetDh[j];
      Dhinv[j][1][1] = Dh[j][0][0]/bdetDh[j];
    }
    
    /* Notice that phi~ = phi(h), so Dphi~ = Dphi*Dh, (chain rule)
       so Dphi~ = Dphi*(Dh)inv    (multiply by (Dh)inv   */       
    /* partial of phi at h(gauss pt) times Dhinv */
    /* loop over gauss, the basis fns, then d/dx or d/dy */
    for( i=0;i<4;i++ ){  /* loop over Gauss points */
      for( j=0;j<4;j++ ){ /* loop over basis functions */
	bdx[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][0] +  phi->DyValues[j][i]*Dhinv[i][1][0];
	bdy[4*j+i] = phi->DxValues[j][i]*Dhinv[i][0][1] + phi->DyValues[j][i]*Dhinv[i][1][1];
      }
    }

 /* set these to phi */
 phi->bdx = bdx;
 phi->bdy = bdy;
 phi->bdetDh  = bdetDh;
 phi->bx = bx; phi->by = by;

PetscFunctionReturn(0);
}
