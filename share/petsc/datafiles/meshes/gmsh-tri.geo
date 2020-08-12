// (3)
//  |\
//  | \
//  |  \
//  |   \
//  |    \
// (1)---(2)

Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {0, 1, 0};

//  +
//  |\
//  | \
//  3  2
//  |   \
//  |    \
//  +--1--+

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};

Line Loop(1) = {1,2,3};
Plane Surface(1) = {1};

Physical Point   (1) = {1};
Physical Point   (2) = {2};
Physical Point   (3) = {3};
Physical Line    (1) = {1};
Physical Line    (2) = {2};
Physical Line    (3) = {3};
Physical Surface (1) = {1};

Transfinite Line "*" = 2;
