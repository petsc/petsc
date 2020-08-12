Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {0, 1, 0};
Point(4) = {0, 0, 1};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,1};
Line(4) = {1,4};
Line(5) = {2,4};
Line(6) = {3,4};

Line Loop(1) = {1,3,2};
Line Loop(2) = {1,5,-4};
Line Loop(3) = {2,6,-5};
Line Loop(4) = {3,4,-6};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};

Surface Loop(1)  = {1,2,3,4};
Volume(1) = {1};

Physical Point   (1) = {1};
Physical Point   (2) = {2};
Physical Point   (3) = {3};
Physical Point   (4) = {4};
Physical Line    (1) = {1};
Physical Line    (2) = {2};
Physical Line    (3) = {3};
Physical Line    (4) = {3};
Physical Line    (5) = {5};
Physical Line    (6) = {6};
Physical Surface (1) = {1};
Physical Surface (2) = {2};
Physical Surface (3) = {3};
Physical Surface (4) = {4};
Physical Volume  (1) = {1};

Transfinite Line "*" = 2;
