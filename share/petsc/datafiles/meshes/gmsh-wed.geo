Point(1) = {0, 0, 0};
Point(2) = {1, 0, 0};
Point(3) = {0, 1, 0};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 1};

Line Loop(1) = {1,2,3};
Plane Surface(1) = {1};

out[] = Extrude{0,0,1}{ Surface{1}; Layers{1}; Recombine 1; };

Surface Loop(1) = {out[]};

Physical Point   (1) = {1};
Physical Point   (2) = {2};
Physical Point   (3) = {3};
Physical Line    (1) = {1};
Physical Line    (2) = {2};
Physical Line    (3) = {3};
Physical Surface (1) = {1};
Physical Surface (2) = {out[0]};
Physical Surface (3) = {out[2]};
Physical Surface (4) = {out[3]};
Physical Surface (5) = {out[4]};
Physical Volume  (1) = {out[1]};

Transfinite Line "*" = 2;
