General.NoPopup = 1;

Ro = GetValue("Outer radius", 1);
Li = GetValue("Inner length", Ro/2);

a = Ro*Sqrt(2)/2;
b = Li*Sqrt(2)/2;
Point(1) = { 0,  0, 0};
Point(2) = {+a, +a, 0};
Point(3) = {-a, +a, 0};
Point(4) = {-a, -a, 0};
Point(5) = {+a, -a, 0};
Point(6) = {+b, +b, 0};
Point(7) = {-b, +b, 0};
Point(8) = {-b, -b, 0};
Point(9) = {+b, -b, 0};

Cx = GetValue("Center X", 0);
Cy = GetValue("Center Y", 0);
Translate {Cx, Cy, 0} { Point{1:9}; }

phi = Pi/180 * GetValue("Rotation (degrees)", 0);
Rotate {{0, 0, 1}, {Cx, Cy, 0}, phi} { Point{1:9}; }

Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 4};
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 2};
Line(5)   = {6, 7};
Line(6)   = {7, 8};
Line(7)   = {8, 9};
Line(8)   = {9, 6};
Line(9)   = {6, 2};
Line(10)  = {7, 3};
Line(11)  = {8, 4};
Line(12)  = {9, 5};

Line Loop(1) = {1, -10, -5,  9};
Line Loop(2) = {2, -11, -6, 10};
Line Loop(3) = {3, -12, -7, 11};
Line Loop(4) = {4,  -9, -8, 12};
Line Loop(5) = {5,   6,  7,  8};

Plane Surface(1) = {1};
Plane Surface(2) = {2};
Plane Surface(3) = {3};
Plane Surface(4) = {4};
Plane Surface(5) = {5};

Physical Line(1)    = {1:4};
Physical Surface(1) = {1:5};

N = GetValue("Segments", 1);
Transfinite Line "*" = N+1;
