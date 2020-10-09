General.NoPopup = 1;

Ro = GetValue("Outer radius", 1);
Li = GetValue("Inner length", Ro/2);

a = Ro*Sqrt(3)/3;
b = Li*Sqrt(3)/3;
Point( 1) = { 0,  0,  0};
Point(11) = {-a, -a, -a};
Point(12) = {+a, -a, -a};
Point(13) = {+a, +a, -a};
Point(14) = {-a, +a, -a};
Point(21) = {-b, -b, -b};
Point(22) = {+b, -b, -b};
Point(23) = {+b, +b, -b};
Point(24) = {-b, +b, -b};

Cx = GetValue("Center X", 0);
Cy = GetValue("Center Y", 0);
Cz = GetValue("Center Z", 0);
Translate {Cx, Cy, Cz} { Point{1};     }
Translate {Cx, Cy, Cz} { Point{11:14}; }
Translate {Cx, Cy, Cz} { Point{21:24}; }

Circle(11) = {11, 1, 12};
Circle(12) = {12, 1, 13};
Circle(13) = {13, 1, 14};
Circle(14) = {14, 1, 11};

Line(21) = {21, 22};
Line(22) = {22, 23};
Line(23) = {23, 24};
Line(24) = {24, 21};
Line(31) = {11, 21};
Line(32) = {12, 22};
Line(33) = {13, 23};
Line(34) = {14, 24};

Line Loop(1) = {11, 12,  13,  14};
Line Loop(2) = {14, 31, -24, -34};
Line Loop(3) = {11, 32, -21, -31};
Line Loop(4) = {12, 33, -22, -32};
Line Loop(5) = {13, 34, -23, -33};
Line Loop(6) = {21, 22,  23,  24};

Surface(1) = {1};
Surface(2) = {2};
Surface(3) = {3};
Surface(4) = {4};
Surface(5) = {5};
Surface(6) = {6};

Surface Loop(1) = {1, 3, 4, 5, 2, 6};

Volume(1) = {1};
Rotate { {0,1,0}, {Cx,Cy,Cz}, +Pi/2 } { Duplicata { Volume {1}; } }
Rotate { {1,0,0}, {Cx,Cy,Cz}, -Pi/2 } { Duplicata { Volume {1}; } }
Rotate { {0,1,0}, {Cx,Cy,Cz}, -Pi/2 } { Duplicata { Volume {1}; } }
Rotate { {1,0,0}, {Cx,Cy,Cz}, +Pi/2 } { Duplicata { Volume {1}; } }
Rotate { {1,0,0}, {Cx,Cy,Cz}, +Pi   } { Duplicata { Volume {1}; } }
Extrude {0, 0, 2*b} { Surface{6}; }

vol[] = Volume "*";
srf[] = CombinedBoundary { Volume{vol[]}; };
Physical Volume  (1) = {vol[]};
Physical Surface (1) = {srf[]};

N = GetValue("Segments", 1);
Transfinite Line "*" = N+1;
