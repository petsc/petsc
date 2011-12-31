
figure (1)

Y = [ 37 21 30 17 15   11 7 16 18 4 1];
bar(Y);
axis([0 12 0 90])
ylabel('MFlops');
title('Iterative Solve: GMRES(30) with ILU(0)');

text(1,22,'IBM SP2','Rotation',90)
text(2,22,'SGI PowerChallenge','Rotation',90)
text(3,22,'Cray T3E','Rotation',90)
text(4,22,'IBM SP1','Rotation',90)
text(5,22,'SGI Indigo 2','Rotation',90)
text(6,22,'166 MH Pentium','Rotation',90)
text(7,22,'Cray T3D','Rotation',90)
text(8,22,'DEC Alpha (old)','Rotation',90)
text(9,22,'Convex HP Exemplar','Rotation',90)
text(10,22,'Sun Sparc5','Rotation',90)
text(11,22,'Paragon','Rotation',90)

figure(2)

Y = [ 82 44 58  30 24   21 17 27 57 7 3 ];
bar(Y);
axis([0 12 0 90])
ylabel('MFlops');
title('Matrix-vector Product');

text(1,22,'IBM SP2','Rotation',90)
text(2,22,'SGI PowerChallenge','Rotation',90)
text(3,22,'Cray T3E','Rotation',90)
text(4,22,'IBM SP1','Rotation',90)
text(5,22,'SGI Indigo 2','Rotation',90)
text(6,22,'166 MH Pentium','Rotation',90)
text(7,22,'Cray T3D','Rotation',90)
text(8,22,'DEC Alpha (old)','Rotation',90)
text(9,22,'Convex HP Exemplar','Rotation',90)
text(10,22,'Sun Sparc5','Rotation',90)
text(11,22,'Paragon','Rotation',90)