
figure (1)

Y = [ 88 64 40 30 31 31 21 16 12 15 15 12 6 4];
bar(Y);
axis([0 15 0 110])
ylabel('MFlops');
title('Iterative Solve: GMRES(30) with ILU(0)');

text(1,22,'IBM SP2 Superchip, 4 memory cards','Rotation',90)
text(2,22,'IBM SP2','Rotation',90)
text(3,22,'SGI PowerChallenge','Rotation',90)
text(4,22,'Cray T3E','Rotation',90)
text(5,22,'200 MH Pentium-Pro NT-MDS','Rotation',90)
text(6,22,'IBM SP1','Rotation',90)
text(7,22,'SGI Indigo 2','Rotation',90)
text(8,22,'200 MH Pentium NT-Gnu','Rotation',90)
text(9,22,'166 MH Pentium Freebsd','Rotation',90)
text(10,22,'Cray T3D','Rotation',90)
text(11,22,'DEC Alpha (old)','Rotation',90)
text(12,22,'Convex HP Exemplar','Rotation',90)
text(13,22,'Sun Sparc5','Rotation',90)
text(14,22,'Paragon','Rotation',90)

figure(2)

Y = [ 109 70 50 36 35 27 23 18 17 17 14 14 7 6];
bar(Y);
axis([0 15 0 110])
ylabel('MFlops');
title('Matrix-vector Product');

text(1,22,'IBM SP2 Superchip, 4 memory cards','Rotation',90)
text(2,22,'IBM SP2','Rotation',90)
text(3,22,'SGI PowerChallenge','Rotation',90)
text(4,22,'Cray T3E','Rotation',90)
text(5,22,'200 MH Pentium-Pro NT-MDS','Rotation',90)
text(6,22,'IBM SP1','Rotation',90)
text(7,22,'SGI Indigo 2','Rotation',90)
text(8,22,'200 MH Pentium-Pro NT-Gnu','Rotation',90)
text(9,22,'166 MH Pentium Freebsd','Rotation',90)
text(10,22,'Cray T3D','Rotation',90)
text(11,22,'DEC Alpha (old)','Rotation',90)
text(12,22,'Convex HP Exemplar','Rotation',90)
text(13,22,'Sun Sparc5','Rotation',90)
text(14,22,'Paragon','Rotation',90)
