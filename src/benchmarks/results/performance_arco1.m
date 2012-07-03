
figure (1)

Y = [ 117 101 88 65 64 50 31 31  21 16 15 15 12 12 6 4 ];
bar(Y);
axis([0 17 0 150])
ylabel('MFlops');
title('Iterative Solve: GMRES(30) with ILU(0). Matrix: arco1, N=1501, NZ=26131');

text(1,22,'SGI PowerChallenge','Rotation',90)
text(2,22,'SGI Origin 2000','Rotation',90)
text(3,22,'IBM SP Superchip, 4 memory cards','Rotation',90)
text(4,22,'IBM SP Superchip, 2 memory cards','Rotation',90)
text(5,22,'IBM SP2','Rotation',90)
text(6,22,'Cray T3E','Rotation',90)
text(7,22,'200 MH Pentium-Pro NT-MDS','Rotation',90)
text(8,22,'IBM SP1','Rotation',90)
text(9,22,'SGI Indigo 2','Rotation',90)
text(10,22,'200 MH Pentium NT-Gnu','Rotation',90)
text(11,22,'Cray T3D','Rotation',90)
text(12,22,'DEC Alpha (old)','Rotation',90)
text(13,22,'166 MH Pentium Freebsd','Rotation',90)
text(14,22,'Convex HP Exemplar','Rotation',90)
text(15,22,'Sun Sparc5','Rotation',90)
text(16,22,'Paragon','Rotation',90)


figure(2)

Y = [ 140 115 109 73 70 69 35 27 23 18 17  14 14 17 7 6 ];
bar(Y);
axis([0 17 0 150])
ylabel('MFlops');
title('Matrix-vector Product. Matrix: arco1, N=1501, NZ=26131');

text(1,22,'SGI PowerChallenge','Rotation',90)
text(2,22,'SGI Origin 2000','Rotation',90)
text(3,22,'IBM SP Superchip, 4 memory cards','Rotation',90)
text(4,22,'IBM SP Superchip, 2 memory cards','Rotation',90)
text(5,22,'IBM SP2','Rotation',90)
text(6,22,'Cray T3E','Rotation',90)
text(7,22,'200 MH Pentium-Pro NT-MDS','Rotation',90)
text(8,22,'IBM SP1','Rotation',90)
text(9,22,'SGI Indigo 2','Rotation',90)
text(10,22,'200 MH Pentium NT-Gnu','Rotation',90)
text(11,22,'Cray T3D','Rotation',90)
text(12,22,'DEC Alpha (old)','Rotation',90)
text(13,22,'166 MH Pentium Freebsd','Rotation',90)
text(14,22,'Convex HP Exemplar','Rotation',90)
text(15,22,'Sun Sparc5','Rotation',90)
text(16,22,'Paragon','Rotation',90)

