
figure (1)

Y = [65 62 56 50 47 33 30 30 25 ];
bar(Y);
axis([0 10 0 150])
ylabel('MFlops');
title('Iterative Solve: GMRES(30) with ILU(0). Matrix: cfd.2.10, N=122880, NZ=4134400');

text(1,22,'IBM SP Superchip, 4 memory cards','Rotation',90)
text(2,22,'IBM SP Superchip, 2 memory cards','Rotation',90)
text(3,22,'IBM SP2','Rotation',90)
text(4,22,'CRAY T3E','Rotation',90)
text(5,22,'SGI Origin 2000','Rotation',90)
text(6,22,'NT Pentium PRO 200Mhz','Rotation',90)
text(7,22,'IBM SP1','Rotation',90)
text(8,22,'Sun Ultra 2 UPA/SBus (168MHz)','Rotation',90)
text(9,22,'SGI PowerChallenge','Rotation',90)

% print -dps performance_cfd_2_10_1.ps

figure(2)

Y = [138 87 100 57 53 39 36 32 28 ];
bar(Y);
axis([0 10 0 150])
ylabel('MFlops');
title('Matrix-vector Product Matrix: cfd.2.10, N=122880, NZ=4134400');

text(1,22,'IBM SP Superchip, 4 memory cards','Rotation',90)
text(2,22,'IBM SP Superchip, 2 memory cards','Rotation',90)
text(3,22,'IBM SP2','Rotation',90)
text(4,22,'CRAY T3E','Rotation',90)
text(5,22,'SGI Origin 2000','Rotation',90)
text(6,22,'NT Pentium PRO 200Mhz ','Rotation',90)
text(7,22,'IBM SP1','Rotation',90)
text(8,22,'Sun Ultra 2 UPA/SBus (168MHz)','Rotation',90)
text(9,22,'SGI PowerChallenge','Rotation',90)

% print -dps performance_cfd_2_10_2.ps
