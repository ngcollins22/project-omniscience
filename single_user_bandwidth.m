clear all
clc
 
X = 8*10^9;
X2 = 12*10^9;
Ka = 26.5*10^9;
Ka2 = 40*10^9;
UHF = 300*10^6;
UHF2 = 3*10^9;
L = 1*10^9;
L2 = 2*10^9;


SNRlin = 10; %assuming SNR 10 DB

nu = log2(1+SNRlin);

%overhead pass times
overhead_pass = 28013; % MGPS overhead pass time in seconds
martian_rotation = 88762; %rotation period in seconds

alpha = overhead_pass./martian_rotation;

RrawX = X.*nu;
RrawKa = Ka.*nu;
RrawUHF = UHF.*nu;
RrawL = L.*nu;

RrawX2 = X2.*nu;
RrawKa2 = Ka2.*nu;
RrawUHF2 = UHF2.*nu;
RrawL2 = L2.*nu;

RX = RrawX.*(1-alpha);
RKa = RrawKa.*(1-alpha);
RUHF = RrawUHF.*(1-alpha);
RL = RrawL.*(1-alpha);

RX2 = RrawX2.*(1-alpha);
RKa2 = RrawKa2.*(1-alpha);
RUHF2 = RrawUHF2.*(1-alpha);
RL2 = RrawL2.*(1-alpha);

DataX_bits_per_rotation = RX*martian_rotation;
DataKa_bits_per_rotation = RKa*martian_rotation;
DataUHF_bits_per_rotation = RUHF*martian_rotation;
DataL_bits_per_rotation = RL*martian_rotation;

DataX2_bits_per_rotation = RX2*martian_rotation;
DataKa2_bits_per_rotation = RKa2*martian_rotation;
DataUHF2_bits_per_rotation = RUHF2*martian_rotation;
DataL2_bits_per_rotation = RL2*martian_rotation;

DataXGbyte = (DataX_bits_per_rotation/8)/(10^9);
DataKaGbyte = (DataKa_bits_per_rotation/8)/(10^9);
DataUHFGbyte = (DataUHF_bits_per_rotation/8)/(10^9);
DataLGbyte = (DataL_bits_per_rotation/8)/(10^9);

DataX2Gbyte = (DataX2_bits_per_rotation/8)/(10^9);
DataKa2Gbyte = (DataKa2_bits_per_rotation/8)/(10^9);
DataUHF2Gbyte = (DataUHF2_bits_per_rotation/8)/(10^9);
DataL2Gbyte = (DataL2_bits_per_rotation/8)/(10^9);

fprintf('The single user bandwidth is min %.3f GB and max %.3f GB if X band is used\n', DataXGbyte, DataX2Gbyte);
fprintf('The single user bandwidth is min %.3f GB and max %.3f GB if Ka band is used\n', DataKaGbyte, DataKa2Gbyte);
fprintf('The single user bandwidth is min %.3f GB and max %.3f GB if UHF band is used\n', DataUHFGbyte, DataUHF2Gbyte);
fprintf('The single user bandwidth is min %.3f GB and max %.3f GB if L band is used', DataLGbyte, DataL2Gbyte);
