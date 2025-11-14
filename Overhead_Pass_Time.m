clear all;
clc;

R = 3396.18255; %Mars radius (km)
h = [1000,1850,4000,5000,13500,15000,15000,15500,17500,19000]; %satellite altitude (km) increments 
%^^^ if circular and the same then just one #
ep = 10; %min elevation angle (rad/s) JUST A FILL IN NEED TO CHANGE
mu = 4.282837*10^4; %Mars' gravitational parameter (km^3/s^2)

A = R+h;
B = R;

cos_psi_max = (B*(cos(ep)^2)+sin(ep)*sqrt(A.^2+B^2*(cos(ep)^2)))/A;
psi_max = acos(cos_psi_max);

w_omega = sqrt(mu./(A.^3)); %mean angular rate for circular orbit around mars
omega_sat = w_omega.*(B./A); %angular rate of sats subpoint on Mars
omega_mars = (2*pi)/88642; %angular rotation of Mars 
O_omega = abs(omega_sat + omega_mars); %or minus depending on the orbit direction


T = (2*psi_max)./O_omega %in seconds
T_m = T./60 %in minutes
T_h = T_m./60 % in hours

