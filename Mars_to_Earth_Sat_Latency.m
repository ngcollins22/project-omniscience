clear all
clc

c = 2.998*10^5; %speed of light km/s

NASA_sat = 160; %NASA comms satellite operational altitudes

Mars_sat = [1000,1850,4000,5000,13500,15000,15000,15500,17500,19000]; %our pos

Earth_to_Mars = 401*10^6; %earth to mars farthest distance in km
distance = Earth_to_Mars-(NASA_sat+Mars_sat);

Latency = distance./c;
