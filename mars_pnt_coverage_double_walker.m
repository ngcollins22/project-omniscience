function mars_pnt_coverage_demo
clc; close all;

%% ---- Mars constants ----
R_Mars_km    = 3389.5;                 % km
muMars_km3s2 = 4.282837e4;             % km^3/s^2
Tsid_sec     = 88642.663;              % Mars sidereal day [s]
omegaMars    = 2*pi/Tsid_sec;          % rad/s

%% ---- Walker-Delta 1: "Equatorial optimized" ----
T1      = 20;          % total satellites
P1      = 5;           % number of planes
F1      = 4;            % phasing parameter
i1_deg  = 30;           % inclination [deg]
h1_km   = 13500;         % altitude [km]

%% ---- Walker-Delta 2: "Upgrade for global coverage" ----
T2      = 6;           % total satellites
P2      = 2;            % number of planes
F2      = 2;            % phasing parameter
i2_deg  = 80;           % inclination [deg]
h2_km   = 13500;        % altitude [km]

%% ---- Build both constellations and concatenate ----
[sat_elems1, S1] = buildWalker(T1,P1,F1,i1_deg,h1_km, ...
                               R_Mars_km,muMars_km3s2,1);
[sat_elems2, S2] = buildWalker(T2,P2,F2,i2_deg,h2_km, ...
                               R_Mars_km,muMars_km3s2,2);

sat_elems = [sat_elems1, sat_elems2];
Nsat1     = numel(sat_elems1);
Nsat2     = numel(sat_elems2);
Nsat      = numel(sat_elems);

%% ---- Time grid ----
t0 = 0;
tF = Tsid_sec;               % one Mars day
dt = 60;                     % 1-min resolution
t  = t0:dt:tF;               % [s]
Nt = numel(t);

%% ---- Latitude band and sampling ----
lat_min_deg = -90;
lat_max_deg =  90;
dlat        = 5;
dlon        = 10;
el_mask_deg = 10;

lat_list   = (lat_min_deg:dlat:lat_max_deg).';
lon0_list  = (0:dlon:350);
Nlat       = numel(lat_list);
Nlon       = numel(lon0_list);

%% ---- Allocate arrays ----
min_inview_vs_time = zeros(1,Nt);
cum_inview_counts  = zeros(Nlat,Nlon);

% Store satellite trajectories for plotting
r_sat_hist = zeros(3, Nsat, Nt);   % [x;y;z] for each sat and time step

%% ---- Main loop ----
for it = 1:Nt
    ti = t(it);

    % Satellite ECI positions at time ti for all sats (both walkers)
    r_sat = zeros(3,Nsat);
    for s = 1:Nsat
        M  = sat_elems(s).M0 + sat_elems(s).n * ti;   % mean anomaly (rad)
        nu = wrapTo2Pi(M);                            % e = 0

        r_pqw = [sat_elems(s).a*cos(nu); 
                 sat_elems(s).a*sin(nu); 
                 0];

        r_eci      = R3(sat_elems(s).raan) * R1(sat_elems(s).i) * r_pqw;
        r_sat(:,s) = r_eci;
    end

    r_sat_hist(:,:,it) = r_sat;

    % Coverage evaluation
    inview_counts = zeros(Nlat,Nlon);
    for ii = 1:Nlat
        phi = deg2rad(lat_list(ii));
        for jj = 1:Nlon
            lambda = deg2rad(lon0_list(jj)) + omegaMars*ti;

            r_g  = sph2eci(R_Mars_km, phi, lambda);
            zhat = r_g / norm(r_g);

            vis = 0;
            for s = 1:Nsat
                svec = r_sat(:,s) - r_g;
                el   = asin( dot(zhat, svec) / norm(svec) );
                if el >= deg2rad(el_mask_deg)
                    vis = vis + 1;
                end
            end
            inview_counts(ii,jj) = vis;
        end
    end

    cum_inview_counts      = cum_inview_counts + inview_counts;
    min_inview_vs_time(it) = min(inview_counts, [], 'all');
end

avg_inview_counts = cum_inview_counts/length(t);

fprintf('Worst-case satellites in view over the band = %d\n', ...
        min(min_inview_vs_time));

%% =========================================================
%           SINGLE FIGURE WITH TILED SUBPLOTS
%% =========================================================
fig = figure;
tiledlayout(fig,2,2,"TileSpacing","compact","Padding","compact");

%% ---- (3) 3D ECI trajectory plot ----
nexttile;
hold on; grid on;

[nx, ny, nz] = sphere(60);
surf(R_Mars_km*nx, R_Mars_km*ny, R_Mars_km*nz, ...
    'FaceAlpha',0.2, 'EdgeColor','none');
colormap(gca,'copper');

% Walker 1 in black
for s = 1:Nsat1
    rtraj = squeeze(r_sat_hist(:,s,:)).';
    plot3(rtraj(:,1), rtraj(:,2), rtraj(:,3), ...
        'k', 'LineWidth',1.0);
end

% Walker 2 in red dashed
for s = (Nsat1+1):Nsat
    rtraj = squeeze(r_sat_hist(:,s,:)).';
    plot3(rtraj(:,1), rtraj(:,2), rtraj(:,3), ...
        'r--', 'LineWidth',1.0);
end

axis equal;
xlabel('x_{ECI} [km]');
ylabel('y_{ECI} [km]');
zlabel('z_{ECI} [km]');
title(sprintf(['ECI Trajectories\n' ...
               'Walker1 %d/%d/%d (i=%d^\\circ, h=%g km)\n' ...
               'Walker2 %d/%d/%d (i=%d^\\circ, h=%g km)'], ...
              T1,P1,F1,i1_deg,h1_km, ...
              T2,P2,F2,i2_deg,h2_km));
view(30,20);
hold off;


%% ---- (1) Min satellites in view vs time ----
nexttile;
plot(t/3600, min_inview_vs_time, 'LineWidth',1.8);
yline(4,'--'); grid on;
xlabel('Time [hours]');
ylabel('Minimum sats in view');
title(sprintf(['Min In-View Over Band [%d,%d]^\\circ\n' ...
               'el\\_mask = %d^\\circ'], ...
              lat_min_deg,lat_max_deg,el_mask_deg));

%% ---- (2) Average satellites in view (heatmap style) ----
nexttile;
imagesc(lon0_list, lat_list, avg_inview_counts);
set(gca,'YDir','normal');
xlabel('Longitude [deg]');
ylabel('Latitude [deg]');
title(sprintf(['Average Sats in View (%.1f hrs)\n' ...
               'Combined Walker1 %d/%d/%d + Walker2 %d/%d/%d'], ...
              tF/3600, T1,P1,F1,T2,P2,F2));
colorbar;
set(gca,'XTick',0:60:360);

%% ---- (4) Ground track plot ----
% Precompute ground-track lat/lon from r_sat_hist
lat_sat = zeros(Nt, Nsat);
lon_sat = zeros(Nt, Nsat);

for s = 1:Nsat
    for it = 1:Nt
        ti    = t(it);
        r_eci = r_sat_hist(:,s,it);

        theta  = omegaMars * ti;
        r_body = R3(-theta) * r_eci;

        r_norm = norm(r_body);
        x = r_body(1);
        y = r_body(2);
        z = r_body(3);

        lat = asin(z / r_norm);
        lon = atan2(y, x);

        lat_sat(it,s) = rad2deg(lat);
        lon_sat(it,s) = mod(rad2deg(lon), 360);
    end
end

nexttile;
hold on; grid on;

% Walker 1 ground tracks in black
for s = 1:Nsat1
    plot(lon_sat(:,s), lat_sat(:,s), 'k', 'LineWidth', 0.8);
end

% Walker 2 ground tracks in red dashed
for s = (Nsat1+1):Nsat
    plot(lon_sat(:,s), lat_sat(:,s), 'r--', 'LineWidth', 0.8);
end

xlim([0 360]);
ylim([-90 90]);
xlabel('Longitude [deg]');
ylabel('Latitude [deg]');
title('Ground Tracks on Mars');
set(gca,'XTick',0:60:360);
hold off;


end

%% ===== Helper to build a Walker-Delta constellation =====
function [sat_elems, S] = buildWalker(T,P,F,i_deg,h_km, ...
                                      R_Mars_km,muMars_km3s2,walker_id)

if T == 0
    sat_elems = struct('a',{},'e',{},'i',{},'raan',{}, ...
                       'argp',{},'M0',{},'n',{}, ...
                       'walker',{},'plane',{});
    S = 0;
    return;
end

S = T / P;
if abs(round(S) - S) > 1e-10
    error('For Walker %d, T must be divisible by P', walker_id);
end
S = round(S);

i_rad   = deg2rad(i_deg);
a_km    = R_Mars_km + h_km;
n_rad_s = sqrt(muMars_km3s2/(a_km^3));

raan_list_deg = (0:P-1)*(360/P);

M0_deg = zeros(P,S);
for j = 1:P
    for m = 1:S
        M0_deg(j,m) = (m-1)*(360/S) + (j-1)*(360*F/T);
    end
end

sat_elems = struct('a',[],'e',[],'i',[],'raan',[], ...
                   'argp',[],'M0',[],'n',[], ...
                   'walker',[],'plane',[]);
k = 0;
for j = 1:P
    for m = 1:S
        k = k + 1;
        sat_elems(k).a      = a_km;
        sat_elems(k).e      = 0;
        sat_elems(k).i      = i_rad;
        sat_elems(k).raan   = deg2rad( raan_list_deg(j) );
        sat_elems(k).argp   = 0;
        sat_elems(k).M0     = deg2rad( M0_deg(j,m) );
        sat_elems(k).n      = n_rad_s;
        sat_elems(k).walker = walker_id;
        sat_elems(k).plane  = j;
    end
end

end

%% ===== Rotation and geometry helpers =====
function R = R1(a)
c = cos(a); s = sin(a);
R = [1 0 0; 0 c -s; 0 s c];
end

function R = R3(a)
c = cos(a); s = sin(a);
R = [c -s 0; s c 0; 0 0 1];
end

function r = sph2eci(R, lat, lon)
clat = cos(lat); slat = sin(lat);
clon = cos(lon); slon = sin(lon);
r = R * [clat*clon; clat*slon; slat];
end

function a = wrapTo2Pi(a)
a = mod(a, 2*pi);
end
