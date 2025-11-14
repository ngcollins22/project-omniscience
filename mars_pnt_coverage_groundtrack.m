function mars_pnt_coverage_demo
clc; close all;

%% ---- Mars constants ----
R_Mars_km   = 3389.5;                 % km
muMars_km3s2 = 4.282837e4;            % km^3/s^2
Tsid_sec    = 88642.663;              % Mars sidereal day [s]
omegaMars   = 2*pi/Tsid_sec;          % rad/s

%% ---- Constellation parameters (Walker-Delta T/P/F) ----
T = 130;            % total satellites
P = 10;             % number of planes
F = 3;             % phasing parameter
i_deg = 30;        % inclination [deg]
h_km  = 1000;   % altitude [km]  
a_km  = R_Mars_km + h_km;             % semi-major axis [km]
n_rad_s = sqrt(muMars_km3s2/(a_km^3));

S = T / P;
if abs(round(S)-S) > 1e-10, error('T must be divisible by P'); end
S = round(S);

%% ---- Time grid ----
t0 = 0;
tF = Tsid_sec;               % one Mars day
dt = 60;                     % 1-min resolution
t = t0:dt:tF;                % [s]
Nt = numel(t);

%% ---- Latitude band and sampling ----
lat_min_deg = -45;
lat_max_deg =  45;
dlat = 5;
dlon = 10;
el_mask_deg = 10;

lat_list   = (lat_min_deg:dlat:lat_max_deg).';
lon0_list  = (0:dlon:350);
Nlat = numel(lat_list);
Nlon = numel(lon0_list);

%% ---- Precompute constellation geometry ----
i = deg2rad(i_deg);
raan_list_deg = (0:P-1)*(360/P);

M0_deg = zeros(P,S);
for j = 1:P
    for m = 1:S
        M0_deg(j,m) = (m-1)*(360/S) + (j-1)*(360*F/T);
    end
end

sat_elems = struct('a',[],'e',[],'i',[],'raan',[],'argp',[],'M0',[]);
k = 0;
for j = 1:P
    for m = 1:S
        k = k + 1;
        sat_elems(k).a    = a_km;
        sat_elems(k).e    = 0;
        sat_elems(k).i    = i;
        sat_elems(k).raan = deg2rad( raan_list_deg(j) );
        sat_elems(k).argp = 0;
        sat_elems(k).M0   = deg2rad( M0_deg(j,m) );
    end
end
Nsat = numel(sat_elems);

%% ---- Allocate arrays ----
min_inview_vs_time = zeros(1,Nt);
cum_inview_counts  = zeros(Nlat,Nlon);

% Store satellite trajectories for plotting
r_sat_hist = zeros(3, Nsat, Nt);   % [x;y;z] for each sat and time step

%% ---- Main loop ----
cosElMask = sind(el_mask_deg); %#ok<NASGU>

for it = 1:Nt
    ti = t(it);

    % Satellite ECI positions at time ti
    r_sat = zeros(3,Nsat);
    for s = 1:Nsat
        M = sat_elems(s).M0 + n_rad_s*ti;   % mean anomaly (rad)
        nu = wrapTo2Pi(M);                  % e = 0

        r_pqw = [a_km*cos(nu); a_km*sin(nu); 0];

        r_eci = R3(sat_elems(s).raan) * R1(sat_elems(s).i) * r_pqw;
        r_sat(:,s) = r_eci;
    end

    r_sat_hist(:,:,it) = r_sat;

    % Coverage evaluation
    inview_counts = zeros(Nlat,Nlon);
    for ii = 1:Nlat
        phi = deg2rad(lat_list(ii));
        for jj = 1:Nlon
            lambda = deg2rad(lon0_list(jj)) + omegaMars*ti;

            r_g = sph2eci(R_Mars_km, phi, lambda);
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

%% ---- Plot: minimum satellites in view vs time + heatmap ----
figure; 
subplot(2,1,1)
plot(t/3600, min_inview_vs_time, 'LineWidth',1.8);
yline(4,'--'); grid on;
xlabel('Time [hours]');
ylabel('Minimum satellites in view over band');
title(sprintf(['Walker %d/%d/%d @ i=%d^\\circ, h=%g km (Mars), ' ...
               'band [%d,%d]^\\circ, el\\_mask=%d^\\circ'], ...
              T,P,F,i_deg,h_km,lat_min_deg,lat_max_deg,el_mask_deg));

fprintf('Worst-case satellites in view over the band = %d\n', ...
        min(min_inview_vs_time));

subplot(2,1,2)
heatmap(lon0_list(:), lat_list(:), avg_inview_counts);
title(sprintf('Average Satellites in View Over T = %d hrs. Min = %.2f', ...
              tF/3600 , min(min(avg_inview_counts))));

%% ---- 3D ECI trajectory plot ----
figure; hold on; grid on;

[nx, ny, nz] = sphere(60);
surf(R_Mars_km*nx, R_Mars_km*ny, R_Mars_km*nz, ...
    'FaceAlpha',0.2, 'EdgeColor','none');
colormap('copper');

for s = 1:Nsat
    rtraj = squeeze(r_sat_hist(:,s,:)).';

    plane_idx = ceil(s / S);             % 1..P
    if P > 1
        plane_gray = 0.6 + 0.4 * ( (plane_idx-1)/(P-1) );
    else
        plane_gray = 0.8;
    end

    plot3(rtraj(:,1), rtraj(:,2), rtraj(:,3), ...
        'LineWidth',1.0, ...
        'Color',[0 0 0]);
end

axis equal;
xlabel('x_{ECI} [km]');
ylabel('y_{ECI} [km]');
zlabel('z_{ECI} [km]');
title(sprintf('Walker %d/%d/%d Trajectories Around Mars', T,P,F));
view(30,20);
hold off;

%% ---- NEW: Ground track plot (lat/lon of sub-satellite points) ----
% We use the assumption that at t = 0, ECI and Mars-fixed are aligned.
% At time t, the planet has rotated by theta = omegaMars * t about z.
% So body-fixed coordinates are: r_body = R3(-theta) * r_eci.

lat_sat = zeros(Nt, Nsat);
lon_sat = zeros(Nt, Nsat);

for s = 1:Nsat
    for it = 1:Nt
        ti   = t(it);
        r_eci = r_sat_hist(:,s,it);

        % Rotate into Mars-fixed frame
        theta   = omegaMars * ti;
        r_body  = R3(-theta) * r_eci;

        r_norm  = norm(r_body);
        x = r_body(1);
        y = r_body(2);
        z = r_body(3);

        % Geocentric latitude and longitude
        lat = asin(z / r_norm);
        lon = atan2(y, x);

        lat_sat(it,s) = rad2deg(lat);
        lon_sat(it,s) = mod(rad2deg(lon), 360);  % [0, 360) for plotting
    end
end

figure; hold on; grid on;
for s = 1:Nsat
    plane_idx = ceil(s / S);
    if P > 1
        plane_gray = 0.6 + 0.4 * ( (plane_idx-1)/(P-1) );
    else
        plane_gray = 0.8;
    end
    plot(lon_sat(:,s), lat_sat(:,s), ...
        'LineWidth', 1.0);
end
xlim([0 360]);
ylim([-90 90]);
xlabel('Longitude [deg]');
ylabel('Latitude [deg]');
title(sprintf('Ground Tracks on Mars (Walker %d/%d/%d, i = %d^\\circ)', ...
              T,P,F,i_deg));
set(gca,'XTick',0:60:360);
hold off;

end

%% ===== Helpers =====
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
