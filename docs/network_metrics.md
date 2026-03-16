# Network Metrics for Mars PNT Constellations

Reliable navigation and communications on Mars depend on how well satellites interconnect at any given epoch. The sweep analyses compute several graph-based metrics from the line-of-sight (LoS) latency tensor output by the propagator. This document summarizes each metric, the exact mathematical definition, and how to interpret the resulting values.

## Constellation Graph Model

At each simulation timestep \(t\):

- Let \(N\) be the number of active satellites.
- Define an undirected adjacency matrix \(A^{(t)} \in \{0,1\}^{N \times N}\) where
  \[
  A^{(t)}_{ij} = \begin{cases}
  1 & \text{if a LoS link with finite latency exists between satellites } i \text{ and } j \\
  0 & \text{otherwise}
  \end{cases}
  \]
- Because latency is symmetric, \(A^{(t)}_{ij} = A^{(t)}_{ji}\) and diagonal entries are zero.

All metrics below are computed per timestep using \(A^{(t)}\) and then averaged over time to obtain a single constellation-level score.

## Number of Links

The number of undirected links present at time \(t\) is
\[
L^{(t)} = \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} A^{(t)}_{ij}
\]
It counts distinct satellite-to-satellite LoS connections. Higher values indicate richer connectivity but also higher hardware or power demands if links must be maintained simultaneously.

## Redundancy Ratio

A constellation with \(N\) nodes needs at least \(N-1\) links to remain connected (a spanning tree). Anything beyond that provides redundant routing options. We normalize the excess link count by the theoretical maximum so the metric lies in \([-1, 1]\):
\[
R^{(t)} = \frac{L^{(t)} - (N-1)}{\frac{N(N-1)}{2} - (N-1)}
\]
- \(R^{(t)} = 1\) for a fully meshed network.
- \(R^{(t)} = 0\) indicates just enough links to stay connected.
- \(R^{(t)} < 0\) means the constellation lacks the minimum links for global connectivity at that timestep.

## Average Degree per Node

The degree of satellite \(i\) at time \(t\) is \(d_i^{(t)} = \sum_j A^{(t)}_{ij}\). The mean degree is
\[
\bar{d}^{(t)} = \frac{1}{N} \sum_{i=1}^{N} d_i^{(t)} = \frac{2 L^{(t)}}{N}
\]
A higher mean degree means each satellite maintains more simultaneous crosslinks, improving resilience and routing flexibility.

## Network Density

Density captures the fraction of possible links that are active:
\[
D^{(t)} = \frac{\bar{d}^{(t)}}{N-1} = \frac{2 L^{(t)}}{N (N-1)}
\]
\(D^{(t)} = 1\) corresponds to a fully connected graph, while \(D^{(t)} = 0\) means no links. Density helps compare constellations of different sizes.

## Average Clustering Coefficient

For satellite \(i\), let \(\mathcal{N}_i^{(t)}\) be the set of neighbors (nodes linked to \(i\)). If \(|\mathcal{N}_i^{(t)}| = k < 2\), the local clustering coefficient is defined as zero. Otherwise:
\[
C_i^{(t)} = \frac{\text{number of links between nodes in } \mathcal{N}_i^{(t)}}{\binom{k}{2}}
\]
The constellation-wide average at time \(t\) is
\[
\bar{C}^{(t)} = \frac{1}{N} \sum_{i=1}^{N} C_i^{(t)}
\]
Clustering measures how well each satellites neighbors are themselves interconnected, which relates to local routing resilience and latency.

## Time Averaging

Each of the above metrics is evaluated at every timestep over the propagation horizon (typically up to 24 hours). The analysis reports the simple arithmetic mean:
\[
\text{metric}_{\text{avg}} = \frac{1}{T} \sum_{t=1}^{T} \text{metric}^{(t)}
\]
This averaging smooths short-term visibility fluctuations while capturing overall constellation behavior.

## Practical Interpretation

- **Number of Links / Density**: highlight overall backbone richness and power requirements.
- **Redundancy Ratio**: quickly indicates whether a constellation has spare routing capacity beyond basic connectivity.
- **Average Degree**: informs per-satellite hardware needs (number of concurrent crosslinks).
- **Clustering Coefficient**: captures the likelihood of short alternative routes if a link fails.

These metrics combine with latency-derived measures and PDOP constraints to assess whether a constellation meets navigation and communications requirements for Mars surface users.
