# Time Series Anomaly Dection with Sequitur and PySAX

A time series anomaly detection program using principles from Kolmogorov Complexity and MDL (Minimum Description Length).
The tool uses compressibility as basis for a score to detect anomalous patterns. This was a project for a a seminar on information theoretic data-ming at LIACS (Leiden Institute of Advanced Computer Science).

## Implementation
Very simply explained it uses the discretization used for time series in PySAX and the grammar based compression of Sequitur as basis for the compression of the time series.
The algorithm then uses the compression to calculate a score of the compressibility of each point in the time-series. If the compressibility of a sequence of points is low for a certain sequence then an anomaly  is detected.

## Data
The data used to run the algorithm is noisy website activity data.

## References
See PDF
