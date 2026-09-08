# CPL monitor microbenchmark results

Median of 9 repetitions, 300 operations each.

| series | value | lifelines | subformulas | variables | local action (us) | send (us) | receive (us) | message metadata (bytes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| running_example | guard | 4 | 16 | 1 | 46.933 | 69.393 | 72.295 | 1231 |
| lifelines | 2 | 2 | 31 | 4 | 135.162 | 158.334 | 153.665 | 1130 |
| lifelines | 4 | 4 | 31 | 4 | 139.661 | 182.015 | 181.922 | 2182 |
| lifelines | 8 | 8 | 31 | 4 | 151.541 | 228.746 | 236.308 | 4286 |
| lifelines | 16 | 16 | 31 | 4 | 168.966 | 318.231 | 342.238 | 8519 |
| lifelines | 32 | 32 | 31 | 4 | 208.528 | 505.918 | 557.008 | 16999 |
| subformulas | 7 | 8 | 7 | 4 | 56.964 | 94.518 | 102.27 | 1910 |
| subformulas | 15 | 8 | 15 | 4 | 89.443 | 141.42 | 148.907 | 2686 |
| subformulas | 63 | 8 | 63 | 4 | 273.217 | 395.863 | 407.983 | 7486 |
| subformulas | 127 | 8 | 127 | 4 | 521.289 | 753.76 | 764.274 | 14102 |
| variables | 1 | 8 | 255 | 1 | 998.033 | 1429.687 | 1447.989 | 27254 |
| variables | 4 | 8 | 255 | 4 | 1011.44 | 1461.912 | 1477.458 | 27926 |
| variables | 16 | 8 | 255 | 16 | 1094.522 | 1588.397 | 1601.598 | 30710 |
| variables | 64 | 8 | 255 | 64 | 1506.353 | 2219.657 | 2195.545 | 42230 |
| variables | 128 | 8 | 255 | 128 | 1976.473 | 2979.484 | 2902.64 | 58038 |
