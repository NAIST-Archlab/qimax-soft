### Data size
## Init

G = <Z0, Z1, ..., Zn>
=> Lambdas: [[1], [1], ..., [1]] (n x 1)
=> Indices: [[3, 0, ..., 0], [0, 3, ..., 0], ..., [0, 0, ..., 3]]

# Flatten
```
G = <
    a0_0*P0_0 + a0_1*P0_1 + ...
    ...
    a(n-1)_0*P(n-1)_0 + a(n-1)_1*P(n-1)_1 + ...
>
```
=> Lambdass: [
    [a0_0,a0_1, ..., a0_(k0)],
    [a1_0,a1_1, ..., a1_(k1)],
    ...
    [a(n-1)_0,a(n-1)_1, ..., a(n-1)_(k(n-1))],
] (n x ?)

=> Indicess: [
    [encode(P0_0), encode(P0_1), ...]
    ...
    [encode(P(n-1)_0), encode(P(n-1)_1) ...]
] (n x ? x n) because encode(P) = [p0, p1, ..., pn] with pj\in{0,1,2,3}, 0:I, 1:X, 2:Y, 3:Z

# Mapped by non-cx operators.

```
G = <
    a0_0*PN0_0 + a0_1*PN0_1 + ...
    ...
    a(n-1)_0*PN(n-1)_0 + a(n-1)_1*PN(n-1)_1 + ...
>
```

Lambdass: n x ?
Indices (n x ? x n) -- (transformed) --> weights (n x ? x n x 4)