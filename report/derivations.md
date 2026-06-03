# Mathematical derivations for the COS SaAvg method

This document derives, step by step, every formula used in `saavg-cos`. It is the
mathematical companion to [`report.md`](report.md) / [`report.tex`](report.tex)
and cross-references the implementation (`saavg_cos/spectral_im.py`,
`saavg_cos/cos_cache.py`, `saavg_cos/fragility.py`, `saavg_cos/risk_hazard.py`).

GitHub renders the `$…$` / `$$…$$` math below.

## Contents

0. [Notation](#0-notation)
1. [The SaAvg model](#1-the-saavg-model)
2. [The conditional Gaussian mixture](#2-the-conditional-gaussian-mixture)
3. [Characteristic function as a Gaussian expectation](#3-characteristic-function-as-a-gaussian-expectation)
4. [Evaluating the expectation: PCA + Gauss–Hermite](#4-evaluating-the-expectation-pca--gausshermite)
5. [Cumulants of the Gaussian mixture](#5-cumulants-of-the-gaussian-mixture)
6. [The cumulant characteristic function](#6-the-cumulant-characteristic-function)
7. [COS inversion (Fang–Oosterlee)](#7-cos-inversion-fangoosterlee)
8. [Fragility convolution and closed-form damage probability](#8-fragility-convolution-and-closed-form-damage-probability)
9. [Hazard and risk aggregation](#9-hazard-and-risk-aggregation)
10. [Approximations and error control](#10-approximations-and-error-control)
11. [References](#11-references)

---

## 0. Notation

| symbol | meaning |
|---|---|
| $P$ | number of FCM periods ($P=10$) |
| $X=(X_1,\dots,X_P)$ | reference log-spectral-acceleration vector, $X_i=\ln \mathrm{Sa}_{\mathrm{ref}}(T_i)$ (cm/s²) |
| $\mu,\ \Sigma$ | mean and covariance of $X$ (from GMM-V7), $X\sim\mathcal N(\mu,\Sigma)$ |
| $\nu_i(\cdot)$ | non-linear amplification median (log) at period $i$ |
| $\tau_i(\cdot)$ | amplification residual standard deviation (log) |
| $R_{\mathrm{AF}}$ | period-to-period correlation of the amplification residual |
| $w_i$ | deterministic surface ("wierde") term |
| $Z=\ln\mathrm{SaAvg}$ | quantity whose distribution we want |
| $\varphi(u)=\mathbb E[e^{iuZ}]$ | characteristic function (CF) of $Z$ |
| $\kappa_1,\kappa_2,\kappa_3$ | first three cumulants of $Z$ (code: `k1,k2,k3`) |
| $\gamma_1=\kappa_3/\kappa_2^{3/2}$ | skewness |

---

## 1. The SaAvg model

The average spectral acceleration is the geometric mean of spectral acceleration
over the $P$ FCM periods [Crowley & Pinho; Kohrangi et al.]:

$$
\mathrm{SaAvg}=\Big(\prod_{i=1}^{P}\mathrm{Sa}(T_i)\Big)^{1/P},
\qquad
Z:=\ln\mathrm{SaAvg}=\frac1P\sum_{i=1}^{P}\ln\mathrm{Sa}(T_i).
$$

Each surface motion is a reference (rock) motion times a site amplification, both
lognormal, so in log-space

$$
\ln\mathrm{Sa}(T_i)=\underbrace{X_i}_{\text{reference}}+\underbrace{\nu_i(X_i)+\eta_i}_{\text{amplification}}+\underbrace{w_i}_{\text{surface}},
\qquad
\eta\sim\mathcal N\!\big(0,\ \mathrm{diag}(\tau)\,R_{\mathrm{AF}}\,\mathrm{diag}(\tau)\big).
$$

The reference vector is multivariate normal with period correlation from the GMM,

$$
X\sim\mathcal N(\mu,\Sigma),\qquad \Sigma_{ij}=\rho_{ij}\,\sigma_i\sigma_j,
$$

where $\sigma_i^2$ is the GMM source+path variance (`gmm_V7.reference_ac_variance`)
and $\rho$ the GMM period correlation. The amplification median $\nu_i$ **saturates**
at strong shaking: in `gmm_V7` it is the clipped non-linear form

$$
\nu_i(x)=\mathrm{clip}\!\Big[\nu^{\rm lin}_i + f_2\,\ln\!\frac{Y_i+f_3}{f_3},\ \ln\mathrm{AF}_{\min},\ \ln\mathrm{AF}_{\max}\Big],
\qquad Y_i=\frac{e^{x}}{\mathrm{AFscale}} .
$$

Because $\nu_i$ is concave and saturates, the upper tail of $Z$ is compressed,
which makes the distribution of $Z$ **negatively skewed** ($\gamma_1<0$). This is
exactly the feature the normal (moment-matching) method discards and the COS
method captures.

> **Units.** $X$ and $Z$ are in cm/s². The amplification dispersion model and the
> fragility thresholds are defined on log-$g$, hence the constant
> $\ln(\mathrm{AFscale})=\ln 981$ that shifts cm/s²$\to g$ before evaluating
> $\tau$ (`spectral_im`: `lnY_g = nodes - log(AFscale)`).

---

## 2. The conditional Gaussian mixture

The key structural fact: **conditional on the reference vector $X$, $Z$ is
Gaussian.** Indeed, given $X$ the only remaining randomness is the residual
average $\frac1P\sum_i\eta_i$, a linear combination of jointly normal variables.
Write

$$
Z=\frac1P\sum_{i=1}^{P}\big(X_i+\nu_i(X_i)+w_i\big)+\frac1P\sum_{i=1}^{P}\eta_i .
$$

The first sum is deterministic given $X$; call it $m(X)$. The second is zero-mean
Gaussian. Hence

$$
\boxed{\,Z\mid X\ \sim\ \mathcal N\big(m(X),\,v(X)\big)\,},\qquad
m(X)=\frac1P\sum_{i=1}^{P}\big(X_i+\nu_i(X_i)+w_i\big).
$$

**Conditional variance.** With $\eta\sim\mathcal N(0,\Sigma_{\rm AF})$,
$\Sigma_{\rm AF}=\mathrm{diag}(\tau)R_{\mathrm{AF}}\mathrm{diag}(\tau)$,

$$
v(X)=\mathrm{Var}\!\Big[\tfrac1P\textstyle\sum_i\eta_i\,\Big|\,X\Big]
=\frac1{P^2}\sum_{i,j}\mathrm{Cov}(\eta_i,\eta_j)
=\frac1{P^2}\sum_{i,j}\tau_i\,(R_{\mathrm{AF}})_{ij}\,\tau_j
=\boxed{\,\frac1{P^2}\,\tau(X)^{\!\top} R_{\mathrm{AF}}\,\tau(X)\,}.
$$

Two site-to-site (s2s) modes used by the GMM are both special cases:

* **Epistemic** (production): the residual is replaced by a deterministic shift
  $s_{2s}\,\tau_i$, so it enters $m(X)$ and **not** $v(X)$:
  $m(X)=\frac1P\sum_i(X_i+\nu_i(X_i)+s_{2s}\tau_i(X_i)+w_i)$, $v(X)=0$ at this level
  (the only variance comes from the omitted reference directions, §4.3).
* **Aleatory**: $\eta$ is random with correlation $R_{\mathrm{AF}}$; $R_{\mathrm{AF}}=I$
  is the "zero-correlation" mode.

(Code: `spectral_im.compute_scenarios`, branch on `s2s_mode`.)

---

## 3. Characteristic function as a Gaussian expectation

The CF of a $\mathcal N(m,v)$ variable is $\mathbb E[e^{iuY}]=e^{ium-\frac12u^2v}$.
By the tower property and the conditional-Gaussian result of §2,

$$
\varphi(u)=\mathbb E\big[e^{iuZ}\big]
=\mathbb E_X\Big[\mathbb E\big[e^{iuZ}\mid X\big]\Big]
=\boxed{\,\mathbb E_X\!\Big[\exp\!\big(iu\,m(X)-\tfrac12u^2 v(X)\big)\Big]\,}.
$$

So the full CF is a single expectation over the $P$-dimensional Gaussian $X$. The
integrand is smooth and bounded, which is what makes the deterministic quadrature
in §4 accurate with very few nodes. Everything downstream (the cumulants, the COS
inversion) is built from this object.

---

## 4. Evaluating the expectation: PCA + Gauss–Hermite

### 4.1 Principal-component (Karhunen–Loève) coordinates

Eigendecompose the reference covariance,

$$
\Sigma=\sum_{r=1}^{P}\lambda_r\,e_r e_r^{\!\top},\qquad \lambda_1\ge\lambda_2\ge\cdots\ge0,
$$

(`np.linalg.eigh`, eigenvalues sorted descending). Then $X$ has the exact
representation

$$
X=\mu+\sum_{r=1}^{P}\sqrt{\lambda_r}\,\xi_r\,e_r,\qquad \xi_r\stackrel{\text{iid}}{\sim}\mathcal N(0,1),
$$

so the expectation in §3 becomes an integral over the independent standard normals
$\xi=(\xi_1,\dots,\xi_P)$:

$$
\varphi(u)=\mathbb E_{\xi}\big[g_u(\xi)\big],\qquad
g_u(\xi)=\exp\!\big(iu\,m(X(\xi))-\tfrac12u^2 v(X(\xi))\big).
$$

### 4.2 Gauss–Hermite quadrature and its normalization

We integrate the leading $\rho$ directions ($\rho=$ `pca_rank`) with a tensor
Gauss–Hermite rule. The physicists' GH rule approximates
$\int_{\mathbb R} e^{-t^2}h(t)\,dt\approx\sum_k\omega_k\,h(t_k)$
(`numpy.polynomial.hermite.hermgauss` returns $\{t_k,\omega_k\}$). For a standard
normal we need $\int g(\xi)\tfrac1{\sqrt{2\pi}}e^{-\xi^2/2}\,d\xi$; substituting
$\xi=\sqrt2\,t$,

$$
\int_{\mathbb R} g(\xi)\,\frac{e^{-\xi^2/2}}{\sqrt{2\pi}}\,d\xi
=\frac1{\sqrt\pi}\int_{\mathbb R} g(\sqrt2\,t)\,e^{-t^2}\,dt
\approx\sum_{k}\underbrace{\frac{\omega_k}{\sqrt\pi}}_{\text{weight}}\,g\big(\underbrace{\sqrt2\,t_k}_{\text{node}}\big).
$$

This is exactly `_gauss_hermite`: `nodes = sqrt(2)*t_k`, `weights = ω_k/sqrt(pi)`.
The order-$O$ rule integrates polynomials up to degree $2O-1$ exactly. For $\rho$
directions the tensor rule has $n=O^\rho$ nodes $\{\xi^{(n)}\}$ with weights
$w_n=\prod_{r=1}^{\rho}(\omega_{k_r}/\sqrt\pi)$ (outer products of the 1-D rule).

The quadrature nodes in the original coordinates are
$X^{(n)}=\mu+\sum_{r\le\rho}\sqrt{\lambda_r}\,\xi^{(n)}_r e_r$ — these are the
mixture component locations. Because $(\mu,\Sigma)$ depend only on $(M,R)$, the
eigendecomposition and nodes are computed **once** and reused across all 160 zones
(only the per-zone amplification $\nu,\tau$ is evaluated inside the zone loop;
`exp(nodes)` is hoisted out).

### 4.3 Omitted directions: delta-method variance + Jensen curvature

The omitted directions $r>\rho$ carry residual covariance

$$
\Sigma_{\rm res}=\sum_{r>\rho}\lambda_r\,e_r e_r^{\!\top}
$$

(`residual_cov`). Rather than add more quadrature dimensions, we fold them back by
a second-order (delta-method) expansion of $m$ around each leading node.

**Variance contribution.** With $X_i=\mu_i+\sum_r\sqrt{\lambda_r}\,\xi_r e_{r,i}$,

$$
\frac{\partial m}{\partial \xi_r}
=\frac1P\sum_i\big(1+\nu_i'(X_i)\big)\sqrt{\lambda_r}\,e_{r,i}
=\sqrt{\lambda_r}\sum_i s_i\,e_{r,i},
\qquad s_i:=\frac{1+\nu_i'(X_i)}{P}.
$$

Treating $m$ as locally linear in the omitted $\xi_{r>\rho}$ gives the residual
variance

$$
v_{\rm ref}=\sum_{r>\rho}\Big(\frac{\partial m}{\partial\xi_r}\Big)^2
=\sum_{r>\rho}\lambda_r\Big(\sum_i s_i e_{r,i}\Big)^2
=\boxed{\,s^{\!\top}\Sigma_{\rm res}\,s\,}
$$

(code: `cond_var = einsum("...ip,pq,...iq->...i", sens, residual_cov, sens)`,
`sens = (1+deriv)/P`).

**Jensen curvature (mean).** A linear treatment leaves a second-order mean bias
because $\nu_i$ is concave. Expanding $X_i+\nu_i(X_i)$ to second order in the
omitted directions and taking the expectation,

$$
\Delta m=\tfrac12\sum_{r>\rho}\lambda_r\,\frac{\partial^2 m}{\partial\xi_r^2}
=\frac1{2P}\sum_i \nu_i''(X_i)\sum_{r>\rho}\lambda_r e_{r,i}^2
=\boxed{\,\frac1{2P}\sum_i \nu_i''(X_i)\,(\Sigma_{\rm res})_{ii}\,}
$$

(code: `cond_mean += 0.5*einsum("...ip,mp->...i", deriv2, res_diag)/P`,
`res_diag = diag(residual_cov)`). This curvature term — not the units handling —
is the dominant correction that brings the mean of the mixture onto the
Monte-Carlo mean.

### 4.4 The amplification derivatives

With $\nu_i(x)=\nu_i^{\rm lin}+f_2\ln\frac{Y+f_3}{f_3}$, $Y=e^{x}/\mathrm{AFscale}$
and $\frac{dY}{dx}=Y$, the first and second derivatives (active, i.e. unclipped,
region) are

$$
\nu_i'(x)=f_2\,\frac{Y}{Y+f_3},\qquad
\nu_i''(x)=f_2 f_3\,\frac{Y}{(Y+f_3)^2}\ (>0\ \text{for the concave log term}),
$$

set to $0$ where $\nu_i$ is clipped at $\ln\mathrm{AF}_{\min/\max}$ (`active` mask).
These feed $s_i$ (§4.3 variance) and $\Delta m$ (§4.3 curvature).

### 4.5 Putting the mixture together

For each leading node $n$ we now have a Gaussian component
$\big(m_n,\,v_n\big)$ with weight $w_n$:

$$
m_n=\frac1P\sum_i\big(X^{(n)}_i+\nu_i(X^{(n)}_i)+[\,s_{2s}\tau_i\,]+w_i\big)+\Delta m_n,
\qquad
v_n=\underbrace{s^{\!\top}\Sigma_{\rm res}s}_{\text{omitted ref}}+\underbrace{\tfrac1{P^2}\tau^{\!\top}R_{\rm AF}\tau}_{\text{aleatory only}} .
$$

So $Z\approx\sum_n w_n\,\mathcal N(m_n,v_n)$ — a small Gaussian mixture (e.g.
$O^\rho=7^2=49$ components at the fast setting).

---

## 5. Cumulants of the Gaussian mixture

We never store the mixture; we reduce it to three numbers. For a mixture with
moment generating function

$$
M(t)=\sum_n w_n\,\exp\!\big(m_n t+\tfrac12 v_n t^2\big),
$$

the cumulants are $\kappa_j=\frac{d^j}{dt^j}\log M(t)\big|_{t=0}$. It is cleanest to
use central moments. Let $\mu=\sum_n w_n m_n$ and $\delta_n=m_n-\mu$. For a single
component $\mathcal N(m_n,v_n)$, writing $Y=Z-\mu=\delta_n+W$ with $W\sim\mathcal N(0,v_n)$
and using $\mathbb E[W]=\mathbb E[W^3]=0$, $\mathbb E[W^2]=v_n$:

$$
\mathbb E[Y^2\mid n]=\delta_n^2+v_n,\qquad
\mathbb E[Y^3\mid n]=\delta_n^3+3\delta_n v_n .
$$

Averaging over components (the first three cumulants equal the first three central
moments):

$$
\boxed{
\kappa_1=\sum_n w_n m_n,\quad
\kappa_2=\sum_n w_n\big(\delta_n^2+v_n\big),\quad
\kappa_3=\sum_n w_n\big(\delta_n^3+3\delta_n v_n\big).}
$$

(code: `k1 = means@w`, `delta = means-k1`, `k2 = (delta**2+var)@w`,
`k3 = (delta**3+3*delta*var)@w`.) The skewness is $\gamma_1=\kappa_3/\kappa_2^{3/2}$.

For completeness, the (neglected) fourth cumulant is
$\kappa_4=\sum_n w_n(\delta_n^4+6\delta_n^2 v_n+3v_n^2)-3\kappa_2^2$; its smallness
(a few percent excess kurtosis here) is why three cumulants suffice (§10).

---

## 6. The cumulant characteristic function

The cumulant generating function is $\log\varphi(u)=\sum_{j\ge1}\kappa_j\frac{(iu)^j}{j!}$.
Truncating after $j=3$ gives the approximation we invert:

$$
\boxed{\ \varphi(u)\ \approx\ \exp\!\Big(i\kappa_1 u-\tfrac12\kappa_2 u^2-\tfrac{i}{6}\kappa_3 u^3\Big)\ }
$$

(code: `phi = exp(-0.5*k2*u**2) * exp(1j*(k1*u - k3*u**3/6))`). Two remarks:

* **Magnitude stays Gaussian-damped.** $|\varphi(u)|=\exp(-\tfrac12\kappa_2u^2)$ —
  the cubic term is purely imaginary (a phase), so it does not affect convergence
  of the inverse transform; it only *rotates* the spectrum, which is precisely how
  the skewness is injected. (This is the practical advantage over an Edgeworth
  *density* series, which can go negative; here the negativity is harmless and is
  cleaned at the CDF level.)
* This is a third-order Edgeworth/Gram–Charlier-type CF [McCullagh 1987;
  Kendall & Stuart]; the saddlepoint approximation is an alternative resummation
  of the same cumulants.

---

## 7. COS inversion (Fang–Oosterlee)

We recover the distribution from $\varphi$ with the Fourier-cosine (COS) method
[Fang & Oosterlee 2008].

### 7.1 Cosine coefficients from the CF

Choose a truncation interval $[a,b]$ on which essentially all probability mass
lives (§7.3). On $[a,b]$ the density has a cosine expansion

$$
f(x)\approx\sideset{}{'}\sum_{k=0}^{N-1}A_k\cos\!\Big(k\pi\frac{x-a}{b-a}\Big),
\qquad
A_k=\frac{2}{b-a}\int_a^b f(x)\cos\!\Big(k\pi\frac{x-a}{b-a}\Big)dx,
$$

where the prime halves the $k=0$ term. Because $f$ is negligible outside $[a,b]$,
extend the integral to $\mathbb R$ and recognize the characteristic function:

$$
A_k\approx\frac{2}{b-a}\,\mathrm{Re}\!\int_{\mathbb R} f(x)\,e^{\,i k\pi\frac{x-a}{b-a}}\,dx
=\frac{2}{b-a}\,\mathrm{Re}\!\Big[e^{-i k\pi\frac{a}{b-a}}\,\varphi\!\Big(\tfrac{k\pi}{b-a}\Big)\Big].
$$

With $u_k=\dfrac{k\pi}{b-a}$ this is exactly the code:
`coeff = (2/width)*Re[ phi(u_k) * exp(-i u_k a) ]`.

### 7.2 The CDF in closed form

Integrating the cosine series term by term, $\int_a^x\cos\!\big(k\pi\frac{t-a}{b-a}\big)dt
=\frac{b-a}{k\pi}\sin\!\big(k\pi\frac{x-a}{b-a}\big)$ for $k\ge1$, and $=(x-a)$ for
$k=0$:

$$
\boxed{\,F(x)\approx \tfrac12 A_0\,(x-a)+\sum_{k=1}^{N-1}A_k\,\frac{b-a}{k\pi}\,
\sin\!\Big(k\pi\frac{x-a}{b-a}\Big)\,}
$$

with $F(x)=0$ for $x\le a$ and $F(x)=1$ for $x\ge b$ (code: `cumulant_cos_cdf`,
the `0.5*coeff[0]*(x-a)` term plus the `sin` sum). The exceedance probability is
$\mathrm{PoE}(x)=1-F(x)$ (`cos_cache.exceedance`). A final
`np.maximum.accumulate(clip(F,0,1))` removes the tiny non-monotone ripples that a
truncated cosine series can produce (worst case here $\sim2\times10^{-5}$).

### 7.3 Truncation interval and number of terms

We center the interval on the distribution using its own cumulants:

$$
[a,b]=\big[\,\kappa_1-L\sqrt{\kappa_2},\ \ \kappa_1+L\sqrt{\kappa_2}\,\big].
$$

The COS error has two parts [Fang & Oosterlee 2008]: a *truncation* error from
mass outside $[a,b]$ that decays as the tail of $f$, and a *series* error from
keeping $N$ terms that, for a CF analytic off the real axis, decays
**exponentially** in $N$. In practice $L=8$ and $N=64$ give machine-level accuracy
over the $q_{0.01}$–$q_{0.99}$ range; deeper tails want $L=10$, $N\ge96$.

### 7.4 Standardized skewness cache

Standardize $z=(x-\kappa_1)/\sqrt{\kappa_2}$. The cumulant CF of the standardized
variable is $\exp(-\tfrac12 v^2-\tfrac{i}{6}\gamma_1 v^3)$ — it depends on the
scenario **only through the skewness** $\gamma_1$. Hence the standardized CDF
$F_{\rm std}(z;\gamma_1)$ can be tabulated **once** on a $(z,\gamma_1)$ grid
(`build_standardized_cache`), and any scenario's CDF obtained by a bilinear
lookup followed by the affine map $x=\kappa_1+\sqrt{\kappa_2}\,z$. This turns each
per-scenario inversion into a table read.

---

## 8. Fragility convolution and closed-form damage probability

A fragility limit state with displacement limit $D_\ell$ is exceeded, given the
shaking, with probability $\Phi\!\big((Z-c)/s\big)$, where (from the lognormal
fragility parameters $b_0,b_1,\sigma$)

$$
c=\frac{\ln D_\ell-b_0}{b_1},\qquad s=\frac{\sigma}{b_1}.
$$

The probability of damage marginal over the shaking is
$\mathbb E_Z\big[\Phi((Z-c)/s)\big]$. Write $\Phi((Z-c)/s)=\mathbb P(s\,\varepsilon\le Z-c)
=\mathbb P(Z+s\,\varepsilon'>c)$ with $\varepsilon'\sim\mathcal N(0,1)$ independent of $Z$.
So with $U:=Z+s\,\varepsilon'$,

$$
\mathbb E[\mathrm{PoD}]=\mathbb P(U>c).
$$

$U$ is the sum of two **independent** variables, so **cumulants add**. The Gaussian
noise $s\varepsilon'$ has cumulants $(0,\,s^2,\,0)$, hence

$$
\boxed{\ \kappa_1(U)=\kappa_1,\quad \kappa_2(U)=\kappa_2+s^2,\quad \kappa_3(U)=\kappa_3\ }
$$

— the demand dispersion simply **adds $s^2$ to $\kappa_2$**, leaving the skew
untouched. Therefore $\mathbb E[\mathrm{PoD}]=\mathrm{PoE}_{\rm COS}\big(c;\ \kappa_1,\kappa_2+s^2,\kappa_3\big)$:
one COS evaluation, no resampling (used in `compare_risk_metrics.py` /
`validate_against_mc.py`).

Equivalently, at the mixture level each component convolves to
$\mathcal N(m_n,v_n+s^2)$ and, since $\mathbb P(\mathcal N(m_n,v_n+s^2)>c)
=\Phi\!\big((m_n-c)/\sqrt{v_n+s^2}\big)$,

$$
\boxed{\ \mathbb E[\mathrm{PoD}]=\sum_n w_n\,\Phi\!\Big(\frac{m_n-c}{\sqrt{v_n+s^2}}\Big)\ }
$$

(`saavg_cos.fragility`). This is the closed form the chain uses for the normal and
COS-mixture estimators; the only difference between "normal" and "COS" downstream
is whether the skew ($\kappa_3$ / the mixture spread $\delta_n$) is kept.

---

## 9. Hazard and risk aggregation

Everything downstream is **linear in the per-scenario exceedance**, so it is a
weighted sum — no resampling, and the logic tree is a final weighted average.

**Hazard.** With annual source rate $\lambda_{\rm src}(M,R)$ (here a Gutenberg–Richter
stand-in, `saavg_cos.source`), the annual rate of exceeding intensity $x$ at a zone is

$$
\lambda(x)=\sum_{M,R}\lambda_{\rm src}(M,R)\,\mathrm{PoE}\big(x\mid \text{zone},M,R\big),
$$

and the return-period level is $x_T:\ \lambda(x_T)=1/T$ (interpolated in log–log;
`make_hazard_map.py`). The 50-year probability of exceedance is $1-e^{-\lambda\cdot50}$.

**Risk.** The expected number of buildings reaching a damage state, and the local
personal risk (LPR), are rate-weighted sums of the per-scenario exceedances:

$$
\lambda_{\rm state}(\text{zone})=\sum_{M,R}\lambda_{\rm src}(M,R)\,\mathbb E[\mathrm{PoD}_{\rm state}],
\qquad
\mathrm{LPR}=\sum_{\rm CS} p_{\rm death}(\mathrm{CS})\,\lambda_{\rm CS},
$$

with $p_{\rm death}$ the consequence model. Expected counts use $1-e^{-\lambda}$ per
building (`compare_risk_metrics.py`). Logic-tree branches combine by their weights
$w_b$ ($\sum_b w_b=1$): any mean quantity is $\sum_b w_b\,(\text{quantity})_b$.

---

## 10. Approximations and error control

The method has **four** controllable approximations, all deterministic (unlike
Monte-Carlo sampling noise):

| approximation | parameter | effect / control |
|---|---|---|
| PCA rank (quadrature dimensions) | `pca_rank` $\rho$ | $\rho=2$ captures the dominant reference variance; omitted directions handled by §4.3. Higher $\rho$ costs $O^\rho$ nodes. |
| Gauss–Hermite order | `gh_order` $O$ | exact to polynomial degree $2O-1$; $O=7$ fast, $O=17$ high-accuracy. |
| cumulant truncation | (3 cumulants) | neglects $\kappa_4$ (excess kurtosis, a few %); add $\kappa_4$ for deeper tails. |
| COS truncation / terms | $L$, $N$ | tail mass + exponential series error; $L=8,N=64$ for $q_{0.01}$–$q_{0.99}$. |

Empirically (see [`report.md`](report.md)): the cumulants match deep Monte-Carlo
to $\sim10^{-3}$; the fragility-convolved exceedance probabilities deviate from MC
by sub-percent across the risk-relevant range, versus tens of percent for the
normal method; and the residual COS error is a *deterministic* three-cumulant
truncation term (worst convolved-PoE deviation $\sim4\times10^{-3}$ at the single
most-skewed cell) that shrinks with $O$ and an added $\kappa_4$ — whereas the
normal method's negative-skew bias is irreducible.

---

## 11. References

1. **F. Fang and C. W. Oosterlee** (2008). *A novel pricing method for European
   options based on Fourier-cosine series expansions.* SIAM J. Sci. Comput.
   **31**(2), 826–848. — the COS method (§7); convergence analysis.
2. **J. W. Baker and N. Jayaram** (2008). *Correlation of spectral acceleration
   values from NGA ground motion models.* Earthquake Spectra **24**(1), 299–317.
   — period-to-period correlation structure (§1).
3. **H. Crowley and R. Pinho**, and **M. Kohrangi, D. Vamvatsikos, P. Bazzurro**
   (2017). *Site-dependent and conditional average spectral acceleration (AvgSa)*
   — the SaAvg intensity measure (§1).
4. **J. J. Bommer et al.** (2017–2021). *Seismic Hazard and Risk Analysis for the
   Groningen Gas Field*, NAM/TNO technical reports — GMM-V7, FCM-V7, the SHRA
   chain (§1, §8, §9).
5. **P. McCullagh** (1987). *Tensor Methods in Statistics.* Chapman & Hall —
   cumulants and Edgeworth expansions (§5, §6).
6. **M. G. Kendall and A. Stuart**, *The Advanced Theory of Statistics, Vol. 1* —
   moments, cumulants, Gram–Charlier/Edgeworth series (§5, §6).
7. **G. H. Golub and J. H. Welsch** (1969). *Calculation of Gauss quadrature
   rules.* Math. Comp. **23**, 221–230; and **M. Abramowitz and I. A. Stegun**
   (1964), *Handbook of Mathematical Functions* §25.4 — Gauss–Hermite quadrature
   (§4.2).
8. **TNO** (2023). *SHRA-Groningen model chain* (hazard-risk-models,
   seismic-source-model), EUPL-1.2. <https://github.com/TNO> — source of the
   GMM-V7/FCM-V7 parameters used here.
