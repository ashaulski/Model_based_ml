# The First-Zone Constraint in RF PA Behavioral Modeling

*Background note for the VDTDNN / AVDTDNN implementation.*

Reference: Y. Zhang, Y. Li, F. Liu, A. Zhu, "Vector Decomposition Based Time-Delay
Neural Network Behavioral Model for Digital Predistortion of RF Power Amplifiers,"
*IEEE Access*, vol. 7, 2019.

markdown.preview.fontSize = 16

---

## 1. Why the constraint exists

A digital predistorter operates on the **complex baseband envelope** $\tilde{x}(n)$, but the
physical distortion happens to a **real passband signal** at the amplifier. Not every
baseband function corresponds to something the amplifier can actually do in-band.

The *first-zone constraint* is the condition that a baseband model only produces terms
that genuinely land in the transmit band. A model that respects it encodes known physics
for free; a model that ignores it must waste capacity rediscovering that physics from
data — and may fit artifacts that can never occur.

---

## 2. Where "zones" come from

The real passband signal is the modulated envelope

$$x_{RF}(t)=\Re\{\tilde{x}(t)e^{j\omega_c t}\}=\tfrac12\left[\tilde{x}e^{j\omega_c t}+\tilde{x}^{*}e^{-j\omega_c t}\right]$$

Passing it through a $p$-th order nonlinearity and expanding the binomial:

$$x_{RF}^{\,p}=\frac{1}{2^{p}}\sum_{q=0}^{p}\binom{p}{q}\,\tilde{x}^{\,q}\,(\tilde{x}^{*})^{\,p-q}\;e^{\,j(2q-p)\omega_c t}$$

Every term is centred at a carrier multiple

$$k=2q-p$$

These spectral clusters are the **zones**:

| $k$ | Location | Meaning |
|:---:|:---|:---|
| $0$ | DC | rectified envelope |
| $\pm1$ | $\pm f_c$ | **first zone** — the transmit band, the only one that survives filtering |
| $\pm2,\pm3,\dots$ | harmonics | removed by the bandpass filter |

Only the first zone matters for DPD.

---

## 3. Odd parity

Because $k=2q-p$, the integers $k$ and $p$ always share the same parity. Therefore

$$k=\pm1\ (\text{odd})\;\Longrightarrow\; p\ \text{must be odd}$$

**Even-order nonlinearities contribute nothing in-band.** They produce only DC and even
harmonics, all of which are filtered away. This is the *odd-parity* half of the constraint.

---

## 4. What the first zone looks like at baseband

Imposing $k=1$ fixes the binomial index to $q=\tfrac{p+1}{2}$. Writing $p=2m+1$, the
surviving baseband term is

$$\tilde{x}^{\,m+1}(\tilde{x}^{*})^{\,m}=\tilde{x}\,(\tilde{x}\tilde{x}^{*})^{m}=\tilde{x}\,|\tilde{x}|^{2m}$$

This **is** the memory-polynomial basis. It is not a modeling convention chosen for
convenience — it is what the physics leaves behind:

> one un-conjugated factor carrying the phase, everything else magnitude.

---

## 5. The constraint as phase equivariance

Apply a common phase rotation $\tilde{x}\to\tilde{x}e^{j\varphi}$. The term
$\tilde{x}^{q}(\tilde{x}^{*})^{p-q}$ acquires

$$e^{j\left(q-(p-q)\right)\varphi}=e^{jk\varphi}$$

**The zone index is exactly the phase-rotation order.** So "lies in the first zone" is
identical to "scales as $e^{j1\cdot\varphi}$", which collapses the entire constraint into
one line:

$$\boxed{\;f\!\left(\tilde{x}e^{j\varphi}\right)=e^{j\varphi}f(\tilde{x})\qquad\forall\varphi\in\mathbb{R}\;}$$

With memory, one common rotation is applied to every tap:

$$f\!\left(\tilde{x}(n)e^{j\varphi},\dots,\tilde{x}(n-M)e^{j\varphi}\right)=e^{j\varphi}\,f\!\left(\tilde{x}(n),\dots,\tilde{x}(n-M)\right)$$

This is the *unitary phase* half of the constraint. It is a testable property: rotate the
input, check that the output rotates by the same amount and nothing else changes.

---

## 6. General form: counting plain and conjugated factors

Sections 2–5 assumed a single power $x_{RF}^p$, i.e. a memoryless nonlinearity. To cover an
**arbitrary** model with memory we expand it as a sum of conjugate Volterra monomials. The
monomial is not postulated — it falls out of the expansion, as follows.

### 6.1 Where the monomial comes from

**Step 1 — a nonlinear system with memory is a Volterra series.**
Any causal, nonlinear, fading-memory system can be written

$$y_{RF}(n)=\sum_{p}\;\sum_{m_1,\dots,m_p} h_p(m_1,\dots,m_p)\;\prod_{i=1}^{p}x_{RF}(n-m_i)$$

The building block is no longer $x_{RF}^{\,p}$ but a **product of $p$ differently-delayed
copies** of the passband signal. Section 2 is the special case $m_1=\dots=m_p$.

**Step 2 — each factor splits in two.**
Because $x_{RF}$ is real its spectrum is conjugate-symmetric, so every factor is the sum of
a positive- and a negative-frequency half:

$$x_{RF}(n-m_i)=\tfrac12\Big[\underbrace{\tilde{x}(n-m_i)e^{\,j\omega_c(n-m_i)}}_{\text{plain branch}}+\underbrace{\tilde{x}^{*}(n-m_i)e^{-j\omega_c(n-m_i)}}_{\text{conjugate branch}}\Big]$$

**Step 3 — expanding the product gives $2^p$ branch choices.**
Multiplying $p$ such binomials yields $2^p$ terms. Each term corresponds to choosing,
independently for every factor, either the plain or the conjugate branch. Let
$S\subseteq\{1,\dots,p\}$ be the set of factors that took the plain branch:

$$\prod_{i=1}^{p}x_{RF}(n-m_i)=\frac{1}{2^{p}}\sum_{S}\;\prod_{i\in S}\tilde{x}(n-m_i)e^{j\omega_c(n-m_i)}\;\prod_{i\notin S}\tilde{x}^{*}(n-m_i)e^{-j\omega_c(n-m_i)}$$

Now simply *name the counts*, $q=|S|$ and $r=p-|S|$, and relabel the delays — $i_1,\dots,i_q$
for the plain factors, $j_1,\dots,j_r$ for the conjugate ones. The baseband part of one such
term is exactly

$$T=\underbrace{\tilde{x}(n-i_1)\cdots\tilde{x}(n-i_q)}_{q\ \text{plain factors}}\;\cdot\;\underbrace{\tilde{x}^{*}(n-j_1)\cdots\tilde{x}^{*}(n-j_r)}_{r\ \text{conjugated factors}}$$

so that

$$q=\#\{\text{un-conjugated factors}\},\qquad
r=\#\{\text{conjugated factors}\},\qquad
p=q+r .$$

The delays $i_1,\dots,i_q,j_1,\dots,j_r$ are free — that is what gives the model memory.

**Step 4 — the carrier exponent identifies the zone.**
Collecting the exponentials of that same term,

$$\exp\Big(j\omega_c\Big[\sum_{i\in S}(n-m_i)-\sum_{i\notin S}(n-m_i)\Big]\Big)
=\underbrace{e^{\,j(q-r)\omega_c n}}_{\text{zone}}\cdot\underbrace{e^{-j\omega_c\left(\sum_k i_k-\sum_k j_k\right)}}_{\text{constant phase}}$$

The second factor is a fixed complex constant absorbed into the kernel coefficient. The
first places the term at carrier multiple $k=q-r$ — the general-memory version of Section 2.

### 6.2 Phase order

The same count governs phase rotation. Under $\tilde{x}\to\tilde{x}e^{j\varphi}$, each plain
factor contributes $e^{+j\varphi}$ and each conjugated factor contributes $e^{-j\varphi}$:

$$T\;\longrightarrow\;e^{jq\varphi}e^{-jr\varphi}\,T=e^{\,j(q-r)\varphi}\,T$$

so the **phase order** of the monomial is $q-r$, matching the zone index found in Step 4.
This is consistent with Section 2, where $r=p-q$ and hence

$$k=2q-p=2q-(q+r)=q-r .$$

Setting all delays to zero recovers Section 2 exactly: $T=\tilde{x}^{\,q}(\tilde{x}^{*})^{\,r}$,
and the number of subsets $S$ of size $q$ is $\binom{p}{q}$ — the binomial coefficient.

### The constraint

$$\boxed{\;q-r=1\;}$$

Odd parity is now a *consequence* rather than a separate rule: substituting $q=r+1$,

$$p=q+r=2r+1 \quad\Rightarrow\quad p\ \text{is always odd.}$$

Both of the paper's stated conditions are therefore the same statement.

### Examples

| Monomial | $q$ | $r$ | $q-r$ | Zone | Allowed |
|:---|:---:|:---:|:---:|:---|:---:|
| $\tilde{x}(n)$ | 1 | 0 | $1$ | first | ✅ linear term |
| $\tilde{x}(n)\tilde{x}(n)\tilde{x}^{*}(n)=\tilde{x}\|\tilde{x}\|^{2}$ | 2 | 1 | $1$ | first | ✅ cubic MP term |
| $\tilde{x}(n)\tilde{x}(n{-}1)\tilde{x}^{*}(n{-}1)=\tilde{x}(n)\|\tilde{x}(n{-}1)\|^{2}$ | 2 | 1 | $1$ | first | ✅ GMP lagging term |
| $\tilde{x}(n)\tilde{x}(n)$ | 2 | 0 | $2$ | 2nd harmonic | ❌ |
| $\tilde{x}(n)\tilde{x}^{*}(n)=\|\tilde{x}\|^{2}$ | 1 | 1 | $0$ | DC | ❌ |
| $\tilde{x}^{*}(n)$ | 0 | 1 | $-1$ | image | ❌ |

In every allowed row the conjugates **pair off** with plain factors to form magnitudes
$\tilde{x}\tilde{x}^{*}=|\tilde{x}|^{2}$, leaving exactly **one unpaired plain factor** to
carry the phase.

### How the delay choice generates each term

Fixing the branch counts still leaves the delays free, and that is what distinguishes the
various admissible models. Taking $p=3$ with $q=2,\ r=1$ (so $q-r=1$, first zone):

| $(i_1,i_2;\,j_1)$ | $T$ | Name |
|:---|:---|:---|
| $(0,0;\,0)$ | $\tilde{x}(n)^{2}\tilde{x}^{*}(n)=\tilde{x}(n)\lvert\tilde{x}(n)\rvert^{2}$ | cubic memory-polynomial term |
| $(0,1;\,1)$ | $\tilde{x}(n)\tilde{x}(n{-}1)\tilde{x}^{*}(n{-}1)=\tilde{x}(n)\lvert\tilde{x}(n{-}1)\rvert^{2}$ | GMP lagging cross-term |
| $(0,1;\,0)$ | $\tilde{x}(n)\tilde{x}(n{-}1)\tilde{x}^{*}(n)$ | admissible, but not a pure magnitude |

The third row is first-zone compliant yet is *not* of the form
$\tilde{x}\cdot|\tilde{x}|^{k}$. GMP is therefore a **strict subset** of the admissible
first-zone terms: it additionally requires each conjugate to pair with a plain factor at the
*same* delay, collapsing every term to a magnitude times a phase-carrying tap.

### Equivalent closed form

A model satisfies the first-zone constraint if and only if it can be written as

$$f=\sum_{l}\underbrace{\tilde{x}(n-l)}_{\text{phase-carrying tap}}\;
\underbrace{G_l\!\left(|\tilde{x}(n)|,\dots,|\tilde{x}(n-M)|\right)}_{\text{phase-blind complex gain}}$$

a magnitude-dependent complex gain applied to a phase-carrying tap.

---

## 7. Consequences for DPD architectures

### GMP satisfies it structurally

The GMP basis term

$$\tilde{x}(n-l)\,|\tilde{x}(n-l-m)|^{k}$$

is compliant by construction: magnitudes are rotation-invariant, and the single $\tilde{x}$
factor supplies exactly one $e^{j\varphi}$. Here $G_l$ is a **fixed polynomial**.

### VDTDNN is the learned generalization

In the vector-decomposed network the hidden activation

$$A_g=h\!\left(\sum_{m=0}^{M} w'_{g,m}\,|\tilde{x}(n-m)|+b_g\right)$$

depends on magnitudes only, hence is phase-invariant; multiplying by $e^{j\theta_{n-l}}$ then
supplies exactly one unit of phase. So $A_g$ plays the role of $G_l$ — a **learned**
nonlinear gain replacing the fixed polynomial $|\tilde{x}|^{k}$. This is precisely why the
architecture is more parameter-efficient than a generic network: the physics is built in,
not learned.

### Why the output layer has no bias

If a constant offset $b$ were added,

$$f=f_{\text{eq}}+b\;\Longrightarrow\;f(\tilde{x}e^{j\varphi})=e^{j\varphi}f_{\text{eq}}+b\neq e^{j\varphi}f
\quad\text{unless } b=0 .$$

A bias is a $q-r=0$ (DC-zone) term, which cannot physically exist in-band. This is the
paper's "nonphysical contribution."

### Why RVTDNN violates it

A generic MLP $g:\mathbb{R}^2\to\mathbb{R}^2$ acting on $(I,Q)$ has no reason to commute
with the rotation matrix,

$$g(R_\varphi v)\neq R_\varphi\,g(v),$$

so the constraint must be **learned** from data. Capacity is spent rediscovering known
physics, and the model generalizes poorly to phase configurations absent from training.
This is the paper's core argument for the vector-decomposed structure.

---

## 8. Note on the parameterization

The paper's output layer uses **four free weights per phase-recovery unit**
$(w''_1,w''_2,w''_3,w''_4)$, which is where its coefficient count $4(G+M+1)$ comes from.
Decomposing that layer, each unit contributes

$$\alpha\,e^{j\theta}+\beta\,e^{-j\theta},
\qquad
\alpha=\frac{(w''_1+w''_4)+j(w''_3-w''_2)}{2},
\qquad
\beta=\frac{(w''_1-w''_4)+j(w''_3+w''_2)}{2}$$

The $e^{-j\theta}$ component is the $q-r=-1$ **image** term from the table in Section 6.
It vanishes if and only if

$$w''_1=w''_4,\qquad w''_3=-w''_2$$

which is exactly the paper's stated "ideal corresponding relationship"
$w''_1=w''_4=a_{gI},\ w''_2=-a_{gQ},\ w''_3=a_{gQ}$.

Importantly, this relationship is **not imposed structurally** — it is left to training to
discover. A model with four free weights per unit therefore spans both $q-r=+1$ and
$q-r=-1$, and is only approximately first-zone compliant.

Tying the weights (equivalently, using a single complex coefficient
$\tilde{a}_g=a_{gI}+ja_{gQ}$ per unit) enforces the constraint **exactly**, at half the
output-layer parameters.

> **Measured in this repository** (M=2, 12 neurons, AVDTDNN), using the relative error
> $\max\big|f(xe^{j\varphi})-e^{j\varphi}f(x)\big|/\max|f(x)|$ with $\varphi=0.7$:
>
> | Variant | Output params | Equivariance error |
> |:---|:---:|:---:|
> | GMP | — | $1.2\times10^{-16}$ |
> | Free weights (paper) | 180 total | $7.0\times10^{-1}$ |
> | Tied weights | 150 total | $1.2\times10^{-7}$ (float32 exact) |
> | GRU | — | $6.9\times10^{-1}$ |
>
> Both variants reached comparable accuracy ($-30.5$ dB vs $-29.7$ dB validation NMSE).

### Edge caveat

The first $M$ output samples are never exactly equivariant, because delayed taps are
zero-padded and $\arg(0)=0$ does not rotate with $\varphi$. Discard the first $M$ samples
when testing or evaluating.

---

## 9. Summary

| Statement | Form |
|:---|:---|
| Zone index | $k=q-r$ |
| First-zone (unitary phase) | $q-r=1$ |
| Odd parity | $p=2r+1$, implied by $q-r=1$ |
| Equivalent test | $f(\tilde{x}e^{j\varphi})=e^{j\varphi}f(\tilde{x})$ |
| Admissible model form | $f=\sum_l \tilde{x}(n-l)\,G_l(\lvert\tilde{x}\rvert\text{-taps})$ |
| Forbidden | output bias (DC), conjugate/image terms, even orders |
