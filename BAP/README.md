# Introduction
Babble and Prune [^1] was introduced by *alkjash* in a five part series of posts on LessWrong. It is a general technique to fine-tune or improve sequential generation models. Roughly speaking it works by first producing a lot of possibilities using a weak, local filter (Babbling) and then prune away the bad ones using a stronger and global filter (Pruning).

In his posts, the author describes this concept at a more philosophical level and connects it to wide-ranging phenomena, from babies learning how to talk and struggling to come up with a good word for Scrabble to *treppenwitze* and writing poetry. The process of babbling is envisioned as traversing a massive implicitly represented Babble graph and pruning is done based on two measures: value, which means you want nodes that are relevant, and connectivity, which helps the babbler not get trapped locally.

The posts are well worth reading, but not intended to be a ML treatise and as such the author doesn't go into too much technical detail.

We would like to apply babble and prune GPT type language models. Babbling in this context would simply mean that we let the model generate a number of different sentences from a single prompt and then pruning would be done by either an external reward model or by the language model itself. We expect the latter case to work because it is much easier to judge whether something is good, than it is to actually make it. $N$ not being equal to $NP$ and all that.

# Cross-Entropy Method

For the cross-entropy method, we want to optimise the change that some value function S(X), where X has a density function f(x) is above a threshold <img src="https://render.githubusercontent.com/render/math?math=\gamma">
<img src="https://render.githubusercontent.com/render/math?math=L = P(S(X) \geq \gamma) = E_{X\sim f(X)}[I[S(X) \geq \gamma]]">
The optimal distribution $f_\text{opt}(X)$ is proportional to $I[S(X) \geq \gamma]f(X)$
\begin{align}
    f_\text{opt}(X) = \frac{I[S(X) \geq \gamma]f(X)}{\mathcal{N}}
\end{align}
This distribution will be approximated by $g(X; u)$, which is found by minimising the kullback-leibler divergence
\begin{align}
    D(f_\text{opt}, g) &= E_g[\log \frac{g}{f_\text{opt}}]\\
    &= \frac{1}{\mathcal{N}}\int dx\ I[S(X) \geq \gamma]f(X) (\log f_\text{opt} - \log \mathcal{N} - \log g(x; u))\\
    &\sim -\int dx\ I[S(X) \geq \gamma]f(X) \log g(x; u)\\
    &\approx -\frac{1}{N}\sum_{i=1}^N I[S(x_i) \geq \gamma]\log g(x_i; u),
\end{align}
where $\sim$ denotes equivalence of optima.

[^1] https://www.lesswrong.com/s/pC6DYFLPMTCbEwH8W
