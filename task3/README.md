# Task III: Open Task

## My View on Quantum Computing and QML

My current view is that quantum computing is not yet a replacement for classical ML pipelines, but it is already useful as a research tool for learning better representations of hard optimization structure. I am especially interested in the “middle zone” between pure quantum speedup claims and pure classical baselines: hybrid methods where quantum circuits are used as trainable feature maps and classical optimizers handle stability.

From my project experience, I think the biggest practical challenge is not writing a quantum circuit; it is making training reliable. In small experiments, I repeatedly saw that optimization quality depends more on ansatz choice, initialization, and shot budget than on the nominal algorithm label. This is why I prefer discussing concrete training behavior (convergence curves, variance across seeds, robustness to noise) instead of only gate-level novelty.

In short, I see QML as promising, but only when we treat it like an optimization-and-generalization problem, not just a quantum-circuit construction problem.

## One Algorithm I Know: QAOA

I am familiar with QAOA because it is conceptually clean and directly tied to combinatorial structure. The part I find most useful is the explicit mapping from a classical objective to a cost Hamiltonian: it makes it easier to reason about what the circuit is trying to optimize.

What I like about QAOA:
- Clear link between objective function and quantum evolution.
- Natural compatibility with warm-start ideas from classical solvers.
- Good playground for studying trainability under realistic noise.

What I find limiting in practice:
- Performance can degrade quickly with depth due to noise and parameter sensitivity.
- Good parameters are often highly instance-dependent.
- “Better than baseline” is hard to show unless the classical baseline is strong and well-tuned.

If I work on QAOA further, I would focus on robust parameter-transfer strategies (for example, learning initial angles from graph statistics) rather than only increasing depth.

## One Quantum Software Stack I Use: Qiskit

Qiskit is the framework I am most comfortable with for rapid prototyping. I like it because it supports a full workflow in one place: circuit design, transpilation, noise-aware simulation, and backend execution. For research, this end-to-end loop is valuable because it lets me iterate quickly from idea to measurable result.

A practical point from my own usage: transpilation choices can materially change outcomes for the same high-level circuit. Because of that, I treat compilation settings as part of the experiment design, not as a post-processing detail.

## Methods I Would Like to Work On

### 1) Meta-learned initialization for variational quantum circuits
I want to study whether a meta-learning outer loop can learn parameter initializations that reduce optimization time across related tasks. My main metric would be “iterations to target AUC/loss” averaged across multiple seeds.

### 2) Hybrid optimization pipelines
I would combine classical pre-optimization with quantum fine-tuning, especially for structured objectives. The goal is to reduce random initialization sensitivity while preserving any quantum advantage in representation.

### 3) Error-aware training rather than error-afterthought evaluation
Instead of training noise-free and only evaluating with noise, I would include realistic noise during training and test mitigation strategies (readout mitigation, simple zero-noise extrapolation) using calibration-aware reporting.

### 4) Benchmark discipline for QML claims
I want to enforce stricter comparisons: same data splits, same budget, multiple seeds, and strong classical baselines. I believe this is essential if we want credible progress in QML for scientific applications.

## A Concrete First Plan I Would Execute

As a first scoped study, I would run a hybrid QAOA-style experiment on a small family of graph optimization instances:
- Baselines: tuned simulated annealing + tuned gradient-based classical solver.
- Quantum side: shallow-depth variational circuit with and without meta-initialization.
- Metrics: best objective value, wall-clock to target, variance over 10 random seeds.

If the hybrid/meta-initialized setup consistently reaches the same target with lower variance and fewer iterations, I would consider that a meaningful step forward.

This is the direction I want to pursue: practical, measurable improvements in trainability and reliability, not only theoretical possibility.