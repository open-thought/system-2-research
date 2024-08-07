# OpenThought - System 2 Research Links

Here you find a collection of material (books, papers, blog-posts etc.) related to reasoning and cognition in AI systems. Specifically we want to cover agents, cognitive architectures, general problem solving strategies and self-improvement.

The term "System 2" in the page title refers to the slower, more deliberative, and more logical mode of thought as described by Daniel Kahneman in his book [Thinking, Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow).

You know a great resource we should add? Please see [How to contribute](#how-to-contribute).


## Cognitive Architectures

(looking for additional links & articles and summaries)

- [SOAR](https://en.wikipedia.org/wiki/Soar_(cognitive_architecture)) by John Laird, Allen Newell, and Paul Rosenbloom
resonance theory) by Stephen Grossberg and Gail Carpenter 
- [ACT-R](https://en.wikipedia.org/wiki/ACT-R) (Adaptive Control of Thought-Rational) by John Anderson at CMU
- [SPAUN](https://www.nengo.ai/nengo-spa/user-guide/spa-intro.html) (Semantic Pointer Architecture Unified Network) by Chris Eliasmith at Waterloo
- [SPAUN 2.0](https://core.ac.uk/download/pdf/158325694.pdf) by Feng-Xuan Choo
- [ART](https://en.wikipedia.org/wiki/Adaptive_resonance_theory) (Adaptive 
- [CLARION](https://en.wikipedia.org/wiki/CLARION_(cognitive_architecture)) (Connectionist Learning with Adaptive Rule Induction ON-line) by Ron Sun
- [EPIC](https://web.eecs.umich.edu/~kieras/epic.html) (Executive Process/Interactive Control) by David Kieras and David Meyer
- [LIDA](https://en.wikipedia.org/wiki/LIDA_(cognitive_architecture)) (Learning Intelligent Distribution Agent) by Stan Franklin
- [Sigma](https://ict.usc.edu/pubs/The%20Sigma%20cognitive%20architecture%20and%20system.pdf) by Paul Rosenbloom
- [OpenCog](https://opencog.org/) by Ben Goertzel
- [NARS](https://cis.temple.edu/~pwang/NARS-Intro.html) (Non-Axiomatic Reasoning System) by Pei Wang
- [Icarus](http://www.isle.org/~langley/papers/icarus.csr17.pdf) by Pat Langley
- [MicroPsi](http://cognitive-ai.com/publications/assets/MicroPsiArchitectureICCM03.pdf) by Joscha Bach
- [Thousand Brains Theory](https://www.numenta.com/blog/2019/01/16/the-thousand-brains-theory-of-intelligence/) & [HTM](https://www.numenta.com/resources/htm/) (Hierarchical Temporal Memory) by Jeff Hawkins
- [SPH](https://ogma.ai/sph-technology-description/) (Sparse Predictive Hierarchie) by Eric Laukien
- [Leabra](https://github.com/emer/leabra), [2016 Paper](https://ccnlab.org/papers/OReillyHazyHerd16.pdf) by [Randall O'Reilly](https://ccnlab.org/people/oreilly/)
- [CogNGen](https://arxiv.org/abs/2310.15177), see also [here](https://osf.io/preprints/osf/cew42) and [here](https://arxiv.org/abs/2204.00619)


## Agent Papers

### LLM Based

- [OpenDevin: An Open Platform for AI Software Developers as Generalist Agents](https://arxiv.org/abs/2407.16741)
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)
- [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Agentless: Demystifying LLM-based Software Engineering Agents](https://arxiv.org/abs/2407.01489)
- [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814)
- [AI Agents That Matter](https://arxiv.org/pdf/2407.01502)
- [Sibyl: Simple yet Effective Agent Framework for Complex Real-world Reasoning](https://arxiv.org/abs/2407.10718)
- [Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents](https://arxiv.org/abs/2403.02502)
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)
- [ArchCode: Incorporating Software Requirements in Code Generation with Large Language Models](https://arxiv.org/abs/2408.00994)
- MedAgent-Zero: [Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents](https://arxiv.org/abs/2405.02957)
- [Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620)
- [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)


#### Massive Sampling / Generate-and-Test

- [Large Language Monkeys: Scaling Inference Compute
with Repeated Sampling](https://arxiv.org/abs/2407.21787)
- [AlphaCode 2 Technical Report](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf)


### World Models

- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080)
- Latent space world-models: [Dreamer](http://arxiv.org/pdf/1912.01603), [V2](https://arxiv.org/abs/2010.02193), [V3](https://arxiv.org/abs/2301.04104), [DayDreamer](https://arxiv.org/pdf/2206.14176)
- [World Models](https://arxiv.org/abs/1803.10122), web: [project page](https://worldmodels.github.io/)


### Neuro-Symbolic Approaches

- [HYSYNTH: Context-Free LLM Approximation for Guiding Program Synthesis](https://arxiv.org/abs/2405.15880)
- [SymbolicAI: A framework for logic-based approaches combining generative models and solvers](https://arxiv.org/abs/2402.00854)
- [DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning](https://arxiv.org/abs/2006.08381)


### Active Inference

- [From pixels to planning: scale-free active inference](https://arxiv.org/abs/2407.20292)


## Prompting Techniques

- Surveys: 
   - (Feb 2024) [A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications](https://arxiv.org/abs/2402.07927)
   - Prompt Engineering Guide [Prompting Techniques](https://www.promptingguide.ai/techniques)
- Chain-of-Thoughts (COT): [Paper](https://arxiv.org/abs/2201.11903)
- Tree-of-Thoughts (ToT): [Paper](https://arxiv.org/pdf/2305.10601)
- Graph-of-Thoughts (GoT): [Paper](https://arxiv.org/abs/2308.09687), [code](https://github.com/spcl/graph-of-thoughts)
- Algorithm of Thoughts (AoT): [Paper](https://arxiv.org/abs/2308.10379)
- Chain-of-Verification (CoVe/CoV): [Paper](https://arxiv.org/abs/2309.11495)
- Mixture-of-Agents (MoA): [Paper](https://arxiv.org/abs/2406.04692)


## Blog Posts

- DeepMind [AlphaProof and AlphaGeometry 2](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)
- [Getting 50% (SoTA) on ARC-AGI with GPT-4o](https://www.lesswrong.com/posts/Rdwui3wHxCeKb7feK/getting-50-sota-on-arc-agi-with-gpt-4o)
- Schmidhuer: [Artificial Curiosity & Creativity](https://people.idsia.ch/~juergen/artificial-curiosity-since-1990.html)
- synthesis.ai: [Do Androids Dream? World Models in Modern AI](https://synthesis.ai/2024/07/02/do-androids-dream-world-models-in-modern-ai/)


## Graph Neural Networks

- Distill [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) (2021)
- [Geometric Deep Learning - Grids, Groups, Graphs, Geodesics, and Gauges](https://geometricdeeplearning.com/)


## Algorithms

- [generate-and-test](https://www.geeksforgeeks.org/generate-and-test-search/)
- [wake-sleep](https://en.wikipedia.org/wiki/Wake-sleep_algorithm)
- [novelty-search](https://algorithmafternoon.com/novelty/novelty_search_algorithm/)
- [monte-carlo-tree-search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)


## Books

- [The Soar Cognitive Architecture](https://a.co/d/cXo8KFu), John E. Laird, MIT Press, 2019
- [How to Build a Brain: A Neural Architecture for Biological Cognition](https://academic.oup.com/book/6263) Chris Eliasmith, Oxford Series on Cognitive Models and Architectures, 2013
- [Active Inference: The Free Energy Principle in Mind, Brain, and Behavior](https://a.co/d/4OAX5dD), Thomas Parr, Giovanni Pezzulo, Karl J. Friston, MIT Press, 2022, [MLST Interview with Thomas Parr](https://www.youtube.com/watch?v=bk_xCikDUDQ)
- [Principles of Synthetic Intelligence PSI: An Architecture of Motivated Cognition](https://amzn.eu/d/46Xwq4s), Joscha Bach, Oxford Series on Cognitive Models and Architectures Book 4, 2009
- [Conscious Mind, Resonant Brain: How Each Brain Makes a Mind](https://a.co/d/5Hl3n7H), Stephen Grossberg, Oxford University Press, 2021
- [The Society of Mind](https://www.amazon.com/Society-Mind-Marvin-Minsky/dp/0671657135), Marvin Minsky, Simon & Schuster, 1986


## Biologically Inspired Approaches

- Ogma Sparse Predictive Hierarchies (SPH): [whitepaper](https://ogma.ai/sph-technology-description/)
- [The Tolman-Eichenbaum Machine: Unifying space and relational memory through generalisation in the hippocampal formation](https://www.biorxiv.org/content/10.1101/770495v2) (TEM), [TEM-t](https://arxiv.org/abs/2112.04035)


## Software Tools & Libraries

- [paul-gauthier/aider](https://github.com/paul-gauthier/aider)
- [OpenDevin](https://github.com/OpenDevin/OpenDevin)
- [princeton-nlp/SWE-agent](https://github.com/princeton-nlp/SWE-agent), [documentation](https://princeton-nlp.github.io/SWE-agent/)
- [meta-llama/llama-agentic-system](https://github.com/meta-llama/llama-agentic-system)
- [stanfordnlp/dspy]( https://github.com/stanfordnlp/dspy)
- [InternLM/lagent](https://github.com/InternLM/lagent) - lightweight framework for building LLM-based agents


## Commercial Offerings

- Software Engineering
  - [Devin](https://preview.devin.ai/)
  - [Cursor](https://www.cursor.com/)
  - [copilot-workspace](https://githubnext.com/projects/copilot-workspace)
  - [textgrad](https://textgrad.com/)


# Competitions & Benchmarks

- [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854), web: [project page](https://webarena.dev/), [Leaderboard](https://docs.google.com/spreadsheets/d/1M801lEpBbKSNwP-vDBkC_pF7LdyGU1f_ufZb_NWNBZQ/edit?usp=sharing)
- [ARC-AGI](https://arcprize.org/arc): [Leaderboad](https://arcprize.org/leaderboard), [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547)
- PlanBench: [Paper](https://arxiv.org/abs/2206.10498), gh: [karthikv792/LLMs-Planning](https://github.com/karthikv792/LLMs-Planning)


## Related Projects

- Nous Research [Open Reasoning Tasks](https://reasoning.nousresearch.com/), a list of reasoning tasks, gh: [NousResearch/Open-Reasoning-Tasks](https://github.com/NousResearch/Open-Reasoning-Tasks)


## Youtube Content

- [Understanding Human Learning and Abstraction Using Cognitive Models & Neural Networks](https://youtu.be/k_9nMt4V8nk) Maria Eckstein (DeepMind)
- [Consciousness as a coherence-inducing operator](https://www.youtube.com/watch?v=qoHCQ1ozswA) Talk by Josha Bach at the [Models of Consciousness Conferences](https://models-of-consciousness.org/)
- [David Shapiro](https://www.youtube.com/@DaveShap)
- [MIT AGI: Cognitive Architecture (Nate Derbinsky)](https://www.youtube.com/watch?v=bfO4EkoGh40)


## Best LLM APIs

- [Anthropic Claude](https://docs.anthropic.com/en/api/getting-started)


## How to contribute

 To share a link related to reasoning in AI systems that is missing here please create a pull request for this file. See [editing files](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files) in the github documentation.
