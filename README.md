# OpenThought - System 2 Research Links

Here you find a collection of material (books, papers, blog-posts etc.) related to reasoning and cognition in AI systems. Specifically we want to cover agents, cognitive architectures, general problem solving strategies and self-improvement.

The term "System 2" in the page title refers to the slower, more deliberative, and more logical mode of thought as described by Daniel Kahneman in his book [Thinking, Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow).

You know a great resource we should add? Please see [How to contribute](#how-to-contribute).


## Cognitive Architectures

(looking for additional links & articles and summaries)

- [SOAR](https://en.wikipedia.org/wiki/Soar_(cognitive_architecture)) (State, Operator, And Result) by John Laird, Allen Newell, and Paul Rosenbloom
- [ACT-R](https://en.wikipedia.org/wiki/ACT-R) (Adaptive Control of Thought-Rational) by John Anderson at CMU
- [SPAUN](https://www.nengo.ai/nengo-spa/user-guide/spa-intro.html) (Semantic Pointer Architecture Unified Network) by Chris Eliasmith at Waterloo, [SPAUN 2.0](https://core.ac.uk/download/pdf/158325694.pdf) by Feng-Xuan Choo
- [ART](https://en.wikipedia.org/wiki/Adaptive_resonance_theory) (Adaptive resonance theory) by Stephen Grossberg and Gail Carpenter
- [CLARION](https://en.wikipedia.org/wiki/CLARION_(cognitive_architecture)) (Connectionist Learning with Adaptive Rule Induction ON-line) by Ron Sun
- [EPIC](https://web.eecs.umich.edu/~kieras/epic.html) (Executive Process/Interactive Control) by David Kieras and David Meyer
- [LIDA](https://en.wikipedia.org/wiki/LIDA_(cognitive_architecture)) (Learning Intelligent Distribution Agent) by Stan Franklin, [2015 Paper](https://digitalcommons.memphis.edu/cgi/viewcontent.cgi?article=1023&context=ccrg_papers)
- [Sigma](https://ict.usc.edu/pubs/The%20Sigma%20cognitive%20architecture%20and%20system.pdf) by Paul Rosenbloom
- [OpenCog](https://opencog.org/) by Ben Goertzel
- [NARS](https://cis.temple.edu/~pwang/NARS-Intro.html) (Non-Axiomatic Reasoning System) by Pei Wang
- [Icarus](http://www.isle.org/~langley/papers/icarus.csr17.pdf) by Pat Langley
- [MicroPsi](http://cognitive-ai.com/publications/assets/MicroPsiArchitectureICCM03.pdf) by Joscha Bach
- [Thousand Brains Theory](https://www.numenta.com/blog/2019/01/16/the-thousand-brains-theory-of-intelligence/) & [HTM](https://www.numenta.com/resources/htm/) (Hierarchical Temporal Memory) by Jeff Hawkins
- [SPH](https://ogma.ai/sph-technology-description/) (Sparse Predictive Hierarchie) by Eric Laukien
- [Leabra](https://github.com/emer/leabra) (Local, Error-driven and Associative, Biologically Realistic Algorithm), [2016 Paper](https://ccnlab.org/papers/OReillyHazyHerd16.pdf) by [Randall O'Reilly](https://ccnlab.org/people/oreilly/)
- [CogNGen](https://arxiv.org/abs/2310.15177) (COGnitive Neural GENerative system) by Alexander Ororbia and Mary Alexandria Kelly, see also [here](https://osf.io/preprints/osf/cew42) and [here](https://arxiv.org/abs/2204.00619)
- [KIX](https://arxiv.org/abs/2402.05346) (KIX: A Metacognitive Generalization Framework) by A. Kumar and Paul Schrater
- [ACE](https://arxiv.org/abs/2310.06775) (Autonomous Cognitive Entity) by David Shapiro et al., gh: [daveshap/ACE_Framework](https://github.com/daveshap/ACE_Framework)
- [Iterative Updating of Working Memory](https://arxiv.org/abs/2203.17255) by Jared Reser, [website](https://aithought.com/), [Video](https://youtu.be/R2H2Pl0I6EA?si=DlO0j-WxhG5TaJeN)


## Agent Papers

### LLM Based
- 06 Jan 2025  [Large language models for artificial general intelligence (AGI): A survey of foundational principles and approaches](https://arxiv.org/abs/2501.03151)
- Nov 2024  [LLaVA-o1: Let Vision Language Models Reason Step-by-Step](https://arxiv.org/html/2411.10440v1)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)
- [The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery](https://arxiv.org/abs/2408.06292), gh: [SakanaAI/AI-Scientist](https://github.com/SakanaAI/AI-Scientist)
- [OpenDevin: An Open Platform for AI Software Developers as Generalist Agents](https://arxiv.org/abs/2407.16741)
- [TextGrad: Automatic "Differentiation" via Text](https://arxiv.org/abs/2406.07496)
- [Trace is the New AutoDiff -- Unlocking Efficient Optimization of Computational Workflows](https://arxiv.org/abs/2406.16218)
- [Agentless: Demystifying LLM-based Software Engineering Agents](https://arxiv.org/abs/2407.01489)
- [Competition-Level Code Generation with AlphaCode](https://arxiv.org/abs/2203.07814)
- [AI Agents That Matter](https://arxiv.org/pdf/2407.01502)
- [Sibyl: Simple yet Effective Agent Framework for Complex Real-world Reasoning](https://arxiv.org/abs/2407.10718)
- [Trial and Error: Exploration-Based Trajectory Optimization for LLM Agents](https://arxiv.org/abs/2403.02502)
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)
- [ArchCode: Incorporating Software Requirements in Code Generation with Large Language Models](https://arxiv.org/abs/2408.00994)
- MedAgent-Zero: [Agent Hospital: A Simulacrum of Hospital with Evolvable Medical Agents](https://arxiv.org/abs/2405.02957)
- [Cognitive Architectures for Language Agents](https://arxiv.org/abs/2309.02427)
- [Large Language Models Can Self-Improve At Web Agent Tasks](https://arxiv.org/abs/2405.20309)
- [AgentGen: Enhancing Planning Abilities for Large Language Model based Agent via Environment and Task Generation](https://arxiv.org/abs/2408.00764)
- [A Prefrontal Cortex-inspired Architecture for Planning in Large Language Models](https://arxiv.org/abs/2310.00194)
- [CLIN: A Continually Learning Language Agent for Rapid Task Adaptation and Generalization](https://arxiv.org/abs/2310.10134)
- [DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for Reinforcement Learning and Monte-Carlo Tree Search](https://arxiv.org/abs/2408.08152)
- GPT-Swarm [Language Agents as Optimizable Graphs](https://arxiv.org/abs/2402.16823)
- Survey: [Reasoning with Large Language Models, a Survey](https://arxiv.org/abs/2407.11511) (Jul 2024)
- Survey: [From LLMs to LLM-based Agents for Software Engineering: A Survey of Current, Challenges and Future](https://arxiv.org/abs/2408.02479) (Aug 2024)
- ADAS: [Automated Design of Agentic Systems](https://arxiv.org/abs/2408.08435)
- [IDEA:Enhancing the Rule Learning Ability of Language Agents through Induction, Deduction, and Abduction](https://arxiv.org/abs/2408.10455)
- LAW: [Language Models, Agent Models, and World Models: The LAW for Machine Reasoning and Planning](https://arxiv.org/abs/2312.05230)
- GenRM: [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240)
- Perspective: [Towards Building Specialized Generalist AI with System 1 and System 2 Fusion](https://arxiv.org/abs/2407.08642)
- CodeAct: [Executable Code Actions Elicit Better LLM Agents](https://arxiv.org/abs/2402.01030)
- PLANSEARCH: [Planning In Natural Language Improves LLM Search For Code Generation](https://arxiv.org/abs/2409.03733)
- [LLMs Still Can't Plan; Can LRMs? A Preliminary Evaluation of OpenAI's o1 on PlanBench](https://arxiv.org/abs/2409.13373)
- [Thinking LLMs: General Instruction Following with Thought Generation](https://arxiv.org/abs/2410.10630)
- [Agent S: An Open Agentic Framework that Uses Computers Like a Human](https://arxiv.org/abs/2410.08164)

### LLM Reasoning Improvements / Training on Synthetic Data

- 05 Feb 2025  LIMO: [Less is More for Reasoning](https://arxiv.org/abs/2502.03387)
- 04 Feb 2025  Satori: [Reinforcement Learning with Chain-of-Action-Thought Enhances LLM Reasoning via Autoregressive Search](https://arxiv.org/abs/2502.02508)
- 31 Jan 2025  s1: [Simple test-time scaling](https://arxiv.org/abs/2501.19393)
- 08 Jan 2025  MetaCoT: [Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought](https://arxiv.org/abs/2501.04682)
- 05 Jan 2025  [Test-time Computing: from System-1 Thinking to System-2 Thinking](https://arxiv.org/abs/2501.02497)
- 30 Dec 2024  [Aviary: training language agents on challenging scientific tasks](https://arxiv.org/abs/2412.21154) - expert iteration & rejection sampling
- 25 Dec 2024  [HuatuoGPT-o1, Towards Medical Complex Reasoning with LLMs](https://arxiv.org/pdf/2412.18925)
- 18 Dec 2024  [A Survey on LLM Inference-Time Self-Improvement](https://arxiv.org/abs/2412.14352)
- 12 Dec 2024  STILL-2: [Imitate, Explore, and Self-Improve: A Reproduction Report on Slow-thinking Reasoning Systems](https://arxiv.org/abs/2412.09413)
- 02 Dec 2024  [Mastering Board Games by External and Internal Planning with Language Models](https://arxiv.org/abs/2412.12119)
- 29 Nov 2024  [Critical Tokens Matter: Token-Level Contrastive Estimation Enhances LLM's Reasoning Capability](https://arxiv.org/abs/2411.19943)
- 14 Oct 2024  TPO: [Thinking LLMs: General Instruction Following with Thought Generation](https://arxiv.org/abs/2410.10630)
- 4 Oct 2024  SWAP: [Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model](https://arxiv.org/abs/2410.03136)
- [Thinking LLMs: General Instruction Following with Thought Generation](https://arxiv.org/abs/2410.10630)
- [Source2Synth: Synthetic Data Generation and Curation Grounded in Real Data Sources](https://arxiv.org/abs/2409.08239)
- [Chain of Thought Imitation with Procedure Cloning](https://arxiv.org/abs/2205.10816)
- [Q*: Improving Multi-step Reasoning for LLMs with Deliberative Planning](https://arxiv.org/abs/2406.14283)
- [Self-Discover: Large Language Models Self-Compose Reasoning Structures](https://arxiv.org/abs/2402.03620)
- TRICE: [Training Chain-of-Thought via Latent-Variable Inference](https://arxiv.org/abs/2312.02179)
- [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629)
- Self-Taught Reasoner: [STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465)
- Self-Notes: [Learning to Reason and Memorize with Self-Notes](https://arxiv.org/abs/2305.00833)
- [From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step](https://arxiv.org/abs/2405.14838)
- LaTRO: [Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding](https://arxiv.org/abs/2411.04282), [code](https://github.com/SalesforceAIResearch/LaTRO)

### Direct o1 Replication Efforts
- [Unsloth GRPO](https://unsloth.ai/blog/r1-reasoning): Train your own R1 reasoning model with Unsloth (GRPO)
- [HF open-r1](https://huggingface.co/open-r1): A fully open reproduction of DeepSeek-R1, gh: [huggingface/open-r1](https://github.com/huggingface/open-r1)
- [TinyZero](https://github.com/Jiayi-Pan/TinyZero): A reproduction of DeepSeek R1 Zero in countdown and multiplication tasks. x: [thread](https://x.com/jiayi_pirate/status/1882839370505621655)
- DeepSeek R-1: (https://chat.deepseek.com/), Tech report: [deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1)
- OpenR: [Technical Report](https://github.com/openreasoner/openr/blob/main/reports/OpenR-Wang.pdf), [Project Page](https://openreasoner.github.io/), code: [openreasoner/openr](https://github.com/openreasoner/openr)
- [GAIR-NLP/O1-Journey](https://github.com/GAIR-NLP/O1-Journey), [O1 Replication Journey: Strategic Progress Report - Part 1](https://arxiv.org/abs/2410.18982)
- [OpenSource-O1/Open-O1](https://github.com/OpenSource-O1/Open-O1)
- [bklieger-groq/g1](https://github.com/bklieger-groq/g1): Using Llama-3.1 70b on Groq to create o1-like reasoning chains
- [ack-sec/toyberry](https://github.com/ack-sec/toyberry): Atlas Reasoning System (Toyberry)
- Blog: [Reverse engineering OpenAI’s o1 ](https://www.interconnects.ai/p/reverse-engineering-openai-o1) by [Nathan Lambert](https://twitter.com/natolambert)

### Reward Models (ORM/PRM)
- 13 Jan 2025  QwQ PRM: [The Lessons of Developing Process Reward Models in Mathematical Reasoning](https://arxiv.org/abs/2501.07301) - consensus of MC estimation & LLM-as-a-judge
- 02 Jan 2025  [Process Reinforcement through Implicit Rewards](https://curvy-check-498.notion.site/-Process-Reinforcement-through-Implicit-Rewards-15f4fcb9c42180f1b498cc9b2eaf896f) - implicit PRM, gh: [PRIME-RL/PRIME](https://github.com/PRIME-RL/PRIME)
- 02 Dec 2024  Implicit PRM: [Free Process Rewards without Process Labels](https://arxiv.org/abs/2412.01981), gh: [PRIME-RL/ImplicitPRM](https://github.com/PRIME-RL/ImplicitPRM)
- [Rewarding Progress: Scaling Automated Process Verifiers for LLM Reasoning](https://arxiv.org/abs/2410.08146)
- [Solving math word problems with process- and outcome-based feedback](https://arxiv.org/abs/2211.14275)
- [Training Verifiers to Solve Math Word Problems](https://arxiv.org/abs/2110.14168)
- [RATIONALYST: Pre-training Process-Supervision for Improving Reasoning](https://arxiv.org/abs/2410.01044)
- 14 Dec 2023  [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935)

### RL
- 04 Jan 2025  REINFORCE++: [A Simple and Efficient Approach for Aligning Large Language Models](https://arxiv.org/abs/2501.03262)
- 20 Dec 2024  OREO: [Offline Reinforcement Learning for LLM Multi-Step Reasoning](https://arxiv.org/abs/2412.16145)
- 11 Oct 2024  DQO: [Enhancing Multi-Step Reasoning Abilities of Language Models through Direct Q-Function Optimization](https://arxiv.org/abs/2410.09302)
- 02 Oct 2024  RLEF: [Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://arxiv.org/abs/2410.02089)
- 08 Feb 2024  R^3: [Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning](https://arxiv.org/abs/2402.05808)
- 05 Feb 2024  GRPO: [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
- [ReFT: Reasoning with Reinforced Fine-Tuning](https://arxiv.org/abs/2401.08967)
- [ARES: Alternating Reinforcement Learning and Supervised Fine-Tuning for Enhanced Multi-Modal Chain-of-Thought Reasoning Through Diverse AI Feedback](https://arxiv.org/pdf/2407.00087)

### MCTS
- 08 Jan 2025  [rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking](https://arxiv.org/abs/2501.04519)
- [Agent Q: Advanced Reasoning and Learning for Autonomous AI Agents](https://arxiv.org/abs/2408.07199)
- [Improve Mathematical Reasoning in Language Models by Automated Process Supervision](https://arxiv.org/abs/2406.06592)
- [Toward Self-Improvement of LLMs via Imagination, Searching, and Criticizing](https://arxiv.org/abs/2404.12253)
- rStar: [Mutual Reasoning Makes Smaller LLMs Stronger Problem-Solvers](https://arxiv.org/abs/2408.06195)
- [LightZero: A Unified Benchmark for Monte Carlo Tree Search in General Sequential Decision Scenarios](https://arxiv.org/abs/2310.08348), code: [opendilab/LightZero](https://github.com/opendilab/LightZero)
- MuZero: [Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model](https://arxiv.org/abs/1911.08265), open-source impl: [werner-duvaud/muzero-general](https://github.com/werner-duvaud/muzero-general)

### Minecraft Agents
- [Voyager: An Open-Ended Embodied Agent with Large Language Models](https://arxiv.org/abs/2305.16291)
- [JARVIS-1: Open-World Multi-task Agents with Memory-Augmented Multimodal Language Models](https://arxiv.org/abs/2311.05997)
- [Optimus-1: Hybrid Multimodal Memory Empowered Agents Excel in Long-Horizon Tasks](https://arxiv.org/abs/2408.03615)
- [STEVE Series: Step-by-Step Construction of Agent Systems in Minecraft](https://arxiv.org/abs/2406.11247)

### Massive Sampling / Generate-and-Test

- Inference survey: [From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models](https://arxiv.org/abs/2406.16838)
- [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)
- [Large Language Monkeys: Scaling Inference Compute
with Repeated Sampling](https://arxiv.org/abs/2407.21787), [code](https://github.com/ScalingIntelligence/large_language_monkeys), [blog](https://scalingintelligence.stanford.edu/blogs/monkeys/)
- [AlphaCode 2 Technical Report](https://storage.googleapis.com/deepmind-media/AlphaCode2/AlphaCode2_Tech_Report.pdf)

### World Models

- GameNGen: [Diffusion Models Are Real-Time Game Engines](https://arxiv.org/abs/2408.14837), [project page](https://gamengen.github.io/)
- [A Path Towards Autonomous Machine Intelligence](https://openreview.net/pdf?id=BZ5a1r-kVsf)
- [GAIA-1: A Generative World Model for Autonomous Driving](https://arxiv.org/abs/2309.17080)
- Latent space world-models: [Dreamer](http://arxiv.org/pdf/1912.01603), [V2](https://arxiv.org/abs/2010.02193), [V3](https://arxiv.org/abs/2301.04104), [DayDreamer](https://arxiv.org/pdf/2206.14176)
- [World Models](https://arxiv.org/abs/1803.10122), web: [project page](https://worldmodels.github.io/)
- [Neural Assets: 3D-Aware Multi-Object Scene Synthesis with Image Diffusion Models](https://arxiv.org/abs/2406.09292)

### Neuro-Symbolic Approaches

- [Deliberate Reasoning for LLMs as Structure-aware Planning with Accurate World Model](https://arxiv.org/abs/2410.03136)
- [HYSYNTH: Context-Free LLM Approximation for Guiding Program Synthesis](https://arxiv.org/abs/2405.15880)
- [SymbolicAI: A framework for logic-based approaches combining generative models and solvers](https://arxiv.org/abs/2402.00854), Library: [ExtensityAI/symbolicai](https://github.com/ExtensityAI/symbolicai)
- [DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning](https://arxiv.org/abs/2006.08381)
- [A Neuro-vector-symbolic Architecture for Solving Raven's Progressive Matrices](https://arxiv.org/abs/2203.04571)
- Reasoning proofs generated by Prolog: [Neuro-Symbolic Integration Brings Causal and Reliable Reasoning Proofs](https://arxiv.org/abs/2311.09802), [Code](https://github.com/DAMO-NLP-SG/CaRing)
- VerityMath: [Advancing Mathematical Reasoning by Self-Verification Through Unit Consistency](https://arxiv.org/abs/2311.07172), [Code](https://github.com/vernontoh/VerityMath)

### Math
- AlphaGeometry: [Solving olympiad geometry without human demonstrations](https://www.nature.com/articles/s41586-023-06747-5)
- [Hologram Reasoning for Solving Algebra Problems with Geometry Diagrams](https://arxiv.org/abs/2408.10592)
- [Metacognitive Capabilities of LLMs: An Exploration in Mathematical Problem Solving](https://arxiv.org/abs/2405.12205)

### Active Inference

- [From pixels to planning: scale-free active inference](https://arxiv.org/abs/2407.20292)
- [Deep active inference agents using Monte-Carlo methods](https://arxiv.org/abs/2006.04176)


## Prompting Techniques

- Surveys:
   - (Jul 2024) [The Prompt Report: A Systematic Survey of Prompting Techniques](https://arxiv.org/abs/2406.06608)
   - (Feb 2024) [A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications](https://arxiv.org/abs/2402.07927)
   - Prompt Engineering Guide [Prompting Techniques](https://www.promptingguide.ai/techniques)
   - [Prompting Fundamentals and How to Apply them Effectively](https://eugeneyan.com/writing/prompting/) by Eugene Yan
- Tools:
   - [priompt](https://github.com/anysphere/priompt): A JSX-based prompting library, [Blog](https://arvid.xyz/posts/prompt-design/)
- Chain-of-Thoughts (COT): [Paper](https://arxiv.org/abs/2201.11903)
- Tree-of-Thoughts (ToT): [Paper](https://arxiv.org/pdf/2305.10601), impl: [Strategic Debate](https://github.com/zbambergerNLP/strategic-debate-tot)
- Graph-of-Thoughts (GoT): [Paper](https://arxiv.org/abs/2308.09687), [code](https://github.com/spcl/graph-of-thoughts)
- Algorithm of Thoughts (AoT): [Paper](https://arxiv.org/abs/2308.10379)
- Chain-of-Verification (CoVe/CoV): [Paper](https://arxiv.org/abs/2309.11495)
- Mixture-of-Agents (MoA): [Paper](https://arxiv.org/abs/2406.04692)
- Tool-Integrated Reasoning (ToRA / TIR): [Paper](https://arxiv.org/abs/2309.17452)
- Program of Thoughts (PoT): [Paper](https://arxiv.org/abs/2211.12588)
- Buffer of Thoughts (BoT): [Paper](https://arxiv.org/abs/2406.04271)
- Chain of Code (CoC): [Paper](https://arxiv.org/abs/2312.04474)
- Thought of Search (ToS): [Paper](https://arxiv.org/abs/2408.11326)
- Re-Reading the question as input (RE2): [Paper](https://arxiv.org/pdf/2309.06275)
- Self-Harmonized Chain of Thought (ECHO): [Paper](https://arxiv.org/abs/2409.04057), [code](https://github.com/Xalp/ECHO)
- Divergent CoT (DCoT), [Paper](https://arxiv.org/abs/2407.03181)
- Iteration of Thought (IoT), [Paper](https://arxiv.org/abs/2409.12618)
- Logic-of-Thought (LoT) [Paper](https://arxiv.org/abs/2409.17539)
- Forest-of-Thought (FoT) [Paper](https://arxiv.org/abs/2412.09078)

### Negative results
- [Chain of Thoughtlessness? An Analysis of CoT in Planning](https://arxiv.org/abs/2405.04776)


## Mechanistic Interpretability

- Anthropic: [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
- [Geometric Space of Hierarchical Concepts in LLM](https://arxiv.org/abs/2406.01506)


## Blog Posts / Presentations

- 05 Feb 2025  Sebastian Raschka [Understanding Reasoning LLMs](https://magazine.sebastianraschka.com/p/understanding-reasoning-llms)
- 02 Feb 2025  Huggingface [Open-R1: Update #1](https://huggingface.co/blog/open-r1/update-1)
- 25 Jan 2025  NLP@HKUST: [https://hkust-nlp.notion.site/simplerl-reason](https://hkust-nlp.notion.site/simplerl-reason), gh: [hkust-nlp/simpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason)
- 23 Jan 2025  Philschmid: [How to align open LLMs in 2025 with DPO & and synthetic data](https://www.philschmid.de/rl-with-llms-in-2025-dpo)
- 08 Jan 2025  ML CMU: [Optimizing LLM Test-Time Compute Involves Solving a Meta-RL Problem](https://blog.ml.cmu.edu/2025/01/08/optimizing-llm-test-time-compute-involves-solving-a-meta-rl-problem/)
- 09 Jan 2025  Notebook: [Agentic RAG with Hugging Face smolagents vs Vanilla RAG](https://colab.research.google.com/drive/1hG3dPgd8wjrO9wSD0K0Feo7EY1iXqrEN?usp=sharing)
- 07 Jan 2025  Chip Huyen: [Agents](https://huyenchip.com/2025/01/07/agents.html)
- HF: [Scaling Test Time Compute with Open Models](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute)
- Nebius: [Leveraging training and search for better software engineering agents](https://nebius.com/blog/posts/training-and-search-for-software-engineering-agents)
- DeepMind [AlphaProof and AlphaGeometry 2](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/)
- [Getting 50% (SoTA) on ARC-AGI with GPT-4o](https://www.lesswrong.com/posts/Rdwui3wHxCeKb7feK/getting-50-sota-on-arc-agi-with-gpt-4o), code: [rgreenblatt/arc_draw_more_samples_pub](https://github.com/rgreenblatt/arc_draw_more_samples_pub)
- Schmidhuber: [Artificial Curiosity & Creativity](https://people.idsia.ch/~juergen/artificial-curiosity-since-1990.html)
- synthesis.ai: [Do Androids Dream? World Models in Modern AI](https://synthesis.ai/2024/07/02/do-androids-dream-world-models-in-modern-ai/)
- [Our Transformers Code Agent beats the GAIA benchmark!](https://huggingface.co/blog/beating-gaia)
- Lil'Log [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) (Jun 2023 )
- BAIR Blog: [The Shift from Models to Compound AI Systems](https://bair.berkeley.edu/blog/2024/02/18/compound-ai-systems/)
- Microsoft Research [Tracing the path to self-adapting AI agents](https://www.microsoft.com/en-us/research/blog/tracing-the-path-to-self-adapting-ai-agents/)
- [LLMs develop their own understanding of reality as their language abilities improve](https://news.mit.edu/2024/llms-develop-own-understanding-of-reality-as-language-abilities-improve-0814), [Emergent Representations Paper](https://arxiv.org/abs/2305.11169)
- LessWrong post: [LLM Generality is a Timeline Crux](https://www.lesswrong.com/posts/k38sJNLk7YbJA72ST/llm-generality-is-a-timeline-crux)
- [Three levels of self-building autonomous agents](https://twitter.com/yoheinakajima/status/1839398354364838366) (Tweet thread by [Yohei
](https://twitter.com/yoheinakajima))
- [Don't Sleep on Single-agent Systems](https://www.all-hands.dev/blog/dont-sleep-on-single-agent-systems)
- Video: [Improving LLM Reasoning using self-generated data: RL and Verifiers](https://youtu.be/jSvXxkwrKlU?si=Ae9WMruDlg37yneP), [Slides](https://drive.google.com/file/d/1komQ7s9kPPvDx_8AxTh9A6tlfJA0j6dR/view) by [Rishabh Agarwal](https://twitter.com/agarwl_) (DeepMind)
- Slides: [Reasoning with inference-time compute](https://wellecks.com/data/welleck2024__inference_compute.pdf) by [Sean Welleck](https://x.com/wellecks), [tweet](https://x.com/wellecks/status/1839011524670796224)


## Graph Neural Networks

- Distill [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) (2021)
- [Geometric Deep Learning - Grids, Groups, Graphs, Geodesics, and Gauges](https://geometricdeeplearning.com/)

### Complex Logical Query Answering (CQLA)
Answering logical queries over Incomplete Knowledge Graphs. Aspirationally this requires combining sparse symbolic index collation (SQL, SPARQL, etc) and dense vector search, preferably in a differentiable manner.
- [Neural Graph Reasoning: Complex Logical Query Answering Meets Graph Databases](https://arxiv.org/abs/2303.14617)
- [Adapting Neural Link Predictors for Data-Efficient Complex Query Answering](https://openreview.net/forum?id=1G7CBp8o7L&referrer=%5Bthe%20profile%20of%20Erik%20Arakelyan%5D(%2Fprofile%3Fid%3D~Erik_Arakelyan1))
- [Generalizing Knowledge Graph Embedding with Universal Orthogonal Parameterization](https://openreview.net/forum?id=Sv4u9PtvT5)
- [Knowledge Sheaves: A Sheaf-Theoretic Framework for Knowledge Graph Embedding](https://arxiv.org/abs/2110.03789)
- [Wasserstein-Fisher-Rao Embedding: Logical Query Embeddings with Local Comparison and Global Transport](https://arxiv.org/abs/2305.04034)
- [GammaE: Gamma Embeddings for Logical Queries on Knowledge Graphs](https://arxiv.org/abs/2210.15578)
- [Soft Reasoning on Uncertain Knowledge Graphs](https://arxiv.org/abs/2403.01508)

### Inductive Reasoning over Heterogeneous Graphs
Similar to the regular CQLA, but with the emphasis on the "Inductive Setting" - i.e. querying over new, unseen during training nodes, edge types or even entire graphs. The latter part is interesting as it relies on the higher order "relations between relations" structure, connecting KG inference to Category Theory.
- [Zero-shot Logical Query Reasoning on any Knowledge Graph](https://arxiv.org/abs/2404.07198)
- [Extending Transductive Knowledge Graph Embedding Models for Inductive Logical Relational Inference](https://arxiv.org/abs/2309.03773)
- [Neural-Symbolic Models for Logical Queries on Knowledge Graphs](https://arxiv.org/abs/2205.10128)
- [InGram: Inductive Knowledge Graph Embedding via Relation Graphs](https://arxiv.org/abs/2305.19987)

### Neural Algorithmic Reasoning (NAR)
Initially attempted back in 2014 with general-purpose but unstable Neural Turing Machines, modern NAR approaches limit their scope to making GNN-based "Algorithmic Processor Networks" which learn to mimic classical algorithms on synthetic data and can be deployed on noisy real-world problems by sandwiching their frozen instances inside Encoder-Processor-Decoder architecture.
- [Neural Turing Machines, 2014](https://arxiv.org/abs/1410.5401)
- [A Generalist Neural Algorithmic Learner](https://openreview.net/forum?id=FebadKZf6Gd)
- [Transformers meet Neural Algorithmic Reasoners](https://arxiv.org/abs/2406.09308)
- [Recursive Algorithmic Reasoning](https://arxiv.org/abs/2307.00337)
- [Dual Algorithmic Reasoning](https://arxiv.org/abs/2302.04496)
- [Learning to Configure Computer Networks with Neural Algorithmic Reasoning](https://arxiv.org/abs/2211.01980)

## Grokking

- [Deep Networks Always Grok and Here is Why](https://imtiazhumayun.github.io/grokking/)
- [Grokfast: Accelerated Grokking by Amplifying Slow Gradients](https://arxiv.org/abs/2405.20233), [review post](https://x.com/_clashluke/status/1820810798693818761) by [Lucas Nestler](https://x.com/_clashluke)
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177)


## Open-Source Agents & Agent Frameworks

- [QwenLM/Qwen-Agent](https://github.com/QwenLM/Qwen-Agent)
- [meta-llama/llama-agentic-system](https://github.com/meta-llama/llama-agentic-system)
- [gpt-researcher](https://github.com/assafelovic/gpt-researcher), [docs](https://docs.gptr.dev/docs/gpt-researcher/introduction)
- [open-interpreter](https://github.com/OpenInterpreter/open-interpreter), [docs](https://docs.openinterpreter.com/getting-started/introduction)
- [ADAS](https://github.com/ShengranHu/ADAS) (Automated Design of Agentic Systems)
- [AI-Scientist](https://github.com/SakanaAI/AI-Scientist.git)
- [Ollama_Agents](https://github.com/MikeyBeez/Ollama_Agents)
- [AgentK](https://github.com/mikekelly/AgentK)
- [Storm](https://github.com/stanford-oval/storm), [Paper](https://arxiv.org/abs/2402.14207)
- [crewAI](https://github.com/crewAIInc/crewAI), [docs](https://docs.crewai.com/)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT), [docs](https://docs.agpt.co/)
- [AutoGen](https://github.com/microsoft/autogen), [docs](https://microsoft.github.io/autogen/), [AutoGen Studio Paper](https://arxiv.org/abs/2408.15247)
- [Trace](https://github.com/microsoft/Trace), [docs](https://microsoft.github.io/Trace/intro.html), [Paper](https://arxiv.org/abs/2406.16218)
- [motleycrew](https://github.com/ShoggothAI/motleycrew), [docs](https://motleycrew.readthedocs.io/en/latest/)
- [langflow](https://github.com/langflow-ai/langflow), [docs](https://docs.langflow.org/)
- [show-me](https://github.com/marlaman/show-me): A Visual and Transparent Reasoning Agent


## Algorithms

- [wake-sleep](https://en.wikipedia.org/wiki/Wake-sleep_algorithm)
- [novelty-search](https://algorithmafternoon.com/novelty/novelty_search_algorithm/)

### Weak Search Methods
Weak methods are general but don't use knowledge (heuristics) to guide the search process.

  - [depth-first-search](https://en.wikipedia.org/wiki/Depth-first_search) (DFS)
  - [breadth-first-search](https://en.wikipedia.org/wiki/Breadth-first_search) (BFS)
  - depth-limited-search, [iterative-deepening-depth-first-search](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search) (IDDFS)
  - [generate-and-test](https://www.geeksforgeeks.org/generate-and-test-search/)
  - [hill-climbing](https://en.wikipedia.org/wiki/Hill_climbing) (borderline case between weak and strong methods)

### Strong Search Methods
- [best-first-search](https://en.wikipedia.org/wiki/Best-first_search)
- [A*](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [beam-search](https://en.wikipedia.org/wiki/Beam_search)
- [monte-carlo-tree-search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (MCTS)


## Books

- [The Soar Cognitive Architecture](https://a.co/d/cXo8KFu), John E. Laird, MIT Press, 2019
- [How to Build a Brain: A Neural Architecture for Biological Cognition](https://academic.oup.com/book/6263) Chris Eliasmith, Oxford Series on Cognitive Models and Architectures, 2013
- [Active Inference: The Free Energy Principle in Mind, Brain, and Behavior](https://a.co/d/4OAX5dD), Thomas Parr, Giovanni Pezzulo, Karl J. Friston, MIT Press, 2022, [MLST Interview with Thomas Parr](https://www.youtube.com/watch?v=bk_xCikDUDQ)
- [Principles of Synthetic Intelligence PSI: An Architecture of Motivated Cognition](https://amzn.eu/d/46Xwq4s), Joscha Bach, Oxford Series on Cognitive Models and Architectures Book 4, 2009
- [Conscious Mind, Resonant Brain: How Each Brain Makes a Mind](https://a.co/d/5Hl3n7H), Stephen Grossberg, Oxford University Press, 2021
- [The Society of Mind](https://www.amazon.com/Society-Mind-Marvin-Minsky/dp/0671657135), Marvin Minsky, Simon & Schuster, 1986
- [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) 2nd Edition, Sutton & Barto, MIT Press, 2018
- [Reinforcement Learning: An Overview](https://arxiv.org/abs/2412.05265), Dec 2024, Kevin Murphy
- [Mathematical Foundations of Reinforcement Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning), Shiyu Zhao, open course on github + [video lectures](https://www.youtube.com/playlist?list=PLEhdbSEZZbDaFWPX4gehhwB9vJZJ1DNm8)
- [Natural Language Cognitive Architecture](https://github.com/daveshap/NaturalLanguageCognitiveArchitecture), David Shapiro, 2022, open source copy
- [An Introduction to Universal Artificial Intelligence](https://a.co/d/b61JiOh), Marcus Hutter, David Quarel, Elliot Catt, CRC Press, 2024 - AIXI, [Slides](http://hutter1.net/ai/suaibook.pdf), [Video](https://cartesiancafe.podbean.com/e/marcus-hutter-universal-artificial-intelligence-and-solomonoff-induction/)


## Biologically Inspired Approaches
Diverse approaches some of which tap into classical PDE systems of biological NNs, some concentrate on Distibuted Sparse Representations (by default non-differentiable), others draw inspiration from Hippocampal Grid Cells, Place Cells, etc. Biological systems surpass most ML methods for Continual and Online Learning, but are hard to implement efficienly on GPU.
- Ogma Sparse Predictive Hierarchies (SPH): [whitepaper](https://ogma.ai/sph-technology-description/)
- [The Tolman-Eichenbaum Machine: Unifying space and relational memory through generalisation in the hippocampal formation](https://www.biorxiv.org/content/10.1101/770495v2) (TEM), [TEM-t](https://arxiv.org/abs/2112.04035)
- [Arousal as a universal embedding for spatiotemporal brain dynamics](https://www.biorxiv.org/content/10.1101/2023.11.06.565918v2)
- [Sparse Distributed Memory is a Continual Learner](https://openreview.net/forum?id=JknGeelZJpHP)
- [Computation with Sequences of Assemblies in a Model of the Brain](https://proceedings.mlr.press/v237/dabagia24a.html)

### Dense Associative Memory
Dense Associative Memory is mainly represented by Modern Hopfield Networks (MHN), which [can be viewed](https://arxiv.org/abs/2008.02217) as a generalized Transformers capable of storing queries, keys and values explicitly (as in Vector Databases) and running recurrent retrival by energy minimization ([relating them](https://openreview.net/forum?id=B1BL9go65H&referrer=%5Bthe%20profile%20of%20Judy%20Hoffman%5D(%2Fprofile%3Fid%3D~Judy_Hoffman1)) to Diffusion models). Application for Continual Learning is possible when combined with uncertainty quantification and differentiable top-k selection.
- [xLSTM repository](https://github.com/NX-AI/xlstm)
- [CAMELoT: Towards Large Language Models with Training-Free Consolidated Associative Memory](https://arxiv.org/abs/2402.13449)
- [Energy Transformer](https://proceedings.neurips.cc/paper_files/paper/2023/hash/57a9b97477b67936298489e3c1417b0a-Abstract-Conference.html)
- [Memorization and consolidation in associative memory networks](https://openreview.net/forum?id=hXef89mdlH)
- [Simplicial Hopfield networks](https://openreview.net/forum?id=_QLsH8gatwx)


## Continual Learning

- [MagMax](https://arxiv.org/abs/2407.06322)


## Software Tools & Libraries

- [paul-gauthier/aider](https://github.com/paul-gauthier/aider)
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [PRIME-RL/PRIME](https://github.com/PRIME-RL/PRIME)
- [claude-engineer](https://github.com/Doriandarko/claude-engineer)
- [continuedev/continue](https://github.com/continuedev/continue)
- [OpenHands](https://github.com/All-Hands-AI/OpenHands) (formerly OpenDevin)
- [princeton-nlp/SWE-agent](https://github.com/princeton-nlp/SWE-agent), [documentation](https://princeton-nlp.github.io/SWE-agent/)
- [stanfordnlp/dspy]( https://github.com/stanfordnlp/dspy), DSPy awesome list: [ganarajpr/awesome-dspy](https://github.com/ganarajpr/awesome-dspy), [paper](https://arxiv.org/abs/2310.03714)
- [InternLM/lagent](https://github.com/InternLM/lagent) - lightweight framework for building LLM-based agents

## Commercial Offerings

- Software Engineering
  - [aide.dev](https://aide.dev/) + [codestoryai/sidecar](https://github.com/codestoryai/sidecar)
  - [Devin](https://preview.devin.ai/)
  - [Cursor](https://www.cursor.com/)
  - [Windsurf](https://codeium.com/windsurf) by Codeium
  - [GitHub Copilot](https://github.com/features/copilot) & [copilot-workspace](https://githubnext.com/projects/copilot-workspace)
  - [lovable.dev](https://lovable.dev/)
  - [textgrad](https://textgrad.com/)
  - [Cosine Genie](https://cosine.sh/genie)
  - [v0.dev](https://v0.dev/) by Vercel
  - [Replit AI](https://replit.com/)
  - [bolt](https://bolt.new/)
  - [continue.dev](https://www.continue.dev/)
  - [Amazon Q Developer](https://aws.amazon.com/de/q/developer/)
  - [Codey](https://sourcegraph.com/cody)by Sourcegraph
- AWS [Automated Reasoning checks](https://aws.amazon.com/en/blogs/aws/prevent-factual-errors-from-llm-hallucinations-with-mathematically-sound-automated-reasoning-checks-preview/)


# Competitions & Benchmarks

- DevAI: [Agent-as-a-Judge: Evaluate Agents with Agents](https://arxiv.org/abs/2410.10934)
- [AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents](https://arxiv.org/abs/2407.18901), web: [project page](https://appworld.dev/), gh: [stonybrooknlp/appworld](https://github.com/stonybrooknlp/appworld/), [Leaderboard](https://appworld.dev/leaderboard)
- [CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents](https://arxiv.org/abs/2407.01511), gh: [camel-ai/crab](https://github.com/camel-ai/crab)
- [WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/abs/2307.13854), web: [project page](https://webarena.dev/), [Leaderboard](https://docs.google.com/spreadsheets/d/1M801lEpBbKSNwP-vDBkC_pF7LdyGU1f_ufZb_NWNBZQ/edit?usp=sharing)
- [ARC-AGI](https://arcprize.org/arc): [Leaderboard](https://arcprize.org/leaderboard), [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547)
- PlanBench: [Paper](https://arxiv.org/abs/2206.10498), gh: [karthikv792/LLMs-Planning](https://github.com/karthikv792/LLMs-Planning)
- [GAIA: a benchmark for General AI Assistants](https://arxiv.org/abs/2311.12983): [Leaderboard](https://gaia-benchmark-leaderboard.hf.space/)
- [StreamBench: Towards Benchmarking Continuous Improvement of Language Agents](https://arxiv.org/pdf/2406.08747), gh: [stream-bench/stream-bench](https://github.com/stream-bench/stream-bench)
- [VisualAgentBench: Towards Large Multimodal Models as Visual Foundation Agents](https://arxiv.org/abs/2408.06327)
- [ZebraLogic](https://huggingface.co/blog/yuchenlin/zebra-logic), [Leaderboard](https://huggingface.co/spaces/allenai/ZebraLogic)
- [Omni-MATH](https://omni-math.github.io/), gh: [KbsdJames/Omni-MATH](https://github.com/KbsdJames/Omni-MATH)
- [BatsResearch/planetarium](https://github.com/BatsResearch/planetarium) - Dataset and benchmark for assessing LLMs in translating natural language descriptions of planning problems into PDDL

### Code
- [SWE-bench](https://www.swebench.com/index.html), [SWE-bench Lite](https://www.swebench.com/lite.html)
- [BigCodeBench: The Next Generation of HumanEval](https://huggingface.co/blog/leaderboard-bigcodebench), [Leaderboard](https://huggingface.co/spaces/bigcode/bigcodebench-leaderboard)
- [SciCode: A Research Coding Benchmark Curated by Scientists](https://arxiv.org/abs/2407.13168), web: https://scicode-bench.github.io/
- [commit-0](https://commit-0.github.io/) The challenge is to rebuild Python core libraries and pass their unit tests, [Leaderboard](https://commit-0.github.io/analysis/)


## Related Projects

- [Awesome LLM Strawberry (OpenAI o1)](https://github.com/hijkzzz/Awesome-LLM-Strawberry)
- [awesome-o1](https://github.com/srush/awesome-o1) literature list by [Sasha Rush](https://twitter.com/srush_nlp)
- [awesome-ai-agents](https://github.com/e2b-dev/awesome-ai-agents)
- Nous Research [Open Reasoning Tasks](https://reasoning.nousresearch.com/), a list of reasoning tasks, gh: [NousResearch/Open-Reasoning-Tasks](https://github.com/NousResearch/Open-Reasoning-Tasks)
- [ARC-AGI Resources](https://docs.google.com/spreadsheets/d/1fR4cgjY1kNKN_dxiidBQbyT6Gv7_Ko7daKOjlYojwTY/edit?pli=1&gid=756763742#gid=756763742) Google table paper list by [ARC price](https://arcprize.org/)


## Youtube Content

- Sasha Rush: [Speculations on Test-Time Scaling (o1)](https://youtu.be/6PEJ96k1kiw)
- François Chollet: [It's Not About Scale, It's About Abstraction](https://youtu.be/s7_NlkBwdj8?feature=shared)
- [Evaluating, Understanding and Improving Approaches for Machine Reasoning](https://www.youtube.com/watch?v=sxbfvVcbIi8)
- Channel: [David Shapiro](https://www.youtube.com/@DaveShap)
- [Artem Kirsanov: Engrams, Building Blocks of Memory in the Brain](https://www.youtube.com/watch?v=X5trRLX7PQY)
- Channel: [Edan Meyer on AI, ML & RL](https://www.youtube.com/@EdanMeyer), [Discrete vs. Continuous RL](https://www.youtube.com/watch?v=s8RqGlU5HEs) + [Paper](https://arxiv.org/abs/2312.01203)
- [MIT AGI: Cognitive Architecture (Nate Derbinsky)](https://www.youtube.com/watch?v=bfO4EkoGh40)
- Channel: [Thinking About Thinking](https://www.youtube.com/@ThoughtChannel/videos) (Mathematics of Neuroscience and AI)
- [Invariance and equivariance in brains and machines](https://youtu.be/xnhhp916JNU?si=VpiDQDbFafyNAlMW)
- [code_your_own_AI: The CORE IDEA of AI Agents Explained](https://youtu.be/xdAKa8jFx3g)


## Best LLM APIs

- [Anthropic Claude](https://docs.anthropic.com/en/api/getting-started)
- [together.ai](https://docs.together.ai/docs/introduction)
- [groq.com](https://console.groq.com)
- [openrouter.ai](https://openrouter.ai/)


## Open-weights Reasoning Models

- [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)
- [NovaSky-AI/Sky-T1-32B-Preview](https://huggingface.co/NovaSky-AI/Sky-T1-32B-Preview), [Blog](https://novasky-ai.github.io/posts/sky-t1/), gh: [NovaSky-AI/SkyThought](https://github.com/NovaSky-AI/SkyThought)
- [ngxson/MiniThinky-v2-1B-Llama-3.2](https://huggingface.co/ngxson/MiniThinky-v2-1B-Llama-3.2)
- [SmallThinker-3B-Preview](https://huggingface.co/PowerInfer/SmallThinker-3B-Preview) (small model trained on [PowerInfer/QWQ-LONGCOT-500K](https://huggingface.co/datasets/PowerInfer/QWQ-LONGCOT-500K))
- [QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview), [Blog post](https://qwenlm.github.io/blog/qwq-32b-preview/)
- [ruliad/deepthought-8b-llama-v0.01-alpha](https://huggingface.co/ruliad/deepthought-8b-llama-v0.01-alpha) JSON format: 1. Problem understanding, 2. Data gathering, 3. Analysis, 4. Calculation (when applicable), 5. Verification, 6. Conclusion drawing, 7. Implementation
- [migtissera/Tess-R1-Limerick-Llama-3.1-70B](https://huggingface.co/migtissera/Tess-R1-Limerick-Llama-3.1-70B) xml tags:
  1. `<thinking>`  tag to indicate when the model is performing CoT.
  2. `<contemplation>` tag when the model contemplate on its answers.
  3. `<alternatively>` tag for alternate suggestions.
  4. `<output>` for the final output


## Novel model architectures

- 20 Jan 2025  [Kimi k1.5: Scaling Reinforcement Learning with LLMs](https://github.com/MoonshotAI/Kimi-k1.5/blob/main/Kimi_k1.5.pdf)
- 20 Jan 2025  [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- 09 Jan 2025  [Transformer^2: Self-adaptive LLMs](https://arxiv.org/abs/2501.06252)
- 31 Dec 2024  [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663)
- 27 Dec 2025  [Xmodel-2 Technical Report](https://arxiv.org/pdf/2412.19638) - Deep-and-Thin Architecture (1.2B, 48 layers)
- [Mirasol3B: A Multimodal Autoregressive model for time-aligned and contextual modalities](https://arxiv.org/abs/2311.05698)
- [Memory3: Language Modeling with Explicit Memory](https://arxiv.org/abs/2407.01178)
- TTT: [Learning to (Learn at Test Time): RNNs with Expressive Hidden States](https://arxiv.org/abs/2407.04620), [Video](https://www.youtube.com/watch?v=I9Ghw2Z7Gqk)
- [TransformerFAM: Feedback attention is working memory](https://arxiv.org/pdf/2404.09173)


## Philosophy: Nature of Intelligence & Consciousness

- [A High Level Theory on the Nature of Intelligence and Consciousness](https://philarchive.org/rec/GARAHL)

### Joscha Bach
- [Machine Consciousness](https://youtu.be/LlLbHm-bJQE)
- [Consciousness as a coherence-inducing operator](https://www.youtube.com/watch?v=qoHCQ1ozswA) Talk by Josha Bach at the [Models of Consciousness Conferences](https://models-of-consciousness.org/)


## Biology / Neuroscience

- [The brain simulates actions and their consequences during REM sleep](https://www.biorxiv.org/content/10.1101/2024.08.13.607810v1)
- CSCG: [Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps](https://www.nature.com/articles/s41467-021-22559-5)
- [System-1 and System-2 realized within the Common Model of Cognition](https://www.researchgate.net/publication/365345252_System-1_and_System-2_realized_within_the_Common_Model_of_Cognition) (2022)

## Workshops

https://s2r-at-scale-workshop.github.io (NeurIPS 2024)

## Tutorials

- [Neurips 2024 Tutorial: Beyond Decoding: Meta-Generation Algorithms for Large Language Models](https://cmu-l3.github.io/neurips2024-inference-tutorial/)


## How to contribute

 To share a link related to reasoning in AI systems that is missing here please create a pull request for this file. See [editing files](https://docs.github.com/en/repositories/working-with-files/managing-files/editing-files) in the github documentation.
