# Decision Engine Architecture

## System Overview

The Decision Engine is a rule-based + ML hybrid system that generates strategic recommendations for race operations. It consists of:

1. **7 Specialized Decision Modules** - Domain-specific logic for different decision types
2. **Central Orchestrator** - Conflict resolution, ranking, caching
3. **Scoring System** - Confidence, risk, priority calculation
4. **Explainability Layer** - Reasoning, logging, auditing
5. **Registry** - Module versioning and management

```
┌─────────────────────────────────────────────────────────────────┐
│                        Decision Engine                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │             Decision Orchestrator (engine.py)             │ │
│  │  • Conflict Resolution  • Caching  • Ranking  • Async     │ │
│  └───────────────────────────────────────────────────────────┘ │
│                            │                                    │
│              ┌─────────────┴─────────────┐                      │
│              ▼                           ▼                      │
│  ┌──────────────────────┐   ┌──────────────────────┐           │
│  │  Priority 10 (SC)    │   │  Priority 9-8        │           │
│  │  • SafetyCarDecision │   │  • PitTimingDecision │           │
│  │  • RainStrategy      │   │  • UndercutOvercut   │           │
│  │  (<100ms latency)    │   │  • StrategyConversion│           │
│  └──────────────────────┘   └──────────────────────┘           │
│              │                           │                      │
│              └─────────────┬─────────────┘                      │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │          Scoring & Explainability (scoring.py)           │  │
│  │  • Confidence   • Risk   • Traffic Lights   • Ranking    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                            │                                    │
│                            ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              Decision Output (schemas.py)                │  │
│  │  • Top 3 Recommendations  • Reasoning  • Alternatives    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
     ┌──────────┐    ┌──────────┐    ┌──────────┐
     │ Strategy │    │ Feature  │    │   ML     │
     │ Sim Tree │    │  Store   │    │ Models   │
     └──────────┘    └──────────┘    └──────────┘
