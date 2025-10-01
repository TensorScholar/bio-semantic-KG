# Bio-Semantic-KG

[![TypeScript](https://img.shields.io/badge/typescript-5.x-blue.svg)](https://www.typescriptlang.org/)
[![Deno](https://img.shields.io/badge/deno-^1.40-lightgrey?logo=deno)](https://deno.land)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A specialized, enterprise-grade framework for extracting and structuring biomedical knowledge from unstructured, bilingual (Persian/English) sources. This project leverages advanced NLP, Computer Vision, and a **Knowledge Graph** (Neo4j) to build a rich, queryable semantic network of medical data.

The entire system is architected using **Domain-Driven Design (DDD)** and **Hexagonal Architecture (Ports & Adapters)**, with a core built on Functional Programming principles (`Result`, `Option`, `Either`) and advanced TypeScript features for maximum type-safety and robustness.

## Core Architectural Principles

- **Domain-Driven Design (DDD)**: The core logic is modeled through rich domain entities, value objects, and specifications, ensuring the software is a true representation of the complex medical domain.
- **Hexagonal Architecture**: A strict separation between the `core` application logic and `infrastructure` details (databases, APIs, parsers), maximizing testability and maintainability.
- **Functional Core**: A robust, error-resistant core using functional concepts and immutable data structures to eliminate side effects and ensure predictable behavior.
- **HIPAA-Compliant Security**: A security-first design featuring a dedicated framework for encryption, access control, and data anonymization to meet medical data standards.

## Key Features

- **Specialized Bilingual NLP**: A custom NLP engine for the medical aesthetics domain, supporting both Persian and English with Named Entity Recognition (NER) and morphological analysis.
- **Knowledge Graph Persistence**: Models complex relationships between clinics, practitioners, procedures, and technologies using a Neo4j graph database.
- **Advanced Type Safety**: Utilizes TypeScript's **Branded Types** to enforce domain-specific constraints at compile time (e.g., `ProcedureId` cannot be a plain string).
- **Comprehensive Mathematical Specifications**: Includes detailed documentation outlining the mathematical and theoretical foundations of all core algorithms, from NLP to knowledge graph construction.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.