# Especificação Técnica dos Modelos de Embedding Avaliados

Este documento consolida a ficha técnica detalhada dos 7 modelos avaliados no experimento comparativo de recuperação sobre o corpus CM Comandos.

---

## 1. Tabela Comparativa de Características

| ID | Identificador | Provedor / Mecanismo | Parâmetros | Quantização | Dimensão Vetorial | Tamanho Armazenamento | Contexto Máximo | Papel no Estudo |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **M1** | `text-embedding-3-small` | OpenAI Cloud API | Não divulgado (~300M) | FP16 (serviço) | 1536 | Nuvem (API) | 8.191 tokens | **Baseline de Referência**: Padrão proprietário de mercado. |
| **M2** | `qwen3-embedding:8b` | Ollama Local | 7.6B | `Q4_K_M` | 4096 | ~4.7 GB | 40.960 tokens | **Candidato Principal**: Modelo denso de alta capacidade em 4-bit. |
| **M3** | `qwen3-embedding:4b` | Ollama Local | 4.0B | `Q4_K_M` | 2560 | ~2.5 GB | 40.960 tokens | **Escalabilidade**: Modelo leve de 4B parâmetros em 4-bit. |
| **M4** | `bge-m3` | Ollama / HuggingFace | 560M | FP16 / GGUF | 1024 | ~2.2 GB | 8.192 tokens | **Referência Aberta Multilíngue**: Líder em benchmarks globais multilíngues com suporte a PT-BR. |
| **M5** | `multilingual-e5-large` | HuggingFace / FastEmbed | 560M | FP16 | 1024 | ~2.2 GB | 512 tokens | **Histórico de Produção**: Embedder já empregado no ambiente `haiku-rag`. |
| **M6** | `nomic-embed-text` | Ollama Local | 137M | `Q4_K_M` | 768 | ~274 MB | 8.192 tokens | **Alta Eficiência**: Modelo ultraleve para inferência rápida e baixo consumo. |
| **M7** | `bertimbau-base-portuguese-sts` | HuggingFace | 110M | FP32 / FP16 | 768 | ~435 MB | 512 tokens | **Especialista em PT-BR**: Baseado no BERTimbau, ajustado especificamente para português brasileiro. |

---

## 2. Detalhamento de Cada Modelo

### M1: OpenAI `text-embedding-3-small`
- **Provedor**: OpenAI (`https://api.openai.com/v1/embeddings`).
- **Autenticação**: Variável `OPENAI_API_KEY` extraída de `/home/joaorura/Desenvolvimento/EASY/cm-comandos-piad/.env`.
- **Características**: Suporta truncamento dimensional (Matryoshka Representation Learning), embora utilizado neste estudo com a dimensão nominal completa de 1536 dimensões.

### M2: `qwen3-embedding:8b` (Q4_K_M)
- **Origem**: Família Qwen 3 (Alibaba Cloud), adaptada para representação densa.
- **Formato**: GGUF com k-quants `Q4_K_M` via Ollama (`http://localhost:11434`).
- **Dimensão**: 4096.
- **Destaque**: Uma das maiores capacidades representativas em modelos abertos locais de embedding, testando a hipótese de que a escala de parâmetros (7.6B) compensa com folga a quantização em 4 bits.

### M3: `qwen3-embedding:4b` (Q4_K_M)
- **Origem**: Família Qwen 3.
- **Formato**: GGUF com k-quants `Q4_K_M` via Ollama.
- **Dimensão**: 2560.
- **Destaque**: Avaliação de sensibilidade à redução no volume de parâmetros em 4-bit mantendo a mesma família arquitetural.

### M4: BAAI `bge-m3`
- **Origem**: Beijing Academy of Artificial Intelligence (BAAI).
- **Características**: Projetado especificamente para recuperação multilíngue extensiva em mais de 100 idiomas. Possui arquitetura otimizada para capturar tanto nuances semânticas densas quanto termos técnicos específicos.

### M5: Intfloat `multilingual-e5-large`
- **Origem**: Microsoft / Intfloat.
- **Treinamento**: Pré-treinado com alinhamento fraco sobre pares de texto em larga escala e pós-treinado com contraste semântico.
- **Nota Operacional**: Requer o prefixo `passage: ` para documentos e `query: ` para consultas para obter o melhor alinhamento latente.

### M6: Nomic `nomic-embed-text`
- **Origem**: Nomic AI.
- **Destaque**: Treinado com pesos totalmente abertos, arquitetura eficiente e capacidade de absorver contextos longos (8192 tokens) em uma pegada de memória mínima.

### M7: `bertimbau-base-portuguese-sts`
- **Origem**: Ricardo Z. / Neuralmind.
- **Base**: `neuralmind/bert-base-portuguese-cased` pré-treinado no corpus BrWaC (Brazilian Web as Corpus).
- **Ajuste Fino**: STS (Semantic Textual Similarity) para alinhamento estrito em português do Brasil.
