# Metodologia Científica: Avaliação Comparativa de Embeddings Quantizados e Multilíngues em RAG Técnico

## 1. Contexto e Problema de Pesquisa

A aplicação de arquiteturas de *Retrieval-Augmented Generation* (RAG) em domínios técnicos corporativos exige representações semânticas precisas para localizar informações críticas em manuais e especificações de engenharia. Embora modelos proprietários de ponta (como o `text-embedding-3-small` da OpenAI) ofereçam excelente desempenho médio, seu uso introduz dependência de terceiros, custos contínuos e riscos à soberania de dados sensíveis.

Por outro lado, o uso de modelos abertos executados localmente enfrenta desafios de consumo de memória (VRAM) e latência. Modelos quantizados em 4 bits via k-quants (`Q4_K_M`) surgem como uma alternativa promissora. Este estudo investiga formalmente:

> **Questão de Pesquisa**: *Em que medida modelos abertos densos quantizados em 4 bits (`Q4_K_M`) e modelos multilíngues abertos preservam a capacidade de recuperação de informação técnica em língua portuguesa quando comparados ao baseline proprietário comercial da OpenAI?*

---

## 2. Embasamento Teórico da Quantização `Q4_K_M` como *Sweet Spot*

### 2.1 Mecanismo de k-quants no llama.cpp
A quantização uniforme clássica (como `Q4_0`) divide o intervalo de pesos linearmente, causando perda acentuada de informação em tensores com distribuições assimétricas ou caudas longas. O formato **k-quants** (Kawrakow, 2023) introduziu superblocos com quantização por blocos e precisão mista adaptativa:
- No perfil **`_M` (*Medium*)**, pesos de camadas críticas — como projeções de atenção (`attention.wq`, `attention.wv`) e as primeiras camadas do perceptron multicamadas (MLP) — são quantizados com 5 ou 6 bits por bloco, enquanto as matrizes restantes utilizam 4 bits.

### 2.2 Preservação Geométrica do Espaço Latente
Trabalhos fundamentais de quantização pós-treinamento, notadamente Dettmers et al. (2023) (*QLoRA*) e Frantar et al. (2022) (*GPTQ*), demonstram que matrizes de pesos quantizadas em 4 bits estruturadas por blocos mantêm a capacidade expressiva original das redes neurais, com elevação de perplexidade marginal ($\Delta \text{ppl} < 0.1$).

No contexto específico de embeddings densos:
1. O objetivo do modelo é projetar trechos de texto em um espaço vetorial $\mathbb{R}^d$ onde a similaridade angular ($\cos \theta$) reflete proximidade semântica.
2. A quantização `Q4_K_M` induz perturbações esféricas quase isotrópicas no espaço latente. Como o cálculo de similaridade depende da direção angular relativa e não da magnitude absoluta, o ordenamento relativo dos vizinhos mais próximos no Top-$K$ permanece altamente estável.
3. Estudos comparativos empíricos apontam que o `Q4_K_M` atinge um trade-off ótimo (*sweet spot*): reduz o uso de memória em ~75% em comparação a FP16 com degradação típica de acurácia entre 1% e 3%, enquanto formatos de 8 bits (`Q8_0`) demandam o dobro de VRAM para ganhos estatisticamente marginais (< 0.5%).

---

## 3. Desenho Experimental

### 3.1 Hipóteses Formais
- **Hipótese Nula ($H_0$)**: $\mu_{\text{HitRate@5}}(M_{\text{aberto}}) < 0.85 \times \mu_{\text{HitRate@5}}(M_{\text{OpenAI}})$  
  *(A quantização ou o menor porte dos modelos abertos resulta em perda de recuperação superior a 15% em relação ao baseline da OpenAI).*
- **Hipótese Alternativa ($H_1$)**: $\mu_{\text{HitRate@5}}(M_{\text{aberto}}) \ge 0.85 \times \mu_{\text{HitRate@5}}(M_{\text{OpenAI}})$  
  *(Pelo menos um modelo aberto local atinge paridade operacional com a referência proprietária).*

### 3.2 Quadro de Variáveis
- **Variável Independente**: Arquitetura e modelo de embedding empregado (7 tratamentos: OpenAI 3-small, Qwen3-8B Q4, Qwen3-4B Q4, BGE-M3, Multilingual-E5-Large, Nomic-Embed-Text, Bertimbau STS).
- **Variáveis Dependentes (Eficácia de Recuperação)**:
  - Hit Rate@K ($K \in \{2, 5, 10\}$)
  - Mean Reciprocal Rank (MRR@K)
  - Context Recall@K
  - Mean Average Precision (MAP@K)
- **Variáveis Dependentes (Eficiência Operacional)**:
  - Latência média por consulta (ms)
  - Tempo de indexação do corpus (s)
  - Tamanho do índice vetorial em disco (MB)
- **Variáveis Controladas**:
  - Corpus textual: 16 manuais técnicos CM Comandos (`data/`).
  - Divisão de nós: `SentenceSplitter` com tamanho de bloco $L = 512$ tokens e sobreposição $O = 50$ tokens.
  - Conjunto de consultas: 132 itens com referências validadas em `testset_openai_4omini.jsonl`.
  - Métrica de distância vetorial: Similaridade de Cosseno.
  - Ambiente de hardware: Mesma GPU/CPU para todos os modelos locais Ollama/HF.

---

## 4. Formulação Matemática das Métricas

Seja $Q$ o conjunto de consultas de teste ($|Q| = 132$). Para cada consulta $q \in Q$, seja $\mathcal{R}_q = \{r_1, r_2, \dots, r_{|\mathcal{R}_q|}\}$ o conjunto de trechos de referência (*ground truth*) e $\mathcal{N}_q^K = [n_{q,1}, n_{q,2}, \dots, n_{q,K}]$ a lista ordenada dos $K$ nós recuperados pelo modelo de embedding.

Definimos a função indicadora de relevância $\text{Rel}(n, \mathcal{R}_q) \in \{0, 1\}$ tal que $\text{Rel}(n, \mathcal{R}_q) = 1$ se o nó $n$ satisfaz ao menos um dos critérios objetivos de correspondência:
1. **Contenção**: $\exists r \in \mathcal{R}_q \mid (r \subseteq n.\text{text}) \lor (n.\text{text} \subseteq r)$.
2. **Similaridade Fuzzy de Tokens**: $\exists r \in \mathcal{R}_q \mid \text{TokenSetRatio}(n.\text{text}, r) \ge 75\%$.
3. **ROUGE-L**: $\exists r \in \mathcal{R}_q \mid F_1^{\text{ROUGE-L}}(n.\text{text}, r) \ge 0.50$.

### 4.1 Hit Rate no Top-K ($\text{HR}@K$)
$$\text{HR}@K = \frac{1}{|Q|} \sum_{q \in Q} \mathbb{I}\left( \sum_{j=1}^K \text{Rel}(n_{q,j}, \mathcal{R}_q) > 0 \right)$$

### 4.2 Mean Reciprocal Rank ($\text{MRR}@K$)
$$\text{MRR}@K = \frac{1}{|Q|} \sum_{q \in Q} \left( \frac{1}{\min \{ j \in [1, K] \mid \text{Rel}(n_{q,j}, \mathcal{R}_q) = 1 \}} \right)$$
*(Caso nenhum nó relevante esteja no Top-$K$, o termo da consulta é $0$).*

### 4.3 Context Recall no Top-K ($\text{CR}@K$)
$$\text{CR}@K = \frac{1}{|Q|} \sum_{q \in Q} \frac{\left| \{ r \in \mathcal{R}_q \mid \exists j \le K \text{ que valida } r \} \right|}{|\mathcal{R}_q|}$$

### 4.4 Mean Average Precision no Top-K ($\text{MAP}@K$)
$$\text{MAP}@K = \frac{1}{|Q_{\text{efetivo}}|} \sum_{q \in Q_{\text{efetivo}}} \frac{1}{\min(|\mathcal{R}_q|, K)} \sum_{j=1}^K \text{Precision}@j(q) \cdot \text{Rel}(n_{q,j}, \mathcal{R}_q)$$

### 4.5 Teste de Hipótese e Significância Estatística
Para testar $H_0$, aplica-se o teste pareado de postos com sinais de **Wilcoxon** (*Wilcoxon signed-rank test*) entre as distribuições pareadas de $HitRate@5$ e $MRR@5$ de cada modelo local vs. OpenAI:
- Nível de significância: $\alpha = 0.05$.
- Intervalos de confiança de 95% calculados por *Bootstrapping* não-paramétrico ($B = 1.000$ reamostragens).

---

## 5. Ameaças à Validade

1. **Validade Interna (Viés de Circularidade do Ground Truth)**: O testset de teste foi sintetizado por GPT-4o-mini sobre índices originalmente montados com embeddings OpenAI. Isso pode conferir vantagem intrínseca à referência M1.  
   *Mitigação*: Documentação explícita do viés e auditoria qualitativa manual de 20 amostras para identificar recuperação semanticamente correta de nós não contidos no gabarito estrito.
2. **Validade Interna (Fronteiras de Chunks e Deduplicação)**: A quebra de blocos no LlamaIndex pode não coincidir perfeitamente com os limites dos trechos salvos no ground truth, e duplicatas no dataset afetam o denominador.  
   *Mitigação*: Deduplicação preliminar e combinação de contenção estrita, fuzzy matching ($\ge 75\%$) e ROUGE-L ($F_1 \ge 0.50$).
3. **Validade Externa (Generalização)**: O corpus é restrito ao nicho técnico industrial de no-breaks e energia crítica. As conclusões refletem a recuperação em linguagem técnica específica em português.
4. **Validade de Conclusão**: Aplicação do teste não-paramétrico de Wilcoxon para garantir que diferenças de métricas não sejam fruto de ruído amostral.
