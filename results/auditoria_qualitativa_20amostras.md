# Auditoria Qualitativa do Ground Truth (20 Amostras)

## Metadados e Contexto Metodológico
- **Data de Execução**: 2026-09-29
- **Arquivo Fonte**: `testset_openai_4omini.jsonl` (Total: 132 amostras)
- **Tamanho da Amostra**: 20 amostras aleatórias (`seed=42`)
- **Finalidade Científica**: Mitigação e quantificação do **viés circular (*circular bias*)** no benchmark de recuperação vetorial para o TCC.

### Por que a auditoria qualitativa manual é necessária?
O conjunto de avaliação sintético foi sintetizado com o pipeline Ragas utilizando o gerador **GPT-4o-mini** e indexador com **OpenAI embeddings** (`text-embedding-3-small`). Esse arranjo metodológico pode introduzir um favorecimento sistemático (viés circular) para modelos da mesma família em detrimento de modelos open-source (como LLaMA, DeepSeek e Mixtral).

A auditoria humana manual substitui o uso de juiz LLM adicional (que introduziria um segundo viés) e avalia diretamente:
1. Se a pergunta (`user_input`) possui formulação técnica coerente com o domínio de No-Breaks / sistemas de energia.
2. Se as passagens extraídas (`reference_contexts`) contêm de fato a resposta substantiva esperada.
3. Se a associação decorre de artefatos de chunking (e.g. cabeçalhos vazios, logotipos, fragmentos incompletos) ou de alucinações do modelo gerador.

### Critérios Operacionais de Classificação:
- **`Relevante`**: O contexto de referência contém a resposta direta e substantiva para a pergunta formulada.
- **`Parcialmente Relevante`**: O contexto aborda o tópico geral da consulta, porém omite informações essenciais ou apresenta conteúdo vago/incompleto.
- **`Irrelevante`**: O contexto não responde à pergunta; resulta de corte indevido de texto (ex: cabeçalhos isolados como *"CM Comandos Lineares"*) ou a pergunta deriva de inferência alucinada do LLM.

---

## 1. Tabela de Amostras para Avaliação

|   Sample # | Query                                                                            | Reference                                                                                            | Classificação   |
|-----------:|:---------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------|:----------------|
|          1 | Quais sã as implicações do uso de comands linears na programção e automação d... | CM Comandos Lineares / CM COMANDOS LINEARESCM COMANDOS LINEARES / CM COMANDOS LINEARES               | ⬜ A avaliar    |
|          2 | Como o sistema Paralelo Multi Ativo muda a confiabilidade dos no-breaks?         | S I S T E M A PA R A L E L O M U LT I AT I V O ( E X C L U S I V O C M C O M A N D O S ) A CM Com... | ⬜ A avaliar    |
|          3 | Quais são os benefícios da tecnologia DSP em relação à alta performance e con... | NO-BREAKS CORPORATIVOSVivemos em um mundo cada v ez mais ág il e prático . Um mundo conec tado po... | ⬜ A avaliar    |
|          4 | Como as abordagens de manutenção e suporte técnico se comparam em relação a c... | O texto aborda serviços de manutenção e suporte técnico para no-breaks, estabilizadores e outras ... | ⬜ A avaliar    |
|          5 | De que maneira as estratégias de redução de custos relacionadas à manutenção ... | O texto aborda serviços de manutenção e suporte técnico para no-breaks, estabilizadores e outras ... | ⬜ A avaliar    |
|          6 | importância sistema No-Break ambientes críticos                                  | No-Break / No-Break                                                                                  | ⬜ A avaliar    |
|          7 | como No-Breaks CM Comandos Lineares garantem continuidade operacional missão ... | APLICAÇÕES Os No -Breaks da CM C omandos Linear es são indicados para aplicações de missão cr íti... | ⬜ A avaliar    |
|          8 | Quais os impactos das variações de tensão e corrente nas medições e desempenh... | ão Elétricas Sub e Sobr etensão de Entrada e Saída Sub e Sobr etensão DC e Bater ia Sobr ecar ga ... | ⬜ A avaliar    |
|          9 | Quais implicações especificações CM MKT OUT 23 sujeitas alterações prévio aviso? | CM. MKT OUT 23 - As especificações estão sujeitas a alterações sem prévio aviso.                     | ⬜ A avaliar    |
|         10 | impacto somatória potências No-Breaks eficiência confiabilidade energia siste... | P O S S I B I L I D A D E D E S O M A T Ó R I A D E P O T Ê N C I A S Permite a ligação para soma... | ⬜ A avaliar    |
|         11 | De que forma o sistema Paralelo Multi Ativo da CM Comandos Lineares ajuda na ... | S I S T E M A PA R A L E L O M U LT I AT I V O ( E X C L U S I V O C M C O M A N D O S ) A CM Com... | ⬜ A avaliar    |
|         12 | Quais os impacto das variaçõe de tensão e corrente nas medições elétrica e no... | ão Elétricas Sub e Sobr etensão de Entrada e Saída Sub e Sobr etensão DC e Bater ia Sobr ecar ga ... | ⬜ A avaliar    |
|         13 | Como as abordagens sobre suporte técnico e manutenção preventiva se comparam ... | O texto aborda serviços de manutenção e suporte técnico para no-breaks, estabilizadores e outras ... | ⬜ A avaliar    |
|         14 | Como abordagens manutenção suporte técnico relatórios se comparam contratos m... | O texto aborda serviços de manutenção e suporte técnico para no-breaks, estabilizadores e outras ... | ⬜ A avaliar    |
|         15 | Quais são as aplicações mais modernas do DSP no processamento digital de sinais? | Alta Tecnolog ia em Processament o Dig ital de Sinais - DSP OS MAIS MODERNOS                         | ⬜ A avaliar    |
|         16 | comparação tecnologias No-Break confiabilidade eficiência ambientes críticos     | O texto menciona um processador de sinal digital sem transformador, destacando a inovação do mode... | ⬜ A avaliar    |
|         17 | Quais são as especificações dos cabos de medição utilizados para a medição de... | ESPECIFICAÇÕES TÉCNICAS Cabos de Medição Para medição de impedância dos módulos Equaliz er Módulo... | ⬜ A avaliar    |
|         18 | De que maneira as abordagens sobre gestao de energia e os benefcios dos contr... | O texto aborda serviços de manutenção e suporte técnico para no-breaks, estabilizadores e outras ... | ⬜ A avaliar    |
|         19 | Qual é a importncia do 'Status chaves internas' no funcinamento do painel?       | es •Status de operação e alar mes do painel •Status chav es inter nasn Proteç                        | ⬜ A avaliar    |
|         20 | Quais são as dimensões físicas e a potência dos modelos mencionados nas certi... | CIFICAÇÕES TÉCNICAS Características Físicas e Mecânicas Dimensões Compac tas Display TFT 4,3" Tou... | ⬜ A avaliar    |

---

## 2. Estatísticas do Resumo

| Classificação | Quantidade (N) | Proporção (%) |
|:---|:---:|:---:|
| Relevante | `__` / 20 | `__%` |
| Parcialmente Relevante | `__` / 20 | `__%` |
| Irrelevante | `__` / 20 | `__%` |
| **Total Avaliado** | **`0` / 20** | **`0.0%`** |

> *Nota: As 20 amostras estão atualmente com status `⬜ A avaliar`. Preencha a coluna 'Classificação' na tabela e execute novamente o script para calcular automaticamente as estatísticas consolidadas.*

---

## 3. Conclusão sobre o Nível de Viés Circular

### Critérios de Diagnóstico de Viés para o TCC:
- **Validade Alta / Baixo Viés (>80% Relevante)**:
  O conjunto sintético possui representatividade substantiva do corpus técnico. A superioridade de modelos de ponta decorre primordialmente da qualidade do espaço latente e não de artefatos metodológicos.
- **Validade Média / Viés Moderado (60% a 80% Relevante)**:
  Verifica-se incidência relevante de perguntas com contextos parciais ou ruídos de extração. Métricas absolutas (MRR, Hit Rate) podem estar infladas para o modelo baseline da OpenAI, tornando indispensável o uso de testes não-paramétricos (Wilcoxon pareado) para validação estatística.
- **Validade Baixa / Alto Viés (<60% Relevante)**:
  Contaminação substancial do conjunto de testes. A aderência dos modelos deve ser interpretada com forte ressalva metodológica na discussão de resultados da monografia.

### Síntese da Avaliação Manual (Template para a Monografia):
- **Diagnóstico Final de Validade**: `[ ] Alta (>80% Relevante) | [ ] Média (60-80% Relevante) | [ ] Baixa (<60% Relevante)`
- **Padrões de Falha e Artefatos Observados**:
  - *Ruídos de Chunking*: [e.g., cabeçalho institucional 'CM Comandos Lineares' sintetizado incorretamente como comando de programação/automação]
  - *Contextos Genéricos*: [e.g., trechos puramente comerciais ou notas de rodapé de catálogo sem informação técnica]
- **Impacto no Ranking Comparativo dos 7 Modelos de Embedding**:
  - [Avaliar em que medida os modelos open-source foram penalizados por artefatos que coincidem com os padrões gerados pela API da OpenAI]
- **Recomendação para a Redação do TCC**:
  - [Registrar explicitamente esta auditoria no capítulo de Metodologia e Limitações do Trabalho]

---

## 4. Detalhamento Integral das Amostras para Auditoria

### Amostra 1 (Índice no testset: `28` | Linha: `29`)

- **Pergunta (`user_input`)**:
  > Quais sã as implicações do uso de comands linears na programção e automação de processos?
- **Tipo de Sintetizador**: `AbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 3 trecho(s))**:

  **Contexto 1:**
  ```text
  CM Comandos Lineares
  ```

  **Contexto 2:**
  ```text
  CM COMANDOS LINEARESCM COMANDOS LINEARES
  ```

  **Contexto 3:**
  ```text
  CM COMANDOS LINEARES
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As implicações do uso de comandos lineares na programação e automação de processos não estão claramente definidas no contexto fornecido.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 2 (Índice no testset: `6` | Linha: `7`)

- **Pergunta (`user_input`)**:
  > Como o sistema Paralelo Multi Ativo muda a confiabilidade dos no-breaks?
- **Tipo de Sintetizador**: `AbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 2 trecho(s))**:

  **Contexto 1:**
  ```text
  S I S T E M A PA R A L E L O M U LT I AT I V O ( E X C L U S I V O C M C O M A N D O S )
A CM Comandos Lineares desenvolveu um sistema inovador e pioneiro no mundo de paralelismo entre No Breaks. O sistema Paralelo 
Multi Ativo permite a expansão do sistema, de acordo com sua demanda de consumo, podendo paralelar quantas unidades forem 
necessárias para atender sua demanda, elevando a capacidade do sistema e a confiabilidade de sua aplicação.
  ```

  **Contexto 2:**
  ```text
  SISTEMA PARALELO MULTI ATIVO (EXCLUSIVO CM COMANDOS)
A CM Comandos Lineares desenvolveu um sistema inovador e pioneiro no mundo de paralelismo entre no-breaks. O sistema 
Paralelo Multi Ativo permite a expansão do sistema, de acordo com sua demanda de consumo, podendo paralelar quantas 
unidades forem necessárias para atender sua demanda, elevando a capacidade do sistema e a confiabilidade de sua aplicação.
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > O sistema Paralelo Multi Ativo muda a confiabilidade dos no-breaks ao permitir a expansão do sistema de acordo com a demanda de consumo, elevando a capacidade do sistema e a confiabilidade de sua aplicação.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 3 (Índice no testset: `70` | Linha: `71`)

- **Pergunta (`user_input`)**:
  > Quais são os benefícios da tecnologia DSP em relação à alta performance e confiabilidade dos no-breaks corporativos?
- **Tipo de Sintetizador**: `SpecificQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 1 trecho(s))**:

  **Contexto 1:**
  ```text
  NO-BREAKS CORPORATIVOSVivemos em um mundo cada v ez mais ág il e prático . Um mundo conec tado por pr ocessador es, chips , softwares e 
periféricos. Mas isso ainda não é o bastant e. Por isso , o mundo caminha para t ecnolog ias de pr ocessament o imediat o 
de dados . A CM C omandos Linear es está um passo à fr ente e of erece aos seus client es pr odut os dotados de uma 
tecnolog ia revolucionár ia, denominada P rocessament o Dig ital de Sinais - - DSP .
Toda a linha de No -Breaks  
conta com a t ecnolog ia DSP .  Carac terística que 
propor ciona alta per formance e confiabilidade ,
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > Os benefícios da tecnologia DSP em relação à alta performance e confiabilidade dos no-breaks corporativos incluem a capacidade de proporcionar alta performance e confiabilidade.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 4 (Índice no testset: `62` | Linha: `63`)

- **Pergunta (`user_input`)**:
  > Como as abordagens de manutenção e suporte técnico se comparam em relação a contratos e suporte 24 horas?
- **Tipo de Sintetizador**: `ComparativeAbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 2 trecho(s))**:

  **Contexto 1:**
  ```text
  O texto aborda serviços de manutenção e suporte técnico para no-breaks, estabilizadores e outras soluções, destacando a importância de contratos de manutenção.
  ```

  **Contexto 2:**
  ```text
  O contrato de manutenção oferece vantagens como a redução da degradação dos equipamentos, prolongando sua vida útil e evitando custos elevados com intervenções corretivas. Ele proporciona suporte técnico 24 horas por dia, minimiza riscos de paradas e garante prioridade no atendimento. A manutenção preventiva pode reduzir a necessidade de manutenções corretivas em até 80%, resultando em economia. Além disso, serviços corretivos inclusos no contrato ajudam a manter os equipamentos em melhores condições. A empresa oferece atendimento em todo o país, com técnicos qualificados e equipados, visando excelência nos serviços prestados.
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As abordagens de manutenção e suporte técnico se comparam em relação a contratos e suporte 24 horas, onde o contrato de manutenção oferece vantagens como redução da degradação dos equipamentos, suporte técnico 24 horas, minimização de riscos de paradas e prioridade no atendimento.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 5 (Índice no testset: `57` | Linha: `58`)

- **Pergunta (`user_input`)**:
  > De que maneira as estratégias de redução de custos relacionadas à manutenção e suporte técnico se comparam em diferentes relatórios, considerando aspectos como contratos de manutenção, redução da degradação dos equipamentos e a eficácia da manutenção preventiva adequada?
- **Tipo de Sintetizador**: `ComparativeAbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 2 trecho(s))**:

  **Contexto 1:**
  ```text
  O texto aborda serviços de manutenção e suporte técnico para no-breaks, estabilizadores e outras soluções, destacando a importância de contratos de manutenção.
  ```

  **Contexto 2:**
  ```text
  O contrato de manutenção oferece vantagens como a redução da degradação dos equipamentos, prolongando sua vida útil e evitando custos elevados com intervenções corretivas. Ele proporciona suporte técnico 24 horas por dia, minimiza riscos de paradas e garante prioridade no atendimento. A manutenção preventiva pode reduzir a necessidade de manutenções corretivas em até 80%, resultando em economia. Além disso, serviços corretivos inclusos no contrato ajudam a manter os equipamentos em melhores condições. A empresa oferece atendimento em todo o país, com técnicos qualificados e equipados, visando excelência nos serviços prestados.
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As estratégias de redução de custos relacionadas à manutenção e suporte técnico se comparam em diferentes relatórios ao destacar a importância dos contratos de manutenção, que reduzem a degradação dos equipamentos e aumentam a eficácia da manutenção preventiva, resultando em economia significativa.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 6 (Índice no testset: `35` | Linha: `36`)

- **Pergunta (`user_input`)**:
  > importância sistema No-Break ambientes críticos
- **Tipo de Sintetizador**: `AbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 2 trecho(s))**:

  **Contexto 1:**
  ```text
  No-Break
  ```

  **Contexto 2:**
  ```text
  No-Break
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > O sistema No-Break é importante em ambientes críticos para garantir a continuidade do fornecimento de energia e proteger equipamentos contra quedas de energia.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 7 (Índice no testset: `26` | Linha: `27`)

- **Pergunta (`user_input`)**:
  > como No-Breaks CM Comandos Lineares garantem continuidade operacional missão crítica
- **Tipo de Sintetizador**: `AbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 4 trecho(s))**:

  **Contexto 1:**
  ```text
  APLICAÇÕES Os No -Breaks da CM C omandos Linear es são indicados para 
aplicações de missão cr ítica, nas mais var iadas cor porações ,  
oferecendo pr oteção contra distúr bios de ener gia elétr ica, tais como: cor tes, micr o cor tes, 
ruídos , dist orção har mônica, picos , transient es, sub e sobr e tensão e var iações de fr equência.
  ```

  **Contexto 2:**
  ```text
  APLICAÇÕES Os No -Breaks da CM C omandos Linear es são indicados para 
aplicações de missão cr ítica, nas mais var iadas cor porações ,  
oferecendo por teção contra distúr bios de ener gia elétr ica, tais como cor tes, micr o cor tes, 
ruídos , dist orção har mônica, picos , transient es, sub e sobr e tensão e var iações de fr equência.
  ```

  **Contexto 3:**
  ```text
  A P L I C A Ç Õ E SOs No Breaks  da CM Comandos Lineares são indicados para aplicações 
de missão crítica, nas mais variadas corporações, oferecendo proteção 
contra distúrbios de energia elétrica, tais como: cortes, micro cortes, ruídos, distorção harmônica, 
picos, transientes, sub e sobre tensão e variações de frequência.
  ```

  <details>
  <summary><i>Ver mais 1 contextos adicionais...</i></summary>

  **Contexto 4:**
  ```text
  APLICA ÇÕES Os No -Breaks da CM C omandos Linear es são indicados para 
aplicações de missão cr ítica, nas mais var iadas cor porações ,  
oferecendo pr oteção contra distúr bios de ener gia elétr ica, tais como: cor tes, micr o cor tes, 
ruídos , dist orção har mônica, picos , transient es, sub e sobr etensão e var iações de fr equência.
  ```
  </details>

- **Resposta de Referência Sintetizada (`reference`)**:
  > Os No-Breaks da CM Comandos Lineares garantem continuidade operacional em aplicações de missão crítica, oferecendo proteção contra distúrbios de energia elétrica, como cortes, micro cortes, ruídos, distorção harmônica, picos, transientes, sub e sobre tensão e variações de frequência.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 8 (Índice no testset: `22` | Linha: `23`)

- **Pergunta (`user_input`)**:
  > Quais os impactos das variações de tensão e corrente nas medições e desempenho dos sistemas?
- **Tipo de Sintetizador**: `AbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 2 trecho(s))**:

  **Contexto 1:**
  ```text
  ão Elétricas
Sub e Sobr etensão de Entrada e Saída
Sub e Sobr etensão DC e Bater ia
Sobr ecar ga e C urto-Circuito
Mínima Descar ga de Bater ian     
n     
n     
n     
Medições
True RMS
Potência de Saída em kV A
 Potência de Saída em kW
Fator de P otência de Saída
Tensão de Saída
Corrente de Saída
Frequência de Saída
Tensão de Bater ia
Tensão de Entrada
Frequência de Entradan   
n   
n  
n   
n   
n   
n   
n   
n   
n   Caract
  ```

  **Contexto 2:**
  ```text
  létricas  
n  
n     
n     
n      Sub e Sobr etensão de Entrada e Saída  
Sub e Sobr etensão DC e Bater ia
Sobr ecar ga e C urto-Circuito
Mínima Descar ga de Bater ia
Medições
True RMS
Potência de Saída em kV A
Potência de Saída em kW
Fator de P otência de Saída
Tensão de Saída
Corrente de Saída
Frequências de Saída
Tensão de Bater ia
Tensão de Entrada
Frequência de Entradan   
n   
n   
n   
n   
n   
n   
n   
n   
n   Caracterí
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As variações de tensão e corrente impactam as medições e o desempenho dos sistemas, afetando a precisão das medições de potência, tensão e corrente, além de influenciar a eficiência operacional dos equipamentos.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 9 (Índice no testset: `108` | Linha: `109`)

- **Pergunta (`user_input`)**:
  > Quais implicações especificações CM MKT OUT 23 sujeitas alterações prévio aviso?
- **Tipo de Sintetizador**: `SpecificQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 1 trecho(s))**:

  **Contexto 1:**
  ```text
  CM. MKT OUT 23 - As especificações estão sujeitas a alterações sem prévio aviso.
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As especificações CM MKT OUT 23 estão sujeitas a alterações sem prévio aviso.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 10 (Índice no testset: `8` | Linha: `9`)

- **Pergunta (`user_input`)**:
  > impacto somatória potências No-Breaks eficiência confiabilidade energia sistemas críticos
- **Tipo de Sintetizador**: `AbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 2 trecho(s))**:

  **Contexto 1:**
  ```text
  P O S S I B I L I D A D E D E S O M A T Ó R I A D E  P O T Ê N C I A S
Permite a ligação para somatória de potência, utilizando dois ou mais No 
Breaks para atingir a demanda da carga total.
Quando utilizado para somatória de potência, possibilita a ligação de 
No Breaks com potências diferentes. Quando a ligação for feita com No 
Breaks de potências diferentes, a carga será distribuída 
proporcionalmente. Permite a
  ```

  **Contexto 2:**
  ```text
  POSSIBILIDADE DE SOMATÓRIA DE  POTÊNCIAS
Permite a ligação para somatória de potência, utilizando dois ou mais 
No-Breaks para atingir a demanda da carga total.
Quando utilizado para somatória de potência, possibilita a ligação de 
No-Breaks com potências diferentes. Quando a ligação for feita com 
No-Breaks de potências diferentes, a carga será distribuída 
proporcionalmente. Permite a ligação em paralelo redundante N + 1,  N + 2, N + 3 … N+X,  
possibilitando o desligamento de uma ou mais unidades, mantendo a 
carga ativa.
A co
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > A somatória de potências em No-Breaks permite a ligação de múltiplas unidades para atender à demanda total da carga, aumentando a eficiência e confiabilidade dos sistemas críticos.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 11 (Índice no testset: `7` | Linha: `8`)

- **Pergunta (`user_input`)**:
  > De que forma o sistema Paralelo Multi Ativo da CM Comandos Lineares ajuda na expanção e confiabilidade em aplicaçõe que precisam de no-breaks?
- **Tipo de Sintetizador**: `AbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 2 trecho(s))**:

  **Contexto 1:**
  ```text
  S I S T E M A PA R A L E L O M U LT I AT I V O ( E X C L U S I V O C M C O M A N D O S )
A CM Comandos Lineares desenvolveu um sistema inovador e pioneiro no mundo de paralelismo entre No Breaks. O sistema Paralelo 
Multi Ativo permite a expansão do sistema, de acordo com sua demanda de consumo, podendo paralelar quantas unidades forem 
necessárias para atender sua demanda, elevando a capacidade do sistema e a confiabilidade de sua aplicação.
  ```

  **Contexto 2:**
  ```text
  SISTEMA PARALELO MULTI ATIVO (EXCLUSIVO CM COMANDOS)
A CM Comandos Lineares desenvolveu um sistema inovador e pioneiro no mundo de paralelismo entre no-breaks. O sistema 
Paralelo Multi Ativo permite a expansão do sistema, de acordo com sua demanda de consumo, podendo paralelar quantas 
unidades forem necessárias para atender sua demanda, elevando a capacidade do sistema e a confiabilidade de sua aplicação.
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > O sistema Paralelo Multi Ativo da CM Comandos Lineares ajuda na expansão e confiabilidade em aplicações que precisam de no-breaks, permitindo a paralelização de quantas unidades forem necessárias para atender a demanda de consumo.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 12 (Índice no testset: `23` | Linha: `24`)

- **Pergunta (`user_input`)**:
  > Quais os impacto das variaçõe de tensão e corrente nas medições elétrica e no desempenho dos sistema elétrico?
- **Tipo de Sintetizador**: `AbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 2 trecho(s))**:

  **Contexto 1:**
  ```text
  ão Elétricas
Sub e Sobr etensão de Entrada e Saída
Sub e Sobr etensão DC e Bater ia
Sobr ecar ga e C urto-Circuito
Mínima Descar ga de Bater ian     
n     
n     
n     
Medições
True RMS
Potência de Saída em kV A
 Potência de Saída em kW
Fator de P otência de Saída
Tensão de Saída
Corrente de Saída
Frequência de Saída
Tensão de Bater ia
Tensão de Entrada
Frequência de Entradan   
n   
n  
n   
n   
n   
n   
n   
n   
n   Caract
  ```

  **Contexto 2:**
  ```text
  létricas  
n  
n     
n     
n      Sub e Sobr etensão de Entrada e Saída  
Sub e Sobr etensão DC e Bater ia
Sobr ecar ga e C urto-Circuito
Mínima Descar ga de Bater ia
Medições
True RMS
Potência de Saída em kV A
Potência de Saída em kW
Fator de P otência de Saída
Tensão de Saída
Corrente de Saída
Frequências de Saída
Tensão de Bater ia
Tensão de Entrada
Frequência de Entradan   
n   
n   
n   
n   
n   
n   
n   
n   
n   Caracterí
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As variações de tensão e corrente impactam as medições elétricas e o desempenho do sistema elétrico, afetando a potência de saída, a precisão das medições e a eficiência geral do sistema.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 13 (Índice no testset: `55` | Linha: `56`)

- **Pergunta (`user_input`)**:
  > Como as abordagens sobre suporte técnico e manutenção preventiva se comparam em diferentes relatórios, considerando aspectos como a redução da degradação dos equipamentos, a importância de contratos de manutenção e a disponibilidade de suporte técnico 24 horas?
- **Tipo de Sintetizador**: `ComparativeAbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 2 trecho(s))**:

  **Contexto 1:**
  ```text
  O texto aborda serviços de manutenção e suporte técnico para no-breaks, estabilizadores e outras soluções, destacando a importância de contratos de manutenção.
  ```

  **Contexto 2:**
  ```text
  O contrato de manutenção oferece vantagens como a redução da degradação dos equipamentos, prolongando sua vida útil e evitando custos elevados com intervenções corretivas. Ele proporciona suporte técnico 24 horas por dia, minimiza riscos de paradas e garante prioridade no atendimento. A manutenção preventiva pode reduzir a necessidade de manutenções corretivas em até 80%, resultando em economia. Além disso, serviços corretivos inclusos no contrato ajudam a manter os equipamentos em melhores condições. A empresa oferece atendimento em todo o país, com técnicos qualificados e equipados, visando excelência nos serviços prestados.
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As abordagens sobre suporte técnico e manutenção preventiva se comparam em diferentes relatórios ao destacar a redução da degradação dos equipamentos, a importância de contratos de manutenção e a disponibilidade de suporte técnico 24 horas, evidenciando que o contrato de manutenção prolonga a vida útil dos equipamentos e minimiza custos com intervenções corretivas.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 14 (Índice no testset: `59` | Linha: `60`)

- **Pergunta (`user_input`)**:
  > Como abordagens manutenção suporte técnico relatórios se comparam contratos manutenção redução degradação equipamentos eficácia manutenção preventiva?
- **Tipo de Sintetizador**: `ComparativeAbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 2 trecho(s))**:

  **Contexto 1:**
  ```text
  O texto aborda serviços de manutenção e suporte técnico para no-breaks, estabilizadores e outras soluções, destacando a importância de contratos de manutenção.
  ```

  **Contexto 2:**
  ```text
  O contrato de manutenção oferece vantagens como a redução da degradação dos equipamentos, prolongando sua vida útil e evitando custos elevados com intervenções corretivas. Ele proporciona suporte técnico 24 horas por dia, minimiza riscos de paradas e garante prioridade no atendimento. A manutenção preventiva pode reduzir a necessidade de manutenções corretivas em até 80%, resultando em economia. Além disso, serviços corretivos inclusos no contrato ajudam a manter os equipamentos em melhores condições. A empresa oferece atendimento em todo o país, com técnicos qualificados e equipados, visando excelência nos serviços prestados.
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As abordagens de manutenção e suporte técnico, especialmente através de contratos de manutenção, se comparam positivamente à manutenção preventiva, pois reduzem a degradação dos equipamentos e aumentam a eficácia da manutenção, podendo diminuir a necessidade de manutenções corretivas em até 80%.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 15 (Índice no testset: `129` | Linha: `130`)

- **Pergunta (`user_input`)**:
  > Quais são as aplicações mais modernas do DSP no processamento digital de sinais?
- **Tipo de Sintetizador**: `SpecificQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 1 trecho(s))**:

  **Contexto 1:**
  ```text
  Alta Tecnolog ia em
Processament o Dig ital
de Sinais - DSP
OS MAIS MODERNOS
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As aplicações mais modernas do DSP no processamento digital de sinais incluem diversas tecnologias avançadas, embora o contexto fornecido não detalhe especificamente quais são essas aplicações.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 16 (Índice no testset: `50` | Linha: `51`)

- **Pergunta (`user_input`)**:
  > comparação tecnologias No-Break confiabilidade eficiência ambientes críticos
- **Tipo de Sintetizador**: `ComparativeAbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 54 trecho(s))**:

  **Contexto 1:**
  ```text
  O texto menciona um processador de sinal digital sem transformador, destacando a inovação do modelo S1 e suas características de comando linear e operação trifásica.
  ```

  **Contexto 2:**
  ```text
  Os No-Breaks Composition trifásico são modulares e on line de dupla conversão, projetados para integração em data centers. Cada módulo possui tecnologia hotswap, permitindo substituição sem parar o sistema. Eles oferecem alta confiabilidade e estão disponíveis em várias capacidades, possibilitando expansão conforme a necessidade. As principais características incluem controle DSP, retificador e inversor a IGBT, e gerenciamento avançado de baterias. O design é modular e escalável, com ampla faixa de tensão admissível e compatibilidade com grupos geradores. Também possuem software de gerenciamento remoto e opções de paralelismo redundante.
  ```

  **Contexto 3:**
  ```text
  O processador de sinal digital é utilizado em sistemas de no-break trifásicos, como o Multi Ativo, para gerenciar comandos lineares e concepção de energia.
  ```

  <details>
  <summary><i>Ver mais 51 contextos adicionais...</i></summary>

  **Contexto 4:**
  ```text
  O Sistema IT Médico é uma supervisão de falhas de isolação em instalações elétricas hospitalares, conforme a norma NBR 13534:2008 da Anvisa. Ele é obrigatório em ambientes hospitalares do Grupo 2, como salas cirúrgicas e UTIs. O sistema aumenta a segurança ao evitar interrupções no fornecimento de energia elétrica durante falhas, permitindo que equipamentos eletrônicos continuem a funcionar. Além disso, reduz correntes de fuga, diminuindo a tensão de contato e o risco de choques elétricos. O sistema inclui dispositivos como o Dispositivo Supervisor de Isolamento (DSI), Supervisor de Temperatura (DST) e Supervisor de Corrente (DSC), que monitoram a resistência e sinalizam falhas. A composição básica do sistema inclui um transformador de separação e outros dispositivos de supervisão e anúncio.
  ```
  **Contexto 5:**
  ```text
  A CM Comandos é uma das maiores fabricantes de No-Breaks da América Latina, com mais de 38 anos de experiência e certificação ISO-9001:2015. A empresa se destaca em soluções para o mercado corporativo, oferecendo proteção contra distúrbios de energia elétrica. Seus No-Breaks são indicados para aplicações de missão crítica e garantem alta confiabilidade e produtividade. A CM Comandos também oferece suporte técnico de excelência, com atendimento 24 horas e profissionais qualificados. Os equipamentos da empresa são reconhecidos por sua precisão e segurança, minimizando falhas e custos de manutenção. Ser cliente da CM Comandos significa retorno do capital investido e garantias de um suporte técnico robusto.
  ```
  **Contexto 6:**
  ```text
  A CM Comandos Lineares é uma empresa auditada pelo sistema de qualidade ISO 9001:2015, localizada em São Paulo. Oferece informações de contato, incluindo telefone e fax, além de um site para mais detalhes.
  ```
  **Contexto 7:**
  ```text
  O sistema de No-Break com Bypass Estático de Manutenção utiliza tecnologia DSP para medições precisas e permite gerenciamento remoto em tempo real. Ele é ideal para cargas não lineares, garantindo proteção e confiabilidade. O sistema pode ser expandido conforme a demanda e possui memória interna para registro de eventos. A tela touch screen de 4,3" facilita o acesso e controle via internet. O Bypass Estático assegura fornecimento contínuo de energia durante manutenções. O retificador converte energia elétrica em corrente contínua, enquanto o inversor transforma essa corrente em alternada. O sistema também conta com interfaces de gerenciamento remoto, permitindo monitoramento e automação de servidores. O software IP Power automatiza o desligamento de múltiplos servidores simultaneamente.
  ```
  **Contexto 8:**
  ```text
  A CM Comandos é uma das maiores fabricantes de No-Breaks da América Latina, com mais de 40 anos de atuação e certificação ISO-9001:2015. A empresa se destaca pela inovação, qualidade e confiabilidade de seus produtos, que são indicados para aplicações de missão crítica. Os No-Breaks oferecem proteção contra distúrbios de energia elétrica, como cortes e picos. Os equipamentos são projetados para operar com alta precisão e segurança, minimizando falhas e custos de manutenção. A CM Comandos também oferece suporte técnico de excelência, com atendimento 24 horas e profissionais qualificados.
  ```
  **Contexto 9:**
  ```text
  O sistema de Equalização Ativa para gerenciamento de baterias maximiza a vida útil das baterias, reduzindo a necessidade de trocas e custos de manutenção. Ele automatiza a manutenção, analisa a vida útil das baterias e minimiza o tempo de inatividade devido a falhas. O monitoramento em tempo real e a medição da impedância garantem que as baterias operem em condições ideais. O sistema é ecológico, reduzindo o consumo de baterias e o descarte ambiental. É aplicável em sistemas No-Break, retificadores de telecomunicações e energia solar e eólica, compatível com diversas marcas e tipos de baterias.
  ```
  **Contexto 10:**
  ```text
  O DominionSP1 é um sistema avançado que utiliza tecnologia DSP para medições precisas e correção do fator de potência. Ele permite o monitoramento em tempo real e o registro de eventos, sendo ideal para cargas não lineares. O sistema possui um modo de bypass estático que permite manutenção sem desligar o equipamento. O retificador converte energia da rede em corrente contínua, enquanto o inversor transforma essa energia em corrente alternada de saída. O banco de baterias armazena energia para garantir autonomia. O transformador isolador melhora a qualidade da energia, filtrando ruídos. O sistema também oferece a possibilidade de upgrade de firmware e gerenciamento remoto via internet, assegurando precisão e confiabilidade.
  ```
  **Contexto 11:**
  ```text
  O documento apresenta informações sobre um sistema de sustentabilidade auditado, incluindo especificações técnicas de modelos trifásicos de transformadores. Detalha potências disponíveis, dimensões físicas, características de operação e proteções elétricas. As potências variam de 10 a 800 kVA, com tensões de entrada e saída específicas. O sistema possui um display LCD, estrutura metálica e ventilação forçada. Inclui alarmes sonoros para subtensão e sobretensão, além de um sistema de registro de eventos. As condições de operação recomendadas incluem temperatura e umidade específicas. O acesso para manutenção é facilitado por uma interface de comunicação serial. O documento também menciona a possibilidade de customização das especificações.
  ```
  **Contexto 12:**
  ```text
  A C M Comandos Lineares oferece No-Breaks Dominion SP1 com tecnologia de Processamento Digital de Sinais (DSP), que proporciona alta performance e confiabilidade para aplicações críticas. Essa tecnologia permite processamento em tempo real, com capacidade de processar dez milhões de amostragens por segundo, sem atrasos. Os No-Breaks são projetados para serem atualizáveis, adicionando novas funções a unidades já instaladas. A tecnologia DSP é amplamente utilizada em sistemas eletrônicos sofisticados, garantindo máxima proteção e precisão digital.
  ```
  **Contexto 13:**
  ```text
  O estabilizador eletrônico Digital Signal Processor PERFECTION Série Premium é um dispositivo que oferece proteção e estabilidade para equipamentos eletrônicos.
  ```
  **Contexto 14:**
  ```text
  O Estabilizador Perfection SP utiliza tecnologia DSP e conversores a IGBTs para corrigir variações de tensão com precisão de ±1%. Possui sistema de proteção contra sub e sobretensão, desligando automaticamente a saída em caso de anomalias. O dispositivo registra até 25.500 eventos, permitindo rastreabilidade total. Com velocidade de correção inferior a 4 milissegundos, é ideal para cargas com altos picos de corrente. O sistema Bypass Automático garante a continuidade do fornecimento de energia em caso de falhas. O estabilizador é projetado para alta performance e eficiência elétrica, gerando menor dissipação térmica e economia de energia. Sua interface LCD exibe status de operação e alarmes, controlados em tempo real pelo microprocessador DSP.
  ```
  **Contexto 15:**
  ```text
  O No-Break é um dispositivo que fornece energia ininterrupta, utilizando um processador digital de sinal para garantir a prevenção de falhas elétricas.
  ```
  **Contexto 16:**
  ```text
  A CM Comandos é uma das maiores fabricantes de No-Breaks da América Latina, com mais de 40 anos de experiência e certificação ISO-9001:2015. A empresa se destaca no mercado corporativo, oferecendo soluções para proteção contra distúrbios de energia elétrica. Seus No-Breaks são indicados para aplicações críticas, garantindo alta confiabilidade e produtividade. A CM Comandos proporciona suporte técnico de excelência, com atendimento 24 horas e profissionais qualificados. Escolher a CM Comandos significa ter retorno do capital investido e garantia de qualidade nos serviços.
  ```
  **Contexto 17:**
  ```text
  A tecnologia de Processamento Digital de Sinais (DSP) é essencial em sistemas eletrônicos modernos, permitindo processamento em tempo real com alta velocidade e confiabilidade. A CM Comandos oferece No-Breaks Composition que utilizam DSP, garantindo alta performance e proteção para aplicações críticas. Esses dispositivos permitem atualizações de firmware, adicionando novas funções a equipamentos já instalados, assegurando tecnologia de ponta e precisão digital.
  ```
  **Contexto 18:**
  ```text
  O No-Break Senoidal Microprocessado possui tensões selecionáveis de 115V a 220V e potências disponíveis de 1,5 kVA a 3,0 kVA. A regulação estática é de ± 2% em modo bateria, com frequência de 60 Hz. O sistema de recarga é automático, levando de 8 a 10 horas para atingir 90% da carga. O equipamento é compacto, com dimensões para montagem em rack de 19 polegadas e possui um display LCD para status e alarmes. A proteção inclui sobrecarga e descarga de bateria baixa, com um ruído audível inferior a 45 dBA. O No-Break é adequado para ambientes internos, com um grau de proteção IP-20 e um MTBF de 55 mil horas.
  ```
  **Contexto 19:**
  ```text
  O recurso de registro permanente de eventos internos permite a atualização do firmware no chip DSP, melhorando a implementação de IAs e preservando investimentos. O DSP armazena até 25.500 registros, que permanecem gravados mesmo com o estabilizador desligado. Funciona como uma 'caixa-preta', rastreando medições e status de alarmes para auxiliar na manutenção. Os eventos são registrados com data e hora através de um relógio em tempo real. O sistema é aplicável em diversas áreas, como automação bancária, industrial e comercial, além de sistemas médicos e telecomunicações.
  ```
  **Contexto 20:**
  ```text
  O retificador converte a energia elétrica em corrente contínua, enquanto o inversor transforma essa corrente em corrente alternada senoidal pura. O sistema de Bypass Estático transfere automaticamente a carga para um circuito alternativo em caso de problemas. O estabilizador eletrônico Perfection MI S2 possui tecnologia de duplos conversores estáticos e é controlado por um microprocessador DSP. Ele corrige variações de tensão, atenua distorções harmônicas e fornece uma forma de onda senoidal pura. O estabilizador é projetado para aplicações sensíveis e pode lidar com picos de alta corrente. Suas especificações incluem potências de 2 a 15 kVA, com regulação estática de ±1%. O equipamento opera em temperaturas de 0ºC a 40ºC e tem um grau de proteção IP-20. Alarmes sonoros e visuais são controlados pelo DSP, e o sistema é projetado para alta confiabilidade e manutenção simplificada.
  ```
  **Contexto 21:**
  ```text
  O Equalizer é uma solução brasileira de Equalização Ativa Individual de Baterias que garante que todas as baterias tenham a mesma tensão, evitando sobrecargas e subcargas. Cada bateria é equipada com um Módulo de Bateria conectado a um Master Controller, que supervisiona e regula a tensão. Isso permite que as baterias atinjam 100% de sua vida útil, reduzindo a necessidade de substituições e reciclagem. O sistema também monitora a resistência interna e a temperatura de cada bateria, emitindo alarmes antes de falhas. Assim, o Equalizer aumenta a confiabilidade do sistema e diminui o tempo de inatividade.
  ```
  **Contexto 22:**
  ```text
  O sistema de Bypass Estático é acionado automaticamente pelo DSP, permitindo manutenção do No-Break sem desligar a carga. Alarmes sonoros e mensagens de alerta são controlados pelo processador DSP e podem ser enviados por e-mail. O log de eventos armazena registros de ocorrências, medições e status de operação. O equipamento possui proteção contra sub e sobretensão, sobrecarga e curto-circuito. As medições incluem potência de saída, tensão e corrente. O design físico é compacto, com um display LCD retroiluminado e estrutura metálica. O sistema é gerenciado por interfaces que suportam múltiplos usuários e protocolos como SNMP e TCP/IP. A configuração elétrica admite variações de tensão e frequência, com potências disponíveis de 5 a 15 kVA. O sistema de baterias é automático, com tempo de recarga de 8 a 10 horas para 90% da carga.
  ```
  **Contexto 23:**
  ```text
  O No-Break D O M I N I O N é um dispositivo que utiliza um processador de sinal digital, o SP1, para garantir a continuidade de energia em caso de falhas elétricas.
  ```
  **Contexto 24:**
  ```text
  O texto apresenta informações sobre um transformador de separação com alta isolação galvânica, que gera um circuito elétrico isolado monitorado por um sistema de alarme. Em caso de falha de isolação, o sistema aciona um alarme para a equipe de manutenção. O sistema IT inclui uma interface homem-máquina (IHM) e um localizador de falhas. O documento também menciona que as especificações técnicas podem ser alteradas sem aviso prévio e incentiva a reciclagem.
  ```
  **Contexto 25:**
  ```text
  O módulo de equalização ativa regula a tensão de cada bateria individualmente, aumentando sua durabilidade e resistência. Ele protege as baterias contra excesso e falta de carga, evitando problemas como sulfatação e secagem do eletrólito. O sistema monitora a tensão, impedância e temperatura, identificando problemas típicos das baterias. A equalização ativa mantém todas as baterias com tensão equilibrada, prevenindo contaminação entre elas. O sistema também otimiza a capacidade das baterias e alerta sobre a necessidade de substituição. A manutenção pode ser realizada remotamente, sem desligar o sistema, e a equipe de manutenção é acionada apenas quando necessário. O hardware e software do sistema garantem monitoramento eficiente e acesso a dados em tempo real.
  ```
  **Contexto 26:**
  ```text
  A CM Comandos Lineares oferece no-breaks corporativos com tecnologia de Processamento Digital de Sinais (DSP), que proporciona alta performance e confiabilidade. Essa tecnologia permite o processamento em tempo real, com capacidade de processar dez milhões de amostragens por segundo. Os no-breaks são projetados para aplicações de missão crítica, garantindo a continuidade das operações. Além disso, os equipamentos podem ser atualizados para adicionar novas funções. A tecnologia DSP é amplamente utilizada em sistemas eletrônicos sofisticados, assegurando máxima proteção e precisão digital.
  ```
  **Contexto 27:**
  ```text
  O documento apresenta especificações técnicas de um No-Break, incluindo tensões de entrada e saída trifásicas, tempo de recarga, proteções elétricas e características de operação. O equipamento possui um display TFT de 4,3" e estrutura metálica, com ventilação forçada e controle digital. Suporta vários servidores e oferece interfaces de gerenciamento, além de alarmes sonoros e mensagens de alerta. A autonomia da NVRAM é de 5 anos e o ruído audível varia entre 55 dBA a 60 dBA. O MTBF é de 200 mil horas e a temperatura ambiente recomendada é de 20ºC a 25ºC. O documento também lista dimensões e pesos de diferentes modelos, tanto com quanto sem baterias internas.
  ```
  **Contexto 28:**
  ```text
  A CM Comandos, com mais de 38 anos de atuação, é uma das maiores fabricantes de No-Breaks da América Latina e líder em soluções para o mercado corporativo. Certificada pela norma ISO-9001:2015, a empresa se compromete com a excelência e satisfação dos clientes. Seus No-Breaks são indicados para aplicações de missão crítica, oferecendo proteção contra distúrbios de energia elétrica. A CM Comandos também se destaca pelo suporte técnico de alta qualidade, disponível 24 horas por dia. Os equipamentos da empresa operam com precisão, agregam funções e são seguros, resultando em alta confiabilidade e produtividade. Ser cliente da CM Comandos garante retorno do capital investido e suporte técnico abrangente.
  ```
  **Contexto 29:**
  ```text
  As especificações técnicas incluem tensões de entrada e saída de 380/400/415V e 200/208/220V, com variações admissíveis de ± 25% e frequências de 50/60 Hz. A regulação estática é de ± 1% e a forma de onda é senoidal, com distorção harmônica THD ≤ 1%. O sistema de recarga é automático, com tempo de recarga de 8 a 10 horas. O by-pass estático permite manutenção sem desligar a carga, e as proteções elétricas incluem sobrecarga e curto-circuito. O dispositivo possui alarmes sonoros e mensagens de alerta via display e e-mail. As características operacionais incluem um ruído audível de 55 dBA a 60 dBA e uma temperatura ambiente recomendada de 20ºC a 25ºC. O gabinete é metálico, com ventilação forçada e controle digital de velocidade. O sistema é compatível com Windows e Linux, e as especificações estão sujeitas a alterações sem aviso prévio.
  ```
  **Contexto 30:**
  ```text
  O S3 C O M P O S I T I O N é um No-Break que utiliza um Processador Digital de Sinais e está disponível nas potências de 3,0, 6,0 e 10 kVA.
  ```
  **Contexto 31:**
  ```text
  A CM Comandos, com mais de 35 anos de atuação, é uma das maiores fabricantes de No-Breaks da América Latina e líder em soluções para o mercado corporativo. Certificada pela norma ISO-9001:2015, a empresa se destaca pela excelência de seus produtos e satisfação do cliente. Seus No-Breaks são indicados para aplicações de missão crítica, protegendo contra distúrbios de energia elétrica. A CM Comandos oferece suporte técnico de alta qualidade, com atendimento 24 horas e processos certificados. Os equipamentos da empresa garantem maior precisão, segurança e confiabilidade, minimizando falhas e custos de manutenção. Ser cliente da CM Comandos significa retorno do capital investido e garantias de um suporte técnico robusto.
  ```
  **Contexto 32:**
  ```text
  O No-Break com Processador Digital de Sinais é modular, com capacidade variando de 25 a 500 kWC.
  ```
  **Contexto 33:**
  ```text
  A CM Comandos Lineares desenvolveu um sistema inovador de paralelismo entre No Breaks, chamado Paralelo Multi Ativo, que permite a expansão conforme a demanda de consumo. O sistema possibilita a somatória de potência utilizando múltiplos No Breaks, mesmo com potências diferentes, distribuindo a carga proporcionalmente. A comunicação entre os No Breaks é feita por um barramento comum, permitindo operação autônoma e controle individual. Não há um No Break mestre, permitindo a paralelização de quantas unidades forem necessárias. O sistema permite a inserção ou retirada de No Breaks sem interrupção. Interfaces de gerenciamento remoto, como o Adaptador SNMP NetMate e o software IP Power, possibilitam monitoramento e automação do shutdown de servidores.
  ```
  **Contexto 34:**
  ```text
  A CM Comandos Lineares criou um sistema pioneiro de paralelismo entre no-breaks, chamado Paralelo Multi Ativo, que permite a expansão conforme a demanda de consumo. O sistema possibilita a somatória de potências, conectando no-breaks de diferentes potências e distribuindo a carga proporcionalmente. Ele suporta configurações redundantes N + 1, N + 2, etc., permitindo o desligamento de unidades sem afetar a carga. A comunicação entre os no-breaks é feita por um barramento comum, onde são trocadas informações sobre carga e sincronismo. Todos os no-breaks operam de forma autônoma, sem a necessidade de um mestre ou escravo. O sistema permite a paralelização ilimitada de unidades e a manutenção sem interrupção. Também é possível o paralelismo dos bancos de baterias, desde que tenham a mesma tensão nominal. O gerenciamento remoto e as interfaces de shutdown são integrados ao sistema, facilitando o controle e a operação.
  ```
  **Contexto 35:**
  ```text
  As Interfaces de Gerenciamento Remoto incluem ferramentas como o Adaptador SNMP para monitoramento via Internet, um software para automatizar o desligamento de servidores e o Adaptador Modbus para integração com Sistemas de Automação Predial. O Bypass Estático transfere automaticamente a carga para um circuito alternativo em caso de problemas, garantindo fornecimento contínuo de energia. O Bypass Estático de Manutenção permite a transferência da carga para manutenção sem interrupções. O Banco de Baterias armazena energia reserva para o No-Break, enquanto o Retificador converte energia da rede elétrica em corrente contínua. O Inversor converte a corrente contínua em corrente alternada de saída, assegurando uma forma de onda senoidal pura.
  ```
  **Contexto 36:**
  ```text
  O documento apresenta especificações técnicas de um sistema de no-break, incluindo detalhes sobre entrada e saída de tensão, potências disponíveis, características físicas, interfaces de gerenciamento e proteções elétricas. O sistema opera com tensão trifásica e possui um fator de potência de até 0,99. A regulação estática é de ± 1% e a forma de onda é senoidal com distorção harmônica inferior a 1%. O tempo de recarga das baterias é de 8 a 10 horas, e o sistema possui um bypass estático para manutenção. As dimensões e peso variam conforme a potência, com modelos que vão de 5 kVA a 300 kVA. O no-break é projetado para operar em ambientes internos, com temperatura e umidade controladas. Alarmes e logs de eventos são gerenciados por um processador DSP, e o sistema é compatível com diversos protocolos de comunicação.
  ```
  **Contexto 37:**
  ```text
  As especificações técnicas incluem tensões de entrada e saída variando entre 100 a 240 VAC, com diferentes capacidades de 3kVA, 6kVA e 10kVA. A frequência é de 50/60 Hz, com variações admissíveis. O fator de potência é ≥ 0,99 e a distorção harmônica THDi é limitada a 5% para 3kVA e 4% para 6kVA/10kVA. As baterias são seladas e isentas de manutenção, com tempo de recarga de 8 a 10 horas. O sistema possui alarmes sonoros para falta de rede, bateria baixa e sobrecarga. As dimensões são compactas, com gabinete metálico e ventilação forçada. O ruído audível é inferior a 55 dBA e a temperatura de operação varia de 0ºC a 40ºC. O grau de proteção é IP-20 e o ambiente recomendado é interno e livre de partículas conducentes.
  ```
  **Contexto 38:**
  ```text
  O texto apresenta especificações técnicas de um sistema de no-break, incluindo características de entrada e saída, como tensões e potências disponíveis. Destaca a presença de um retificador, inversor e banco de baterias, além de um controle digital por microprocessador DSP. O sistema possui alarmes sonoros e mensagens de alerta, além de um log de eventos armazenados. A proteção elétrica abrange sobrecarga, curto-circuito e variações de tensão. O no-break é projetado para ser compacto, com um display LCD e estrutura metálica. Também menciona interfaces de gerenciamento e compatibilidade com diversos sistemas operacionais. As dimensões e pesos dos modelos disponíveis são especificados, assim como a certificação de sustentabilidade.
  ```
  **Contexto 39:**
  ```text
  A CM Comandos é uma das maiores fabricantes de No-Breaks da América Latina, com 38 anos de atuação e certificação ISO-9001:2015. A empresa é líder em soluções para o mercado corporativo, oferecendo produtos que protegem contra distúrbios de energia elétrica. Seus No-Breaks são indicados para aplicações de missão crítica, garantindo qualidade e confiabilidade. A CM Comandos também se destaca pelo suporte técnico de excelência, disponível 24 horas por dia. Os equipamentos da empresa são projetados para operar com alta precisão e segurança, minimizando falhas e custos de manutenção. Ser cliente da CM Comandos significa garantir retorno do capital investido e suporte técnico robusto.
  ```
  **Contexto 40:**
  ```text
  No-Breaks com Processadores de Sinal Digital estão disponíveis em potências de 2 a 15 kVA.
  ```
  **Contexto 41:**
  ```text
  O Transformador de Separação é projetado para suprir energia em sistemas médicos, seguindo normas IEC 61558-2-15 e NBR 13534. Possui isolamento galvânico reforçado e blindagem eletrostática entre os enrolamentos. O dispositivo Supervisor de Isolamento (DSI) monitora continuamente o isolamento da rede elétrica, emitindo alarmes em caso de falhas. O DSI é equipado com um microprocessador e comunicação Modbus via RS485. O transformador garante segurança em aplicações cirúrgicas, com sensores de temperatura do tipo PT-100. Suas características incluem potência monofásica de até 10 kVA e tensões primárias de até 1000 V. O equipamento é projetado para ser preciso, seguro e confiável, com testes de autodiagnóstico e injeção de sinais de controle. As dimensões e pesos variam conforme a potência do transformador.
  ```
  **Contexto 42:**
  ```text
  O Dispositivo Anunciador permite sinalização visual e auditiva de alarmes em enfermarias e salas de médicos, podendo ser instalado em até 4 locais simultaneamente. Ele possui LEDs para indicar o estado do equipamento e botões para silenciar alarmes e testar a eficiência do dispositivo. O Gerador de Sinais BT4000 identifica falhas de isolação individual, injetando um sinal inferior a 1mA. O dispositivo localizador de falhas BT0074 e BT0078 é projetado para localizar falhas de isolação em sistemas de aterramento IT médico, conforme a norma IEC 61557. A Interface Homem-Máquina (IHM) permite monitorar parâmetros em tempo real, com tela touch screen e proteção por senha.
  ```
  **Contexto 43:**
  ```text
  A CM Comandos Lineares oferece produtos com tecnologia de Processamento Digital de Sinais (DSP), que proporciona alta performance e confiabilidade. Os No Breaks Innovation S1 Trifásico Transformerless são ideais para aplicações críticas, garantindo continuidade operacional. Essa tecnologia permite processamento em tempo real, com capacidade de dez milhões de amostragens por segundo. Os equipamentos podem ser atualizados para adicionar novas funções. A DSP é amplamente utilizada em sistemas eletrônicos sofisticados, destacando-se pela sua velocidade e confiabilidade. A CM Comandos assegura tecnologia de ponta e máxima proteção em suas soluções.
  ```
  **Contexto 44:**
  ```text
  O sistema analisa as curvas de carga e descarga das baterias remotamente, enviando alarmes antes que falhas comprometam o sistema. A manutenção é proativa, com intervenções diretas em caso de alarmes. Oferece um dashboard responsivo com visualizações vertical e horizontal. A análise do ciclo de vida das baterias segue as normas IEEE 1188 e 1491, registrando a evolução do ciclo de vida desde a instalação. O gerenciamento web inclui medições individuais de impedância, tensão e temperatura, além de indicadores de status configuráveis. Gráficos de descarga com filtros dinâmicos e envio de e-mails em caso de alarmes são recursos adicionais. O sistema armazena dados em um banco de dados ANSI SQL com frequência de gravação configurável e controle de acesso por senha em três níveis, utilizando protocolos SNMP e Modbus Over IP.
  ```
  **Contexto 45:**
  ```text
  As especificações técnicas incluem um sistema online de dupla conversão com forma de onda senoidal pura, controle digital por microprocessador DSP e inversor a módulo IGBT de alta frequência. A entrada é bivolt automático (110 ou 220 V) com variação admissível de ± 15% e frequência de 50 ou 60 Hz. A saída tem tensão de 110 V, potências disponíveis de 1,25 kVA e 2,0 kVA, e regulação estática de ± 1%. O sistema de recarga é automático, levando 3 horas para 90% da carga, e possui proteções elétricas contra sobrecarga e descarga de bateria baixa. O display é LCD retroiluminado, com dimensões compactas e estrutura em rack metálico. O ruído audível é inferior a 45 dBA e a temperatura ambiente recomendada é de 20ºC a 25ºC. A interface de gerenciamento suporta múltiplos usuários e protocolos como SNMP e TCP/IP. O sistema é projetado para ambientes internos, com grau de proteção IP-20.
  ```
  **Contexto 46:**
  ```text
  O Sistema IT Médico é voltado para a proteção elétrica hospitalar, garantindo a segurança e eficiência no ambiente de saúde.
  ```
  **Contexto 47:**
  ```text
  O estabilizador eletrônico digital Signal Processor PERFECTION MI S2 é um dispositivo projetado para regular a tensão elétrica, garantindo a proteção de equipamentos eletrônicos contra flutuações de energia.
  ```
  **Contexto 48:**
  ```text
  O No-Break Microprocessado CREATION S2 é um dispositivo que fornece energia ininterrupta, garantindo a continuidade do funcionamento de equipamentos eletrônicos durante quedas de energia.
  ```
  **Contexto 49:**
  ```text
  O DSP registra eventos e medições em sua memória, funcionando como uma 'caixa-preta' para detectar erros operacionais. O sistema PFC corrige o Fator de Potência do No-Break para 0.99, mitigando efeitos de cargas não lineares. As interfaces de gerenciamento remoto incluem o adaptador SNMP NetMate, que permite monitorar o No-Break pela internet e enviar alertas por e-mail. O software IP Power automatiza o desligamento de servidores, enquanto o adaptador ArmModbus integra o No-Break a sistemas de automação via protocolo Modbus. A porta de gerenciamento remoto é uma interface microprocessada DB9A, suportando múltiplas plataformas e sistemas operacionais.
  ```
  **Contexto 50:**
  ```text
  A Equalização Ativa de Baterias é um sistema de gerenciamento que otimiza o desempenho e a vida útil das baterias, garantindo que todas as células operem de forma equilibrada.
  ```
  **Contexto 51:**
  ```text
  A CM Comandos Lineares oferece estabilizadores de tensão de alta tecnologia digital, projetados para um mundo que exige processamento imediato de dados. Esses estabilizadores são ideais para cargas que geram picos de alta corrente, proporcionando alta performance e confiabilidade. A linha Perfection Série Premium utiliza conversores estáticos com módulos IGBTs de última geração, controlados por microprocessador DSP. Eles corrigem instantaneamente variações de tensão da rede elétrica, sem retardos. Todos os setups são realizados por software, com registro log de eventos para rastreabilidade em caso de anomalias.
  ```
  **Contexto 52:**
  ```text
  As especificações técnicas incluem um sistema de no-break com tecnologia de dupla conversão, controle digital DSP e inversor IGBT. A tensão de entrada pode ser 110V ou 220V, com variação admissível de ±15%. A saída padrão é de 110V ou 220V, com potências disponíveis de 2 a 15 kVA e regulação estática de ±1%. O sistema possui proteção contra sobrecarga e curto-circuito, com rendimento de 90%. As baterias são seladas e isentas de manutenção, com recarga automática em 8 a 10 horas. O no-break opera em ambientes internos, com temperatura de 0ºC a 40ºC e umidade de 0% a 95%. O software IP Power permite conexão remota e gerenciamento de eventos. As dimensões variam conforme o modelo, com pesos que vão de 70 kg a 245 kg.
  ```
  **Contexto 53:**
  ```text
  A CM Comandos é uma das maiores fabricantes de No-Breaks da América Latina, com mais de 40 anos de atuação e certificação ISO-9001:2015. A empresa se destaca pela inovação, qualidade e confiabilidade de seus produtos, que são indicados para aplicações de missão crítica. Os No-Breaks oferecem proteção contra distúrbios de energia elétrica, como cortes e picos. Os equipamentos são projetados para operar com alta precisão e segurança, minimizando falhas e custos de manutenção. A CM Comandos também oferece suporte técnico de excelência, com atendimento 24 horas e profissionais qualificados.
  ```
  **Contexto 54:**
  ```text
  O texto menciona um No-Break e um processador de sinal digital relacionado a ele.
  ```
  </details>

- **Resposta de Referência Sintetizada (`reference`)**:
  > Os No-Breaks Composition trifásico oferecem alta confiabilidade e eficiência, sendo projetados para ambientes críticos como data centers e hospitais, com características como tecnologia hotswap, controle DSP e gerenciamento avançado de baterias.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 17 (Índice no testset: `107` | Linha: `108`)

- **Pergunta (`user_input`)**:
  > Quais são as especificações dos cabos de medição utilizados para a medição de impedância dos módulos Equalizer?
- **Tipo de Sintetizador**: `SpecificQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 1 trecho(s))**:

  **Contexto 1:**
  ```text
  ESPECIFICAÇÕES  TÉCNICAS
Cabos de Medição
Para medição de impedância dos módulos Equaliz er 
Módulos M12 - 12VDC ( Terminais M6,  M8 e M10) 
Cabos de C omunicação
Cabos para comunicação entre os módulos Equaliz er (025m 
a 10m)Master C ontr oller 
Contr ola os módulos de bater ias e f ornece a inter face para ger enciamento r emoto .
Har dware:  
Processador:   ARM Cor tex A7 696 MH z • 512 MB memór ia RAM e 8GB F lash.
Memória:   Flash ar maz ena até 12 meses conf orme quantidade de bater ias.
Capacidade:   até 256 str ings de bater ias
Cada string:    até 256 módulos Equaliz er por str ing
Capacidade total:   até 65.536 módulos Equaliz er
RTC:  Relóg io em Tempo Real e Watchdog inter no
 Inter faces:  
    •1 Porta RJ11 - Comunicação   
    •1 Porta RJ45 10/100 Mbit - Ether net 
 Dimensões:   28 x 97 x 111 mm - 180 g
o
Ambiente:   abrigado , IP-20, 0 a 60 C, umidade r elativa de 20% a 90% não condensante .
Alimentação:   5 VDC, estabilizada.
Fonte:
   Adaptador ex terno de 110/220V AC para 5VDC estabilizada -  r equer tomada com No -Break.n   
n   
n   
n   
n   
n   
n   
n  
n  
n   
n   
n     
n     
n   
n   
n   
n    
n  
n    
n   
n   
n   
n   
n    
n   
n    
n    
n    
n    
n   
 
n   
n    Software:
 Sistema Operacional Embar cado
Web Ser ver Integ rado 
Contr ole de A cesso por Senha em dois nív eis
Acesso por Nav egador Web
 Layout Responsiv o, pronto para PCs , Tablets e 
   Smar tphones 
Envia E-mails de Aler ta
Acesso Ser vidor E-mail com Senha Cr iptog rafada
Protocolo SSL Secur e Sock et Lay er 
Relóg io Interno em Tempo Real – R TC 
Sincr onismo com Time Ser ver ex terno
Gráficos de P erformance dir eto no Nav egador
Protocolo SNMP V1, V3 e M odbus O ver IP 
Armaz ena L og de Ev entos de Alar mes
Armaz ena Data L ogger M edições da Bater ia
Banco de Dados ANSI SQL
Capacidade de Ar maz enamento 1 Ano
Frequência de Gravação no Banco de Dados
   Configuráv el
Opção salvar Banco de Dados via TCP/IP
Rotinas de manutenção do Banco de Dados .
Módulo de Ba teria Equaliz er
Processador:   DSP
Inter faces:  2 Portas RJ11
Sensor Temper atura:  Interno
Gabinete:   ABS
Dimensões:   12 x 65 x 113 mm - 50 g
o
  Ambiente:   abrigado , IP-20, 0 a 60 C, umidade
  relativa máx 90% não condensante .
Modelo Equaliz er M12 - MB02 
Tensão nominal:   12 VDC - 02VDC 
Capacidade:  7 a 600 Ah - 7 a 5.000 Ah n   
n    
n    
n    
n   
n 
n   
n   
n   
Módulo de C orrente 
 Descrição:
Acessório (opcional) destinado à medição da 
corrente elétrica que circula pelo banco de 
baterias com indicação do sentido da corrente 
elétrica (C arga ou Descar ga).
Especificações:
Tensão de alimentação:  5V
Erro:  ≤ 1,5% @ corrente nominal
Comunicação:  RS-232 (Nív el TTL)n 
n
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As especificações dos cabos de medição utilizados para a medição de impedância dos módulos Equalizer não estão detalhadas no texto fornecido.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 18 (Índice no testset: `56` | Linha: `57`)

- **Pergunta (`user_input`)**:
  > De que maneira as abordagens sobre gestao de energia e os benefcios dos contratos de manutencao se comparam?
- **Tipo de Sintetizador**: `ComparativeAbstractQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 2 trecho(s))**:

  **Contexto 1:**
  ```text
  O texto aborda serviços de manutenção e suporte técnico para no-breaks, estabilizadores e outras soluções, destacando a importância de contratos de manutenção.
  ```

  **Contexto 2:**
  ```text
  O contrato de manutenção oferece vantagens como a redução da degradação dos equipamentos, prolongando sua vida útil e evitando custos elevados com intervenções corretivas. Ele proporciona suporte técnico 24 horas por dia, minimiza riscos de paradas e garante prioridade no atendimento. A manutenção preventiva pode reduzir a necessidade de manutenções corretivas em até 80%, resultando em economia. Além disso, serviços corretivos inclusos no contrato ajudam a manter os equipamentos em melhores condições. A empresa oferece atendimento em todo o país, com técnicos qualificados e equipados, visando excelência nos serviços prestados.
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As abordagens sobre gestão de energia e os benefícios dos contratos de manutenção se comparam na medida em que ambos visam otimizar o desempenho e a durabilidade dos equipamentos, com os contratos de manutenção oferecendo vantagens como a redução da degradação dos equipamentos e a economia em manutenções corretivas.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 19 (Índice no testset: `114` | Linha: `115`)

- **Pergunta (`user_input`)**:
  > Qual é a importncia do 'Status chaves internas' no funcinamento do painel?
- **Tipo de Sintetizador**: `SpecificQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 1 trecho(s))**:

  **Contexto 1:**
  ```text
  es
    
    •Status de operação e alar mes do painel
   
    •Status chav es inter nasn   
   
Proteç
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > O 'Status chaves internas' é importante para o funcionamento do painel, pois indica o status de operação e alarme do painel.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`

---

### Amostra 20 (Índice no testset: `71` | Linha: `72`)

- **Pergunta (`user_input`)**:
  > Quais são as dimensões físicas e a potência dos modelos mencionados nas certificações técnicas?
- **Tipo de Sintetizador**: `SpecificQuerySynthesizer`
- **Contextos de Referência Esperados (`reference_contexts` - 1 trecho(s))**:

  **Contexto 1:**
  ```text
  CIFICAÇÕES  TÉCNICAS
Características Físicas e Mecânicas
Dimensões Compac tas 
Display TFT 4,3" Touch Scr een 
Estrutura  do G abinete: 
    
    •Rack: metálico
    
    •Tampas laterais e super ior removív eis
    
    •Acabamento: pintura epó xi-pó na cor g rafite
       com tratamento tér mico e anticor rosivo 
Ventilação: f orçada , com contr ole dig ital de
      velocidade  pelo DSP
Transf ormador Isolador : com blindagem
    eletr ostática
Inter faces de G erenciamento
Mono e multiusuár io, cliente -server, multi-ser ver
Vários ser vidor es em um único No -Break
Ferramentas de Shutdown e Ger enciamento
Protocolos:
   
    •Serial RS232 
    
    •Serial RS485*
    
    •SNMP / Telnet / http / TCP/IP*
Softwares de Ger enciamento*
   
    •IP Power e IP P ower SE
    
    •Adaptador SNMP NetM ate 
    
    •Adaptador Ar mModbus RS485
    
    •SiteP ro 
Ambientes e Sistemas Operacionais
   
    •Windows 8 / 10 / 2010 
    
    •Linux * 
       (Mar cas dos r espec tivos fabricantes)
 * Opcionaln
n
n
n
n
n
n
n
n
n
n
n   
   
   
   
   
   Porta de Comunicação: 
    
    •Serial RS232C Isolada F ull Duplex -DB9 F êmea*
   
    •Contato Seco DB9 F êmea
    
    •Ether net RJ 45
   
   
   
   
   
   Medições
True RMS
Potência de Saída em kV A
Potência de Saída em kW
Fator de P otência de Saída
Tensão de Saída
 Corrente de Saída
Frequência de Saída
Tensão de Bater ia
Corrente de Bater ia*
Fator de P otência de Entrada*
Tensão de Entrada
Corrente de Entrada*
 Frequência de Entradan   
n   
n   
n   
n   
n  
n   
n   
n   
n   
n   
n   
n  
Alarmes
Contr olados pelo pr ocessador DSP
Tipos de Alarmes:
   
    •Sonor os:
       »  F alta de Rede: 1 toque a cada 4 s  
       »  P ré-alar me das Bater ias: 1 toque por segundo  
       »  F alha I nterna do No -Break: alar me contínuo
    
    •Mensagens de Aler ta:
       »  Display de Cr istal Líquido
       »  Sof tware IP P ower via TCP/IP
       »  M ensagem por e -mail , celular ou pop -up:
           -  Operação Nor mal
           -  F alha de Rede
           -  P ré-alar me de Bater ias
           -  B ypass Estático Ativ o
           -  B ypass M anual Ativ o
           -  Sobr ecar ga de Saída
           -  F alha n   
n   
Log de E ventos
Autonomia da EEPROM:  
    
    •5 anos (com No -Break desligado)n
n   Registr os Armaz enados:
   
    •Memór ia com 25.500 r egistros (sendo 15 logs de 
configuração e 495 logs de ev entos)
    
       •Indicação de data, hora e ocor rência
    
    •Medições
    
    •Status de operação e alar mes do painel
   
    •Status chav es inter nas
   
Características de Oper ação
Ruído A udív el: 55 dBA a 60 dBA a 1 metr o
MTBF (M ean Time Bet ween F ailur es): 200 mil horas
MTTR (M ean Time To Repair): 30 minutos
Temperatura Ambiente: 
   
    •No-Break: 0ºC a 40ºC 
    
    •Bater ias: 0ºC a 30ºC
    
    •Recomendada: 20ºC e 25ºC
Umidade Relativa:
    
    •0% a 95% sem condensação
    
    •Recomendada: 45% a 55%
Altitude: até 1.800m
Tipo de Ambiente Recomendado:
    
    •Interno, instalação abr igada
   
    •Atmosf era: limpa, livr e de par tículas condutivas , 
      gases tó xicos , líquidos e inflamáv eis.
Grau de P roteção: IP -20n
n
n
n
n
n
n
n   
   
   
   
   
   
   
       5000      5,0 900 X 480 X 980    295
  10000    10,0     900 X     480 X 980    305
  15000    15,0      900 X     480 X 980      409
  20000    20,0   1180 X     550 X 980      560
  30000    30,0   1180 X     550 X 980      640         Modelo Potência
kVADimensões Físicas
alt x larg x prof 
mmPeso (com 
baterias)
kg
Dimensões Físicas
alt x larg x prof 
mmPotência
kVAPeso (sem 
baterias)
kgModelo
  4
  60000   60,0         1355 X  720 X 940          560
  80000   80,0          1902 X 783 X 925          730
100000  100,0     1902 X 783 X 925             870
120000  120,0     1902 X 783 X 925     1000
160000  160,0   1900 X 1265 X 1000  1200
200000  200,0   1900 X 1265 X 1000  1300
300000  300,0   2010 X 2050 X 1000  22800000   40,0         1355 X  720 X 940          490Modelos c om baterias internas 
Modelos sem baterias  Tecnologia
n On Line - Dupla Con versão
  ```

- **Resposta de Referência Sintetizada (`reference`)**:
  > As dimensões físicas e a potência dos modelos mencionados nas certificações técnicas variam conforme o modelo, com dimensões que vão de 900 x 480 x 980 mm a 2010 x 2050 x 1000 mm e potências que vão de 5,0 kVA a 300,0 kVA.
- **Classificação**: `⬜ A avaliar`
- **Observações Qualitativas / Análise de Viés**: `[Anotações do revisor...]`
