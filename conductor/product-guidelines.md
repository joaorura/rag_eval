# Product Guidelines: CM Comandos RAG Benchmark

## 1. Precisão Técnica e Vocabulário de Domínio (Português pt-BR)
- **Fidelidade à Engenharia Elétrica**: Toda documentação, saídas de avaliação e respostas do chat devem manter fidelidade estrita à terminologia industrial de sistemas de energia crítica (nobreaks monofásicos/trifásicos, estabilizadores, modulação PWM, tiristores, retificadores IGBT, inversores, chave estática de bypass, baterias VRLA seladas, barramentos DC, topologia *true-online* dupla conversão, paralelismo ativo redundante e equalizadores).
- **Normas Técnicas**: Observância às normas ABNT aplicáveis e respeito integral às especificações dos manuais da fabricante brasileira CM Comandos Lineares.

## 2. Reprodutibilidade Científica e Determinismo
- **Sementes Aleatórias Fixadas**: Toda rotina que envolva amostragem, inicialização de pesos ou embaralhamento deve fixar explicitamente `SEED = 42` (`random`, `numpy`, `torch`).
- **Isolamento de Índices em Disco**: Cada modelo avaliado deve manter persistência isolada em `storage/index_<model_id>/`, prevenindo contaminações cruzadas de coleções vetoriais ou metadados de nós.
- **Deduplicação Determinística**: O conjunto de teste deve ser saneado antes do cômputo de métricas através da tupla unívoca `(user_input, reference_contexts)`.

## 3. Rigor no Benchmark e Métricas de Recuperação
- **Matching Multicamada**: A verificação de relevância contra o *ground truth* deve seguir lógica em cascata (substring estrita $\ge 40$ chars $\rightarrow$ RapidFuzz $\ge 85\%$ $\rightarrow$ ROUGE-L $F_1 \ge 0.50$), mitigando falsos negativos gerados por fronteiras de quebra de parágrafo.
- **Validação Estatística Não-Paramétrica**: Toda comparação pareada contra baselines deve reportar a estatística $W$ e o $p$-valor do teste pareado de Wilcoxon com nível de significância $\alpha = 0.05$, acompanhado de intervalos de confiança de 95% via Bootstrap ($B = 1.000$ iterações).

## 4. Gestão Rigorosa de Recursos Computacionais (Engenharia)
- **Execução Sequencial na GPU**: Modelos locais devem ser executados sequencialmente em placas gráficas de 8 GB VRAM, com liberação compulsória de memória (`keep_alive=0` no Ollama, `torch.cuda.empty_cache()` e `gc.collect()`) antes de alternar de modelo.
- **Monitoramento de Latência**: Medição de latência por consulta em milissegundos, com limite aceitável de $\le 150\text{ ms}$ para operações interativas em produção.
