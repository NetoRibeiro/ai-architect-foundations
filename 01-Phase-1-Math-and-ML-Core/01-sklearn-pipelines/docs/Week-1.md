# Semana 1: A Ponte entre ML Clássico e Arquitetura de IA

**Objetivo:** Revisar conceitos fundamentais de Machine Learning sob a ótica da Engenharia de IA Corporativa (Deep Learning e LLMs).

---

## Tópico 1: Transformação de Dados (O que a Rede Neural enxerga)

Embora o tratamento de dados estruturados já seja rotina na Análise de Dados, no Deep Learning isso dita se os gradientes matemáticos do modelo vão convergir ou quebrar.

### Variáveis Categóricas (One-Hot vs. Label)
*   **O viés da ordem:** Usar *Label Encoding* (0=Maçã, 1=Banana, 2=Laranja) faz a matemática do modelo assumir que Laranja (2) é maior que Maçã (0). Em árvores de decisão, isso importa pouco. Em Deep Learning (álgebra linear pura), isso destrói o modelo introduzindo ordens falsas.
*   **A Ponte GenAI (Embeddings):** É por isso que LLMs não leem palavras diretamente. O *One-Hot Encoding* é o ancestral primitivo do Embedding: ele cria vetores ortogonais independentes para garantir que o modelo não assuma relações matemáticas falsas. Modelos modernos usam embeddings densos para mapear essas palavras em um espaço vetorial semântico.

### Escalonamento (StandardScaler vs. MinMaxScaler)
*   **O Problema Matemático:** Redes neurais atualizam seus pesos multiplicando o erro de trás para frente (Regra da Cadeia). Se uma coluna possui valores na casa dos milhões e outra na casa dos decimais, os cálculos dos gradientes vão "explodir" ou "desaparecer" (*Exploding/Vanishing Gradient*).
*   **A Regra de Ouro:** O `StandardScaler` (centralizando a média em 0 e desvio padrão em 1) mantém a matemática da rede neural estável e simétrica.

---

## Tópico 2: A Mentira da Acurácia (O Paradoxo do Desbalanceamento)

*   **O Paradoxo:** Em um cenário de detecção de fraudes (onde apenas 0,1% das transações são fraudulentas), um modelo que simplesmente chuta "Não é fraude" para todas as linhas terá **99,9% de Acurácia**. Para executivos, parece perfeito; para engenharia, é um modelo inútil.
*   **A Ponte GenAI (RAG e Alucinações):** Na arquitetura de IA corporativa, focar na acurácia global mascara problemas críticos. Se um assistente virtual acerta 95 respostas usando a base de conhecimento, mas em 5 ele inventa uma cláusula jurídica (alucinação), a acurácia é 95%, mas o risco sistêmico e legal torna a aplicação inviável.

---

## Tópico 3: O Santo Graal das Métricas (Trabalhando nos Erros)

O verdadeiro trabalho do Arquiteto de IA vive na análise profunda dos erros gerados pelo modelo. Esqueça os "Acertos" e foque na Matriz de Confusão.

### 1. Falso Positivo (FP) - O Alarme Falso
*   **Conceito:** O modelo previu que ERA (Positivo), mas a realidade é que NÃO ERA (Falso).
*   **No ML Clássico:** Um filtro de spam que joga um e-mail crítico no lixo.
*   **No Deep Learning/LLM:** O LLM cita um documento interno como fonte de uma resposta, mas o documento na verdade não diz aquilo (*Alucinação de Citação*).

### 2. Falso Negativo (FN) - A Omissão Crítica
*   **Conceito:** O modelo previu que NÃO ERA (Negativo), mas a realidade é que ERA (Falso).
*   **No ML Clássico:** Um modelo médico atestar que um paciente não tem câncer, mandando-o para casa, mas ele tem.
*   **No Deep Learning/LLM:** O filtro de *Safety Guardrail* diz que o prompt do usuário é seguro, mas na verdade era uma tentativa de injeção de prompt (*Prompt Injection*) para vazar dados.

### Precisão, Recall e F1-Score
*   **Precisão (Precision = TP / TP + FP):** Das previsões positivas, quantas eram reais?
    *   *Foco do Arquiteto:* Maximizar Precisão quando o custo de um **FALSO POSITIVO** for catastrófico (ex: IA inventando jurisprudência para um advogado).
*   **Recall (Revocação = TP / TP + FN):** Dos casos positivos reais, quantos o modelo encontrou?
    *   *Foco do Arquiteto:* Maximizar Recall quando o custo de um **FALSO NEGATIVO** for catastrófico (ex: deixar passar uma injeção de prompt destrutiva).
*   **F1-Score:** A Média Harmônica entre Precisão e Recall. O número definitivo para penalizar o modelo caso ele favoreça uma métrica em detrimento drástico da outra.

---

## Prática de Laboratório: Recomendação de Dataset
**Credit Card Fraud Detection (Kaggle)**
Dataset altamente desbalanceado (0.17% de fraudes). Ideal para testar a armadilha da Acurácia alta vs. Recall baixo na classe minoritária.

**Desafio do Analista no Colab:**
1. Construir pipeline com `StandardScaler` e um classificador (`RandomForest` ou outro).
2. Constatar a acurácia global alta mascarando o fracasso em detectar fraudes.
3. Testar a técnica de `class_weight='balanced'` para forçar o algoritmo a salvar o Recall das fraudes, sacrificando parte da Acurácia global.