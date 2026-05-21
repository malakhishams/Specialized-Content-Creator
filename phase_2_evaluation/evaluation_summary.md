
# Evaluation Summary: Prompt Engineering Analysis

## A. Task Description

The task selected for this evaluation report is **IT Support Tickets Summarization** the goal is to summariza a given IT support ticket into one clear sentence that capture the main problem, who is affected and what resolving steps have been taken.

A dataset of 10 technical support tickets have been selected from the IT support tickets dataset. three different prompting technoques were applied to generate the summaries using **Llama 3.1 8B** via a **Groq API**, The generated summaries have been compared to 10 human written refrences to evaluate them using **BLEU-1** and **ROUGE-1** metrics.

## B. Prompting Strategies

### 1. Zero-Shot Prompting

This technique involves asking the AI model to perform a task without providing any examples to clarify that task, so the model completely relies on its previous knowledge. It is useful when a task is straight forward and doesn't have complexity.

### 2. Few-Shot Prompting

This technique provides several examples to the model which look like the desired task, so the model catches patterns from the examples and understands what it is asked to do, this guarantees more accurate results over zero-shot prompting, but still depends on the desired task.

### 3. Chain-of-Thought (CoT) Prompting

Chain-of-Thought prompting breaks down a complex task into clear sequential steps that the model follows to reach the final answer. In this task, the model was instructed to identify the main problem, who is affected, what has been tried, and then write a one sentence summary. While this technique produces more detailed and structured reasoning, the longer outputs can negatively affect BLEU and ROUGE-L scores when compared against short reference answers.

## C. Evaluation Metrics

### 1. BLEU-1

Mainly focuses on precision, how much of the generated text matches the reference. It is used mostly in machine translation tasks, comparing the model generated text against a human written reference. BLEU-1 specifically measures unigram overlap, meaning it checks how many individual words match between the generated and reference text. Its values range from 0 to 1, where values closer to 1 are better.

### 2. ROUGE-L

Mainly focuses on recall, how much of the reference content is covered by the generated text. It is used mostly in text summarization tasks, comparing the model generated summary against a human written reference. ROUGE-L specifically measures the longest common subsequence between both texts, capturing word order as well as word overlap. Its values range from 0 to 1, where values closer to 1 are better.

## D. Results

The following table summarizes the BLEU-1 and ROUGE-L scores for each prompting strategy:

|Prompt Strategy|BLEU-1|ROUGE-L|
|----------------|--------|---------|
|Zero-Shot|0.1669|0.3786|
|Few-Shot|0.3948|0.6622|
|CoT|0.0520|0.1798|
|CoT Low Temperature (0.2)|0.0358|0.1498|
|CoT High Temperature (0.9)|0.0481|0.1602|

## E. Analysis of the Result

### 1. Best Performing Strategy

Obviuosly, best berforming strategy is **Few-shot** which shows  BLEU-1 equals 0.3948 and ROUGE-L score equals 0.6622 . and it is natural that ROUGE-L is greater than BLEU-1 as it is the most suitable metric to evaluate this task.

### 2. Temperature Analysis

Among the CoT temperature variations, CoT with high temperature (0.9) achieved slightly better scores with BLEU-1=0.0481 and ROUGE-L=0.1602 compared to low temperature (0.2) with BLEU-1=0.0358 and ROUGE-L=0.1498. However the difference is minimal, suggesting that temperature had little impact on summarization quality for this specific task.

## F. Trade-offs: Low vs High Temperature

Low temperature (0.2) reduces randomness and produces more predictable and consistent outputs. This makes it suitable for tasks requiring precision and factual accuracy such as technical documentation or medical records. However it can produce repetitive and less varied responses over multiple runs.

High temperature (0.9) increases randomness and produces more diverse and creative outputs, making it ideal for creative tasks like writing or brainstorming. However it may also lead to less coherent and less consistent responses, which explains the slightly lower precision in this summarization task.

In the context of IT support ticket summarization, low temperature is generally preferable as accuracy and consistency are more important than creativity.
