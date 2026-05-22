#  Specialized Content Generator
### A Full NLP Pipeline: Transformer Implementation, Fine-Tuning & Prompt Engineering

---

## Project Overview

This project is a complete NLP pipeline built as part of Sprints AI scholarship program. It covers three phases — from implementing a Transformer block from scratch, to fine-tuning a language model, to evaluating advanced prompt engineering strategies on real IT support ticket data.

---

## Project Structure

```
specialized-content-generator/
├── data/
│   ├── fine_tuning_data/
│   │   └── custom_dataset.txt          # 60 IT support tickets used for fine-tuning
│   └── it_support_tickets/
│       └── IT Support Ticket Data.csv  # Raw IT support ticket dataset
├── fine_tuned_model_checkpoint/
│   └── model_weights.pt                # Not included — run fine_tuning_script.py to generate
├── src/
│   ├── basic_transformer_block.py      # Custom Transformer block implementation
│   └── fine_tuning_script.py           # GPT-2 fine-tuning pipeline
├── phase_2_evaluation/
│   ├── prompt_templates.json           # Zero-Shot, Few-Shot, CoT prompts
│   ├── generated_responses.json        # 40 generated responses
│   ├── evaluation_script.py            # BLEU + ROUGE evaluation script
│   └── evaluation_summary.md           # Analysis report
├── .gitattributes
├── .gitignore
├── comparison_report.md                # Base vs fine-tuned GPT-2 comparison
└── README.md
```

---

## 🔧 Phase 1 — Transformer Implementation & Domain-Specific Fine-Tuning

### BasicTransformerBlock
A custom Transformer block implemented from scratch using PyTorch:
- Multi-head self-attention (`nn.MultiheadAttention`)
- Feed-forward network (Linear → ReLU → Linear)
- Layer normalization after each sub-layer
- Residual connections

```python
block = BasicTransformerBlock(embed_dim=64, num_heads=4, ff_dim=256)
x = torch.rand(2, 10, 64)
output = block(x)
# Input shape:  torch.Size([2, 10, 64])
# Output shape: torch.Size([2, 10, 64])
```

### GPT-2 Fine-Tuning
- **Base model:** `erwanf/gpt2-mini`
- **Dataset:** 60 IT support tickets (Technical Support department)
- **Training:** 5 epochs on CPU/GPU
- **Result:** Fine-tuned model generates domain-specific IT support text

> **Note:** Model weights are not included in this repository. Run `python src/fine_tuning_script.py` to generate them locally.

---

## 🎯 Phase 2 — Prompt Engineering & Quantitative Evaluation

### Task
Summarize IT support tickets into one concise sentence using 3 prompting strategies.

### Prompt Strategies

| Strategy | Description |
|----------|-------------|
| Zero-Shot | Ask directly with no examples |
| Few-Shot | Provide 3 examples before asking |
| Chain-of-Thought | Step-by-step reasoning instructions |

### Generation
- **Model:** Llama 3.1 8B via Groq API
- **Total responses:** 40
  - 10 Zero-Shot
  - 10 Few-Shot
  - 10 Chain-of-Thought
  - 5 CoT Low Temperature (0.2)
  - 5 CoT High Temperature (0.9)

### Evaluation Results

| Prompt Strategy | BLEU-1 | ROUGE-L |
|----------------|--------|---------|
| Zero-Shot | 0.1669 | 0.3786 |
| **Few-Shot** | **0.3948** | **0.6622** |
| CoT | 0.0520 | 0.1798 |
| CoT Low Temp (0.2) | 0.0358 | 0.1498 |
| CoT High Temp (0.9) | 0.0481 | 0.1602 |

**Winner: Few-Shot Prompting** with BLEU-1=0.3948 and ROUGE-L=0.6622

---

## 🤖 Phase 3 — Extractive Question Answering with BERT

### Model
`google-bert/bert-large-uncased-whole-word-masking-finetuned-squad`

### Task
Extract the primary technical component mentioned in each IT support ticket.

### Question
> *"What is the primary technical component or system mentioned?"*

### Sample Results

| Ticket Topic | Predicted Answer | Score |
|-------------|-----------------|-------|
| Account portal outage | centralized account management portal | 5.78 |
| Laser Printer issue | printer firmware | 4.67 |
| MySQL dashboard crash | data overflow or an analytics software error | 4.13 |
| Smart Home firmware | firmware upgrade | 4.32 |
| Kubernetes outage | barcode scanners, raid controllers | 4.22 |

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python | Core language |
| PyTorch | Transformer block + fine-tuning |
| HuggingFace Transformers | BERT QA + GPT-2 |
| Groq API | LLM inference (Llama 3.1 8B) |
| NLTK / evaluate | BLEU + ROUGE metrics |
| Pandas | Data handling |
| Google Colab | GPU training |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/malakhishams/Specialized-Content-Creator.git
cd Specialized-Content-Creator
```

### 2. Install dependencies
```bash
pip install torch transformers pandas evaluate rouge_score groq nltk
```

### 3. Set up API key
```bash
# Windows
set GROQ_API_KEY=your_key_here

# Mac/Linux
export GROQ_API_KEY=your_key_here
```

### 4. Run fine-tuning
```bash
python src/fine_tuning_script.py
```

### 5. Run evaluation
```bash
python phase_2_evaluation/evaluation_script.py
```

---

## Key Findings

- **Few-Shot prompting** consistently outperforms Zero-Shot and CoT on structured summarization tasks
- **Fine-tuning** on domain-specific data clearly improves output style and vocabulary
- **Transformers** eliminate the need for manual text preprocessing unlike traditional ML approaches
- **BERT's bidirectional attention** enables accurate extraction of answer spans even from long contexts
- **CoT prompting** produces detailed reasoning but lower BLEU/ROUGE scores due to longer outputs

---

## License

This project was built as part of Sprints AI scholarship program for educational purposes.

---

## 🏆 Achievement

The Transformer Implementation & Domain-Specific Fine-Tuning part received a perfect score of **100/100 for Quality** and **100/100 for Scope** as part of the scholarship program evaluation.
