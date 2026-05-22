from groq import Groq
import os
import pandas as pd
import json
import time
import evaluate

api_key = os.environ.get("GROQ_API_KEY")  # Set your Groq API key here
df = pd.read_csv(r"data\it_support_tickets\IT Support Ticket Data.csv")
MODEL = "llama-3.1-8b-instant"

client = Groq(api_key=api_key)

# task will be to summarize it support ticket in one sentence
# Take 5 Random it support tickets from the dataset
# first -> clean the data

def clean_text(text):
    text = str(text)
    text = text.replace('<br>', ' ')
    text = text.replace('<br/>', ' ')
    text = ' '.join(text.split())
    return text

df['Body'] = df['Body'].apply(clean_text)
df = df.dropna(subset=['Body'])

# pick random 5 technical support tickets
tickets = df[df['Department'] == 'Technical Support'].sample(n=10, random_state = 42)

# print tickets
for i, row in enumerate(tickets['Body'].tolist()):
    print(f"Ticket {i+1}:")
    print(row)
    print("="*120)

reference_answers = [
    "User requests resolution of issues related to digital marketing campaigns.",
    "Customer seeks details on digital marketing services including social media management, content creation, SEO optimization, and analytics tools.",
    "User reports project tasks not syncing due to third-party tool integration problems despite restarting and reviewing settings.",
    "User requests guidance on optimizing investments using Elasticsearch and PyTorch for data analytics.",
    "User reports digital marketing campaign analytics not updating correctly despite restarting tools and clearing cache.",
    "User reports unauthorized access to medical data due to outdated Apache Hadoop and MongoDB configurations.",
    "User reports inaccessible digital marketing materials due to SSD file corruption possibly caused by a USB transfer error, despite rebooting, disk check, and file recovery attempts.",
    "Customer requests advice on optimizing investment data analytics techniques and enhancing performance of analytics tools for handling large data volumes.",
    "Customer seeks insights on digital strategies for brand growth and product integration to benefit their business.",
    "Customer reports SaaS project management integration failures across multiple devices possibly due to OS or driver incompatibilities, despite updating device drivers and reinstalling the SaaS application."
    ]



zero_shot_prompt = """ Summarize the following IT support ticket in one sentence:
Ticket: {ticket}
Summary: """

few_shot_prompt = """ Summarize the following IT support ticket in one sentence:
Here are some examples:

Ticket: Resolve issues related to digital marketing campaigns.
Summary: User requests resolution of issues related to digital marketing campaigns.

Ticket: Provide details on digital marketing services including social media management, content creation, SEO optimization, and analytics tools.
Summary: Customer seeks details on digital marketing services including social media management, content creation, SEO optimization, and analytics tools.

Ticket: Printer not working, tried restarting and checking connections after installing new driver.
Summary: User reports printer not working despite restarting and checking connections after installing new driver

Now sumarize this ticket:
Ticket: {ticket}
Summary: """

cot_prompt = """Summarize the following IT support ticket in one concise sentence. Think step by step:
1. Identify the main problem
2. Identify who is affected
3. Identify what has been tried
4. Write a one sentence summary

Ticket: {ticket}

Step by step:"""


def generate_responses(prompt, temperature=0.7):
    time.sleep(1) 
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content


responses = {       # total of 40 responses
    "Zero-Shot": [],        # 10 responses
    "Few-Shot":[],      #10 responses
    "CoT": [],      # 10 rresponses
    "CoT_low_temp":[],       # 5 responses
    "CoT_high_temp": []      # 5 responses
}


if os.path.exists(r"E:\VS Code stuff\NLP Workspace\generated_responses.json"):
    print("Responses already exist, skipping generation...")
else:

    print("Generating Zero-Shot Responses")
    print("="*120)

    tickets_list = tickets['Body'].tolist()

    for ticket in tickets_list:
        prompt = zero_shot_prompt.format(ticket=ticket)
        response = generate_responses(prompt)
        responses["Zero-Shot"].append({"Ticket": ticket, "Response": response})

    ########################################################################################

    print("Generating Few-Shot Responses")
    print("="*120)

    for ticket in tickets_list:
        prompt = few_shot_prompt.format(ticket=ticket)
        response = generate_responses(prompt)
        responses["Few-Shot"].append({"Ticket": ticket, "Response": response})

    ########################################################################################

    print("Generating CoT Responses")
    print("="*120)

    for ticket in tickets_list:
        prompt = cot_prompt.format(ticket=ticket)
        response = generate_responses(prompt)
        responses["CoT"].append({"Ticket": ticket, "Response": response})

    ########################################################################################

    print("Generating CoT Low Temperature Responses")
    print("="*120)

    for ticket in tickets_list[:5]:
        prompt = cot_prompt.format(ticket=ticket)
        response = generate_responses(prompt, temperature=0.2)
        responses["CoT_low_temp"].append({"Ticket": ticket, "Response": response})

    ########################################################################################

    print("Generating CoT High Temperature Responses")
    print("="*120)

    for ticket in tickets_list[:5]:
        prompt = cot_prompt.format(ticket=ticket)
        response = generate_responses(prompt, temperature=0.9)
        responses["CoT_high_temp"].append({"Ticket": ticket, "Response": response})

    ###########################################################################################
    with open("generated_responses.json", "w") as f:
        json.dump(responses, f, indent=2)

    print("Responses saved to generated_responses.json ")

print("Starting Evaluation...")

#######################################################################################################


# EVALUATION

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")

# Load responses from JSON (more safe than using the list above)
with open("generated_responses.json", "r") as f:
    all_responses = json.load(f)

# Calculate scores for each prompt type
def evaluate_prompt(responses, references):
    predictions = [r["Response"] for r in responses]
    refs_for_bleu = [[ref] for ref in references]
    
    bleu_score = bleu.compute(
        predictions=predictions,
        references=refs_for_bleu
    )
    
    rouge_score = rouge.compute(
        predictions=predictions,
        references=references
    )
    
    return {
        "BLEU-1": round(bleu_score["bleu"], 4),
        "ROUGE-L": round(rouge_score["rougeL"], 4)
    }

# References for each prompt type
refs_10 = reference_answers        # 10 references for main prompts
refs_5 = reference_answers[:5]     # 5 references for temp prompts

# Evaluate all prompt types
results = {}
results["Zero-Shot"] = evaluate_prompt(all_responses["Zero-Shot"], refs_10)
results["Few-Shot"] = evaluate_prompt(all_responses["Few-Shot"], refs_10)
results["CoT"] = evaluate_prompt(all_responses["CoT"], refs_10)
results["CoT_low_temp"] = evaluate_prompt(all_responses["CoT_low_temp"], refs_5)
results["CoT_high_temp"] = evaluate_prompt(all_responses["CoT_high_temp"], refs_5)

# results
print("===== EVALUATION RESULTS =====")
for prompt_type, scores in results.items():
    print(f"{prompt_type}: BLEU-1={scores['BLEU-1']}, ROUGE-L={scores['ROUGE-L']}")


with open("evaluation_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Evaluation complete!")
