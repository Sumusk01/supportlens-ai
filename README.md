# SupportLens AI

A retrieval-grounded AI support agent for Apple customer support conversations.

SupportLens AI takes an incoming customer message and performs three tasks:

1. Classifies the customer's primary support intent.
2. Retrieves historically similar AppleSupport cases and extracts the resolution actions used in those cases.
3. Drafts a response grounded in the retrieved evidence and decides whether the case should be auto-handled or escalated to a human.

The system is intentionally conservative: historical evidence is treated as evidence, not as permission to invent a solution.

---

## 1. Problem

Customer-support agents repeatedly solve similar problems, but useful historical resolutions are difficult to find quickly.

The goal of SupportLens AI is to provide an evidence-grounded first response rather than a generic chatbot response.

For every incoming message, the system produces:

* predicted intent
* confidence score
* top intent alternatives
* historically similar customer cases
* supported resolution actions
* a grounded draft response
* AUTO-HANDLE or ESCALATE decision
* explanation for the decision

### Design principle

> Proof is worth more than the system.

The system therefore prioritizes evidence from historical support interactions and escalates when the available evidence is insufficient.

---

## 2. Dataset

The project uses the Customer Support on Twitter dataset:

`thoughtvector/customer-support-on-twitter`

The raw dataset is approximately 2.8 million tweets.

For this project, AppleSupport conversations were selected from the dataset.

The raw dataset is intentionally not committed to the repository because of its size and dataset redistribution considerations.

Expected location:

`data/raw/twcs.csv`

After constructing customer -> AppleSupport reply pairs, the project contains:

* 106,260 usable historical customer/support pairs
* 1,760 duplicate customer messages
* 0 empty customer messages
* 0 duplicate customer+reply pairs

The training corpus excludes the locked Golden Set examples to reduce evaluation leakage.

---

## 3. System Architecture

```text
Customer Message
       |
       v
+----------------------+
| Intent Classifier    |
| TF-IDF + Logistic    |
| Regression           |
+----------------------+
       |
       v
+----------------------+
| Historical Retrieval |
| TF-IDF + Cosine      |
| Similarity           |
+----------------------+
       |
       v
+----------------------+
| Evidence Aggregator  |
| Resolution Actions   |
+----------------------+
       |
       +--------------------+
       |                    |
       v                    v
+----------------+   +-------------------+
| Grounded Draft |   | Escalation Policy |
| Response       |   |                   |
+----------------+   +-------------------+
       |                    |
       v                    v
   Draft Reply       AUTO-HANDLE /
                     ESCALATE
```

---

## 4. Intent Classification

The system uses 12 support intents:

* `IOS_UPDATE`
* `BATTERY_POWER`
* `APP_PROBLEM`
* `CONNECTIVITY`
* `ACCOUNT_APPLE_ID`
* `APP_STORE_PURCHASE`
* `MUSIC_MEDIA`
* `SCREEN_DISPLAY`
* `SETTINGS_FEATURE`
* `PERFORMANCE_STABILITY`
* `HARDWARE_DEVICE`
* `OTHER`

The intent represents the customer's primary problem rather than simply detecting which product or device is mentioned.

For example, a message mentioning an iPhone is not automatically classified as a hardware problem.

### Weak-label training

Because the full Twitter dataset does not provide the required intent labels, high-confidence heuristic rules were used to create weak labels.

The weak-labeling stage:

* applies strong phrases and domain keywords
* abstains when evidence is insufficient
* rejects ambiguous/tied classifications
* retains only high-confidence examples for classifier training

This produced:

**13,098 high-confidence weak-labelled examples**

The classifier uses:

* TF-IDF
* unigrams + bigrams
* sublinear TF
* Logistic Regression
* balanced class weights

---

## 5. Historical Retrieval

Historical support cases are indexed using TF-IDF.

For an incoming customer message, the system retrieves the top 5 historically similar customer messages and their AppleSupport replies.

The retrieval layer does not assume that the highest lexical similarity is automatically the best resolution.

Instead, the retrieved replies are analyzed to identify resolution actions.

Examples of resolution actions include:

* `UPDATE_SOFTWARE`
* `RESTART_DEVICE`
* `CHECK_SETTINGS`
* `CHECK_CONNECTION`
* `CHECK_ACCOUNT`
* `CHECK_APP_STORE`
* `CHECK_BATTERY`
* `PROVIDE_TROUBLESHOOTING`
* `PROVIDE_WORKAROUND`
* `PROVIDE_ARTICLE`
* `ASK_DIAGNOSTIC_QUESTION`
* `ASK_FOR_DETAILS`
* `CONTACT_SUPPORT_DM`
* `ROUTE_TO_SPECIALIST`
* `ROUTE_FEEDBACK`
* `ROUTE_LANGUAGE_SUPPORT`

An action is considered supported when it appears across multiple retrieved historical cases.

---

## 6. Grounded Response Drafting

The response drafter is intentionally deterministic.

It does not pretend to be an LLM-generated answer.

Instead, it converts historically supported resolution actions into a concise response template.

For example:

```text
Customer:
My iPhone battery is draining very quickly.

Historical evidence:
CONTACT_SUPPORT_DM: 2/5 cases
PROVIDE_TROUBLESHOOTING: 2/5 cases

Supported actions:
CONTACT_SUPPORT_DM
PROVIDE_TROUBLESHOOTING
```

The resulting response is generated only from supported actions.

This design reduces hallucination risk and makes the behavior easy to inspect and reproduce.

---

## 7. Auto-Handle vs Escalate

The escalation policy is deliberately conservative.

A case is escalated when:

* the predicted intent is `OTHER`
* intent confidence is below 0.60
* no supported historical resolution action exists
* the strongest supported action is not observed in at least 2 retrieved cases

Otherwise the system can auto-handle the case by producing the grounded response.

### Important distinction

`AUTO-HANDLE` means that the system is confident enough to automatically produce a grounded first response.

It does not mean that the customer's underlying issue has necessarily been resolved.

---

# 8. Evaluation

The evaluation is separated into component-level and end-to-end tests.

## Golden Set

A locked manually labelled Golden Set contains:

**196 customer examples**

Each example was labelled using the rule:

> Assign the customer's primary support problem, not every problem mentioned in the message.

The Golden Set was created independently from the weak-label training data and is not modified based on classifier predictions.

---

## Intent Classification Results

### Weak-label validation

| Metric   | Result |
| -------- | -----: |
| Accuracy | 94.73% |
| Macro F1 | 92.73% |

However, this number is not treated as the main performance result because both training and validation labels originate from the same heuristic labeling system.

### Independent Golden Set

| Metric   | Result |
| -------- | -----: |
| Accuracy | 41.33% |
| Macro F1 | 41.68% |

The gap between these results is an important finding rather than something hidden.

It shows that the classifier learned the structure of the weak labels much better than it learned the real human-labelled task.

---

## Baselines

Two baselines were evaluated on the locked Golden Set.

| Model                        | Accuracy | Macro F1 |
| ---------------------------- | -------: | -------: |
| Majority class               |   11.73% |    1.75% |
| Keyword baseline             |   50.51% |   47.14% |
| TF-IDF + Logistic Regression |   41.33% |   41.68% |

The keyword baseline currently outperforms the learned classifier on the independent Golden Set.

This is an important negative result: the weak-label training strategy is not yet sufficient for robust intent generalization.

---

# 9. Retrieval Evaluation

Two retrieval evaluations were performed.

## Similarity evaluation

On 500 held-out historical examples:

| Metric                   | Result |
| ------------------------ | -----: |
| Average Top-1 similarity | 0.3995 |
| Median Top-1 similarity  | 0.3349 |
| Average Top-5 similarity | 0.3442 |

Similarity is not treated as retrieval accuracy because lexical similarity does not guarantee semantic relevance.

## Human relevance evaluation

100 customer queries were manually reviewed across their top 5 retrieved cases.

A relevance score was assigned:

* `0` = unrelated/generic overlap
* `1` = same domain or potentially useful
* `2` = core problem directly matches

Results:

| Metric                              |    Result |
| ----------------------------------- | --------: |
| Top-1 directly relevant             |       14% |
| Top-1 useful (1 or 2)               |       67% |
| Top-5 directly relevant             |       34% |
| Top-5 useful (1 or 2)               |       81% |
| Mean relevance across Top-5         | 0.794 / 2 |
| Queries with no useful Top-5 result |       19% |

### Headline retrieval result

**81% of reviewed queries had at least one useful historical example in the top 5.**

However:

**Only 34% had at least one directly relevant historical example.**

Therefore, 81% should be interpreted as evidence availability, not resolution accuracy.

---

# 10. Resolution Action Extraction

A manually reviewed set of 100 historical AppleSupport replies was used to test the resolution-action extractor.

Result:

**93/100 exact matches = 93%**

The remaining errors were primarily annotation-boundary cases such as distinguishing:

* diagnostic questions vs requests for details
* workaround vs article
* article + DM combinations

The extractor was frozen at 93% rather than tuned indefinitely against the review set.

---

# 11. End-to-End Agent Evaluation

A fixed random sample of 50 examples from the locked Golden Set was used for end-to-end evaluation.

| Metric                                   | Result |
| ---------------------------------------- | -----: |
| Intent accuracy                          |  34.0% |
| Mean intent confidence                   | 0.3987 |
| Mean confidence on correct predictions   | 0.4486 |
| Mean confidence on incorrect predictions | 0.3730 |
| AUTO-HANDLE                              |    20% |
| ESCALATE                                 |    80% |

The conservative escalation policy intentionally favors safety when intent confidence or historical evidence is weak.

---

# 12. Top Failure Modes

## 1. Weak-label generalization

The classifier reaches 94.73% accuracy on weak-label validation but only 41.33% on the independent Golden Set.

**Hypothesis:** the model is learning the heuristic labeling function rather than the true intent boundary.

**Next step:** increase the amount of manually labelled intent data.

---

## 2. APP_PROBLEM overprediction

Generic messages containing words such as "app" frequently get classified as `APP_PROBLEM`.

**Hypothesis:** APP_PROBLEM is the largest weak-labelled class and acts as an attractive fallback for ambiguous lexical patterns.

**Next step:** rebalance the manually labelled training set and add hard negative examples.

---

## 3. OTHER is not represented in classifier training

`OTHER` examples were not used as a learned training class because the weak-labeling system uses `OTHER` primarily as an abstention/fallback category.

As a result, the classifier cannot reliably predict `OTHER`.

**Next step:** manually label sufficient `OTHER` examples and train an explicit fallback/abstention mechanism.

---

## 4. Lexical similarity is not semantic relevance

TF-IDF can retrieve tweets with shared words but different underlying problems.

Conversely, a semantically useful example can have relatively low lexical similarity.

**Hypothesis:** short Twitter messages and vocabulary variation limit bag-of-words retrieval.

**Next step:** use embedding-based retrieval followed by a lightweight reranker.

---

## 5. Context-poor and multi-intent tweets

Twitter support conversations often contain extremely short messages such as:

* "Same here"
* "It was 11.1"
* "Thanks"
* "Wales, UK"

Other messages contain several simultaneous problems such as an update causing both performance and battery issues.

**Hypothesis:** single-message classification loses conversation context and forces a multi-intent problem into a single label.

**Next step:** reconstruct conversation threads and use preceding customer/support turns during classification and retrieval.

---

# 13. What Is Misleading About My Headline Number?

The most misleading number is the **94.73% intent validation accuracy**.

It looks like excellent classification performance, but the validation set uses labels produced by the same weak-labeling rules that generated the training data.

The independent manually labelled Golden Set gives a much lower:

**41.33% accuracy / 41.68% Macro F1**

Therefore the 94.73% number demonstrates consistency with the weak-labeling strategy, not real-world intent accuracy.

Similarly, the retrieval headline of 81% should not be described as an 81% resolution rate.

It means:

> At least one of the top 5 retrieved historical cases was judged useful for 81% of manually reviewed queries.

It does not prove that the retrieved answer would solve the customer's issue.

---

# 14. LLM-as-Judge Pilot

An LLM judge was implemented using Gemini to independently score retrieval relevance using the same 0/1/2 rubric.

A small 10-example pilot was conducted because the available free API quota was limited.

9 calls completed successfully.

Human-vs-judge agreement on those 9 examples was:

**55.56%**

This result is treated only as a rubric-validation pilot, not as a statistically reliable benchmark.

Human labels remain the ground truth.

The disagreements highlighted an important issue: context-poor and multi-turn Twitter messages are difficult even for an LLM judge without conversation context.

---

# 15. Reproducibility

The complete local evaluation harness runs all major evaluations without requiring an LLM API call.

## Prerequisites

The project uses Python 3.11+.

The raw TWCS dataset is not committed to this repository because of its size. Obtain the dataset separately and place the CSV at:

```text
data/raw/twcs.csv
```

The generated ML datasets are also excluded from Git because of their size. They can be regenerated from the raw dataset.

## Setup

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Prepare the training data

Generate the customer → AppleSupport training pairs:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_ml_data_corrected.py
```

This creates:

```text
data/ml_training_data.csv
```

Then generate the weak labels:

```powershell
.\.venv\Scripts\python.exe scripts\create_weak_labels.py
```

This creates:

```text
data/weak_labeled_all.csv
data/weak_labeled_data.csv
```

`weak_labeled_data.csv` contains the high-confidence examples used to train the intent classifier.

## Run the evaluation harness

```powershell
.\.venv\Scripts\python.exe scripts\run_evaluation.py
```

The full harness currently completes in approximately:

**4 minutes on the development machine**

This is comfortably below the assignment requirement of 15 minutes.

The evaluation harness runs:

1. baseline evaluation
2. retrieval similarity evaluation
3. retrieval relevance evaluation
4. resolution-action extraction test
5. end-to-end agent evaluation

The LLM judge is intentionally not part of the default evaluation harness because it requires an external API and may be subject to free-tier quota limits.

The Golden Set (`data/golden_set.csv`) and evaluation artifacts are committed to the repository so that the reported evaluation evidence can be inspected directly.
---

# 16. Running the Agent

Start the interactive agent:

```powershell
.\.venv\Scripts\python.exe scripts\run_agent.py
```

Then enter a customer message, for example:

```text
My iPhone battery is draining very quickly
```

The agent will display:

* intent
* confidence
* top predictions
* historical evidence
* supported actions
* grounded draft response
* escalation decision

---

# 17. Project Structure

```text
SupportLens AI/
|
+-- data/
|   +-- golden_set.csv
|   +-- ml_training_data.csv
|   +-- weak_labeled_data.csv
|   +-- retrieval_review_set.csv
|   +-- agent_evaluation_50.csv
|   +-- baseline_predictions.csv
|   +-- retrieval_evaluation.csv
|   +-- retrieval_relevance_summary.csv
|   +-- resolution_action_test_results.csv
|   +-- historical_replies_sample_100.csv
|   +-- historical_replies_actions_100.csv
|   +-- llm_judge_test_10.csv
|   |
|   +-- raw/
|       +-- twcs.csv       # obtained separately
|
+-- scripts/
|   +-- run_agent.py
|   +-- run_evaluation.py
|   +-- intent_predictor.py
|   +-- retrieve_responses.py
|   +-- evidence_aggregator.py
|   +-- draft_response.py
|   +-- escalation.py
|   +-- extract_resolution_actions.py
|   +-- evaluate_agent.py
|   +-- evaluate_baselines.py
|   +-- evaluate_retrieval.py
|   +-- evaluate_retrieval_relevance.py
|   +-- test_resolution_actions.py
|   +-- llm_judge.py
|   +-- test_llm_judge.py
|   +-- prepare_ml_data_corrected.py
|   +-- create_weak_labels.py
|   +-- create_golden_set.py
|   +-- create_retrieval_review.py
|   +-- generate_grounded_response.py
|   +-- analyze_customer_evidence.py
|   +-- analyze_intents.py
|   +-- diagnose_apple_support.py
|   +-- inspect_*.py
|   +-- measure_*.py
|   +-- train_intent_model.py
|
+-- requirements.txt
+-- .gitignore
+-- README.md
```

---

# 18. Decision Log

The following decisions were intentionally made during development:

1. **AppleSupport was selected as the brand** because it provides a sufficiently large support history and a clear product domain.

2. **The customer's primary problem defines the intent** instead of assigning every possible topic mentioned in a tweet.

3. **Generic iPhone/device mentions are not automatically hardware intents.**

4. **`OTHER` is retained as an explicit fallback category** because real support traffic contains ambiguous and out-of-scope messages.

5. **Weak labels are used only to bootstrap the classifier**, not as a substitute for human evaluation.

6. **The Golden Set is locked before model evaluation** to prevent tuning directly against the test set.

7. **Golden examples are excluded from training and retrieval** to reduce leakage.

8. **TF-IDF was selected for retrieval** because it is lightweight, deterministic, locally runnable, and appropriate for a zero-cost prototype.

9. **Top-5 retrieval is used instead of blindly trusting rank 1** because lexical similarity does not always correspond to support relevance.

10. **Resolution actions are aggregated across multiple historical cases** so a single noisy example cannot determine the response.

11. **The response drafter is deterministic rather than LLM-generated** so every response can be traced to observed historical actions and reproduced without API cost.

12. **Escalation is deliberately conservative** because unsupported automated advice is more harmful than asking a human to review the case.

13. **The 94.73% weak-label validation score is explicitly not treated as the headline model result.**

14. **The retrieval 81% useful@5 number is reported as evidence availability rather than resolution accuracy.**

15. **The full evaluation harness avoids external LLM calls** so the core submission remains reproducible under a zero-cost setup.

---

# 19. One-Week Improvement Plan

If one additional week were available, the priority order would be:

### 1. Expand human-labelled intent data

Create several hundred additional labels focused on:

* APP_PROBLEM hard negatives
* OTHER
* multi-intent messages
* ambiguous cases

### 2. Improve retrieval

Replace or augment TF-IDF with embedding retrieval and a lightweight reranker.

The goal would be to improve directly relevant Top-5 retrieval rather than simply increasing lexical similarity.

### 3. Add conversation context

Reconstruct Twitter conversation threads and use previous turns when the incoming message is too short to classify independently.

### 4. Calibrate confidence

Use a separate calibration/validation set to make the AUTO-HANDLE threshold reflect actual error risk.

### 5. Improve response grounding

Associate response actions with the exact historical evidence supporting them and preserve relevant historical support links where appropriate.

---

# 20. Limitations

This is a prototype support-assistance system rather than a production customer-service agent.

Known limitations include:

* weakly supervised intent labels
* limited manually labelled intent data
* bag-of-words retrieval
* lack of full conversation context
* multi-intent customer messages
* conservative deterministic response generation
* no live customer-support integration
* no production monitoring
* limited LLM-judge validation due API quota

The evaluation results are therefore presented as evidence about the prototype rather than claims of production-level performance.

---

## License / Dataset Note

The project code is provided for evaluation purposes.

The raw Customer Support on Twitter dataset should be obtained from its original source and used according to its applicable terms. It is not included in the repository.
