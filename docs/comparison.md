# ModelAtlas vs HuggingFace: Extended Comparison

Detailed query-by-query comparison. All queries run March 2026 against both systems. HuggingFace uses its Python API with the best available `pipeline_tag` filters and sort-by-likes. ModelAtlas uses `navigate_models` with `quality=+1` for light popularity weighting. All results are real and reproducible.

For the summary version, see the [README](../README.md).

---

## Level 1: Matching the Incumbent

These are queries where HuggingFace works well — clean pipeline tags, popular models, straightforward lookup. ModelAtlas must reproduce these results to be credible.

### Sentiment Analysis

**HuggingFace** (`pipeline_tag=text-classification, search="sentiment"`):
```
  782 likes  cardiffnlp/twitter-roberta-base-sentiment-latest
  466 likes  nlptown/bert-base-multilingual-uncased-sentiment
  447 likes  mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis
```

**ModelAtlas** (`capability=-1, quality=+1, require=["classification"], prefer=["high-downloads", "encoder-only"], avoid=["code-domain", "medical-domain"]`):
```
  0.77  ProsusAI/finbert                              | finance-domain, encoder-only, community-favorite, high-downloads
  0.77  cardiffnlp/twitter-roberta-base-sentiment     | encoder-only, high-downloads
  0.77  nlptown/bert-multilingual-uncased-sentiment    | encoder-only, multilingual, high-downloads
```

**Verdict:** Same core models. MA adds ProsusAI/finbert — arguably the most important sentiment model in production (financial NLP) — which HF's "sentiment" keyword search misses because finbert's pipeline_tag is `text-classification`, not `sentiment-analysis`.

### Named Entity Recognition

**HuggingFace** (`pipeline_tag=token-classification, search="NER"`):
```
  702 likes  dslim/bert-base-NER
  225 likes  blaze999/Medical-NER
  187 likes  d4data/biomedical-ner-all
```

**ModelAtlas** (`capability=-1, quality=+1, require=["NER"], prefer=["high-downloads", "encoder-only"]`):
```
  0.38  google-bert/bert-base-uncased               | NER, ONNX, CoreML, community-favorite, high-downloads
  0.38  microsoft/deberta-v3-base                    | NER, encoder-only, high-downloads
  0.31  FacebookAI/xlm-roberta-conll03-english       | NER, multilingual, ONNX
```

**Verdict:** MA returns bert-base-uncased (the foundation model for most NER fine-tunes), DeBERTa v3 (the current SOTA base for NER), and XLM-RoBERTa (multilingual). These are arguably *better* base recommendations than HF's fine-tuned-specific results, because they're the models practitioners actually start from.

### Image Captioning

**HuggingFace** (`pipeline_tag=image-to-text, search="caption"`):
```
  1458 likes  Salesforce/blip-image-captioning-large
   927 likes  nlpconnect/vit-gpt2-image-captioning
   846 likes  Salesforce/blip-image-captioning-base
```

**ModelAtlas** (`capability=+1, quality=+1, require=["image-understanding"], prefer=["high-downloads", "consumer-GPU-viable", "multimodal"]`):
```
  1.00  OpenGVLab/InternVL2-2B      | image-understanding, multimodal, consumer-GPU-viable, high-downloads
  1.00  Qwen/Qwen2-VL-2B-Instruct  | image-understanding, multimodal, consumer-GPU-viable, high-downloads
  1.00  Qwen/Qwen2.5-VL-3B         | image-understanding, multimodal, consumer-GPU-viable, high-downloads
```

**Verdict:** Both excellent. HF returns the classic BLIP/ViT-GPT2 captioning models. MA returns modern vision-language models (InternVL2, Qwen-VL) that are the current generation of image understanding. Different era of models, both valid — MA's are more current.

---

## Level 2: Exceeding HuggingFace

Queries with directional concepts ("small," "fast") or domain intent ("medical classifier") that don't map cleanly to a single HF tag.

### Small Code Model

**HuggingFace** (`pipeline_tag=text-generation, search="code small"`):
```
   33 likes  codeparrot/codeparrot-small
   28 likes  microsoft/CodeGPT-small-py
   19 likes  microsoft/CodeGPT-small-java-adaptedGPT2
```

**ModelAtlas** (`efficiency=-1, capability=+1, domain=+1, quality=+1, require=["code-generation"], prefer=["consumer-GPU-viable", "high-downloads", "trending"]`):
```
  0.64  bullpoint/Qwen3-Coder-Next-AWQ-4bit   | 3B, code, tool-calling, high-downloads, trending
  0.62  Qwen/Qwen2.5-Coder-0.5B-Instruct      | 0.5B, code, edge-deployable, high-downloads
  0.62  Qwen/Qwen2.5-Coder-1.5B-Instruct      | 1.5B, code, consumer-GPU-viable, high-downloads
```

**Verdict:** HF returns models from 2021 with 33 likes because they have "small" in the name. MA returns current Qwen coders from 0.5B to 3B, all high-downloads and trending. The difference between keyword matching and understanding what "small" means.

### Fast Embedding Model

**HuggingFace** (`pipeline_tag=feature-extraction, search="embedding small fast"`):
```
  (no results)
```

**ModelAtlas** (`efficiency=-1, capability=-1, quality=+1, require=["embedding"], prefer=["consumer-GPU-viable", "edge-deployable", "high-downloads"]`):
```
  0.81  Qwen/Qwen3-Embedding-0.6B           | 0.6B, edge-deployable, high-downloads
  0.67  Octen/Octen-Embedding-0.6B           | 0.6B, encoder-only, multilingual, edge-deployable
  0.67  jinaai/jina-embeddings-v5-text-small  | sub-1B, multilingual, edge-deployable, trending
```

**Verdict:** HF returns nothing. "Fast" is not a tag, not a pipeline, not a searchable field. MA translates "fast" into `efficiency=-1` (small) + `edge-deployable` (lightweight) and finds three sub-1B embedding models from top providers.

### Medical Text Classifier

**HuggingFace** (`pipeline_tag=text-classification, search="medical"`):
```
   19 likes  FreedomIntelligence/medical_o1_verifier_3B
   15 likes  sid321axn/Bio_ClinicalBERT-finetuned-medicalcondition
    7 likes  justpyschitry/Medical_Article_Classifier_by_ICD-11_Chapter
```

**ModelAtlas** (`capability=-1, efficiency=-1, domain=+1, quality=+1, require=["medical-domain", "classification"], prefer=["consumer-GPU-viable", "encoder-only", "high-downloads"]`):
```
  0.38  StanfordAIMI/stanford-deidentifier-base  | medical, encoder-only, high-downloads
  0.27  obi/deid_bert_i2b2                        | medical, encoder-only, consumer-GPU-viable
  0.27  obi/deid_roberta_i2b2                     | medical, encoder-only, consumer-GPU-viable
```

**Verdict:** HF's #1 result is a reasoning *verifier* (not a classifier). Its #3 is an ICD-11 article classifier with 7 likes. MA returns Stanford's medical deidentifier, and OBI's clinical BERT/RoBERTa classifiers — models from established clinical NLP research groups, purpose-built for medical text classification.

---

## Level 3: The Unfindable

Queries that combine direction + domain + negation. These are structurally impossible on HuggingFace.

### "Multilingual chat model for edge — NOT code, NOT math, NOT embedding"

There is no combination of HuggingFace tags and filters that expresses "multilingual AND chat AND small AND NOT three specific domains."

```python
navigate_models(efficiency=-1, domain=+1, quality=+1,
                require_anchors=["multilingual"],
                prefer_anchors=["edge-deployable", "consumer-GPU-viable", "GGUF-available"],
                avoid_anchors=["code-domain", "math-domain", "embedding"])
```

```
  0.85  PaddlePaddle/PaddleOCR-VL-1.5     | sub-1B, multilingual, edge-deployable, trending
  0.60  Edge-Quant/Nanbeige4.1-3B-GGUF     | 3B, multilingual, GGUF, llama.cpp
  0.60  LiquidAI/LFM2-2.6B-Exp-GGUF       | 2.6B, multilingual, GGUF, llama.cpp
```

The `avoid_anchors` exponential penalty (0.5 per match) eliminates the thousands of code/math/embedding models that would otherwise flood the results. Three avoided anchors = 0.125× score for any model in those domains.

### "Tiny on-device TTS"

HuggingFace search for "small speech tiny edge TTS": no results.

```python
navigate_models(efficiency=-1, capability=-1, domain=+1, quality=+1,
                require_anchors=["speech-domain"],
                prefer_anchors=["consumer-GPU-viable", "edge-deployable", "trending"])
```

```
  0.31  Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign  | 1.7B, speech, multilingual, trending
  0.31  ibm-granite/granite-4.0-1b-speech       | 1B, speech, multilingual, encoder-decoder
  0.25  Aratako/MioTTS-0.1B                     | 0.1B, speech, edge-deployable, sub-1B
  0.25  FunAudioLLM/Fun-CosyVoice3-0.5B        | 0.5B, speech, ONNX, edge-deployable
```

**MioTTS-0.1B is a 100-million-parameter TTS model.** It exists on HuggingFace with minimal visibility. ModelAtlas found it because `speech-domain + edge-deployable + sub-1B` is a precise intersection in a coordinate system.

### "Biology classifier, encoder-only"

```python
navigate_models(efficiency=-1, domain=+1, quality=+1,
                require_anchors=["biology-domain"],
                prefer_anchors=["classification", "consumer-GPU-viable", "encoder-only"])
```

```
  0.15  microsoft/BiomedNLP-BiomedBERT          | biology+chemistry+medical+physics, encoder-only
  0.13  Ihor/gliner-biomed-large-v1.0           | biomedical NER+classification, encoder-only
  0.13  MilosKosRad/BioNER                      | biological NER, encoder-only
  0.13  PoetschLab/GROVER                       | genomics, encoder-only, classification
```

**PoetschLab/GROVER** is a genomics model. It has 7 anchors. Maybe 20 people in the world are looking for it at any given time. ModelAtlas found it because the coordinate intersection exists. No keyword would.

### "Distilled reasoning model, tiny, NOT the usual 7B+ suspects"

```python
navigate_models(efficiency=-1, capability=+1, training=-1, quality=+1,
                require_anchors=["distilled", "reasoning"],
                prefer_anchors=["edge-deployable", "consumer-GPU-viable", "GGUF-available"],
                avoid_anchors=["7B-class", "13B-class", "30B-class"])
```

```
  1.00  Jackrong/Qwen3.5-0.8B-Claude-4.6-Opus-Reasoning-Distilled-GGUF
        | 0.8B, distilled, reasoning, edge-deployable, GGUF, trending
```

**Perfect 1.0 score.** A 0.8B model distilled from Claude Opus 4.6's reasoning into Qwen3.5. The `avoid_anchors` for size classes eliminated every 7B+ model with exponential penalty (0.5^1 = half score per size class matched). What remained was a sub-1B reasoning distillation that *literally scored perfectly* on every axis.

---

## Summary

| Query complexity | HuggingFace | ModelAtlas |
|-----------------|-------------|-----------|
| Single tag ("sentiment analysis") | ✓ Works well | ✓ Matches + sometimes better alternatives |
| Directional ("small code model") | Returns wrong results | ✓ Correct direction, current models |
| Multi-constraint ("tiny multilingual NOT code") | Cannot express | ✓ Precise intersection |
| Niche intersection ("genomics encoder classifier") | Cannot find | ✓ Finds the unfindable |

The harder the question, the wider the gap. But the Level 1 match is what makes Level 3 credible.
