# crystal-0.8b — training data

crystal-0.8b is a weight merge of two fine-tunes of Qwen3.5-0.8B:
- **g6:** `v3-gold6`, 104 sources, 699,960 rows, 1 epoch
- **g7:** `v3-gold7`, 113 sources, 999,961 rows, 3 epochs (the epoch 2 checkpoint is used)

The g7 sources include every g6 source. Every row is a choice or yes/no question built from the source's training split (validation where noted). The question phrasings are our own.

- **Labels:** `human` = annotators, `programmatic` = derived by rule or generator code, `model` = fixed by the generator that built the dataset. No teacher-model distributions were used.
- **Decontamination:** training rows that overlap the Decision Index public rows or JEVal are removed (13-word shingles, plus exact matches for short texts). So are rows that overlap our fresh-v1 evaluation set.
- **Index train splits:** sources marked *train split of index benchmark* use the training split of a dataset whose test split is a Decision Index benchmark. They are disclosed here as the board's rules ask.
- **Licences:** as recorded from each dataset's card or repository when the data was built. They were not legally reviewed. Each dataset stays under its own licence. Bold marks a non-commercial condition or a missing or unverified licence.
  - Because of the non-commercial labels, the weights are released under **CC BY-NC 4.0**.

| Source | Upstream | Licence (as recorded) | Labels | Rows in g6 | Rows in g7 | Note |
|:--|:--|:--|:--|--:|--:|:--|
| `abcd-train` | github asappresearch/abcd@6b8700ce abcd_v1.1 train | MIT | human | 9,657 | 12,708 |  |
| `acos-train` | NUSTM/ACOS (data/*/*_quad_train.tsv, *_quad_dev.tsv) | **no licence file** · no licence file (research release; SemEval-2016 restaurant reviews, Amazon laptop reviews) | human | 7,513 | 9,887 | train split of index benchmark **ACOS** |
| `aegis-prompts` | nvidia/Aegis-AI-Content-Safety-Dataset-2.0@d86bb8bedff5 train.json + validation… | CC-BY-4.0 | human | 0 | 10,849 |  |
| `apibank-train` | liminghao1630/API-Bank | MIT | model | 4,541 | 5,976 | train split of index benchmark **API-Bank** |
| `aqua-rat-train` | deepmind/aqua_rat (raw) | Apache-2.0 | human | 5,897 | 7,765 |  |
| `arc-train` | allenai/ai2_arc (ARC-Easy + ARC-Challenge) | CC-BY-SA-4.0 | human | 2,770 | 3,036 |  |
| `arena55k-train` | lmarena-ai/arena-human-preference-55k@18c29834 train.csv | Apache-2.0 | human | 9,369 | 12,327 |  |
| `argq-train` | ibm-research/argument_quality_ranking_30k@590726b3 train.csv | CC-BY-SA-3.0 (card licensing text; YAML says cc-by-3.0) | human | 7,108 | 9,354 |  |
| `argsme-train` | webis/args_me@042bdf645bc9 args-me.jsonl (idebate, debatewise, debatepedia only) | CC-BY-4.0 | human | 0 | 6,177 |  |
| `banking77-train` | legacy-datasets/banking77 | CC-BY-4.0 | human | 9,661 | 9,661 | train split of index benchmark **BANKING77** |
| `boolq-train` | google/boolq | CC-BY-SA-3.0 | human | 4,684 | 6,160 |  |
| `chid-train` | thu-coai/chid@47ad8022 original/train_data.txt | Apache-2.0 | human | 7,052 | 9,280 |  |
| `civilcomments-train` | google/civil_comments@f2970eb3 data/train-*.parquet | CC0-1.0 | human | 9,358 | 12,315 |  |
| `claimstance-train` | ibm-research/claim_stance@ec4e2c2e train.csv | CC-BY-SA-3.0 (card licensing text; YAML says cc-by-3.0) | human | 895 | 910 |  |
| `clinc-train` | clinc/oos-eval data/data_full.json | CC-BY-3.0 | human | 6,702 | 8,819 | train split of index benchmark **CLINC150+OOS** |
| `cmrc-train` | hfl/cmrc2018@137f2c45 train + github DRCKnowledgeTeam/DRCD@b944790d DRCD_traini… | CMRC2018 CC-BY-SA-4.0; DRCD CC-BY-SA-3.0 | human | 6,079 | 7,993 |  |
| `code-output-gen` | generated (qwen_decision/v3/sources/code_output.py templates) | ours (generated) | programmatic | 3,445 | 4,534 |  |
| `commonsenseqa-train` | tau/commonsense_qa (default) | MIT | human | 4,852 | 6,385 |  |
| `condaqa-train` | lasha-nlp/CONDAQA train+dev | Apache-2.0 | human | 4,046 | 5,315 |  |
| `contractnli-train` | stanfordnlp/contract-nli (resources/contract-nli.zip) | CC-BY-4.0 | human | 4,743 | 4,743 | train split of index benchmark **ContractNLI** |
| `corr2cause-train` | causalnlp/corr2cause (train.csv) | MIT (GitHub causalNLP/corr2cause; no licence tag on the Hub card) | programmatic | 4,949 | 6,513 |  |
| `cosmosqa-train` | allenai/cosmos_qa (refs/convert/parquet, train) | CC-BY-4.0 | human | 7,934 | 10,445 |  |
| `creak-train` | github yasumasaonoe/creak@5e892b28 data/creak/train.json | MIT (LICENSE.txt) | human | 4,994 | 6,571 |  |
| `csqa2-train` | github allenai/csqa2@3f57a4a6 dataset/CSQA2_{train,dev}.json.gz | CC-BY-4.0 | human | 0 | 6,443 |  |
| `cuad-train` | theatticusproject/cuad-qa (Hub parquet conversion, default/train) | CC-BY-4.0 (card YAML and Licensing Information; contracts from SEC EDGAR) | human | 4,497 | 4,484 |  |
| `drop-train` | ucinlp/drop train | CC-BY-SA-4.0 | human | 13,379 | 17,606 |  |
| `ecare-train` | github Waste-Wood/e-CARE dataset/{train,dev}_full.jsonl | MIT | human | 6,498 | 8,551 |  |
| `esci-train` | amazon-science/esci-data shopping_queries_dataset | Apache-2.0 | human | 5,928 | 7,804 | train split of index benchmark **Amazon ESCI** |
| `ezstance-train` | EZ-STANCE zip from the Google Drive link in github chenyez/EZ-STANCE README (sh… | **no licence file** · no licence file (tweets; research release): review before release | human | 0 | 13,054 |  |
| `faithdial-train` | McGill-NLP/FaithDial train+valid | MIT | human | 5,602 | 7,387 |  |
| `fewnerd-train` | DFKI-SLT/few-nerd supervised train | CC-BY-SA-4.0 | human | 10,012 | 13,176 |  |
| `fewrel-train` | github thunlp/FewRel@278a2315 data/train_wiki.json + pid2name.json | MIT | human | 8,892 | 11,702 |  |
| `finer-train` | nlpaueb/finer-139 finer139.zip train (every 4th sentence) | CC-BY-SA-4.0 | human | 10,580 | 13,924 |  |
| `fintweet-train` | zeroshot/twitter-financial-news-topic + -sentiment train | MIT | human | 6,287 | 8,273 |  |
| `folio-train` | github Yale-LILY/FOLIO@5d7bb84c data/v0.0 train+validation | CC-BY-SA-4.0 | human | 1,081 | 1,085 |  |
| `gap-train` | github google-research-datasets/gap-coreference@83135f26 gap-development.tsv + … | Apache-2.0 | human | 2,190 | 2,190 |  |
| `goemotions-train` | google-research-datasets/go_emotions simplified train | Apache-2.0 | human | 19,044 | 25,062 |  |
| `gsm8k-train` | openai/gsm8k (main) | MIT | human | 5,529 | 7,276 | train split of index benchmark **GSM8K** |
| `habermas-train` | google-deepmind/habermas_machine (hm_all_candidate_comparisons.parquet) | CC-BY-4.0 (data), Apache-2.0 (code) | human | 2,004 | 2,003 | train split of index benchmark **Habermas Machine** |
| `hatexplain-train` | github punyajoy/HateXplain@01d74227 Data/dataset.json, train ids of Data/post_i… | MIT | human | 6,170 | 8,119 |  |
| `headqa-train` | dvilares/head_qa@d444ead9 data/head-qa-es-en-pdfs.zip HEAD/train_HEAD.json + HE… | MIT (dataset); exams reused under the Spanish Ministerio de Sanidad terms (no distortion, cite the source, st… | human | 3,504 | 4,612 |  |
| `hellaswag-train` | Rowan/hellaswag | MIT | human | 19,916 | 26,209 | train split of index benchmark **HellaSwag** |
| `helpsteer2-train` | nvidia/HelpSteer2 preference (train) + train attribute ratings | CC-BY-4.0 | human | 4,780 | 6,288 |  |
| `helpsteer3-train` | nvidia/HelpSteer3@f6d14577 preference/train.jsonl.gz | CC-BY-4.0 | human | 5,597 | 7,366 |  |
| `hhrlhf-train` | Anthropic/hh-rlhf@09be8c5b helpful-base + harmless-base train.jsonl.gz | MIT | human | 8,540 | 11,237 |  |
| `hover-train` | hover-nlp/hover (train release v1.1) + wiki_wo_links.db | CC-BY-SA-4.0 | human | 6,108 | 8,039 | train split of index benchmark **HoVer** |
| `humicroedit-train` | SemEval-2020 Task 7 (semeval-2020-task-7-dataset.zip, subtask-2) | **unverified** · none stated (SemEval-2020 task release; Hub mirror: unknown) | human | 5,071 | 6,674 | train split of index benchmark **Humicroedit** |
| `icl-classify` | DeveloperOats/DBPedia_Classes train; zeroshot/twitter-financial-news-topic train | DBpedia: card CC0-1.0, text from Wikipedia (CC-BY-SA, per dbpedia.org); tweets MIT | human | 8,064 | 8,065 |  |
| `infotabs-train` | github infotabs/infotabs@34e7dc09 data/maindata/infotabs_train.tsv + data/table… | Apache-2.0 (repository LICENSE) | human | 6,338 | 8,340 |  |
| `isarcasm-train` | iabufarha/iSarcasmEval (train/train.En.csv, train/train.Ar.csv) | MIT | human | 5,743 | 7,557 | train split of index benchmark **iSarcasmEval** |
| `jglue-train` | github yahoojapan/JGLUE@6f071c09 jcommonsenseqa-v1.3 + jnli-v1.3 train | CC-BY-SA-4.0 | human | 8,513 | 11,203 |  |
| `klue-train` | klue/klue@349481ec ynat, nli, re train | CC-BY-SA-4.0 | human | 8,662 | 11,399 |  |
| `kv-lookup` | generated | ours | programmatic | 3,700 | 3,700 |  |
| `lexglue-short-train` | coastalcph/lex_glue (parquet on main, ledgar, unfair_tos, case_hold) | CC-BY-4.0 | human | 7,559 | 9,948 |  |
| `lexglue-train` | coastalcph/lex_glue (parquet on main, ecthr_a, ecthr_b, scotus) | **NC labels (SCOTUS/SCDB)** · CC-BY-4.0 | human | 11,860 | 23,190 |  |
| `logic-puzzles-gen` | generated (qwen_decision/v3/sources/logic_puzzles.py) | ours (generated) | programmatic | 3,445 | 4,534 |  |
| `massive-intents` | AmazonScience/massive | CC-BY-4.0 | human | 5,298 | 6,973 |  |
| `mathqa-train` | allenai/math_qa (refs/convert/parquet, default) | Apache-2.0 | human | 5,901 | 7,790 |  |
| `maud-train` | theatticusproject/maud@37d5c3b9 MAUD_v1/MAUD_train.csv + MAUD_dev.csv | CC-BY-4.0 | human | 8,789 | 11,578 |  |
| `medmcqa-train` | openlifescienceai/medmcqa | Apache-2.0 | human | 4,950 | 6,514 |  |
| `minicheck-synth` | lytang/C2D-and-D2C-MiniCheck@1f698000d2f0 (c2d + d2c parquet) | MIT | model | 0 | 7,820 |  |
| `mmlu-aux-train` | cais/mmlu (auxiliary_train) | MIT | human | 7,040 | 9,268 |  |
| `mnli-train` | nyu-mll/multi_nli train | CC-BY-3.0 (card); genre sources per the MultiNLI paper | human | 9,480 | 12,475 |  |
| `moralstories-train` | demelin/moral_stories moral_stories_full.jsonl | MIT | human | 5,436 | 7,154 |  |
| `mtop-train` | dl.fbaipublicfiles.com/mtop/mtop.zip <lang>/train.txt (en all, 3k per other lan… | CC-BY-SA-4.0 | human | 8,752 | 11,518 |  |
| `multiwoz-train` | github budzianowski/multiwoz@fe0c8e65 data/MultiWOZ_2.2 train + schema.json | MIT | human | 8,568 | 11,275 |  |
| `musique-train` | bdsaglam/musique (official MuSiQue v1.0 files: musique_ans_v1.0_train.jsonl, mu… | CC-BY-4.0 | human | 7,936 | 11,621 |  |
| `newyorker-matching-train` | jmhessel/newyorker_caption_contest (matching, cross-validation split 0) | CC-BY-4.0 | human | 4,878 | 6,425 | train split of index benchmark **New Yorker caption matching** |
| `nli4ct-train` | ai-systems/Task-2-SemEval-2024 (training_data.zip) | **no licence file** · no licence file (SemEval-2024 Task 2 data; CTRs from ClinicalTrials.gov) | human | 1,680 | 1,674 | train split of index benchmark **NLI4CT** |
| `nvd-cve` | NVD CVE JSON 2.0 yearly feeds (sha256 per year below; NVD updates them daily) +… | NVD data: no licensing restrictions (NIST); CVE descriptions: MITRE CVE terms (free use with notice); CWE: MI… | human | 0 | 13,161 |  |
| `oasst2-train` | OpenAssistant/oasst2@179dd21f data/train parquet | Apache-2.0 | human | 7,885 | 10,376 |  |
| `openbookqa-train` | allenai/openbookqa (main) | **unverified** · unknown on the Hub (AllenAI release Apache-2.0; unverified) | human | 3,425 | 4,506 |  |
| `paws-train` | google-research-datasets/paws labeled_final train | free for any purpose (google-research-datasets/paws LICENSE) | human | 7,053 | 9,281 |  |
| `phishing` | kudzaiprichard/aura-phishing-email-corpus; Dizzzy0x00/LLMGen-Phishing-Email-Dat… | CC-BY-4.0 (Zenodo record 8339691); Apache-2.0 | <function _phishing_origin at 0x7ba415b7e340> | 4,137 | 5,441 |  |
| `poem-pairs-gen` | biglam/gutenberg-poetry-corpus@fcd42e249fed (CC0); github cmusphinx/cmudict@747… | public-domain poems; CC0 corpus; generated perturbations (ours) | programmatic | 0 | 8,546 |  |
| `pstance-train` | P-Stance CSVs from the Google Drive folder in github chuchun8/PStance README (t… | repository MIT; tweets (Twitter terms) | human | 0 | 9,144 |  |
| `pubhealth-train` | PUBHEALTH train.tsv (Google Drive 1eTtRs5cUlBP5dXsx-FTAlmXuB6JQi2qj, linked fro… | MIT | human | 4,899 | 6,448 |  |
| `pubmedqa-labeled` | qiaojin/PubMedQA (pqa_labeled) | MIT | human | 895 | 895 |  |
| `quac-train` | https://s3.amazonaws.com/my89public/quac/train_v0.2.json | CC-BY-SA-4.0 | human | 8,516 | 11,207 |  |
| `quality-train` | tasksource/QuALITY (mirror of QuALITY v1.0.1, data/train-*.parquet) | CC-BY-4.0 (dataset); articles: Project Gutenberg / OANC (Slate) / CC-BY-4.0 | human | 2,263 | 2,262 |  |
| `quoref-train` | quoref-train-dev-v0.1.zip (quoref-dataset S3) quoref-train-v0.1.json | CC-BY-4.0 | human | 6,150 | 8,096 |  |
| `ragtruth-train` | ParticleMedia/RAGTruth | MIT annotations; source texts CNN/DM, MS MARCO, Yelp (training only) | human | 5,257 | 6,918 | train split of index benchmark **RAGTruth** |
| `redial-train` | community-datasets/re_dial@eb588549 data/train-00000-of-00001.parquet | CC-BY-4.0 | human | 8,073 | 10,623 |  |
| `ropes-train` | allenai/ropes train+validation | CC-BY-4.0 | human | 4,993 | 6,565 |  |
| `scitail-train` | allenai/scitail dgem_format train | Apache-2.0 (github allenai/scitail LICENSE) | human | 7,577 | 9,960 |  |
| `scruples-train` | allenai/scruples v1.0 anecdotes + dilemmas (train, dev) from the README's stora… | Apache-2.0 (repository); anecdotes are Reddit posts | human | 0 | 12,774 |  |
| `semcor-train` | nltk_data@550b6625 corpora/semcor.zip (SemCor 3.0) + corpora/wordnet.zip (WordN… | Princeton WordNet licence (SemCor 1.6/WordNet; SemCor 3.0 mapping by R. Mihalcea, same terms) | human | 8,655 | 11,382 |  |
| `sentfin-train` | github.com/pyRis/SEntFiN SEntFiN.csv (human annotations only) | MIT (repository LICENSE) | human | 5,967 | 7,853 |  |
| `sharc-train` | sharc1-official.zip json/sharc_train.json | CC-BY-SA-3.0 | human | 7,370 | 9,698 |  |
| `siqa-train` | allenai/social_i_qa (refs/convert/parquet, train) | CC-BY-4.0 (AllenAI release; Hub card untagged) | human | 9,137 | 12,024 |  |
| `snips-train` | github sonos/nlu-benchmark@b86ac7f1 2017-06-custom-intent-engines/<Intent>/trai… | CC0-1.0 | human | 5,819 | 7,658 |  |
| `snli-train` | stanfordnlp/snli@cdb5c3d5 plain_text train (30k sampled) | CC-BY-SA-4.0 | human | 8,661 | 11,398 |  |
| `spartqa-train` | tasksource/spartqa-yn@150c819e + tasksource/spartqa-mchoice@d29ba456 train | MIT (HLR/SpartQA_generation) | programmatic | 7,054 | 9,283 |  |
| `squad2-answerability` | rajpurkar/squad_v2 | CC-BY-SA-4.0 | human | 4,898 | 6,442 |  |
| `stepgame-train` | ZhengyanShi/StepGame@6d859381 train.jsonl, wrong-template stories removed per S… | MIT | programmatic | 6,092 | 8,016 |  |
| `strategyqa-train` | ChilleD/StrategyQA | MIT | human | 1,443 | 1,443 |  |
| `svamp-train` | ChilleD/SVAMP@5e0bf1e5 data/train parquet | MIT | human | 629 | 629 |  |
| `tabfact-train` | github wenhuchen/Table-Fact-Checking@2ab782ba collected_data r1+r2, train_id ta… | MIT (repository LICENSE); tables from Wikipedia (CC-BY-SA) | human | 14,924 | 19,639 |  |
| `taskmaster-train` | github google-research-datasets/Taskmaster@d92cb6af TM-1-2019 train | CC-BY-4.0 | human | 2,164 | 2,162 |  |
| `tatqa-train` | github.com/NExTplusplus/TAT-QA dataset_raw (train, dev; test never read) | CC BY 4.0 (README) | human | 5,883 | 7,742 |  |
| `tatqa-yesno` | TAT-QA train+dev (as finance.py) | CC-BY-4.0 | human | 11,768 | 13,703 |  |
| `tldr-feedback-train` | openai/summarize_from_feedback refs/convert/parquet@a4d82a10 comparisons/train | Modified MIT (feedback data); Reddit TL;DR posts CC-BY-4.0 (Webis-TLDR-17) | human | 8,635 | 11,363 |  |
| `tools-percall` | Team-ACE/ToolACE; glaiveai/glaive-function-calling-v2 | Apache-2.0; Apache-2.0 | model | 7,714 | 10,140 |  |
| `tydiqa-train` | google-research-datasets/tydiqa@da78f23f primary_task train | Apache-2.0 | human | 8,625 | 11,352 |  |
| `ukpconvarg-train` | github UKPLab/acl2016-convincing-arguments@ca9d24e4 data/UKPConvArg1Strict-CSV | CC-BY-4.0 (arguments: createdebate CC-BY-3.0, convinceme public domain) | human | 5,374 | 7,072 |  |
| `vast-train` | emilyallaway/zero-shot-stance (data/VAST) | **no licence file** · no licence file (research release; NYT comments) | human | 5,911 | 7,776 | train split of index benchmark **VAST** |
| `vitaminc-train` | tals/vitaminc train (real revisions only) | CC-BY-SA-3.0 | human | 8,609 | 11,331 |  |
| `wanli-train` | alisawuffles/WANLI | CC-BY-4.0 | human | 7,049 | 9,276 |  |
| `when2call-train` | nvidia/When2Call | CC-BY-4.0 | model | 6,736 | 8,868 | train split of index benchmark **When2Call** |
| `winogrande-train` | allenai/winogrande (winogrande_xl) | Apache-2.0 (GitHub release; Hub card blank) | human | 20,130 | 26,490 | train split of index benchmark **WinoGrande** |
| `wned-el` | facebook/kilt_tasks wned validation | **upstream WNED: no licence** · KILT MIT; text from Wikipedia (CC-BY-SA) | human | 3,049 | 3,010 |  |
| `wtq-train` | WikiTableQuestions v1.0.2 compact zip (GitHub release) data/training.tsv | CC-BY-SA-4.0 (repository LICENSE) | human | 5,411 | 7,119 |  |
| `zeshel-train` | github lajanugen/zeshel (Google Drive archive) mentions/train | CC-BY-SA (Fandom wikis) | human | 11,080 | 14,581 |  |
