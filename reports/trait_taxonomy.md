# Trait Taxonomy — from this POC to a Production Customer-Trait System

**Context.** The vision for this task was a rich, many-trait customer representation, in the spirit of how
large platforms model users from many observed interactions. This document maps that vision onto what
was actually built, what could be added responsibly, and what should not be built.

**Principle.** A trait is included only if it is (1) **measured** from data the business is entitled to use,
(2) **defined** precisely enough to compute and test, and (3) **useful** for a decision. The target is not a
count of traits; it is every trait that meets those three tests. This POC has 33 — as many as this dataset supports.

## Tier A — Built in this POC (public, anonymised transaction data)

| Family | Example traits (this repo) | Count | Layer |
| --- | --- | --- | --- |
| Recency / frequency (RFM) | days since last purchase, orders in 12 months / 90 days / all history, tenure | 5 | L1–L3 |
| Monetary | spend in 90 days / 12 months / total | 3 | L2–L3 |
| Basket | average and median order value, products per order, units per order, quantity per line | 5 | L2 |
| Price | average unit price paid, price paid vs typical price | 2 | L2 |
| Diversity | distinct products, product groups, spend spread (entropy), main-group share | 4 | L2 |
| Seasonality | Christmas-product share, Oct–Dec spend share | 2 | L2–L3 |
| Loyalty | share of lines re-ordering a previously bought product | 1 | L2 |
| Cancellations | cancellation rate, cancelled value share | 2 | L2 |
| Rhythm | mean/median gap between orders, gap irregularity, overdue ratio | 4 | L3 |
| Trend / consistency | 90-day spend and order trends, 12-month slope, active months in 12 | 4 | L3 |
| Timing | share of orders placed before noon | 1 | L3 |
| Model outputs | segment, 90-day purchase propensity, decile | 3 | L4 |
| Explanations | top features raising / lowering each score, family contributions | per customer | L5 |

## Tier B — Can be added with consented first-party data (production architecture)

Each row needs a lawful basis, a clear notice, purpose limitation and a retention limit before it is built.

| Family | Traits it would add | Data needed | Condition |
| --- | --- | --- | --- |
| Engagement | visit frequency, product views before purchase, cart abandonment, search terms | own website / app event logs | cookie and tracking consent; first-party only |
| Marketing response | email open / click rates, campaign response, channel preference, opt-out status | own CRM / campaign logs | marketing consent; respect opt-outs |
| Service experience | support contacts, resolution time, return reasons | own support and returns systems | notice at collection |
| **Expressed** feedback | rating given, **stated** satisfaction, topic of a complaint, sentiment of a message the customer sent to the business | reviews, surveys, conversations with the business | consent; measure what the customer *said*, never infer what they *are* |
| Stated preferences | preferred categories, sizes, contact times — declared by the customer | preference centre / profile | customer-controlled and editable |
| Causal response | uplift: who responds *because of* a contact | randomised hold-out tests | experiment design; this turns associations into decisions |

For a conversational-AI business, the natural extension is **conversation-derived, expressed traits**: intent,
topic and sentiment of messages a customer chose to send, with notice and consent, used to serve that
customer better — not to build a hidden psychological profile.

## Tier C — Deliberately excluded

| Excluded | Why |
| --- | --- |
| Personality, emotional state, "impulsiveness", self-control or other psychological inference | Not validly measurable from purchase data; stigmatising if wrong; the customer never agreed to it. This POC describes behaviour, not minds |
| Scraped social-media or messaging data; cross-platform tracking without consent | Violates platform terms of service and lacks consent and notice |
| Inferred sensitive attributes (health, religion, politics, sexuality, caste, financial distress) | High harm if wrong or leaked; discriminatory use; should not be inferred even when statistically possible |
| Protected attributes or close proxies as model inputs | Fairness: `country` is used only for auditing in this POC, never as a model input |
| Re-identification or enrichment of anonymised records | Breaks the dataset licence's intent and individuals' reasonable expectations |

## Legal and governance notes (India) — not legal advice

India's Digital Personal Data Protection Act, 2023 is operationalised by the DPDP Rules, 2025, notified on
14 November 2025. Rules on the Data Protection Board applied immediately; the core obligations for data
fiduciaries — purpose-specific consent notices, security safeguards, breach reporting, retention and erasure —
apply 18 months after notification. A production version of this system should be designed for them now:

- **Notice and consent** for each purpose (profiling for marketing is a distinct purpose), with easy withdrawal.
- **Purpose limitation and data minimisation**: collect only what a listed trait needs.
- **Retention limits and erasure**: traits are deleted when the purpose ends or consent is withdrawn.
- **Children's data**: verifiable parental consent; no tracking, behavioural monitoring or targeted advertising
  directed at children.
- **Data-principal rights**: a customer can see and correct their profile. The layered, plain-language output
  of `get_profile` is designed to be showable to the customer themselves.
- **Accountability**: documented features, audited models, and fairness checks such as the one in this POC.

## How this POC scales

| Stage | Traits | What changes |
| --- | --- | --- |
| This POC | 33 behavioural + segment + propensity + drivers | public data, batch scoring |
| + first-party behaviour | ~60–100 | web/app events, campaigns, service data, all consented |
| + expressed feedback | + intent, topic, sentiment of customer messages | consented conversation data |
| + experiments | uplift scores per action | randomised hold-outs turn "who is likely" into "who is influenced" |

The number of traits grows with the data a business is **entitled** to use, and every new trait passes the
same three tests: measured, defined, useful.
