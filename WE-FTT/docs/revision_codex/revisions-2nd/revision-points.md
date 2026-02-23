

# Response to Reviewers (Minor Revision)

**Manuscript title:** *Resolving signal ambiguity in earthquake precursor detection with an environment-specific deep learning framework for AMSR-2 data*

We sincerely thank the Associate Editor and Reviewer #1 for their constructive comments. We have revised the manuscript to strengthen the physical interpretation, especially regarding (i) the **near-surface sensing limitation** of C-band passive microwave observations, and (ii) a **quantitative, first-order comparison** between soil-moisture-driven dielectric variability and plausible stress-related dielectric perturbations. All changes will be highlighted in the revised manuscript (Track Changes).

---

## Associate Editor (Joseph Awange)

### 1) Comment (verbatim, unchanged)

Associate Editor (Joseph Awange): Minor revision is recommended.

### 2) Response

We thank the Associate Editor for the recommendation. Following this guidance, we revised the manuscript mainly to (1) improve the physical soundness of our interpretation by explicitly stating the C-band near-surface sensing depth limitation, (2) add a quantitative sensitivity comparison (soil moisture vs. plausible stress-induced dielectric perturbations), and (3) remove/soften wording that could be interpreted as deep “subsurface” sensing or quasi-tomographic capability.

### 3) Change in the manuscript

* **Section 2.1**: Removed the “quasi-tomographic” wording and clarified that low-frequency channels are primarily sensitive to **cm-scale near-surface** dielectric variability (top few centimeters) over land.
* **Section 4.2.1–4.2.2**: Revised the physical mechanism discussion to avoid claims of deep subsurface sensing; reframed mechanism statements (e.g., P-hole) as **plausible interpretations** rather than definitive proof.
* **Section 4.2.2** (new text): Added a **first-order Fresnel emissivity sensitivity calculation** that quantitatively contrasts the TB magnitude expected from (a) soil moisture changes and (b) percent-level dielectric perturbations potentially related to stress. (Full insertion text provided below under Reviewer #1, Comment 2.)

---

## Reviewer #1

### Comment 1

#### 1) Comment (verbatim, unchanged)

Reviewer #1: Thanks for the revisions and responses. This work is potentially important and valuable. I would recommend another round of revision to ensure the methods and results are physically sounding.

#### 2) Response

We sincerely thank the reviewer for the positive evaluation and for stressing the importance of physical consistency. In this revision we focused on (i) explicitly clarifying the physical sensing depth of AMSR-2 channels over land, (ii) adding a quantitative, physically grounded sensitivity comparison requested by the reviewer, and (iii) revising phrasing that could be read as deep subsurface sensing or overly strong mechanism attribution.

#### 3) Change in the manuscript

* We revised the physical interpretation sections (4.2.1–4.2.2) and removed “quasi-tomographic”/deep-subsurface implications.
* We added a dedicated quantitative comparison paragraph (and an optional small table) in Section 4.2.2 (text provided below).

---

### Comment 2

#### 1) Comment (verbatim, unchanged)

My main concern is still the physical basis of this work. Let's just focus on land area, whose surface dielectric properties are highly variable due to soil moisture, vegetation, and snow properties. Could the authors provide quantitative comparisons of the dielectric changes caused by (a) soil moisture (e.g., from 0.05 cm3/cm3 to 0.40 cm3/cm3 or/and the opposite) and (b) earthquake precursor signals or stress changes? Please note that C-band is only sensitive to surface 1- to 5-cm soil layer dielectric properties under sparsely vegetated conditions, so the comparisons should be made for the surface soil layer.

#### 2) Response

We appreciate this key comment and fully agree that, over land, **soil moisture–driven dielectric variability** is the dominant confounder. To address the reviewer’s request without introducing new field experiments, we added a **quantitative first-order sensitivity analysis** based on Fresnel emissivity physics and representative near-surface permittivity values consistent with established microwave soil dielectric mixing models.

**(a) Soil moisture effect (surface 1–5 cm):**
For near-surface volumetric soil moisture changing from 0.05 cm3/cm3 (dry) to 0.40 cm3/cm3 (wet), the real part of soil permittivity typically increases from **ε′ ~ O(4)** to **ε′ ~ O(20–30)** in the C-band. Using Fresnel emissivity at a representative AMSR-2 incidence angle (~55°) and Ts=300 K, this implies a **very large TB decrease** at 6.9 GHz on the order of **tens to ~100 K**, depending on polarization (H-pol being particularly sensitive).

**(b) Stress-related dielectric perturbation (surface manifestation):**
Laboratory measurements of stress-dependent dielectric properties at microwave frequencies report **percent-level** perturbations of ε′ under compressive loading for common minerals/rocks (e.g., Mao et al., 2020b). Using the same Fresnel sensitivity, a conservative **1–5%** perturbation of ε′ produces only **O(1–a few K)** changes in TB.

Therefore, the land-surface soil-moisture-driven dielectric effect is expected to be **one to two orders of magnitude larger** than plausible stress-related dielectric perturbations detectable by C-band passive radiometry. This supports the need for our **environment-specific stratification and statistical background screening** (support-difference confidence intervals and global random-date false-positive mapping), which are designed precisely to suppress background variability over land.

Finally, we explicitly incorporated the reviewer’s point about sensing depth: over land, the C-band channels in AMSR-2 are sensitive mainly to the **very shallow near-surface layer (order of centimeters, ~1–5 cm under sparse vegetation)**, so our quantitative comparison and revised mechanism discussion are strictly framed at the near-surface/skin-layer level.

#### 3) Change in the manuscript

**(A) Section 2.1: remove “quasi-tomographic / deeper penetration into the soil column” and clarify “near-surface (cm-scale)”**

* **Original sentence (current manuscript):**
  “The multi-frequency nature of AMSR-2 is particularly advantageous, as lower frequencies (e.g., 6.9 GHz) provide deeper penetration into the soil column, while higher frequencies (e.g., 89.0 GHz) are more sensitive to surface skin temperature and atmospheric effects, enabling a quasi-tomographic analysis of potential pre-seismic phenomena.”

* **Replace with (revised text to paste):**
  “The multi-frequency nature of AMSR-2 provides complementary sensitivity to land-surface states. Over land, low-frequency channels (e.g., 6.9 GHz) are relatively more sensitive to **cm-scale near-surface** dielectric/soil-moisture variability (typically the top few centimeters, ~1–5 cm under sparsely vegetated conditions), whereas higher frequencies (e.g., 89.0 GHz) are more influenced by surface skin temperature, vegetation, and atmospheric contributions. This spectral complementarity supports context-aware screening of candidate pre-seismic perturbations, but **does not imply deep subsurface sensing**.”

**(B) Section 4.2.2: add a new quantitative paragraph (and optional table) answering the “quantitative comparisons” request**
Insert the following paragraph at the beginning of **Section 4.2.2 (Separating precursor signals from background variability)**, right after its first sentence (or as the first paragraph of 4.2.2). The original section begins at.

* **[New text to insert in Section 4.2.2]**
  “**Quantitative magnitude comparison (soil moisture vs. stress-related dielectric perturbations).** Over land, the observed brightness temperature (TB) variability is strongly controlled by near-surface dielectric changes driven by soil moisture. To quantify the relative magnitude, we provide a first-order sensitivity estimate using Fresnel emissivity for a smooth dielectric half-space at a representative AMSR-2 incidence angle (θ≈55°), with (TB_p \approx e_p T_s) and (e_p = 1 - R_p), where (R_p) is the Fresnel power reflectivity for polarization (p). Using representative near-surface permittivity endpoints consistent with established microwave dielectric mixing models, a soil moisture change from (mv=0.05) to (mv=0.40) can increase the real permittivity from approximately (\varepsilon' \approx 4) (dry) to (\varepsilon' \approx 25) (wet). For (T_s=300) K, this implies a TB decrease of approximately **106 K (H-pol)** and **68 K (V-pol)** at 6.9 GHz. In contrast, laboratory studies report that stress-dependent dielectric perturbations of common minerals/rocks at microwave frequencies are typically at the **percent level** (e.g., Mao et al., 2020b). A conservative **1–5%** perturbation in (\varepsilon') around (\varepsilon' \approx 10) yields only **~0.6–2.8 K (H-pol)** and **~0.4–1.9 K (V-pol)** TB changes in the same Fresnel sensitivity calculation. Although real TB is influenced by roughness, vegetation optical depth, and atmospheric emission, this order-of-magnitude contrast illustrates why soil moisture dominates land TB variability, and motivates the need for our environment-specific stratification and statistical background screening to suppress background confounders.”

* **Optional (recommended) small table:** Add a compact table (either in main text or Supplementary) summarizing the above two scenarios (mv endpoints vs. percent-level ε′ perturbations). This is not a new experiment—just a summary of the above calculation.

---

### Comment 3

#### 1) Comment (verbatim, unchanged)

In addition, the authors mentioned that "In wetland and arid zones (Zones D and E), the significant detection capabilities of low-frequency channels (e.g., 6.9 GHz H-pol) strongly support the P-hole activation hypothesis, where stress-induced positive charge carriers migrate from the crust to the surface, modifying subsurface dielectric properties". So here what is the depth of "subsurface"? Since AMSR is only sensitive to the very shallow surface and anything changed beneath may not be captured.

#### 2) Response

We agree with the reviewer that our previous wording was ambiguous and could be interpreted as implying that AMSR-2 senses deeper subsurface layers. This was not our intention. We have revised the manuscript to explicitly state that AMSR-2 C-band observations are sensitive mainly to the **very shallow near-surface/skin layer (order of centimeters)** over land, and therefore any deeper lithospheric process (including the P-hole hypothesis) can only be discussed in terms of its **surface manifestation** detectable by passive microwave emissivity changes.

Accordingly, we replaced “subsurface dielectric properties” with “**near-surface (skin-layer) dielectric/emissivity properties**” and removed “quasi-tomographic” wording. We also softened the strength of the mechanism claim by framing the P-hole hypothesis as a **plausible interpretation** rather than definitive evidence.

#### 3) Change in the manuscript

**(A) Section 4.2.1: rewrite the P-hole paragraph to remove subsurface depth implication and tone down causal certainty**
The current text contains the quoted sentence and “quasi-tomographic” claim here:.

* **Original (current manuscript, excerpt):**
  “In wetland and arid zones (Zones D and E), the significant detection capabilities of low-frequency channels (e.g., 6.9 GHz H-pol) strongly support the P-hole activation hypothesis, where stress-induced positive charge carriers migrate from the crust to the surface, modifying subsurface dielectric properties (Freund, 2011; Mao et al., 2020b). The deeper penetration of lower frequencies enables quasi-tomographic monitoring of these charge distributions, …”

* **Replace the whole Section 4.2.1 paragraph (recommended revised version to paste):**
  “The observed zone-dependent frequency–polarization responses (Table 1) likely reflect differences in microwave radiative transfer and in how potential seismo-lithospheric processes, if any, couple to the Earth’s surface. Over land (Zones D and E), the usefulness of low-frequency channels (e.g., 6.9 GHz H-pol) is **consistent with** mechanisms that modulate **near-surface (skin-layer) dielectric/emissivity properties**, which are the portion of the land surface sensed by C-band passive microwave radiometry (order of centimeters under sparsely vegetated conditions). One plausible coupling pathway discussed in the literature is the activation and migration of positive charge carriers (P-holes) under stress (Freund, 2011), which could alter the electrical state and effective emissivity of the very shallow surface layer. However, we emphasize that AMSR-2 does **not** directly sense deeper subsurface changes; any deep process must manifest through **near-surface** dielectric/emissivity changes to be observable. In marine environments (Zone A), the dominant sensitivity of high-frequency channels (e.g., 89 GHz) to sea-surface roughness and skin temperature can produce distinct spectral–polarization signatures, potentially associated with sea-state/thermal perturbations (Liu et al., 2023b). The polarization dependence in Figure 5 provides additional discriminatory information, as different physical perturbations can imprint different H/V emissivity responses.”

---

### Comment 4

#### 1) Comment (verbatim, unchanged)

My feeling is that the authors had interesting findings but with wrong explanations.

#### 2) Response

We appreciate this candid assessment and agree that the explanation must be physically conservative and consistent with the sensing physics of passive microwave radiometry. In the revision, we have:

1. **softened** deterministic language (e.g., “strongly support”) and reframed the mechanism discussion as **plausible interpretations**;
2. removed wording that could imply deep “subsurface” sensing or “quasi-tomographic monitoring”;
3. added a quantitative sensitivity comparison showing that soil moisture–driven dielectric variability can produce O(10^2 K) TB changes, whereas plausible stress-related dielectric perturbations are expected to be O(1–few K) at the near-surface sensing depth;
4. expanded the limitations to state explicitly that our framework detects **statistically robust patterns** but does not uniquely identify a single physical mechanism.

#### 3) Change in the manuscript

* **Section 4.2.1 and 4.2.2**: revised mechanism wording and removed “quasi-tomographic”/deep-subsurface implications.
* **Section 4.2.2**: inserted the new quantitative comparison paragraph (provided above).
* **Section 3.2 (Results)**: replaced wording that implies deep subsurface mechanisms.

  * **Original sentence (current manuscript):** “... indicating complex subsurface precursor mechanisms.”
  * **Replace with:** “... indicating complex **near-surface coupled** precursor processes.”
* **(Optional but recommended) Section 4.4 (Limitations):** add one sentence explicitly:
  “Because AMSR-2 primarily senses the near-surface (cm-scale) layer over land, the physical mechanisms discussed here should be interpreted as plausible surface-coupling pathways rather than definitive evidence of deep subsurface processes.”
