/* ==================================================================
   ▸▸▸  EDIT ONLY THIS FILE  ◂◂◂
   This is the entire content of your page. Fill in your specifics
   below and the site builds itself. You shouldn't need to touch the
   HTML or CSS at all.

   RULES
   - Leave a value as an empty string ""  to HIDE it.
     (e.g. no PubMed entry yet?  pubmed: ""  → the button disappears)
   - Keep the quotes and commas exactly as shown.
   - Lists (authors, highlights, gifs) can have any number of items —
     just add or remove blocks following the same pattern.
   ================================================================== */

const CONFIG = {

  /* ---------- 0. LOOK & FEEL  (leave any value "" to keep the default) */
  theme: {
    colorMain:   "#88503c",       // burgundy brand colour (matches site title)
    colorMainDark:   "#623f2a",
    colorSecondary:   "#5b864c",
    colorAccent: "#6d9d49",       // hover/accent burgundy
  },

  /* ---------- 0b. BRAND LOGO  (optional) ------------------------------ */
  // Put an image in assets/ to replace the default coloured mark in the
  // header + footer. `link` is where clicking the logo/title takes you.
  brandLogo: {
    src:  "assets/sycamore_tree.svg",            // e.g. "assets/logo.svg"   ("" keeps the default mark)
    link: "#top",        // e.g. "https://your-lab.org"  (default: back to top)
    alt:  "Sycamore",
  },

  /* ---------- 1. PAPER ---------------------------------------- */
  brand:        "Sycamore",                 // short name in nav + footer
  badge:        "Preprint, 2026", // small pill above title
  title:        "Sycamore: Characterizing Synthetic Personas for",          // first line of title
  titleEm:      "Evaluating Genomics Visualization Retrieval",               // accented line (set "" to skip)
  tagline:      "Sycamore studies what LLM-based synthetic personas actually produce as evaluators: ungrounded vs. grounded in real interview data vs. a real-expert baseline, using a genomics visualization search engine, [Geranium](https://gosling-lang.github.io/geranium/), as its case study.",
  venue:        "Preprint",
  year:         "2026",
  doi:          "10.48550/arXiv.2605.08630",

  /* ---------- 2. AUTHORS -------------------------------------- */
  // `aff` lists the affiliation numbers (see `affiliations` below).
  // Any link left "" is hidden for that author.
  authors: [
    { name: "Huyen N. Nguyen",   aff: [1],    website: "https://huyennguyen.com/", scholar: "https://scholar.google.com/citations?user=tsrO-ZgAAAAJ&hl=en", orcid: "https://orcid.org/0000-0001-6554-2327" },
    { name: "Astrid van den Brandt",   aff: [1],    website: "https://hidivelab.org/team/members/astrid-vandenbrandt/", scholar: "", orcid: "" },
    { name: "Nils Gehlenborg", aff: [1], website: "https://hidivelab.org/team/members/nils-gehlenborg/", scholar: "https://scholar.google.com/citations?user=YEcBVFAAAAAJ&hl=en", orcid: "https://orcid.org/0000-0003-0327-8297"},
  ],
  affiliations: [
    "Harvard Medical School, Harvard University",   // 1
  ],

  /* ---------- 3. LINKS  (leave "" to hide the button) --------- */
  links: {
    // The main preprint link used by the "Download preprint" buttons AND, by
    // default, by the PDF Preprint viewer below. It can be:
    //   • an arxiv link   -> "https://arxiv.org/pdf/2407.20571"  (embeds natively)
    //   • an OSF link     -> "https://osf.io/preprints/osf/zatw9_v7"  (see note in §10)
    //   • a local file    -> "assets/preprint.pdf"
    pdf:        "https://arxiv.org/pdf/2605.08630",
    code:       "https://github.com/huyen-nguyen/sycamore",   // GitHub repository
    pubmed:     "",   // PubMed entry
    ieeexplore: "",   // IEEE Xplore publication page
    ismb:       "",   // ISMB presentation materials
    ieeevis:    "",   // IEEE VIS 2026 presentation details
    license:    "",   // license link used in footer
    supplement: "https://osf.io/kdfr3/",
    video: "https://www.youtube.com/watch?v=h_Qm7L_C4CA",

    // OPTIONAL override for ONLY the embedded viewer (the download buttons still
    // use `pdf` above). Set this to a local copy when the remote host won't embed
    // (e.g. OSF), e.g. "assets/preprint.pdf". Leave "" to just use `pdf`.
    preprintPdf: "",
  },

  /* ---------- 3b. CUSTOM CHIPS  (extra quick-link buttons) ---- */
  customChips: [
    // { icon: "youtube", label: "YouTube", href: "https://www.youtube.com/watch?v=VIDEO_ID" },
    // { icon: "results", label: "Project page", href: "https://your-lab.org/project" },
    // { icon: "star", label: "Icons", href: "https://huyen-nguyen.github.io/iframe/icons" }
    // List of icons: https://huyen-nguyen.github.io/iframe/icons
    // No matching icon in icons.js? Paste raw SVG instead:
    // { icon: "<svg viewBox='0 0 24 24' width='24' height='24'>...</svg>", label: "Custom", href: "#" },
  ],

  /* ---------- 4. ABSTRACT  (one string per paragraph) -------- */
  abstract: [
    "Evaluating visualization systems in niche domains such as genomics is challenging due to scarcity of domain experts and difficulty recruiting a representative user base. While LLM-based synthetic personas are increasingly used to ease evaluation bottlenecks, they face well-founded skepticism. Rather than weighing synthetic personas as substitutes for real users, we ask a fundamental open question: when synthetic personas evaluate a real visualization system, what do they actually produce, and how does that output change when grounded in documented human contexts? We present Sycamore, an exploratory three-condition probe design using [Geranium](https://gosling-lang.github.io/geranium/), a search engine for multimodal genomics visualization, as a case study.",
    "Sycamore evaluates Geranium using: (1) ungrounded synthetic personas from generic LLM priors; (2) grounded synthetic personas constrained by voice-of-customer artifacts from a prior interview study; and (3) a published baseline study of real domain experts. We observe that grounding shifts synthetic feedback toward the language and concerns of documented users, while ungrounded evaluators drift toward operational specifics that real participants did not raise; both synthetic conditions, however, converge on a find-and-adapt frame and miss the image-modality preference observed in the expert study. We discuss what these observations imply for where synthetic personas might fit alongside expert studies in domain-specific visualization evaluation. All supplemental materials are available at [this https URL](https://osf.io/kdfr3/)."
  ],

  /* ---------- 5. CITATION ------------------------------------- */
  // The formatted citation is built automatically from the fields above.
  // Edit the BibTeX below directly (BibTeX needs "Last, First" name order).
  bibtex:
      `@article{nguyen2026sycamore,
  author={Nguyen, Huyen N. and van den Brandt, Astrid and Gehlenborg, Nils},
  journal={arXiv Preprint}, 
  title={Sycamore: Characterizing Synthetic Personas for Evaluating Genomics Visualization Retrieval}, 
  year={2026},
  doi={10.48550/arXiv.2605.08630},
  url={https://arxiv.org/abs/2605.08630}}`,

  /* ---------- 6. HIGHLIGHTS  (cards; add/remove freely) ------- */
  highlights: [
    { title: "A three-condition probe for synthetic evaluators", text: "Sycamore compares ungrounded LLM personas, personas grounded in interview data, and a published real-expert baseline. All three use a shared protocol to evaluate the same Geranium system for genomics visualization retrieval." },
    { title: "Grounding shapes what personas say", text: "Grounding shifts synthetic feedback toward the language and concerns of documented users, while ungrounded evaluators drift into operational specifics real participants never raised. Both *miss* the image-modality preference seen in the expert study." },
    { title: "Role in the evaluation pipeline", text: "Synthetic evaluators **cannot** replace real users, but they can be useful for (1) debugging protocols and interfaces, (2) reaching personas recruitment cannot, and (3) generating query patterns that real users can validate. They are not suitable for studying acceptance or trust, which require human judgment." },
  ],

  /* ---------- 6c. ABOUT THE NAME  (note under Highlights; "" hides it) */
  aboutName:
      "**Sycamore** stands for ***Sy**nthetic **C**h**a**racterization for Evaluating Geno**m**ics Visualizati**o**n **Re**trieval*. The bird singing in the sycamore is a nod to the song *Dream a Little Dream of Me* (1931).",

  /* ---------- 6b. TEASER FIGURE  (shown under Highlights) ----- */
  // A single overview / method figure. Drop the image in assets/.
  // Set src "" to hide the whole figure.
  teaser: {
    src:     "https://hidivelab.org/assets/img/publications/fullsize/nguyen-2026-sycamore-synthetic-personas.png",   // e.g. "assets/teaser.png"   ("" hides it)
    alt:     "System overview of Sycamore",
    caption: "Overview of the Sycamore: The viewer (left) and overall architecture (right).",
  },

  /* ---------- 7. DEMO ----------------------------------------- */
  demo: {
    youtubeId: "h_Qm7L_C4CA",     // just the id, e.g. "dQw4w9WgXcQ"  ("" hides the player)
    // gifs: [                    // drop files in assets/  ("" / empty list hides this row)
    //   { src: "assets/demo-1.gif", caption: "Fluid zoom & filter across a genomic region." },
    //   { src: "assets/demo-2.gif", caption: "Side-by-side multi-sample comparison." },
    // ],
  },

  /* ---------- 8. LOGOS  (institutional / conference) ---------- */
  // Add { src, alt, link } image objects (link is optional).
  // Empty list = show placeholder slots.
  logos: [
    { src: "https://hms.harvard.edu/themes/shared/harvardmedical/logo.svg", alt: "Harvard Medical School", link: "https://hms.harvard.edu/" },
    { src: "assets/hidivelogo.png", alt: "HIDIVE Lab", link: "https://hidivelab.org/" },

  ],
  /* ---------- 9. FOOTER / CONTACT ----------------------------- */
  contactEmail:    "huyen_nguyen@hms.harvard.edu",
  contactNote:     "Harvard Medical School",
  copyrightHolder: "The Authors",
  licenseName:     "CC BY 4.0",
  // conferenceName:  "IEEE VIS 2026",   // shown in footer ("Built for ...")

  /* ---------- 10. PDF PREPRINT VIEWER ------------------------- */
  // Controls how the "PDF Preprint" section (after the Demo) renders the file
  // from links.pdf (or links.preprintPdf if set). Options:
  //   "auto"   (default) – arxiv & local files embed directly; OSF links are
  //                        routed through Google's Docs Viewer so they display
  //                        instead of forcing a download.
  //   "direct" – always use a plain <iframe> (best quality; arxiv + local files).
  //   "google" – always route through Google's Docs Viewer (use for any host
  //              that refuses to be framed).
  //
  // NOTES
  //   • arxiv: works out of the box — paste either the /abs/ or /pdf/ link.
  //   • OSF:   paste the preprint page link (e.g.
  //              "https://osf.io/preprints/osf/zatw9_v7")
  //            and leave this on "auto". The viewer converts it to the OSF
  //            download URL and shows it via the Docs Viewer. This is best-effort
  //            (depends on Google's viewer + a public file). If it doesn't render,
  //            download the PDF into assets/ and set links.preprintPdf to it.
  preprintViewer: "auto",
};
